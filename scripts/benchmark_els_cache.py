"""GPU benchmark: exact trace ranking replay and controlled, real ELS search.

The controlled search uses a restricted pool, NOT a full 36-layer ELS run.
Q is collected once; no large scores or Q tensors are written to disk.
"""
import argparse
import gc
import hashlib
import json
import os
import socket
import subprocess
from pathlib import Path
import time


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--reference-run', type=Path)
    p.add_argument('--pool', type=int, nargs='+', default=[6, 22, 24])
    p.add_argument('--paired-repeats', type=int, default=2,
                   help='Pairs: first baseline/cache, second cache/baseline, etc.')
    p.add_argument('--replay-only', action='store_true')
    return p


def candidates_from_reference(path):
    if path is None:
        return [[24], [22], [6], [22, 24], [6, 24], [6, 22, 24]], 'fixed_fallback'
    progress = path / 'els_progress.json'
    fit = path / 'fit.json'
    if progress.exists():
        trace = json.loads(progress.read_text())
        source = str(progress)
    elif fit.exists():
        trace = json.loads(fit.read_text())['els']['trace']
        source = str(fit)
    else:
        raise FileNotFoundError(f'No els_progress.json or fit.json in {path}')
    candidates = []
    for row in trace:
        key = tuple(sorted(row['layers']))
        if key:
            candidates.append(list(key))
    if not candidates:
        raise ValueError('Reference trace contains no candidates')
    return candidates, source


def main():
    args = parser().parse_args()
    if args.paired_repeats < 1 or len(set(args.pool)) != len(args.pool):
        raise ValueError('Require positive paired repeats and a unique pool')
    # Keep --help and module import independent of the GPU environment.
    import torch
    from blade_plus_iti import atomic_json, file_hash, legacy_split, median_positive
    from ttsafety.behaviors import bestfirst_layers
    from ttsafety.els_cache import ELSCandidateCache
    from ttsafety.eval import load_c4_text, teacher_forced_ppl
    from ttsafety.generic_importance import collect_c4_generic_importance
    from ttsafety.hooks import capture_last_token, get_decoder_layers
    from ttsafety.iti_composition import collect_writer_means, generate_batch, qwen_no_thinking, tensor_dict_hash
    from ttsafety.models import env_info, load_model
    from ttsafety.sycophancy import score_edges, score_edges_g
    from ttsafety.weight_prune import rank_weight_indices, selection_from_ranking

    # Snapshot provenance before this invocation creates its result files.
    repo_root = Path(__file__).resolve().parents[1]
    def git_output(*argv):
        return subprocess.check_output(['git', '-C', str(repo_root), *argv], text=True).strip()
    startup_status = git_output('status', '--porcelain')
    tracked_status = git_output('status', '--porcelain', '--untracked-files=no')
    git_provenance = {
        'commit': git_output('rev-parse', 'HEAD'),
        'status_porcelain_at_start': startup_status,
        'dirty_at_start': bool(tracked_status),
        'dirty_scope': 'tracked files; untracked runner logs/results excluded',
        'tracked_status_porcelain_at_start': tracked_status,
        'recorded_before_result_creation': True,
    }
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if (args.run_dir / 'summary.json').exists():
        raise FileExistsError('Use a fresh run directory; benchmark results will not be overwritten')
    torch.manual_seed(0)
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '8')))
    if not torch.cuda.is_available():
        raise RuntimeError('This benchmark requires CUDA')

    def sync():
        torch.cuda.synchronize()

    def timed(fn):
        sync()
        begin = time.perf_counter()
        value = fn()
        sync()
        return value, time.perf_counter() - begin

    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    summary = {'status': 'running', 'git': git_provenance, 'hostname': socket.gethostname(), 'coverage': {
        'replay': 'ranking-only actual reference candidates or explicitly labeled fixed fallback',
        'controlled': 'real bestfirst_layers with restricted candidate pool; NOT full 36-layer end-to-end',
        'pool': args.pool, 'solo_screening': False, 'iti': False},
        'settings': {'Q_mode': 'g1scalar', 'Q_max_tokens': 65536, 'Q_seqlen': 2048,
                     'Q_batch_size': 2, 'generation_batch_size': 16, 'max_new_tokens': 128,
                     'ppl_max_tokens': 5000, 'ppl_window': 1024, 'ppl_batch_size': 8,
                     'beta': .05, 'eps': .005, 'test_frac': .005,
                     'final_rho': .01, 'final_rank_fraction': .05, 'thinking': False,
                     'final_mask_comparison_scope': 'same zero-clamped ELS scorer at rho=.01; not production strict-positive final-mask audit',
                     'paired_repeats': args.paired_repeats, 'torch_threads': torch.get_num_threads()},
        'replay': [], 'controlled_runs': [], 'comparisons': []}

    def save():
        atomic_json(args.run_dir / 'summary.json', summary)

    try:
        inputs = json.loads(args.inputs.read_text())
        candidates, source = candidates_from_reference(args.reference_run)
        summary.update(inputs_hash=file_hash(args.inputs), model=inputs['model'],
                       replay_source=source, replay_candidates=candidates,
                       replay_candidate_count=len(candidates),
                       replay_unique_count=len({tuple(x) for x in candidates}),
                       replay_reference_hash=file_hash(source) if source != 'fixed_fallback' else None,
                       source_hashes={str(p): file_hash(p) for p in [Path(__file__),
                           Path(__file__).parent / 'blade_plus_iti.py',
                           Path(__file__).parents[1] / 'src/ttsafety/els_cache.py',
                           Path(__file__).parents[1] / 'src/ttsafety/behaviors.py']})
        save()
        (model, tok), load_seconds = timed(lambda: load_model(inputs['model']))
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        assert '</think>' in qwen_no_thinking(tok, 'What is the capital of France?')
        layers = list(range(len(get_decoder_layers(model))))
        if not set(args.pool).issubset(layers):
            raise ValueError('Pool outside model layers')
        summary.update(env=env_info(), model_commit=getattr(model.config, '_commit_hash', None),
                       model_hash_source='Hugging Face config commit; no full weight-file hash',
                       template_hash=hashlib.sha256(tok.chat_template.encode()).hexdigest(),
                       load_seconds=load_seconds, actual_model_layers=len(layers))
        train, select, _ = legacy_split(inputs['contrast_rows'])
        unc = [r['question'] for r in train if r['label'] == 1]
        cert = [r['question'] for r in train if r['label'] == 0]
        prompts = [r['question'] for r in select if r['label'] == 1]
        summary['select_uncertain_count'] = len(prompts)
        setup_start = time.perf_counter()
        au = capture_last_token(model, tok, [qwen_no_thinking(tok, p) for p in unc], batch_size=16)
        ac = capture_last_token(model, tok, [qwen_no_thinking(tok, p) for p in cert], batch_size=16)
        directions = {i: au[i].mean(0) - ac[i].mean(0) for i in layers}
        del au, ac
        mu_a = collect_writer_means(model, tok, unc, layers)
        mu_b = collect_writer_means(model, tok, cert, layers)
        c4 = load_c4_text()
        print('Collecting Q once with original g1scalar settings', flush=True)
        (q, qmeta), qseconds = timed(lambda: collect_c4_generic_importance(
            model, tok, layers, 'both', text=c4, seqlen=2048, batch_size=2,
            mode='g1scalar', max_tokens=65536))
        raw = score_edges(model, directions, mu_a, mu_b, layers, 'both')
        lam = median_positive(raw) / median_positive(q)
        del raw
        gc.collect()
        sync()
        summary.update(Q=qmeta, Q_actual_tokens=qmeta['n_tokens'], Q_seconds=qseconds, lambda_value=lam,
                       c4_text_hash=hashlib.sha256(c4.encode()).hexdigest(),
                       setup_seconds=time.perf_counter() - setup_start)
        save()

        def make_ranker(cached):
            state = {'score_seconds': 0., 'score_calls': 0}

            def score(m, d, a, b, ls, comp):
                def work():
                    raw = score_edges_g(m, d, a, b, ls, comp, Q=q, lam=lam, abstain=True)
                    return {k: torch.where(torch.isfinite(v), v, torch.zeros_like(v)) for k, v in raw.items()}
                result, elapsed = timed(work)
                state['score_seconds'] += elapsed
                state['score_calls'] += 1
                return result

            cache, construction_seconds = timed(lambda: ELSCandidateCache(score) if cached else None)
            state['construction_seconds'] = construction_seconds

            def rank(m, d, a, b, ls, comp, max_fraction):
                if cache is not None:
                    return cache.rank(m, d, a, b, ls, comp, max_fraction)
                return rank_weight_indices(score(m, d, a, b, ls, comp), max_fraction)
            return rank, state, cache

        def rank_call(rank, ls, fraction=.01):
            return rank(model, directions, mu_a, mu_b, ls, 'both', fraction)

        def same_selection(a, b):
            return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)

        def ranking_equal(a, b):
            return set(a) == set(b) and all(
                torch.equal(a[k], b[k]) if torch.is_tensor(a[k]) else a[k] == b[k] for k in a)

        # Warm GPU scoring kernels with an independent disposable instance.
        warm_rank, _, _ = make_ranker(False)
        warm, warm_seconds = timed(lambda: rank_call(warm_rank, args.pool[:1]))
        del warm, warm_rank
        summary['ranking_warmup_seconds_excluded'] = warm_seconds
        baseline, bs, _ = make_ranker(False)
        cached, cs, cache = make_ranker(True)
        for phase in ('cold', 'warm'):
            for i, ls in enumerate(candidates):
                outputs, row = {}, {'phase': phase, 'layers': ls}
                for mode in (('baseline', 'cache') if i % 2 == 0 else ('cache', 'baseline')):
                    rank, state = (baseline, bs) if mode == 'baseline' else (cached, cs)
                    before = state['score_seconds']
                    torch.cuda.reset_peak_memory_stats()
                    outputs[mode], seconds = timed(lambda: rank_call(rank, ls))
                    score_seconds = state['score_seconds'] - before
                    row[mode] = {'total_seconds': seconds, 'score_seconds': score_seconds,
                                 'ranking_and_bookkeeping_seconds': seconds - score_seconds,
                                 'peak_cuda_bytes': torch.cuda.max_memory_allocated()}
                def compare():
                    a, b = outputs['baseline'], outputs['cache']
                    sa, sb = selection_from_ranking(a, .005), selection_from_ranking(b, .005)
                    return {'ranking_exact': ranking_equal(a, b),
                            'mask_exact': same_selection(sa, sb), 'mask_hash': tensor_dict_hash(sa)}
                checks, overhead = timed(compare)
                row.update(checks, comparison_seconds_excluded=overhead, cache_stats=dict(cache.stats))
                summary['replay'].append(row)
                save()
                print(f'Replay {phase} {ls}: exact={checks}', flush=True)
                if not checks['ranking_exact'] or not checks['mask_exact']:
                    raise AssertionError('Replay ranking or mask mismatch')
                del outputs
        summary['replay_totals'] = {}
        for phase in ('cold', 'warm'):
            rows = [r for r in summary['replay'] if r['phase'] == phase]
            b = sum(r['baseline']['total_seconds'] for r in rows) + (bs['construction_seconds'] if phase == 'cold' else 0)
            c = sum(r['cache']['total_seconds'] for r in rows) + (cs['construction_seconds'] if phase == 'cold' else 0)
            summary['replay_totals'][phase] = {'baseline_seconds': b, 'cache_seconds': c,
                                             'speedup': b / c, 'cache_stats': rows[-1]['cache_stats']}
        del baseline, cached, cache
        gc.collect()

        def evaluation():
            outputs = []
            for start in range(0, len(prompts), 16):
                outputs.extend(generate_batch(model, tok, prompts[start:start + 16], max_new_tokens=128))
            rate = sum(any(marker in x['generation'].lower() for marker in inputs['uncertainty_markers'])
                       for x in outputs) / len(outputs)
            ppl = teacher_forced_ppl(model, tok, c4, max_tokens=5000, window=1024, batch_size=8)
            return outputs, rate, ppl

        if not args.replay_only:
            # Full evaluation warms generation/PPL; fresh rank caches start afterwards.
            (_, base_rate, base_ppl), warm_seconds = timed(evaluation)
            summary.update(base_rate=base_rate, base_ppl=base_ppl,
                           evaluation_warmup_seconds_excluded=warm_seconds)
            save()
            for repeat in range(args.paired_repeats):
                paired = {}
                for mode in (('baseline', 'cache') if repeat % 2 == 0 else ('cache', 'baseline')):
                    rank, state, cache = make_ranker(mode == 'cache')
                    run = {'mode': mode, 'repeat': repeat, 'candidates': [], 'hash_seconds': 0.,
                           'ranking_seconds': 0., 'evaluation_seconds': 0., 'logging_seconds': 0.,
                           'cache_construction_seconds': state['construction_seconds']}
                    current = {}

                    def observed_rank(m, d, a, b, ls, comp, max_fraction):
                        ranking, elapsed = timed(lambda: rank(m, d, a, b, ls, comp, max_fraction))
                        run['ranking_seconds'] += elapsed
                        mask_hash, hash_seconds = timed(lambda: tensor_dict_hash(selection_from_ranking(ranking, .005)))
                        run['hash_seconds'] += hash_seconds
                        current.clear()
                        current.update(layers=list(ls), mask_hash=mask_hash, ranking_seconds=elapsed)
                        return ranking

                    def measure():
                        (outputs, rate, ppl), seconds = timed(evaluation)
                        generation_hash, overhead = timed(lambda: digest(outputs))
                        run['hash_seconds'] += overhead
                        run['evaluation_seconds'] += seconds
                        run['candidates'].append(dict(current, ppl=ppl, uncertainty_rate=rate,
                                                      generation_hash=generation_hash, outputs=outputs,
                                                      evaluation_seconds=seconds))
                        log_start = time.perf_counter()
                        atomic_json(args.run_dir / f'controlled_{repeat}_{mode}.json', run)
                        run['logging_seconds'] += time.perf_counter() - log_start
                        print(f'Controlled {repeat} {mode} {current["layers"]} rate={rate} ppl={ppl}', flush=True)
                        return rate, ppl

                    torch.cuda.reset_peak_memory_stats()
                    selected, seconds = timed(lambda: bestfirst_layers(
                        model, directions, mu_a, mu_b, args.pool, 'both', measure,
                        base_rate, base_ppl, beta=.05, eps=.005, test_frac=.005,
                        ranking_fn=observed_rank))
                    run.update(L_star=selected, wall_seconds=seconds,
                               timed_seconds=seconds - run['hash_seconds'] - run['logging_seconds'] + state['construction_seconds'],
                               peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                               score_seconds=state['score_seconds'],
                               cache_stats=dict(cache.stats) if cache is not None else None)
                    if selected:
                        final_rank, final_seconds = timed(lambda: rank_call(rank, sorted(selected), .05))
                        final_selection = selection_from_ranking(final_rank, .01)
                        run.update(final_mask_hash=tensor_dict_hash(final_selection),
                                   final_mask_edges=sum(t.numel() for t in final_selection.values()),
                                   final_mask_seconds_excluded=final_seconds)
                        del final_rank, final_selection
                    else:
                        run['final_mask_hash'] = None
                    atomic_json(args.run_dir / f'controlled_{repeat}_{mode}.json', run)
                    paired[mode] = run
                    summary['controlled_runs'].append({k: v for k, v in run.items() if k != 'candidates'})
                    save()
                    del rank, cache
                    gc.collect()
                a, b = paired['baseline'], paired['cache']
                aa = {tuple(r['layers']): r for r in a['candidates']}
                bb = {tuple(r['layers']): r for r in b['candidates']}
                checks = []
                for ls in sorted(set(aa) | set(bb)):
                    x, y = aa.get(ls), bb.get(ls)
                    checks.append({'layers': list(ls), 'both_visited': x is not None and y is not None,
                                   **({k + '_exact': x[k] == y[k] for k in
                                       ['mask_hash', 'generation_hash', 'outputs', 'ppl', 'uncertainty_rate']}
                                      if x is not None and y is not None else {})})
                comparison = {'repeat': repeat, 'candidate_checks': checks,
                              'L_star_exact': a['L_star'] == b['L_star'],
                              'final_mask_exact': a['final_mask_hash'] == b['final_mask_hash'],
                              'candidate_trace_exact': [x['layers'] for x in a['candidates']] == [x['layers'] for x in b['candidates']],
                              'speedup': a['timed_seconds'] / b['timed_seconds']}
                summary['comparisons'].append(comparison)
                save()
                if any(not c.get('mask_hash_exact', True) for c in checks) or not comparison['final_mask_exact']:
                    raise AssertionError('Controlled benchmark mask mismatch')
        summary['status'] = 'complete'
        summary['all_observed_equal'] = all(
            all(v for k, v in comp.items() if k.endswith('_exact')) and
            all(all(v for k, v in check.items() if k.endswith('_exact') or k == 'both_visited')
                for check in comp['candidate_checks']) for comp in summary['comparisons'])
        save()
    except Exception as err:
        summary.update(status='failed', error={'type': type(err).__name__, 'message': str(err)})
        save()
        raise


if __name__ == '__main__':
    main()
