"""Full-layer ELS: exact binary-rate bounds versus complete evaluation.

Both modes use the same candidate cache and original generation batches. Jobs
run on separate GPUs; measured wall-time ratios include system noise.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    tmp.replace(path)


def audit(directory):
    """CPU-only audit of complete baseline against bounded evaluation prefixes."""
    directory = Path(directory)
    a = json.loads((directory / 'baseline/summary.json').read_text())
    b = json.loads((directory / 'bounded/summary.json').read_text())
    ar = json.loads((directory / 'baseline/candidate_records.json').read_text())
    br = json.loads((directory / 'bounded/candidate_records.json').read_text())
    errors = []

    def check(ok, message):
        if not ok:
            errors.append(message)

    for key in ('settings', 'inputs_hash', 'directions_hash', 'mu_a_hash', 'mu_b_hash',
                'lambda_value', 'Q', 'c4_hash', 'template_hash', 'model_commit',
                'source_commit', 'source_hashes', 'uncertainty_markers', 'pool', 'L_star', 'final_mask_hash', 'base'):
        check(a.get(key) == b.get(key), 'mismatch: ' + key)
    check(a.get('status') == b.get('status') == 'complete', 'runs incomplete')
    check(not a.get('code_dirty', True) and not b.get('code_dirty', True), 'tracked code dirty')
    check(a.get('mode') == 'baseline' and b.get('mode') == 'bounded', 'incorrect modes')
    check(len(ar) == len(br), 'candidate count mismatch')
    for i, (x, y) in enumerate(zip(ar, br)):
        label = f'candidate {i} {x.get("layers")}: '
        for key in ('stage', 'layers', 'mask_hash', 'incumbent_best'):
            check(x.get(key) == y.get(key), label + key)
        check(y['outputs'] == x['outputs'][:len(y['outputs'])], label + 'output prefix')
        if y['ppl'] is not None:
            check(x['ppl'] == y['ppl'], label + 'PPL')
        if y['stage'] == 'solo':
            continue
        total = a['settings']['n_select_uncertain']
        batch_sizes = a['settings']['batch_sizes']
        boundaries = [0]
        for size in batch_sizes:
            boundaries.append(boundaries[-1] + size)
        check(sum(batch_sizes) == total, label + 'batch size total')
        check(x['n_evaluated'] == len(x['outputs']) == total, label + 'baseline incomplete')
        baseline_pos = sum(any(m in out['generation'].lower() for m in a['uncertainty_markers']) for out in x['outputs'])
        check(x['positives'] == baseline_pos, label + 'baseline positive count')
        check(x['full_metric'] == baseline_pos / total, label + 'baseline full metric')
        check(x['ppl'] is not None and x['early_rejected'] is False, label + 'baseline rejected or missing PPL')
        for name, row in [('baseline', x), ('bounded', y)]:
            check(row['n_evaluated'] in boundaries, label + name + 'batch boundary')
            expected_batches = boundaries.index(row['n_evaluated']) if row['n_evaluated'] in boundaries else -1
            check(row['generation_batches'] == expected_batches, label + name + 'batch count')
        check(y['n_evaluated'] == len(y['outputs']), label + 'output count')
        check(y['n_total'] == x['n_total'] == a['settings']['n_select_uncertain'], label + 'total count')
        actual_pos = sum(any(m in out['generation'].lower() for m in a['uncertainty_markers']) for out in y['outputs'])
        check(y['positives'] == actual_pos, label + 'positive count')
        check(y['lower_bound'] == y['positives'] / y['n_total'], label + 'lower bound arithmetic')
        if y['early_rejected']:
            check(y['lower_bound'] >= y['incumbent_best'], label + 'invalid rejection certificate')
            check(x['full_metric'] >= y['incumbent_best'], label + 'baseline could win rejected candidate')
        else:
            check(y['n_evaluated'] == y['n_total'], label + 'accepted incomplete evaluation')
            check(x['full_metric'] == y['full_metric'], label + 'full metric')
        if y['full_metric'] is not None:
            check(x['full_metric'] == y['full_metric'], label + 'computed full metric')
    def totals(summary, rows):
        return {'generated_questions': sum(r['n_evaluated'] for r in rows),
                'generated_tokens': sum(r['generated_tokens'] for r in rows),
                'generation_batches': sum(r['generation_batches'] for r in rows),
                'ppl_calls': sum(r['ppl'] is not None for r in rows),
                'els_seconds': summary['els_seconds'],
                'solo_seconds': summary['solo_seconds'],
                'bestfirst_seconds': summary['bestfirst_seconds']}
    ta, tb = totals(a, ar), totals(b, br)
    result = {'status': 'pass' if not errors else 'fail', 'errors': errors,
              'baseline': ta, 'bounded': tb,
              'saved': {k: ta[k] - tb[k] for k in ta},
              'wall_time_speedup': ta['els_seconds'] / tb['els_seconds'],
              'timing_caveat': 'Parallel jobs on different GPUs, not paired same-GPU ABBA; system noise affects wall time.',
              'coverage': 'Full model-layer solo screening plus full best-first search; both use ELSCandidateCache.'}
    atomic_json(directory / 'audit.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['baseline', 'bounded', 'audit'], required=True)
    p.add_argument('--inputs', type=Path)
    p.add_argument('--run-dir', type=Path)
    p.add_argument('--audit-dir', type=Path)
    args = p.parse_args()
    if args.mode == 'audit':
        result = audit(args.audit_dir)
        print(json.dumps(result, indent=2))
        if result['status'] != 'pass':
            raise SystemExit(1)
        return
    if args.inputs is None or args.run_dir is None:
        p.error('--inputs and --run-dir required for GPU modes')
    run(args)


def run(args):
    import gc
    import torch
    from blade_plus_iti import legacy_split, median_positive, file_hash
    from ttsafety.behaviors import solo_layer_pool, bestfirst_layers
    from ttsafety.els_cache import ELSCandidateCache
    from ttsafety.els_bounds import can_prune_binary_rate
    from ttsafety.eval import load_c4_text, teacher_forced_ppl
    from ttsafety.generic_importance import collect_c4_generic_importance
    from ttsafety.hooks import capture_last_token, get_decoder_layers
    from ttsafety.iti_composition import collect_writer_means, generate_batch, qwen_no_thinking, tensor_dict_hash
    from ttsafety.models import env_info, load_model
    from ttsafety.sycophancy import score_edges, score_edges_g
    from ttsafety.weight_prune import selection_from_ranking

    root = Path(__file__).resolve().parents[1]
    def git(*argv):
        return subprocess.check_output(['git', '-C', str(root), *argv], text=True).strip()
    provenance = {'source_commit': git('rev-parse', 'HEAD'),
                  'code_dirty': bool(git('status', '--porcelain', '--untracked-files=no')),
                  'startup_status': git('status', '--porcelain'), 'hostname': socket.gethostname(),
                  'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                  'slurm_job_id': os.environ.get('SLURM_JOB_ID')}
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if (args.run_dir / 'summary.json').exists():
        raise FileExistsError('Use a fresh run directory')
    torch.manual_seed(0)
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '8')))
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    def timed(fn):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = fn()
        torch.cuda.synchronize()
        return result, time.perf_counter() - start
    records = []
    summary = dict(provenance, mode=args.mode, status='running', settings={
        'layers': 'all', 'components': 'both', 'thinking': False, 'Q_mode': 'g1scalar',
        'Q_max_tokens': 65536, 'Q_seqlen': 2048, 'Q_batch_size': 2,
        'generation_batch_size': 16, 'max_new_tokens': 128, 'ppl_max_tokens': 5000,
        'ppl_window': 1024, 'ppl_batch_size': 8, 'screen_frac': .005, 'test_frac': .005,
        'beta': .05, 'eps': .005, 'final_rho': .01, 'final_rank_fraction': .05,
        'torch_threads': torch.get_num_threads()},
        final_mask_scope='same zero-clamped ELS scorer at rho=.01, not production strict-positive final mask',
        timing_caveat='Parallel separate-GPU jobs; wall-time comparisons include system noise.',
        overhead={'hash_seconds': 0., 'logging_seconds': 0.})
    def save():
        start = time.perf_counter()
        atomic_json(args.run_dir / 'candidate_records.json', records)
        atomic_json(args.run_dir / 'summary.json', summary)
        summary['overhead']['logging_seconds'] += time.perf_counter() - start
    def hashed(fn):
        result, seconds = timed(fn)
        summary['overhead']['hash_seconds'] += seconds
        return result
    try:
        summary['inputs_hash'] = file_hash(args.inputs)
        summary['source_hashes'] = {str(path.relative_to(root)): file_hash(path) for path in
            [Path(__file__).resolve(), root / 'scripts/blade_plus_iti.py',
             root / 'src/ttsafety/behaviors.py', root / 'src/ttsafety/els_bounds.py',
             root / 'src/ttsafety/els_cache.py']}
        inputs = json.loads(args.inputs.read_text())
        summary['uncertainty_markers'] = inputs['uncertainty_markers']
        (model, tok), summary['load_seconds'] = timed(lambda: load_model(inputs['model']))
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        assert '</think>' in qwen_no_thinking(tok, 'What is the capital of France?')
        layers = list(range(len(get_decoder_layers(model))))
        summary.update(env=env_info(), model=inputs['model'], model_commit=getattr(model.config, '_commit_hash', None),
                       template_hash=hashlib.sha256(tok.chat_template.encode()).hexdigest(),
                       gpu_name=torch.cuda.get_device_name(), gpu_uuid=str(getattr(torch.cuda.get_device_properties(0), 'uuid', 'unavailable')),
                       all_layers=layers)
        train, select, _ = legacy_split(inputs['contrast_rows'])
        unc = [r['question'] for r in train if r['label'] == 1]
        cert = [r['question'] for r in train if r['label'] == 0]
        prompts = [r['question'] for r in select if r['label'] == 1]
        summary['settings']['n_select_uncertain'] = len(prompts)
        summary['settings']['batch_sizes'] = [len(prompts[i:i + 16]) for i in range(0, len(prompts), 16)]
        setup_start = time.perf_counter()
        au = capture_last_token(model, tok, [qwen_no_thinking(tok, p) for p in unc], batch_size=16)
        ac = capture_last_token(model, tok, [qwen_no_thinking(tok, p) for p in cert], batch_size=16)
        directions = {i: au[i].mean(0) - ac[i].mean(0) for i in layers}
        del au, ac
        mu_a = collect_writer_means(model, tok, unc, layers)
        mu_b = collect_writer_means(model, tok, cert, layers)
        summary.update(directions_hash=hashed(lambda: tensor_dict_hash(directions)),
                       mu_a_hash=hashed(lambda: tensor_dict_hash(mu_a)),
                       mu_b_hash=hashed(lambda: tensor_dict_hash(mu_b)))
        c4 = load_c4_text()
        summary['c4_hash'] = hashlib.sha256(c4.encode()).hexdigest()
        print('Collecting original g1scalar Q once', flush=True)
        (q, qmeta), summary['Q_seconds'] = timed(lambda: collect_c4_generic_importance(
            model, tok, layers, 'both', text=c4, seqlen=2048, batch_size=2,
            mode='g1scalar', max_tokens=65536))
        raw = score_edges(model, directions, mu_a, mu_b, layers, 'both')
        lam = median_positive(raw) / median_positive(q)
        del raw
        gc.collect()
        summary.update(Q=qmeta, Q_actual_tokens=qmeta['n_tokens'], lambda_value=lam,
                       setup_seconds=time.perf_counter() - setup_start)
        save()
        def score(m, d, a, b, ls, comp):
            raw = score_edges_g(m, d, a, b, ls, comp, Q=q, lam=lam, abstain=True)
            return {k: torch.where(torch.isfinite(v), v, torch.zeros_like(v)) for k, v in raw.items()}
        def ppl():
            return teacher_forced_ppl(model, tok, c4, max_tokens=5000, window=1024, batch_size=8)
        def is_positive(row):
            return any(marker in row['generation'].lower() for marker in inputs['uncertainty_markers'])
        # Same original batch composition; warm-up excludes cache construction.
        def base_eval():
            out = []
            for i in range(0, len(prompts), 16):
                out.extend(generate_batch(model, tok, prompts[i:i + 16], max_new_tokens=128))
            return {'outputs': out, 'metric': sum(map(is_positive, out)) / len(prompts), 'ppl': ppl()}
        summary['base'], summary['base_evaluation_seconds_excluded'] = timed(base_eval)
        cache, construction_seconds = timed(lambda: ELSCandidateCache(score))
        summary['cache_construction_seconds'] = construction_seconds
        current = {}
        stage = 'solo'
        def rank(m, d, a, b, ls, comp, max_fraction):
            ranking, seconds = timed(lambda: cache.rank(m, d, a, b, ls, comp, max_fraction))
            current.clear()
            current.update(stage=stage, layers=list(ls), ranking_seconds=seconds,
                mask_hash=hashed(lambda: tensor_dict_hash(selection_from_ranking(ranking, .005))))
            return ranking
        def blank_record():
            return dict(current, outputs=[], n_total=len(prompts), n_evaluated=0, positives=0,
                        lower_bound=0., incumbent_best=None, early_rejected=False,
                        full_metric=None, ppl=None, generation_batches=0, generated_tokens=0,
                        generation_seconds=0., ppl_seconds=0.)
        def solo_ppl():
            row = blank_record()
            row['ppl'], row['ppl_seconds'] = timed(ppl)
            records.append(row)
            save()
            print(f'solo {row["layers"]} ppl={row["ppl"]}', flush=True)
            return row['ppl']
        def overhead():
            return sum(summary['overhead'].values())
        before = overhead()
        pool, wall = timed(lambda: solo_layer_pool(model, directions, mu_a, mu_b, layers, 'both',
            solo_ppl, summary['base']['ppl'], screen_frac=.005, beta=.05, ranking_fn=rank))
        summary.update(pool=pool, solo_wall_seconds=wall,
                       solo_seconds=wall - (overhead() - before) + construction_seconds)
        stage = 'bestfirst'
        save()
        def measure(incumbent):
            row = blank_record()
            row['incumbent_best'] = incumbent
            for start in range(0, len(prompts), 16):
                if args.mode == 'bounded' and can_prune_binary_rate(row['positives'], len(prompts), incumbent):
                    row['early_rejected'] = True
                    break
                out, seconds = timed(lambda: generate_batch(model, tok, prompts[start:start + 16], max_new_tokens=128))
                row['outputs'].extend(out)
                row['generation_seconds'] += seconds
                row['generation_batches'] += 1
                row['n_evaluated'] += len(out)
                row['generated_tokens'] += sum(x['n_tokens'] for x in out)
                row['positives'] += sum(map(is_positive, out))
                row['lower_bound'] = row['positives'] / len(prompts)
            if row['n_evaluated'] == len(prompts):
                row['full_metric'] = row['lower_bound']
            if args.mode == 'bounded' and can_prune_binary_rate(row['positives'], len(prompts), incumbent):
                row['early_rejected'] = True
            if not row['early_rejected']:
                row['ppl'], row['ppl_seconds'] = timed(ppl)
            row['generation_hash'] = hashed(lambda: digest(row['outputs']))
            row['questions_saved'] = len(prompts) - row['n_evaluated']
            row['ppl_saved'] = int(row['ppl'] is None)
            records.append(row)
            save()
            print(f'{args.mode} {row["layers"]} n={row["n_evaluated"]} pos={row["positives"]} incumbent={incumbent} rejected={row["early_rejected"]}', flush=True)
            return None if row['early_rejected'] else (row['full_metric'], row['ppl'])
        before = overhead()
        selected, wall = timed(lambda: bestfirst_layers(model, directions, mu_a, mu_b, pool, 'both',
            lambda: (_ for _ in ()).throw(AssertionError('bounded callback expected')),
            summary['base']['metric'], summary['base']['ppl'], beta=.05, eps=.005,
            test_frac=.005, ranking_fn=rank, bounded_measure=measure))
        summary.update(L_star=selected, bestfirst_wall_seconds=wall,
                       bestfirst_seconds=wall - (overhead() - before), cache_stats=cache.stats,
                       peak_cuda_bytes=torch.cuda.max_memory_allocated())
        summary['els_seconds'] = summary['solo_seconds'] + summary['bestfirst_seconds']
        if selected:
            ranking, summary['final_mask_seconds_excluded'] = timed(lambda: cache.rank(
                model, directions, mu_a, mu_b, sorted(selected), 'both', .05))
            summary['final_mask_hash'] = hashed(lambda: tensor_dict_hash(selection_from_ranking(ranking, .01)))
        else:
            summary['final_mask_hash'] = None
        summary['status'] = 'complete'
        save()
    except Exception as err:
        summary.update(status='failed', error={'type': type(err).__name__, 'message': str(err)})
        save()
        raise


if __name__ == '__main__':
    main()
