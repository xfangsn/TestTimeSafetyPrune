"""Audit must reject invalid certificates and changed evaluated prefixes."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    'benchmark_els_bounds', Path(__file__).parents[1] / 'scripts/benchmark_els_bounds.py')
_bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bench)


def pair(tmp_path):
    common = dict(status='complete', code_dirty=False, settings={'n_select_uncertain': 4, 'batch_sizes': [2, 2]},
                  uncertainty_markers=['maybe'], inputs_hash='x', directions_hash='d',
                  mu_a_hash='a', mu_b_hash='b', lambda_value=1., Q={'n_tokens': 32},
                  c4_hash='c', template_hash='t', model_commit='m', source_commit='s',
                  source_hashes={}, pool=[1], L_star=[], final_mask_hash=None,
                  base={'metric': .5, 'ppl': 10.}, els_seconds=10., solo_seconds=2.,
                  bestfirst_seconds=8.)
    x = dict(stage='bestfirst', layers=[1], mask_hash='mask', incumbent_best=.5,
             outputs=[{'generation': 'maybe'}, {'generation': 'maybe'},
                      {'generation': 'yes'}, {'generation': 'yes'}],
             n_evaluated=4, n_total=4, positives=2, lower_bound=.5,
             early_rejected=False, full_metric=.5, ppl=10., generated_tokens=4,
             generation_batches=2)
    y = copy.deepcopy(x)
    y.update(outputs=x['outputs'][:2], n_evaluated=2, early_rejected=True,
             full_metric=None, ppl=None, generated_tokens=2, generation_batches=1)
    return common, x, y


def write_pair(tmp_path, common, x, y):
    for mode, row in [('baseline', x), ('bounded', y)]:
        path = tmp_path / mode
        path.mkdir(exist_ok=True)
        (path / 'summary.json').write_text(json.dumps(dict(common, mode=mode)))
        (path / 'candidate_records.json').write_text(json.dumps([row]))


def test_valid_prefix_certificate_and_exact_savings(tmp_path):
    common, x, y = pair(tmp_path)
    write_pair(tmp_path, common, x, y)
    result = _bench.audit(tmp_path)
    assert result['status'] == 'pass'
    assert result['saved']['generated_questions'] == 2
    assert result['saved']['ppl_calls'] == 1


@pytest.mark.parametrize('change', ['prefix', 'certificate', 'mask', 'positives', 'ppl'])
def test_audit_detects_difference(tmp_path, change):
    common, x, y = pair(tmp_path)
    if change == 'prefix':
        y['outputs'][0] = {'generation': 'perhaps'}
    elif change == 'certificate':
        y['lower_bound'] = .25
    elif change == 'mask':
        y['mask_hash'] = 'wrong'
    elif change == 'positives':
        y['positives'] = 1
    else:
        y['ppl'] = 11.
    write_pair(tmp_path, common, x, y)
    assert _bench.audit(tmp_path)['status'] == 'fail'


def test_completed_last_batch_can_skip_only_ppl(tmp_path):
    common, x, y = pair(tmp_path)
    y = copy.deepcopy(x)
    y.update(early_rejected=True, ppl=None)
    write_pair(tmp_path, common, x, y)
    result = _bench.audit(tmp_path)
    assert result['status'] == 'pass'
    assert result['saved']['generated_questions'] == 0
    assert result['saved']['ppl_calls'] == 1


def test_baseline_metric_is_independently_recomputed(tmp_path):
    common, x, y = pair(tmp_path)
    x['full_metric'] = .75
    write_pair(tmp_path, common, x, y)
    result = _bench.audit(tmp_path)
    assert result['status'] == 'fail'
    assert any('baseline full metric' in error for error in result['errors'])


@pytest.mark.parametrize('tamper', ['boundary', 'batches'])
def test_original_batch_boundaries_are_enforced(tmp_path, tamper):
    common, x, y = pair(tmp_path)
    if tamper == 'boundary':
        y['n_evaluated'] = 3
        y['outputs'] = x['outputs'][:3]
    else:
        y['generation_batches'] = 2
    write_pair(tmp_path, common, x, y)
    result = _bench.audit(tmp_path)
    assert result['status'] == 'fail'
    assert any('batch ' in error for error in result['errors'])
