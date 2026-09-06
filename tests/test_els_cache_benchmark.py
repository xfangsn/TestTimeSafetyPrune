"""Reference replay must retain the original repeated ELS rank calls."""
import importlib.util
import json
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    'benchmark_els_cache', Path(__file__).parents[1] / 'scripts/benchmark_els_cache.py')
_benchmark = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_benchmark)


def test_progress_preserves_repeats_and_omits_baseline(tmp_path):
    path = tmp_path / 'els_progress.json'
    path.write_text(json.dumps([
        {'layers': []}, {'layers': [24]}, {'layers': [24]},
        {'layers': [24, 22]}, {'layers': []}]))
    candidates, source = _benchmark.candidates_from_reference(tmp_path)
    assert candidates == [[24], [24], [22, 24]]
    assert source == str(path)


def test_fit_trace_fallback(tmp_path):
    path = tmp_path / 'fit.json'
    path.write_text(json.dumps({'els': {'trace': [{'layers': []}, {'layers': [6]}]}}))
    assert _benchmark.candidates_from_reference(tmp_path) == ([[6]], str(path))


def test_fixed_fallback_is_explicit():
    candidates, source = _benchmark.candidates_from_reference(None)
    assert source == 'fixed_fallback'
    assert candidates == [[24], [22], [6], [22, 24], [6, 24], [6, 22, 24]]


def test_missing_reference_is_not_silent_fallback(tmp_path):
    with pytest.raises(FileNotFoundError):
        _benchmark.candidates_from_reference(tmp_path)


def test_empty_trace_is_rejected(tmp_path):
    (tmp_path / 'els_progress.json').write_text('[{"layers": []}]')
    with pytest.raises(ValueError):
        _benchmark.candidates_from_reference(tmp_path)
