import pytest
import torch
from torch import nn

from ttsafety.iti_composition import fit_legacy_iti, iti_hook, scaled_selection, strict_selection


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.o_proj = nn.Linear(4, 4, bias=False)
        self.self_attn.o_proj.weight.data.copy_(torch.eye(4))


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Block()])

    def forward(self, x):
        return self.layers[0].self_attn.o_proj(x)


def test_legacy_hook_skips_prefill_and_steers_decode():
    model = Toy()
    vector = torch.tensor([1., 2., 3., 4.])
    prompt = torch.zeros(2, 3, 4)
    step = torch.zeros(2, 1, 4)
    with iti_hook(model, {0: vector}):
        assert torch.equal(model(prompt), prompt)
        assert torch.equal(model(step), vector.expand_as(step))
    assert torch.equal(model(step), step)


def test_composition_uses_edited_o_proj_and_restores_on_failure():
    model = Toy()
    original = model.layers[0].self_attn.o_proj.weight.detach().clone()
    selection = {"layers.0.self_attn.o_proj": torch.tensor([0])}
    with pytest.raises(RuntimeError, match="intentional"):
        with scaled_selection(model, selection, 2), iti_hook(model, {0: torch.ones(4)}):
            assert torch.equal(model(torch.zeros(1, 1, 4)), torch.tensor([[[2., 1., 1., 1.]]]))
            raise RuntimeError("intentional")
    assert torch.equal(model.layers[0].self_attn.o_proj.weight, original)
    assert len(model.layers[0].self_attn.o_proj._forward_pre_hooks) == 0


def test_partial_weight_edit_setup_failure_restores_earlier_matrix():
    model = Toy()
    # Failure after the first edit (invalid index) must also restore it.
    model.layers.append(Block())
    original = model.layers[0].self_attn.o_proj.weight.detach().clone()
    with pytest.raises(IndexError):
        with scaled_selection(model, {"layers.0.self_attn.o_proj": torch.tensor([0]),
                                      "layers.1.self_attn.o_proj": torch.tensor([999])}, 2):
            pass
    assert torch.equal(model.layers[0].self_attn.o_proj.weight, original)


def test_all_position_policy_is_explicitly_different():
    model = Toy()
    x = torch.zeros(2, 3, 4)
    with iti_hook(model, {0: torch.ones(4)}, policy="all_positions"):
        assert torch.equal(model(x), torch.ones_like(x))


def test_positive_mask_cannot_backfill():
    scores = {"w": torch.full((10, 10), -torch.inf)}
    scores["w"][0, 0] = 2
    with pytest.raises(ValueError, match="Infeasible"):
        strict_selection(scores, 0.02)
    assert strict_selection(scores, 0.01)["w"].tolist() == [0]


def test_refit_reselects_heads_while_transfer_preserves_ids():
    acts = {0: torch.tensor([[[1., 0.], [5., 0.]], [[3., 0.], [7., 0.]],
                             [[-1., 0.], [-5., 0.]], [[-3., 0.], [-7., 0.]]])}
    labels = [1, 1, 0, 0]
    transfer = fit_legacy_iti(acts, labels, k=1, fixed_heads=[(0, 0)])
    refit = fit_legacy_iti(acts, labels, k=1)
    assert transfer["heads"] == [(0, 0)]
    assert refit["heads"] == [(0, 1)]
    assert transfer["sigmas"][(0, 0)] == pytest.approx(5 ** 0.5)
    assert refit["sigmas"][(0, 1)] == pytest.approx(37 ** 0.5)
    assert torch.equal(transfer["directions"][(0, 0)], torch.tensor([1., 0.]))


def test_fit_split_exactly_matches_existing_blade():
    import ast
    from collections import defaultdict
    from pathlib import Path
    import json
    import random
    import runpy

    tree = ast.parse(Path("scripts/blade_epistemic_p0.py").read_text())
    node = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "split_3way")
    namespace = {"defaultdict": defaultdict, "random": random}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "historical_split", "exec"), namespace)
    new = runpy.run_path("scripts/blade_plus_iti.py")["legacy_split"]
    rows = [{"entity": str(i // 2), "family": str(i % 3), "label": i % 2} for i in range(120)]
    assert new(rows) == namespace["split_3way"](rows)
