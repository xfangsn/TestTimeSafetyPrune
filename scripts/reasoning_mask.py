"""Single audited BLADE-G mask-builder for reasoning-pattern edits (removal & amplification).

Objective (stated once): rank residual-writer weights by S = [c]_+ − λ_eff · Q, where c is the
signed BLADE numerator, Q the generic-importance penalty (g1scalar, C4), and λ_eff = λ0 · |α−1| with
λ0 = median([c]_+>0) / median(Q>0). Removal is α=0 → λ_eff=λ0; amplification α>1 → λ_eff=λ0(α−1).
Abstention is PRESERVED (S≤0 → ineligible); we FAIL if fewer than the requested top-k positive-score
weights exist (no silent back-fill of abstained weights). Returns the selection dict + provenance.
"""
import hashlib

import torch

from ttsafety.generic_importance import collect_c4_generic_importance
from ttsafety.eval import load_c4_text
from ttsafety.sycophancy import score_edges, score_edges_g
from ttsafety.weight_prune import rank_weight_indices, selection_from_ranking


def _medpos(d):
    t = torch.cat([v.flatten() for v in d.values()]).float()
    return t[t > 0].median().item()


def collect_Q(model, tok, layers, components="both", q_tokens=65536):
    Q, meta = collect_c4_generic_importance(model, tok, layers, components, text=load_c4_text(),
                                            seqlen=2048, batch_size=2, mode="g1scalar",
                                            max_tokens=q_tokens)
    return Q, meta


def build_mask(model, direction, muC, muG, layers, *, Q, alpha, rho, components="both", screen=0.03):
    """BLADE-G selection for editing at `layers` with target sparsity `rho`, intended edit scale
    `alpha` (0=remove, >1=amplify). `Q` covers at least `layers`. Raises if positive candidates < k."""
    c = score_edges(model, direction, muC, muG, layers, components)          # [c]_+ (already relu'd)
    lam0 = _medpos(c) / _medpos({k: Q[k] for k in c})
    lam_eff = lam0 * abs(alpha - 1.0)
    S = score_edges_g(model, direction, muC, muG, layers, components, Q=Q, lam=lam_eff, abstain=True)
    n_pos = int(sum((torch.isfinite(v) & (v > 0)).sum().item() for v in S.values()))
    total = int(sum(v.numel() for v in S.values()))
    k = max(1, round(rho * total))
    if n_pos < k:
        raise ValueError(f"BLADE-G abstention: only {n_pos} positive-score weights < requested k={k} "
                         f"(rho={rho}, layers={layers}, alpha={alpha}). Lower rho or widen layers.")
    S = {kk: torch.where(torch.isfinite(v), v, torch.full_like(v, float("-inf"))) for kk, v in S.items()}
    sel = selection_from_ranking(rank_weight_indices(S, max(screen, rho)), rho)
    prov = {"alpha": alpha, "rho": rho, "layers": list(layers), "lam0": lam0, "lam_eff": lam_eff,
            "n_positive": n_pos, "n_total": total, "k": k,
            "sel_sha256": hashlib.sha256(
                repr({kk: sorted(v.tolist()) for kk, v in sel.items()}).encode()).hexdigest()[:16]}
    return sel, prov
