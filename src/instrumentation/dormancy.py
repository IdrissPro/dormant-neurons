# src/instrumentation/dormancy.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


@torch.no_grad()
def tau_dormant_mask(acts_bh: torch.Tensor, tau: float, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Activation-based τ-dormancy (Sokar et al., 2023).

    acts_bh: [B, H] layer output activations for a probe batch.
    Returns:
      mask:  [H] bool (True = dormant)
      score: [H] normalized mean abs activation
    """
    if acts_bh.ndim != 2:
        raise ValueError(f"Expected acts_bh [B,H], got shape={tuple(acts_bh.shape)}")
    m = acts_bh.abs().mean(dim=0)           # [H]
    m_bar = m.mean()                        # scalar
    s = m / (m_bar + eps)                   # [H]
    return (s <= tau), s


@torch.no_grad()
def grad_quantile_dormant_mask(row_grad_norms_h: torch.Tensor, q: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gradient-based dormancy: mark bottom-q quantile as dormant.

    row_grad_norms_h: [H] per-row grad norm for a Linear weight matrix
    q: fraction in (0,1). Example q=0.10 marks 10% smallest gradients as dormant.

    Returns:
      mask: [H] bool
      score: [H] = row_grad_norms (for logging)
    """
    if row_grad_norms_h.ndim != 1:
        raise ValueError(f"Expected row_grad_norms_h [H], got shape={tuple(row_grad_norms_h.shape)}")
    if not (0.0 < q < 1.0):
        raise ValueError(f"q must be in (0,1), got {q}")

    H = row_grad_norms_h.numel()
    k = max(1, int(round(q * H)))
    # kthvalue gives k-th smallest; we want threshold so that <= threshold is dormant.
    thresh = torch.kthvalue(row_grad_norms_h, k).values
    mask = row_grad_norms_h <= thresh
    return mask, row_grad_norms_h


@torch.no_grad()
def dormant_fraction(mask_h: torch.Tensor) -> float:
    return float(mask_h.float().mean().item())


@torch.no_grad()
def overlap_coefficient(mask_a: torch.Tensor, mask_b: torch.Tensor, eps: float = 1e-8) -> float:
    """
    overlap(X,Y) = |X ∩ Y| / min(|X|, |Y|)
    where X,Y are dormant sets (mask=True means in set).
    """
    a = mask_a.bool()
    b = mask_b.bool()
    inter = (a & b).sum().item()
    ca = a.sum().item()
    cb = b.sum().item()
    denom = max(1.0, float(min(ca, cb)))
    return float(inter / denom)


@torch.no_grad()
def dormancy_events(prev_mask: torch.Tensor, curr_mask: torch.Tensor) -> Tuple[float, float]:
    """
    Returns:
      death_rate: fraction of neurons that became dormant (active -> dormant)
      revival_rate: fraction that became active (dormant -> active)
    """
    prev = prev_mask.bool()
    curr = curr_mask.bool()
    death = (~prev) & (curr)
    revival = (prev) & (~curr)
    return float(death.float().mean().item()), float(revival.float().mean().item())


@dataclass
class LayerDormancyStats:
    frac_dormant: float
    death_rate: Optional[float] = None
    revival_rate: Optional[float] = None
    overlap_prev: Optional[float] = None
    # Optional scalar summaries for debugging / plotting
    score_mean: Optional[float] = None
    score_min: Optional[float] = None
    score_max: Optional[float] = None


@dataclass
class DormancyReport:
    """
    Holds per-layer dormancy results for a checkpoint in time.
    """
    selection: str  # "activation" or "gradient"
    layer_stats: Dict[str, LayerDormancyStats]
    layer_masks: Dict[str, torch.Tensor]  # layer_name -> [H] bool mask


@torch.no_grad()
def summarize_scores(score_h: torch.Tensor) -> Tuple[float, float, float]:
    return float(score_h.mean().item()), float(score_h.min().item()), float(score_h.max().item())


@torch.no_grad()
def compute_activation_dormancy(
    activations: Dict[str, torch.Tensor],
    tau: float,
    eps: float = 1e-8,
    prev_masks: Optional[Dict[str, torch.Tensor]] = None,
) -> DormancyReport:
    """
    activations: layer_name -> [B,H] activations
    prev_masks: optional previous masks to compute events/overlap
    """
    layer_stats: Dict[str, LayerDormancyStats] = {}
    layer_masks: Dict[str, torch.Tensor] = {}

    for name, acts in activations.items():
        if acts.ndim != 2:
            # ignore weird shapes for now
            continue
        mask, score = tau_dormant_mask(acts, tau=tau, eps=eps)
        frac = dormant_fraction(mask)

        st = LayerDormancyStats(frac_dormant=frac)
        sm, smin, smax = summarize_scores(score)
        st.score_mean, st.score_min, st.score_max = sm, smin, smax

        if prev_masks is not None and name in prev_masks:
            pm = prev_masks[name]
            st.death_rate, st.revival_rate = dormancy_events(pm, mask)
            st.overlap_prev = overlap_coefficient(pm, mask, eps=eps)

        layer_stats[name] = st
        layer_masks[name] = mask

    return DormancyReport(selection="activation", layer_stats=layer_stats, layer_masks=layer_masks)


@torch.no_grad()
def compute_gradient_dormancy(
    row_grad_norms: Dict[str, torch.Tensor],
    q: float,
    prev_masks: Optional[Dict[str, torch.Tensor]] = None,
) -> DormancyReport:
    """
    row_grad_norms: layer_name -> [H] row grad norms
    prev_masks: optional previous masks to compute events/overlap
    """
    layer_stats: Dict[str, LayerDormancyStats] = {}
    layer_masks: Dict[str, torch.Tensor] = {}

    for name, g in row_grad_norms.items():
        if g.ndim != 1:
            continue
        mask, score = grad_quantile_dormant_mask(g, q=q)
        frac = dormant_fraction(mask)

        st = LayerDormancyStats(frac_dormant=frac)
        sm, smin, smax = summarize_scores(score)
        st.score_mean, st.score_min, st.score_max = sm, smin, smax

        if prev_masks is not None and name in prev_masks:
            pm = prev_masks[name]
            st.death_rate, st.revival_rate = dormancy_events(pm, mask)
            st.overlap_prev = overlap_coefficient(pm, mask)

        layer_stats[name] = st
        layer_masks[name] = mask

    return DormancyReport(selection="gradient", layer_stats=layer_stats, layer_masks=layer_masks)

