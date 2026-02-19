from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


@torch.no_grad()
def _center(Z: torch.Tensor) -> torch.Tensor:
    return Z - Z.mean(dim=0, keepdim=True)


@torch.no_grad()
def effective_rank(Z: torch.Tensor, eps: float = 1e-12) -> Tuple[float, torch.Tensor]:
    """
    Effective rank = exp( -sum_i p_i log p_i ) where p_i = s_i / sum s.
    Z: [B, d]
    Returns:
      r_eff (float), singular values s (1D tensor)
    """
    if Z.ndim != 2:
        raise ValueError(f"Expected Z [B,d], got {tuple(Z.shape)}")
    Zc = _center(Z)
    s = torch.linalg.svdvals(Zc)  # [min(B,d)]
    p = s / (s.sum() + eps)
    H = -(p * (p + eps).log()).sum()
    r_eff = torch.exp(H)
    return float(r_eff.item()), s


@torch.no_grad()
def cosine_diversity(Z: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Diversity = 1 - mean_{i!=j} cos(z_i, z_j)
    Z: [B, d]
    """
    if Z.ndim != 2:
        raise ValueError(f"Expected Z [B,d], got {tuple(Z.shape)}")
    Z = Z / (Z.norm(dim=1, keepdim=True) + eps)
    S = Z @ Z.t()  # [B,B]
    B = Z.size(0)
    off = (S.sum() - S.diag().sum()) / (B * (B - 1) + eps)
    return float((1.0 - off).item())


@torch.no_grad()
def topk_singular_values(s: torch.Tensor, k: int) -> torch.Tensor:
    """
    Return top-k singular values as a 1D tensor (length <= k).
    """
    if s.ndim != 1:
        raise ValueError(f"Expected s 1D, got {tuple(s.shape)}")
    k = int(k)
    if k <= 0:
        return s[:0]
    k = min(k, s.numel())
    return s[:k].detach()


@torch.no_grad()
def linear_cka(Z1: torch.Tensor, Z2: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Linear CKA between representations Z1 and Z2 on the same batch.
    Implements: CKA(K,L) = <HKH,HLH>_F / (||HKH||_F ||HLH||_F)

    Z1, Z2: [B, d1], [B, d2]
    """
    if Z1.ndim != 2 or Z2.ndim != 2:
        raise ValueError(f"Expected Z1/Z2 2D, got {tuple(Z1.shape)}, {tuple(Z2.shape)}")
    if Z1.size(0) != Z2.size(0):
        raise ValueError("Z1 and Z2 must have same batch dimension")

    def gram(Z: torch.Tensor) -> torch.Tensor:
        Zc = _center(Z)
        return Zc @ Zc.t()

    K = gram(Z1)
    L = gram(Z2)
    B = K.size(0)
    H = torch.eye(B, device=K.device) - torch.ones((B, B), device=K.device) / B
    Kc = H @ K @ H
    Lc = H @ L @ H
    num = (Kc * Lc).sum()
    den = torch.sqrt((Kc * Kc).sum() * (Lc * Lc).sum() + eps)
    return float((num / den).item())


@dataclass
class CKAReference:
    """
    Keeps a reference representation for CKA comparisons.
    Modes:
      - first: first observed Z is fixed reference
      - ema: exponentially-smoothed reference: ref <- beta*ref + (1-beta)*Z
    """
    mode: str = "ema"         # "first" | "ema"
    beta: float = 0.99
    ref: Optional[torch.Tensor] = None

    @torch.no_grad()
    def update(self, Z: torch.Tensor) -> None:
        if self.ref is None:
            self.ref = Z.detach().clone()
            return
        if self.mode == "first":
            return
        if self.mode == "ema":
            # keep on same device
            self.ref.mul_(self.beta).add_(Z.detach(), alpha=(1.0 - self.beta))
            return
        raise ValueError(f"Unknown CKAReference mode: {self.mode}")

    @torch.no_grad()
    def cka(self, Z: torch.Tensor) -> Optional[float]:
        if self.ref is None:
            return None
        return linear_cka(self.ref, Z)


@torch.no_grad()
def compute_repr_metrics(
    Z: torch.Tensor,
    do_effective_rank: bool = True,
    do_cosine_div: bool = True,
    svd_topk: int = 0,
    cka_ref: Optional[CKAReference] = None,
) -> Dict[str, object]:
    """
    Returns dict suitable for logging (scalars + optional small arrays).

    Note: arrays (like sv_topk) are returned as python lists to go into JSONL.
    TensorBoard logging for arrays is handled elsewhere (or as text).
    """
    out: Dict[str, object] = {}

    svals = None
    if do_effective_rank:
        r_eff, svals = effective_rank(Z)
        out["repr/effective_rank"] = r_eff

    if do_cosine_div:
        out["repr/cosine_diversity"] = cosine_diversity(Z)

    if svd_topk and svd_topk > 0:
        if svals is None:
            _, svals = effective_rank(Z)
        sv = topk_singular_values(svals, svd_topk)
        out["repr/svd_topk"] = sv.detach().cpu().tolist()

    if cka_ref is not None:
        # compute CKA against current reference, then update reference
        c = cka_ref.cka(Z)
        if c is not None:
            out["repr/cka_to_ref"] = float(c)
        cka_ref.update(Z)

    return out
