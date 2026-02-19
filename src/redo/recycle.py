# src/redo/recycle.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import init


# -----------------------------
# Optimizer-state handling
# -----------------------------

@torch.no_grad()
def _zero_adam_moments_rows(adam: torch.optim.Optimizer, param: torch.Tensor, row_idx: torch.Tensor) -> None:
    st = adam.state.get(param, None)
    if not st:
        return
    for k in ("exp_avg", "exp_avg_sq"):
        if k in st and st[k] is not None:
            # assumes 2D param: [H_out, H_in]
            st[k][row_idx, :] = 0.0


@torch.no_grad()
def _zero_adam_moments_cols(adam: torch.optim.Optimizer, param: torch.Tensor, col_idx: torch.Tensor) -> None:
    st = adam.state.get(param, None)
    if not st:
        return
    for k in ("exp_avg", "exp_avg_sq"):
        if k in st and st[k] is not None:
            # assumes 2D param: [H_out, H_in]
            st[k][:, col_idx] = 0.0


@torch.no_grad()
def _zero_adam_moments_bias(adam: torch.optim.Optimizer, param: torch.Tensor, idx: torch.Tensor) -> None:
    st = adam.state.get(param, None)
    if not st:
        return
    for k in ("exp_avg", "exp_avg_sq"):
        if k in st and st[k] is not None:
            st[k][idx] = 0.0


# -----------------------------
# Reinitialization helpers
# -----------------------------

@torch.no_grad()
def _init_rows_like(weight: torch.Tensor, init_mode: str) -> torch.Tensor:
    tmp = torch.empty_like(weight)
    if init_mode == "xavier_uniform":
        init.xavier_uniform_(tmp)
    elif init_mode == "orthogonal":
        init.orthogonal_(tmp)
    else:
        raise ValueError(f"Unknown init_mode: {init_mode}")
    return tmp


# -----------------------------
# Core ReDo primitive
# -----------------------------

@torch.no_grad()
def redo_recycle_linear_pair(
    lin_in: nn.Linear,
    lin_out: nn.Linear,
    dormant_mask: torch.Tensor,   # [H] True=dormant for output neurons of lin_in
    optimizer: Optional[torch.optim.Optimizer] = None,
    init_mode: str = "xavier_uniform",
    reset_bias: bool = True,
    outgoing: str = "zero",       # "zero" | "random"
    max_frac: float = 0.5,
) -> int:
    """
    ReDo for a pair of Linear layers: lin_in -> (activation) -> lin_out.
      - Reinit incoming weights: rows of lin_in.weight for dormant neurons
      - Reset bias for those neurons (optional)
      - Set outgoing weights: columns of lin_out.weight for those neurons (zero or random)

    Returns: number of neurons recycled.
    """
    if dormant_mask.ndim != 1:
        raise ValueError(f"dormant_mask must be [H], got {tuple(dormant_mask.shape)}")

    H = dormant_mask.numel()
    idx = dormant_mask.nonzero(as_tuple=False).squeeze(-1)
    if idx.numel() == 0:
        return 0

    # Safety cap: avoid catastrophic resets.
    max_n = int(max(1, round(max_frac * H)))
    if idx.numel() > max_n:
        # Keep the first max_n indices (deterministic). Could also sample.
        idx = idx[:max_n]

    # 1) reinit incoming weights (rows of lin_in.weight)
    w_in = lin_in.weight  # [H, in_dim]
    tmp = _init_rows_like(w_in, init_mode=init_mode)
    w_in[idx, :] = tmp[idx, :]

    # bias
    if reset_bias and lin_in.bias is not None:
        lin_in.bias[idx] = 0.0

    # 2) outgoing weights (columns of lin_out.weight)
    w_out = lin_out.weight  # [out_dim, H]
    if outgoing == "zero":
        w_out[:, idx] = 0.0
    elif outgoing == "random":
        tmp2 = _init_rows_like(w_out, init_mode=init_mode)  # same init ok
        w_out[:, idx] = tmp2[:, idx]
    else:
        raise ValueError(f"Unknown outgoing strategy: {outgoing}")

    # 3) optimizer state reset (Adam moments) for affected entries
    if optimizer is not None:
        # We only handle Adam-like optimizers with exp_avg/exp_avg_sq in state.
        try:
            _zero_adam_moments_rows(optimizer, lin_in.weight, idx)
            if lin_in.bias is not None and reset_bias:
                _zero_adam_moments_bias(optimizer, lin_in.bias, idx)
            _zero_adam_moments_cols(optimizer, lin_out.weight, idx)
        except Exception:
            # If optimizer state structure differs, ignore silently.
            pass

    return int(idx.numel())


# -----------------------------
# Model-wide wiring
# -----------------------------

@dataclass
class ReDoResult:
    recycled_by_layer: Dict[str, int]
    total_recycled: int


def _build_name_to_linear(model: nn.Module) -> Dict[str, nn.Linear]:
    out: Dict[str, nn.Linear] = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            out[name] = m
    return out


@torch.no_grad()
def redo_apply_on_sequential_linears(
    model: nn.Module,
    dormant_masks: Dict[str, torch.Tensor],
    optimizer: Optional[torch.optim.Optimizer] = None,
    init_mode: str = "xavier_uniform",
    reset_bias: bool = True,
    outgoing: str = "zero",
    max_frac: float = 0.5,
) -> ReDoResult:
    """
    Apply ReDo across an MLP-like stack of Linear layers inside 'model'.

    Assumption: if a Linear named L is followed downstream by another Linear L_next
    in the same "scope", we can treat (L, L_next) as a pair for outgoing resets.
    In practice, we will:
      - sort Linear layer names lexicographically
      - apply pairwise for consecutive linears that appear in dormant_masks

    This is simple and works well for backbones defined as fc0, fc1, fc2...
    For more complex topologies, you can provide explicit pairs later.

    dormant_masks keys must be names of nn.Linear modules in model.
    """
    name_to_lin = _build_name_to_linear(model)
    layer_names = sorted([n for n in dormant_masks.keys() if n in name_to_lin])

    recycled: Dict[str, int] = {}
    total = 0

    # Pair consecutive linears
    for i in range(len(layer_names) - 1):
        n_in = layer_names[i]
        n_out = layer_names[i + 1]
        lin_in = name_to_lin[n_in]
        lin_out = name_to_lin[n_out]
        mask = dormant_masks[n_in].to(lin_in.weight.device)

        k = redo_recycle_linear_pair(
            lin_in=lin_in,
            lin_out=lin_out,
            dormant_mask=mask,
            optimizer=optimizer,
            init_mode=init_mode,
            reset_bias=reset_bias,
            outgoing=outgoing,
            max_frac=max_frac,
        )
        recycled[n_in] = k
        total += k

    return ReDoResult(recycled_by_layer=recycled, total_recycled=total)
