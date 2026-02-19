from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable, Any

import torch
import torch.nn as nn


def list_named_linears(model: nn.Module) -> List[Tuple[str, nn.Linear]]:
    """Return (name, module) for all nn.Linear in a model (stable order)."""
    out: List[Tuple[str, nn.Linear]] = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            out.append((name, m))
    return out


def filter_module_names(
    names: List[str],
    include_prefixes: Optional[List[str]] = None,
    exclude_prefixes: Optional[List[str]] = None,
) -> List[str]:
    """
    Utility to select module names by prefix.
    include_prefixes=None means include all.
    """
    include_prefixes = include_prefixes or []
    exclude_prefixes = exclude_prefixes or []

    def included(n: str) -> bool:
        if include_prefixes:
            ok = any(n.startswith(p) for p in include_prefixes)
        else:
            ok = True
        if exclude_prefixes and any(n.startswith(p) for p in exclude_prefixes):
            ok = False
        return ok

    return [n for n in names if included(n)]


@dataclass
class ActivationSnapshot:
    """
    Stores one probe snapshot of activations.

    activations[name] has shape [B, H] for Linear outputs (expected),
    but can also hold other shapes if you hook other modules.
    """
    activations: Dict[str, torch.Tensor]


class ActivationCatcher:
    """
    Activation catcher that can be turned on/off.
    Designed to be used only during periodic "probe" forward passes.

    Example:
      catcher = ActivationCatcher(model, layer_names)
      catcher.register()

      with catcher.capture():
         _ = model(obs_probe)

      acts = catcher.latest.activations
    """

    def __init__(self, model: nn.Module, layer_names: List[str]):
        self.model = model
        self.layer_names = set(layer_names)
        self.handles: List[Any] = []
        self.enabled: bool = False
        self.latest = ActivationSnapshot(activations={})

    def _hook(self, name: str):
        def fn(_module, _inp, out):
            if not self.enabled:
                return
            # Keep only a detached tensor on the same device.
            # Avoid cloning unless necessary (detach is cheap).
            try:
                self.latest.activations[name] = out.detach()
            except Exception:
                # some modules return tuples; store first tensor
                if isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]):
                    self.latest.activations[name] = out[0].detach()
        return fn

    def register(self) -> None:
        # Attach hooks to named_modules that match layer_names
        for name, m in self.model.named_modules():
            if name in self.layer_names:
                self.handles.append(m.register_forward_hook(self._hook(name)))

    def clear(self) -> None:
        self.latest.activations.clear()

    def close(self) -> None:
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles.clear()

    def capture(self):
        """
        Context manager enabling capture for a single probe pass.
        """
        catcher = self

        class _Ctx:
            def __enter__(self_):
                catcher.enabled = True
                catcher.clear()
                return catcher

            def __exit__(self_, exc_type, exc, tb):
                catcher.enabled = False
                return False

        return _Ctx()


@torch.no_grad()
def linear_row_grad_norms(model: nn.Module, layer_names: List[str]) -> Dict[str, torch.Tensor]:
    """
    After loss.backward(), compute per-row L2 grad norm for specified Linear layers.

    Returns dict: layer_name -> grad_norms [H_out]
    """
    name_set = set(layer_names)
    out: Dict[str, torch.Tensor] = {}
    for name, m in model.named_modules():
        if name not in name_set:
            continue
        if not isinstance(m, nn.Linear):
            continue
        if m.weight.grad is None:
            continue
        g = m.weight.grad.detach()  # [H_out, H_in]
        out[name] = torch.linalg.vector_norm(g, ord=2, dim=1)  # [H_out]
    return out


def auto_select_linear_layers(
    model: nn.Module,
    include_prefixes: Optional[List[str]] = None,
    exclude_prefixes: Optional[List[str]] = None,
) -> List[str]:
    """
    Convenience: list all Linear layer module names, then filter by prefixes.

    include_prefixes example:
      ["actor.backbone", "critic.backbone", "q1.backbone"]
    But this depends on how the algorithm composes networks.
    """
    all_linears = [name for name, _ in list_named_linears(model)]
    return filter_module_names(all_linears, include_prefixes=include_prefixes, exclude_prefixes=exclude_prefixes)


def assert_activation_shapes(snapshot: ActivationSnapshot, expected_batch: int, max_print: int = 10) -> None:
    """
    Debug helper: ensure hooked activations are [B, H] and batch matches expected.
    """
    bad = []
    for name, a in snapshot.activations.items():
        if not torch.is_tensor(a):
            bad.append((name, "not tensor"))
            continue
        if a.ndim < 2:
            bad.append((name, f"ndim={a.ndim}"))
            continue
        if a.shape[0] != expected_batch:
            bad.append((name, f"batch={a.shape[0]} expected={expected_batch}"))
    if bad:
        msg = "Activation shape check failed:\n"
        for i, (n, r) in enumerate(bad[:max_print]):
            msg += f"  - {n}: {r}\n"
        raise ValueError(msg)

