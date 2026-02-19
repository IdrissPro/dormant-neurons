from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name in ("silu", "swish"):
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unknown activation: {name}")


@dataclass
class MLPConfig:
    in_dim: int
    hidden_dims: List[int]
    activation: str = "relu"
    layernorm: bool = False


class MLPBackbone(nn.Module):
    """
    MLP trunk that exposes:
      - named Linear modules: trunk.fc0, trunk.fc1, ...
      - penultimate features: output of last hidden layer (after activation/ln)
    """

    def __init__(self, cfg: MLPConfig, name_prefix: str = "trunk"):
        super().__init__()
        self.cfg = cfg
        self.name_prefix = name_prefix

        act = get_activation(cfg.activation)
        self._act_name = cfg.activation.lower()
        self._use_ln = bool(cfg.layernorm)

        dims = [cfg.in_dim] + list(cfg.hidden_dims)
        self.fcs: nn.ModuleList = nn.ModuleList()
        self.lns: nn.ModuleList = nn.ModuleList()

        for i in range(len(dims) - 1):
            fc = nn.Linear(dims[i], dims[i + 1])
            self.fcs.append(fc)
            if self._use_ln:
                self.lns.append(nn.LayerNorm(dims[i + 1]))
            else:
                self.lns.append(nn.Identity())

            # Give stable, human-readable names (important for hook selection).
            setattr(self, f"{name_prefix}.fc{i}".replace(".", "_"), fc)
            setattr(self, f"{name_prefix}.ln{i}".replace(".", "_"), self.lns[-1])

        # store activation module once
        self.act = act

    def named_linear_layer_names(self) -> List[str]:
        # We expose names as they appear in named_modules().
        # Because we attach modules via attributes with underscores,
        # we'll provide the actual attribute paths.
        names = []
        for i in range(len(self.fcs)):
            names.append(f"{self.name_prefix}_fc{i}")
        return names

    def forward(self, x: torch.Tensor, return_features: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        h = x
        feat = None
        for i, (fc, ln) in enumerate(zip(self.fcs, self.lns)):
            h = fc(h)
            h = ln(h)
            h = self.act(h)
            if i == len(self.fcs) - 1:
                feat = h  # penultimate (last hidden) features
        if return_features:
            assert feat is not None
            return h, feat
        return h


class CategoricalPolicy(nn.Module):
    """Discrete-action policy head for PPO/DQN-like discrete actor-critic usage."""

    def __init__(self, backbone: MLPBackbone, hidden_dim: int, n_actions: int):
        super().__init__()
        self.backbone = backbone
        self.logits = nn.Linear(hidden_dim, n_actions)

    def forward(self, obs: torch.Tensor, return_features: bool = False):
        h, feat = self.backbone(obs, return_features=True)
        logits = self.logits(h)
        if return_features:
            return logits, feat
        return logits

    def dist(self, obs: torch.Tensor):
        logits = self.forward(obs, return_features=False)
        return torch.distributions.Categorical(logits=logits)


class DiagGaussianPolicy(nn.Module):
    """Continuous-action policy head with state-independent log_std (common in PPO)."""

    def __init__(self, backbone: MLPBackbone, hidden_dim: int, act_dim: int, log_std_init: float = -0.5):
        super().__init__()
        self.backbone = backbone
        self.mu = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Parameter(torch.ones(act_dim) * log_std_init)

    def forward(self, obs: torch.Tensor, return_features: bool = False):
        h, feat = self.backbone(obs, return_features=True)
        mu = self.mu(h)
        log_std = self.log_std.expand_as(mu)
        if return_features:
            return mu, log_std, feat
        return mu, log_std

    def dist(self, obs: torch.Tensor):
        mu, log_std = self.forward(obs, return_features=False)
        std = torch.exp(log_std)
        return torch.distributions.Normal(mu, std)


class ValueHead(nn.Module):
    def __init__(self, backbone: MLPBackbone, hidden_dim: int):
        super().__init__()
        self.backbone = backbone
        self.v = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor, return_features: bool = False):
        h, feat = self.backbone(obs, return_features=True)
        v = self.v(h).squeeze(-1)
        if return_features:
            return v, feat
        return v


class QNetwork(nn.Module):
    """
    Q network for DQN or SAC critics.
    For SAC critics: input is [obs, action] concatenated.
    """

    def __init__(self, backbone: MLPBackbone, hidden_dim: int):
        super().__init__()
        self.backbone = backbone
        self.q = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        h, feat = self.backbone(x, return_features=True)
        q = self.q(h).squeeze(-1)
        if return_features:
            return q, feat
        return q


def infer_obs_dim(observation_space) -> int:
    import numpy as np

    shape = observation_space.shape
    if shape is None:
        raise ValueError("Observation space has no shape (unsupported).")
    if len(shape) != 1:
        # For now we only support vector obs in this project baseline.
        # Visual/CNN extension can be added later.
        raise ValueError(f"Only 1D vector obs supported; got shape={shape}")
    return int(shape[0])


def infer_act_dims(action_space) -> Tuple[str, int]:
    """
    Returns:
      (kind, dim) where kind in {"discrete", "continuous"}
    """
    if hasattr(action_space, "n"):
        return "discrete", int(action_space.n)
    # Box
    shape = action_space.shape
    if shape is None or len(shape) != 1:
        raise ValueError(f"Only 1D action spaces supported; got {shape}")
    return "continuous", int(shape[0])


def build_mlp_backbone(in_dim: int, hidden_dims: List[int], activation: str, layernorm: bool, name_prefix: str) -> MLPBackbone:
    cfg = MLPConfig(in_dim=in_dim, hidden_dims=hidden_dims, activation=activation, layernorm=layernorm)
    return MLPBackbone(cfg=cfg, name_prefix=name_prefix)
