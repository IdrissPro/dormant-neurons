from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from omegaconf import DictConfig

from src.logging.logger import RunLogger
from src.rl.networks import (
    infer_obs_dim,
    infer_act_dims,
    build_mlp_backbone,
    CategoricalPolicy,
    DiagGaussianPolicy,
    ValueHead,
)
from src.instrumentation.dormancy import (
    compute_activation_dormancy,
    compute_gradient_dormancy,
)
from src.instrumentation.repr_metrics import (
    CKAReference,
    compute_repr_metrics,
)
from src.instrumentation.hooks import (
    linear_row_grad_norms,
    list_named_linears,
)
from src.redo.recycle import redo_apply_on_sequential_linears
from src.redo.schedules import ReDoScheduler

#    
# PPO config defaults
#    

@dataclass
class PPOConfig:
    # rollout
    num_steps: int = 2048
    # optimization
    update_epochs: int = 10
    num_minibatches: int = 32
    lr: float = 3e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None
    # misc
    norm_adv: bool = True
    clip_vloss: bool = True
    # net
    hidden_dims: Tuple[int, ...] = (64, 64)
    activation: str = "tanh"  # will map tanh manually
    layernorm: bool = False
    shared_backbone: bool = False  # if true, actor+critic share a trunk


def _get(cfg: DictConfig, path: str, default: Any) -> Any:
    cur: Any = cfg
    for key in path.split("."):
        if not hasattr(cur, key):
            return default
        cur = getattr(cur, key)
    return cur if cur is not None else default


def load_ppo_config(cfg: DictConfig) -> PPOConfig:
    # Note: we allow configs to omit PPO-specific keys; defaults apply.
    hidden = _get(cfg, "algo.hidden_dims", list(PPOConfig.hidden_dims))
    activation = _get(cfg, "algo.activation", "relu")
    # Support tanh if desired (common in PPO). We handle it in backbone build below.
    return PPOConfig(
        num_steps=int(_get(cfg, "algo.num_steps", PPOConfig.num_steps)),
        update_epochs=int(_get(cfg, "algo.update_epochs", PPOConfig.update_epochs)),
        num_minibatches=int(_get(cfg, "algo.num_minibatches", PPOConfig.num_minibatches)),
        lr=float(_get(cfg, "algo.lr", PPOConfig.lr)),
        anneal_lr=bool(_get(cfg, "algo.anneal_lr", PPOConfig.anneal_lr)),
        gamma=float(_get(cfg, "algo.gamma", PPOConfig.gamma)),
        gae_lambda=float(_get(cfg, "algo.gae_lambda", PPOConfig.gae_lambda)),
        clip_coef=float(_get(cfg, "algo.clip_coef", PPOConfig.clip_coef)),
        ent_coef=float(_get(cfg, "algo.ent_coef", PPOConfig.ent_coef)),
        vf_coef=float(_get(cfg, "algo.vf_coef", PPOConfig.vf_coef)),
        max_grad_norm=float(_get(cfg, "algo.max_grad_norm", PPOConfig.max_grad_norm)),
        target_kl=_get(cfg, "algo.target_kl", PPOConfig.target_kl),
        norm_adv=bool(_get(cfg, "algo.norm_adv", PPOConfig.norm_adv)),
        clip_vloss=bool(_get(cfg, "algo.clip_vloss", PPOConfig.clip_vloss)),
        hidden_dims=tuple(int(x) for x in hidden),
        activation=str(activation),
        layernorm=bool(_get(cfg, "algo.layernorm", PPOConfig.layernorm)),
        shared_backbone=bool(_get(cfg, "algo.shared_backbone", PPOConfig.shared_backbone)),
    )


#    
# Small utilities
#    

def _maybe_tanh_activation(name: str) -> str:
    # networks.py supports relu/silu/gelu. PPO often uses tanh.
    # We'll treat tanh by implementing it in a local backbone wrapper below.
    return name.lower()


class TanhBackboneAdapter(nn.Module):
    """
    Wraps an MLPBackbone-like module but applies tanh nonlinearity between layers.
    We use this only if cfg requests activation="tanh".
    """
    def __init__(self, in_dim: int, hidden_dims: Tuple[int, ...], layernorm: bool, name_prefix: str):
        super().__init__()
        dims = [in_dim] + list(hidden_dims)
        self.fcs = nn.ModuleList()
        self.lns = nn.ModuleList()
        for i in range(len(dims) - 1):
            fc = nn.Linear(dims[i], dims[i + 1])
            self.fcs.append(fc)
            self.lns.append(nn.LayerNorm(dims[i + 1]) if layernorm else nn.Identity())
            # stable attr names for named_modules
            setattr(self, f"{name_prefix}_fc{i}", fc)
            setattr(self, f"{name_prefix}_ln{i}", self.lns[-1])

    def forward(self, x: torch.Tensor, return_features: bool = False):
        h = x
        feat = None
        for i, (fc, ln) in enumerate(zip(self.fcs, self.lns)):
            h = torch.tanh(ln(fc(h)))
            if i == len(self.fcs) - 1:
                feat = h
        if return_features:
            return h, feat
        return h


def build_backbone(in_dim: int, ppo_cfg: PPOConfig, name_prefix: str) -> nn.Module:
    act = _maybe_tanh_activation(ppo_cfg.activation)
    if act == "tanh":
        return TanhBackboneAdapter(in_dim, ppo_cfg.hidden_dims, ppo_cfg.layernorm, name_prefix=name_prefix)
    # default path: use networks.py backbone (relu/silu/gelu)
    return build_mlp_backbone(
        in_dim=in_dim,
        hidden_dims=list(ppo_cfg.hidden_dims),
        activation=act,
        layernorm=ppo_cfg.layernorm,
        name_prefix=name_prefix,
    )


def flatten_obs(obs: np.ndarray) -> np.ndarray:
    # For vector observations, vector env already returns [N, obs_dim]
    return obs


def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    # 1 - Var[y-yhat]/Var[y]
    var_y = torch.var(y_true)
    if var_y.item() < 1e-8:
        return float("nan")
    return float((1.0 - torch.var(y_true - y_pred) / (var_y + 1e-8)).item())


#    
# Rollout storage
#    

class RolloutBuffer:
    def __init__(self, num_steps: int, num_envs: int, obs_dim: int, action_shape: Tuple[int, ...], device: torch.device):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device

        self.obs = torch.zeros((num_steps, num_envs, obs_dim), device=device)
        self.actions = torch.zeros((num_steps, num_envs, *action_shape), device=device)
        self.logprobs = torch.zeros((num_steps, num_envs), device=device)
        self.rewards = torch.zeros((num_steps, num_envs), device=device)
        self.dones = torch.zeros((num_steps, num_envs), device=device)
        self.values = torch.zeros((num_steps, num_envs), device=device)

        self.advantages = torch.zeros((num_steps, num_envs), device=device)
        self.returns = torch.zeros((num_steps, num_envs), device=device)

        self._t = 0

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        logprob: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        t = self._t
        self.obs[t] = obs
        self.actions[t] = action
        self.logprobs[t] = logprob
        self.rewards[t] = reward
        self.dones[t] = done
        self.values[t] = value
        self._t += 1

    def compute_returns_and_advantages(self, next_value: torch.Tensor, next_done: torch.Tensor, gamma: float, gae_lambda: float) -> None:
        """
        GAE-Lambda advantage estimation.
        next_value: [num_envs]
        next_done: [num_envs] (0/1)
        """
        lastgaelam = torch.zeros((self.num_envs,), device=self.device)
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - self.dones[t + 1]
                nextvalues = self.values[t + 1]
            delta = self.rewards[t] + gamma * nextvalues * nextnonterminal - self.values[t]
            lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            self.advantages[t] = lastgaelam
        self.returns = self.advantages + self.values

    def get(self) -> Dict[str, torch.Tensor]:
        # flatten [T, N, ...] -> [T*N, ...]
        T, N = self.num_steps, self.num_envs
        b_obs = self.obs.reshape(T * N, -1)
        b_actions = self.actions.reshape(T * N, *self.actions.shape[2:])
        b_logprobs = self.logprobs.reshape(T * N)
        b_adv = self.advantages.reshape(T * N)
        b_returns = self.returns.reshape(T * N)
        b_values = self.values.reshape(T * N)
        return {
            "obs": b_obs,
            "actions": b_actions,
            "logprobs": b_logprobs,
            "advantages": b_adv,
            "returns": b_returns,
            "values": b_values,
        }


#    
# Instrumentation helpers
#    

@torch.no_grad()
def build_linear_name_map(model: nn.Module) -> Dict[int, str]:
    """
    Map id(module) -> name for all nn.Linear modules in a model.
    Useful to label activations by exact module name.
    """
    out: Dict[int, str] = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            out[id(m)] = name
    return out


@torch.no_grad()
def probe_backbone_postact_activations(
    backbone: nn.Module,
    backbone_name_map: Dict[int, str],
    obs_b: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Compute per-layer *post-activation* activations for an MLP backbone.

    Returns: dict linear_module_name -> acts [B,H]
    where acts correspond to post-(ln)->activation output for that layer.

    Supports:
      - build_mlp_backbone() from networks.py (has fcs, lns, act)
      - TanhBackboneAdapter (has fcs, lns)
    """
    acts: Dict[str, torch.Tensor] = {}
    h = obs_b

    # Common contract: backbone has fcs and lns lists
    fcs = getattr(backbone, "fcs", None)
    lns = getattr(backbone, "lns", None)
    if fcs is None or lns is None:
        raise ValueError("Backbone must expose .fcs and .lns for probing.")

    for fc, ln in zip(fcs, lns):
        pre = fc(h)
        post_ln = ln(pre)
        if isinstance(backbone, TanhBackboneAdapter):
            h = torch.tanh(post_ln)
        else:
            # networks.py backbone has .act
            act = getattr(backbone, "act", None)
            if act is None:
                raise ValueError("Non-tanh backbone must expose .act")
            h = act(post_ln)

        name = backbone_name_map.get(id(fc), None)
        if name is None:
            # Fallback: stable but less ideal
            name = f"linear_{len(acts)}"
        acts[name] = h.detach()

    return acts


@torch.no_grad()
def compute_global_dormant_frac(layer_stats: Dict[str, Any]) -> float:
    if not layer_stats:
        return 0.0
    return float(np.mean([st.frac_dormant for st in layer_stats.values()]))


def log_dormancy_report(logger: RunLogger, report, prefix: str, step: int) -> None:
    # log per-layer + global summaries
    layer_fracs = {f"{prefix}/layer_frac/{k}": v.frac_dormant for k, v in report.layer_stats.items()}
    logger.log_dict(layer_fracs, step=step)

    global_frac = compute_global_dormant_frac(report.layer_stats)
    logger.log_scalar(f"{prefix}/global_frac", global_frac, step=step)

    # optional dynamics keys if present
    for lname, st in report.layer_stats.items():
        if st.death_rate is not None:
            logger.log_scalar(f"{prefix}/death_rate/{lname}", st.death_rate, step=step)
        if st.revival_rate is not None:
            logger.log_scalar(f"{prefix}/revival_rate/{lname}", st.revival_rate, step=step)
        if st.overlap_prev is not None:
            logger.log_scalar(f"{prefix}/overlap_prev/{lname}", st.overlap_prev, step=step)


def pick_probe_batch_from_rollout(b_obs: torch.Tensor, batch_size: int) -> torch.Tensor:
    """
    b_obs: [T*N, obs_dim]
    Returns a (deterministic) probe batch: first batch_size samples.
    """
    B = min(int(batch_size), b_obs.size(0))
    return b_obs[:B]

def _compute_logprob_and_entropy(dist, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(dist, torch.distributions.Categorical):
        logprob = dist.log_prob(action.long())
        entropy = dist.entropy()
        return logprob, entropy
    # Normal for continuous (independent dims)
    logprob = dist.log_prob(action).sum(-1)
    entropy = dist.entropy().sum(-1)
    return logprob, entropy


def _make_policy_value(
    cfg: DictConfig,
    obs_dim: int,
    act_kind: str,
    act_dim: int,
    device: torch.device,
):
    ppo_cfg = load_ppo_config(cfg)
    hidden_dims = ppo_cfg.hidden_dims
    hidden_last = hidden_dims[-1]

    if ppo_cfg.shared_backbone:
        trunk = build_backbone(obs_dim, ppo_cfg, name_prefix="shared")
        if act_kind == "discrete":
            actor_head = nn.Linear(hidden_last, act_dim)
            actor = None
        else:
            actor_mu = nn.Linear(hidden_last, act_dim)
            actor_log_std = nn.Parameter(torch.ones(act_dim) * -0.5)
            actor = None
        critic_head = nn.Linear(hidden_last, 1)

        class SharedActorCritic(nn.Module):
            def __init__(self):
                super().__init__()
                self.trunk = trunk
                self.actor_head = actor_head if act_kind == "discrete" else actor_mu
                self.actor_log_std = actor_log_std if act_kind != "discrete" else None
                self.critic_head = critic_head

            def get_features(self, obs: torch.Tensor) -> torch.Tensor:
                _, feat = self.trunk(obs, return_features=True)
                return feat

            def get_value(self, obs: torch.Tensor) -> torch.Tensor:
                h = self.trunk(obs, return_features=False)
                return self.critic_head(h).squeeze(-1)

            def get_dist(self, obs: torch.Tensor):
                h = self.trunk(obs, return_features=False)
                if act_kind == "discrete":
                    logits = self.actor_head(h)
                    return torch.distributions.Categorical(logits=logits)
                mu = self.actor_head(h)
                log_std = self.actor_log_std.expand_as(mu)
                return torch.distributions.Normal(mu, torch.exp(log_std))

        model = SharedActorCritic().to(device)
        return model, ppo_cfg

    # Separate actor/critic backbones (recommended for clean ReDo scoping)
    actor_backbone = build_backbone(obs_dim, ppo_cfg, name_prefix="actor")
    critic_backbone = build_backbone(obs_dim, ppo_cfg, name_prefix="critic")

    if act_kind == "discrete":
        actor = nn.Sequential(actor_backbone, nn.Linear(hidden_last, act_dim))
    else:
        # state-independent log_std like common PPO
        class GaussianActor(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = actor_backbone
                self.mu = nn.Linear(hidden_last, act_dim)
                self.log_std = nn.Parameter(torch.ones(act_dim) * -0.5)

            def forward(self, obs: torch.Tensor):
                h = self.backbone(obs, return_features=False)
                mu = self.mu(h)
                log_std = self.log_std.expand_as(mu)
                return mu, log_std

            def dist(self, obs: torch.Tensor):
                mu, log_std = self.forward(obs)
                return torch.distributions.Normal(mu, torch.exp(log_std))

            def get_features(self, obs: torch.Tensor) -> torch.Tensor:
                _, feat = self.backbone(obs, return_features=True)
                return feat

        actor = GaussianActor()

    class Critic(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = critic_backbone
            self.v = nn.Linear(hidden_last, 1)

        def forward(self, obs: torch.Tensor):
            h = self.backbone(obs, return_features=False)
            return self.v(h).squeeze(-1)

        def get_features(self, obs: torch.Tensor) -> torch.Tensor:
            _, feat = self.backbone(obs, return_features=True)
            return feat

    critic = Critic()
    actor = actor.to(device)
    critic = critic.to(device)

    return (actor, critic), ppo_cfg


def _get_actor_dist(model, obs: torch.Tensor, act_kind: str):
    if isinstance(model, nn.Module) and hasattr(model, "get_dist"):
        return model.get_dist(obs)
    if act_kind == "discrete":
        logits = model(obs)
        return torch.distributions.Categorical(logits=logits)
    # GaussianActor
    return model.dist(obs)


def _get_value(model_or_critic, obs: torch.Tensor):
    if isinstance(model_or_critic, nn.Module) and hasattr(model_or_critic, "get_value"):
        return model_or_critic.get_value(obs)
    return model_or_critic(obs)


def _get_features(model_or_actorcritic, obs: torch.Tensor, which: str) -> torch.Tensor:
    """
    which: "shared" | "actor" | "critic"
    """
    if which == "shared":
        return model_or_actorcritic.get_features(obs)
    # separate
    if which == "actor":
        return model_or_actorcritic.get_features(obs)
    if which == "critic":
        return model_or_actorcritic.get_features(obs)
    raise ValueError(which)


def train(cfg: DictConfig, envs: gym.vector.VectorEnv, device: torch.device, logger: RunLogger) -> None:
    ppo_cfg = load_ppo_config(cfg)

    obs_dim = infer_obs_dim(envs.single_observation_space)
    act_kind, act_dim = infer_act_dims(envs.single_action_space)
    num_envs = envs.num_envs

    # Build model(s)
    model_bundle, ppo_cfg = _make_policy_value(cfg, obs_dim, act_kind, act_dim, device=device)

    # Optimizer over all params
    if isinstance(model_bundle, nn.Module):
        params = list(model_bundle.parameters())
    else:
        actor, critic = model_bundle
        params = list(actor.parameters()) + list(critic.parameters())
    optimizer = optim.Adam(params, lr=ppo_cfg.lr, eps=1e-5)

    # Rollout buffer
    action_shape = () if act_kind == "discrete" else (act_dim,)
    rb = RolloutBuffer(ppo_cfg.num_steps, num_envs, obs_dim, action_shape, device=device)

    # Instrumentation setup
    instr = cfg.instrumentation
    instr_enabled = bool(instr.enabled)

    metric_every = int(instr.metric_every_updates)
    heavy_every = int(instr.heavy_metric_every_updates)
    probe_bs = int(instr.probe_batch_size)

    tau = float(instr.tau)
    grad_q = float(instr.grad_quantile)
    eps = float(instr.eps)

    # CKA refs
    cka_mode = str(instr.repr_metrics.cka_reference)
    cka_beta = float(instr.repr_metrics.cka_ema_beta)
    cka_ref_actor = CKAReference(mode=cka_mode, beta=cka_beta) if bool(instr.repr_metrics.linear_cka) else None
    cka_ref_critic = CKAReference(mode=cka_mode, beta=cka_beta) if bool(instr.repr_metrics.linear_cka) else None

    # Track previous dormancy masks for event rates
    prev_act_masks: Dict[str, torch.Tensor] = {}
    prev_grad_masks: Dict[str, torch.Tensor] = {}

    # ReDo scheduler
    redo_cfg = cfg.redo
    redo_enabled = bool(redo_cfg.enabled)
    scheduler = ReDoScheduler(
        mode=str(redo_cfg.mode),
        redo_every_updates=int(redo_cfg.redo_every_updates),
        target_dormant_frac=float(redo_cfg.target_dormant_frac),
        patience=int(redo_cfg.patience),
    )

    # Determine which "scope" to probe/recycle
    # For PPO we’ll probe & recycle backbones (actor/critic or shared trunk).
    shared = isinstance(model_bundle, nn.Module)
    if shared:
        trunk = model_bundle.trunk
        trunk_name_map = build_linear_name_map(trunk)
    else:
        actor, critic = model_bundle
        actor_trunk = actor.backbone if hasattr(actor, "backbone") else actor[0]  # type: ignore
        critic_trunk = critic.backbone
        actor_name_map = build_linear_name_map(actor_trunk)
        critic_name_map = build_linear_name_map(critic_trunk)

    # Training loop counters
    total_env_steps = int(cfg.run.total_env_steps)
    batch_size = ppo_cfg.num_steps * num_envs
    minibatch_size = batch_size // ppo_cfg.num_minibatches
    num_updates = total_env_steps // batch_size

    # Init env
    obs, _ = envs.reset(seed=int(cfg.seed))
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)

    start_time = time.time()
    global_env_step = 0
    global_update_step = 0

    for update in range(1, num_updates + 1):
        global_update_step = update
        logger.set_update_step(update)

        # LR annealing
        if ppo_cfg.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lr_now = frac * ppo_cfg.lr
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now
            logger.log_scalar("charts/lr", lr_now, step=global_env_step)

        #  - Rollout collection  -
        rb._t = 0
        for step in range(ppo_cfg.num_steps):
            global_env_step += num_envs
            logger.set_env_step(global_env_step)

            with torch.no_grad():
                if shared:
                    dist = model_bundle.get_dist(obs_t)
                    value = model_bundle.get_value(obs_t)
                else:
                    dist = _get_actor_dist(actor, obs_t, act_kind)
                    value = critic(obs_t)

                if act_kind == "discrete":
                    action = dist.sample()
                else:
                    action = dist.sample()

                logprob, _entropy = _compute_logprob_and_entropy(dist, action)

            # Step env
            act_np = action.cpu().numpy()
            next_obs, reward, terminated, truncated, infos = envs.step(act_np)
            done = np.logical_or(terminated, truncated)

            # store
            rb.add(
                obs=obs_t,
                action=action if act_kind != "discrete" else action.unsqueeze(-1),  # store as tensor
                logprob=logprob,
                reward=torch.tensor(reward, dtype=torch.float32, device=device),
                done=torch.tensor(done, dtype=torch.float32, device=device),
                value=value,
            )

            # Logging episodic stats
            if "final_info" in infos:
                for fi in infos["final_info"]:
                    if fi is None:
                        continue
                    if "episode" in fi:
                        ep = fi["episode"]
                        logger.log_scalar("charts/episodic_return", ep["r"], step=global_env_step)
                        logger.log_scalar("charts/episodic_length", ep["l"], step=global_env_step)

            obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)

        # bootstrap value
        with torch.no_grad():
            next_done = rb.dones[-1]
            if shared:
                next_value = model_bundle.get_value(obs_t)
            else:
                next_value = critic(obs_t)

        rb.compute_returns_and_advantages(next_value, next_done, gamma=ppo_cfg.gamma, gae_lambda=ppo_cfg.gae_lambda)

        batch = rb.get()
        b_obs = batch["obs"]
        b_actions = batch["actions"]
        b_logprobs = batch["logprobs"]
        b_adv = batch["advantages"]
        b_returns = batch["returns"]
        b_values = batch["values"]

        if ppo_cfg.norm_adv:
            b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        #  - PPO optimization  -
        inds = np.arange(batch_size)
        approx_kl = 0.0

        for epoch in range(ppo_cfg.update_epochs):
            np.random.shuffle(inds)
            for start in range(0, batch_size, minibatch_size):
                mb_inds = inds[start : start + minibatch_size]

                mb_obs = b_obs[mb_inds]
                if act_kind == "discrete":
                    mb_actions = b_actions[mb_inds].long().squeeze(-1)
                else:
                    mb_actions = b_actions[mb_inds]

                # forward
                if shared:
                    dist = model_bundle.get_dist(mb_obs)
                    newvalue = model_bundle.get_value(mb_obs)
                else:
                    dist = _get_actor_dist(actor, mb_obs, act_kind)
                    newvalue = critic(mb_obs)

                newlogprob, entropy = _compute_logprob_and_entropy(dist, mb_actions)
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                # policy loss
                pg_loss1 = -b_adv[mb_inds] * ratio
                pg_loss2 = -b_adv[mb_inds] * torch.clamp(ratio, 1 - ppo_cfg.clip_coef, 1 + ppo_cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # value loss
                if ppo_cfg.clip_vloss:
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -ppo_cfg.clip_coef, ppo_cfg.clip_coef
                    )
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ppo_cfg.ent_coef * entropy_loss + ppo_cfg.vf_coef * v_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(params, ppo_cfg.max_grad_norm)

                # Gradient-based dormancy needs grads: measure before step if scheduled this update
                do_metrics = instr_enabled and (update % metric_every == 0) and (epoch == 0) and (start == 0)
                grad_norms_actor = None
                grad_norms_critic = None
                if do_metrics:
                    if shared:
                        # measure trunk linears
                        layer_names = [n for n, _ in list_named_linears(model_bundle.trunk)]
                        grad_norms_actor = linear_row_grad_norms(model_bundle.trunk, layer_names)
                        grad_norms_critic = grad_norms_actor
                    else:
                        a_trunk = actor.backbone if hasattr(actor, "backbone") else actor[0]  # type: ignore
                        c_trunk = critic.backbone
                        layer_names_a = [n for n, _ in list_named_linears(a_trunk)]
                        layer_names_c = [n for n, _ in list_named_linears(c_trunk)]
                        grad_norms_actor = linear_row_grad_norms(a_trunk, layer_names_a)
                        grad_norms_critic = linear_row_grad_norms(c_trunk, layer_names_c)

                optimizer.step()

                # KL diagnostics
                with torch.no_grad():
                    approx_kl = float(((ratio - 1) - logratio).mean().item())

                if ppo_cfg.target_kl is not None and approx_kl > float(ppo_cfg.target_kl):
                    break

            if ppo_cfg.target_kl is not None and approx_kl > float(ppo_cfg.target_kl):
                break

        #  - Logging PPO losses (once per update)  -
        y_pred = b_values
        y_true = b_returns
        ev = explained_variance(y_pred, y_true)
        logger.log_dict(
            {
                "losses/approx_kl": approx_kl,
                "losses/explained_variance": ev,
            },
            step=global_env_step,
        )
        logger.log_scalar("charts/sps", int(global_env_step / (time.time() - start_time)), step=global_env_step)

        #  - Instrumentation: dormancy + representation  -
        if instr_enabled and (update % metric_every == 0):
            with torch.no_grad():
                probe_obs = pick_probe_batch_from_rollout(b_obs, probe_bs)

                # Activation-based dormancy uses post-act per layer for backbones
                if shared:
                    act_map = probe_backbone_postact_activations(model_bundle.trunk, trunk_name_map, probe_obs)
                    act_report = compute_activation_dormancy(
                        activations=act_map, tau=tau, eps=eps, prev_masks=prev_act_masks
                    )
                    prev_act_masks = {k: v.clone() for k, v in act_report.layer_masks.items()}
                    log_dormancy_report(logger, act_report, prefix="dormancy/activation/shared", step=global_env_step)

                    # Gradient-based dormancy
                    if grad_norms_actor is not None:
                        grad_report = compute_gradient_dormancy(
                            row_grad_norms=grad_norms_actor, q=grad_q, prev_masks=prev_grad_masks
                        )
                        prev_grad_masks = {k: v.clone() for k, v in grad_report.layer_masks.items()}
                        log_dormancy_report(logger, grad_report, prefix="dormancy/gradient/shared", step=global_env_step)

                    # Representation metrics on trunk features (penultimate)
                    Z = model_bundle.get_features(probe_obs)
                    rm = compute_repr_metrics(
                        Z,
                        do_effective_rank=bool(instr.repr_metrics.effective_rank),
                        do_cosine_div=bool(instr.repr_metrics.cosine_diversity),
                        svd_topk=int(instr.repr_metrics.svd_topk),
                        cka_ref=cka_ref_actor,
                    )
                    # log scalars + store svd list as text/jsonl
                    for k, v in rm.items():
                        if isinstance(v, (int, float)):
                            logger.log_scalar(k, v, step=global_env_step)
                        else:
                            logger.log_text(k, str(v), step=global_env_step)

                else:
                    a_trunk = actor.backbone if hasattr(actor, "backbone") else actor[0]  # type: ignore
                    c_trunk = critic.backbone

                    act_a = probe_backbone_postact_activations(a_trunk, actor_name_map, probe_obs)
                    act_c = probe_backbone_postact_activations(c_trunk, critic_name_map, probe_obs)

                    act_report_a = compute_activation_dormancy(act_a, tau=tau, eps=eps, prev_masks=prev_act_masks)
                    prev_act_masks = {k: v.clone() for k, v in act_report_a.layer_masks.items()}
                    log_dormancy_report(logger, act_report_a, prefix="dormancy/activation/actor", step=global_env_step)

                    act_report_c = compute_activation_dormancy(act_c, tau=tau, eps=eps, prev_masks=None)
                    log_dormancy_report(logger, act_report_c, prefix="dormancy/activation/critic", step=global_env_step)

                    if grad_norms_actor is not None:
                        grad_report_a = compute_gradient_dormancy(grad_norms_actor, q=grad_q, prev_masks=prev_grad_masks)
                        prev_grad_masks = {k: v.clone() for k, v in grad_report_a.layer_masks.items()}
                        log_dormancy_report(logger, grad_report_a, prefix="dormancy/gradient/actor", step=global_env_step)

                    if grad_norms_critic is not None:
                        grad_report_c = compute_gradient_dormancy(grad_norms_critic, q=grad_q, prev_masks=None)
                        log_dormancy_report(logger, grad_report_c, prefix="dormancy/gradient/critic", step=global_env_step)

                    # Representation metrics
                    Za = actor.get_features(probe_obs) if hasattr(actor, "get_features") else a_trunk(probe_obs)  # type: ignore
                    Zc = critic.get_features(probe_obs)
                    rm_a = compute_repr_metrics(
                        Za,
                        do_effective_rank=bool(instr.repr_metrics.effective_rank),
                        do_cosine_div=bool(instr.repr_metrics.cosine_diversity),
                        svd_topk=int(instr.repr_metrics.svd_topk),
                        cka_ref=cka_ref_actor,
                    )
                    rm_c = compute_repr_metrics(
                        Zc,
                        do_effective_rank=bool(instr.repr_metrics.effective_rank),
                        do_cosine_div=bool(instr.repr_metrics.cosine_diversity),
                        svd_topk=int(instr.repr_metrics.svd_topk),
                        cka_ref=cka_ref_critic,
                    )
                    for k, v in rm_a.items():
                        tag = f"actor_{k}"
                        if isinstance(v, (int, float)):
                            logger.log_scalar(tag, v, step=global_env_step)
                        else:
                            logger.log_text(tag, str(v), step=global_env_step)
                    for k, v in rm_c.items():
                        tag = f"critic_{k}"
                        if isinstance(v, (int, float)):
                            logger.log_scalar(tag, v, step=global_env_step)
                        else:
                            logger.log_text(tag, str(v), step=global_env_step)

        #  - ReDo integration  -
        if redo_enabled:
            # Decide which mask to use for recycling:
            # If selection=activation, use most recent activation masks.
            # If selection=gradient, use most recent gradient masks.
            selection = str(redo_cfg.selection).lower()

            # Build a "global dormant frac" from whichever report is available.
            # For conditioned mode, we compute from last stored masks.
            dormant_frac_global = None
            if selection == "activation" and prev_act_masks:
                dormant_frac_global = float(np.mean([m.float().mean().item() for m in prev_act_masks.values()]))
            if selection == "gradient" and prev_grad_masks:
                dormant_frac_global = float(np.mean([m.float().mean().item() for m in prev_grad_masks.values()]))

            if scheduler.should_redo(update_step=update, dormant_frac_global=dormant_frac_global):
                # Apply to scope
                init_mode = str(redo_cfg.init_mode)
                outgoing = str(redo_cfg.outgoing)
                reset_bias = bool(redo_cfg.reset_bias)
                max_frac = float(redo_cfg.max_recycled_frac_per_layer)
                opt_for_reset = optimizer if bool(redo_cfg.reset_optimizer_state) else None

                # Determine mask dict based on selection
                masks = prev_act_masks if selection == "activation" else prev_grad_masks

                # Safety: if no masks, skip
                if masks:
                    if shared:
                        # recycle sequential linears in trunk
                        result = redo_apply_on_sequential_linears(
                            model=model_bundle.trunk,
                            dormant_masks=masks,
                            optimizer=opt_for_reset,
                            init_mode=init_mode,
                            reset_bias=reset_bias,
                            outgoing=outgoing,
                            max_frac=max_frac,
                        )
                        logger.log_scalar("redo/total_recycled", result.total_recycled, step=global_env_step)
                        for lname, k in result.recycled_by_layer.items():
                            logger.log_scalar(f"redo/recycled/{lname}", k, step=global_env_step)
                    else:
                        a_trunk = actor.backbone if hasattr(actor, "backbone") else actor[0]  # type: ignore
                        c_trunk = critic.backbone

                        scope = str(redo_cfg.scope).lower()
                        if scope in ("shared_trunk", "policy_only"):
                            res_a = redo_apply_on_sequential_linears(
                                model=a_trunk,
                                dormant_masks=masks,
                                optimizer=opt_for_reset,
                                init_mode=init_mode,
                                reset_bias=reset_bias,
                                outgoing=outgoing,
                                max_frac=max_frac,
                            )
                            logger.log_scalar("redo/actor_total_recycled", res_a.total_recycled, step=global_env_step)

                        if scope in ("shared_trunk", "value_only", "all"):
                            res_c = redo_apply_on_sequential_linears(
                                model=c_trunk,
                                dormant_masks=masks,
                                optimizer=opt_for_reset,
                                init_mode=init_mode,
                                reset_bias=reset_bias,
                                outgoing=outgoing,
                                max_frac=max_frac,
                            )
                            logger.log_scalar("redo/critic_total_recycled", res_c.total_recycled, step=global_env_step)

                # reset conditioned state so we don't immediately redo again
                if str(redo_cfg.mode).lower() == "conditioned":
                    scheduler.reset()

        logger.flush()
