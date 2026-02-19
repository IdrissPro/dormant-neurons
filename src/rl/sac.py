
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from omegaconf import DictConfig

from src.logging.logger import RunLogger
from src.rl.networks import infer_obs_dim, infer_act_dims, build_mlp_backbone
from src.instrumentation.dormancy import compute_activation_dormancy, compute_gradient_dormancy
from src.instrumentation.repr_metrics import CKAReference, compute_repr_metrics
from src.instrumentation.hooks import linear_row_grad_norms, list_named_linears
from src.redo.recycle import redo_apply_on_sequential_linears
from src.redo.schedules import ReDoScheduler


#    
# Config
#    

@dataclass
class SACConfig:
    # Replay / training
    buffer_size: int = 1_000_000
    learning_starts: int = 10_000
    batch_size: int = 256
    train_frequency: int = 1
    gradient_steps: int = 1

    # Optimization
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    alpha: float = 0.2              # if autotune disabled
    autotune: bool = True
    target_entropy: Optional[float] = None

    # Network
    hidden_dims: Tuple[int, ...] = (256, 256)
    activation: str = "relu"
    layernorm: bool = False

    # Misc
    policy_delay: int = 1           # update actor every N critic updates (often 1)
    seed: int = 1


def _get(cfg: DictConfig, path: str, default: Any) -> Any:
    cur: Any = cfg
    for key in path.split("."):
        if not hasattr(cur, key):
            return default
        cur = getattr(cur, key)
    return cur if cur is not None else default


def load_sac_config(cfg: DictConfig) -> SACConfig:
    hidden = _get(cfg, "algo.hidden_dims", list(SACConfig.hidden_dims))
    return SACConfig(
        buffer_size=int(_get(cfg, "algo.buffer_size", SACConfig.buffer_size)),
        learning_starts=int(_get(cfg, "algo.learning_starts", SACConfig.learning_starts)),
        batch_size=int(_get(cfg, "algo.batch_size", SACConfig.batch_size)),
        train_frequency=int(_get(cfg, "algo.train_frequency", SACConfig.train_frequency)),
        gradient_steps=int(_get(cfg, "algo.gradient_steps", SACConfig.gradient_steps)),
        gamma=float(_get(cfg, "algo.gamma", SACConfig.gamma)),
        tau=float(_get(cfg, "algo.tau", SACConfig.tau)),
        lr=float(_get(cfg, "algo.lr", SACConfig.lr)),
        alpha=float(_get(cfg, "algo.alpha", SACConfig.alpha)),
        autotune=bool(_get(cfg, "algo.autotune", SACConfig.autotune)),
        target_entropy=_get(cfg, "algo.target_entropy", SACConfig.target_entropy),
        hidden_dims=tuple(int(x) for x in hidden),
        activation=str(_get(cfg, "algo.activation", SACConfig.activation)),
        layernorm=bool(_get(cfg, "algo.layernorm", SACConfig.layernorm)),
        policy_delay=int(_get(cfg, "algo.policy_delay", SACConfig.policy_delay)),
        seed=int(_get(cfg, "seed", SACConfig.seed)),
    )


#    
# Replay buffer (vector-env aware)
#    

class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, size: int):
        self.size = int(size)
        self.obs = np.zeros((self.size, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.size, act_dim), dtype=np.float32)
        self.rewards = np.zeros((self.size,), dtype=np.float32)
        self.dones = np.zeros((self.size,), dtype=np.float32)

        self._idx = 0
        self._full = False

    def __len__(self) -> int:
        return self.size if self._full else self._idx

    def add_batch(
        self,
        obs: np.ndarray,          # [N, obs_dim]
        actions: np.ndarray,      # [N, act_dim]
        rewards: np.ndarray,      # [N]
        next_obs: np.ndarray,     # [N, obs_dim]
        dones: np.ndarray,        # [N]
    ) -> None:
        N = obs.shape[0]
        for i in range(N):
            self.obs[self._idx] = obs[i]
            self.actions[self._idx] = actions[i]
            self.rewards[self._idx] = rewards[i]
            self.next_obs[self._idx] = next_obs[i]
            self.dones[self._idx] = float(dones[i])

            self._idx += 1
            if self._idx >= self.size:
                self._idx = 0
                self._full = True

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        n = len(self)
        if n <= 0:
            raise RuntimeError("ReplayBuffer is empty")
        idx = np.random.randint(0, n, size=int(batch_size))
        return {
            "obs": self.obs[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_obs": self.next_obs[idx],
            "dones": self.dones[idx],
        }


#    
# Networks
#    

LOG_STD_MIN = -20
LOG_STD_MAX = 2


class SquashedGaussianActor(nn.Module):
    """
    Actor outputs a squashed Gaussian policy:
      a = tanh(u), u ~ N(mu, std)
    Includes correction term in logprob.
    Exposes:
      - backbone for dormancy probing
      - get_features(obs)
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: Tuple[int, ...], activation: str, layernorm: bool):
        super().__init__()
        self.backbone = build_mlp_backbone(
            in_dim=obs_dim,
            hidden_dims=list(hidden_dims),
            activation=activation,
            layernorm=layernorm,
            name_prefix="actor",
        )
        hdim = hidden_dims[-1]
        self.mu = nn.Linear(hdim, act_dim)
        self.log_std = nn.Linear(hdim, act_dim)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(obs, return_features=False)
        mu = self.mu(h)
        log_std = self.log_std(h)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    @torch.no_grad()
    def get_features(self, obs: torch.Tensor) -> torch.Tensor:
        _, feat = self.backbone(obs, return_features=True)
        return feat

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          action: [B, act_dim] in [-1,1]
          log_prob: [B]
        """
        mu, log_std = self.forward(obs)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std)
        u = dist.rsample()
        a = torch.tanh(u)

        # logprob correction: log(1 - tanh(u)^2)
        log_prob = dist.log_prob(u).sum(-1)
        log_prob -= torch.log(1.0 - a.pow(2) + 1e-6).sum(-1)
        return a, log_prob


class CriticQ(nn.Module):
    """
    Q(s,a) with input concat([obs, action]).
    Exposes backbone for dormancy probing, and get_features(x).
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: Tuple[int, ...], activation: str, layernorm: bool, name_prefix: str):
        super().__init__()
        in_dim = obs_dim + act_dim
        self.backbone = build_mlp_backbone(
            in_dim=in_dim,
            hidden_dims=list(hidden_dims),
            activation=activation,
            layernorm=layernorm,
            name_prefix=name_prefix,
        )
        hdim = hidden_dims[-1]
        self.q = nn.Linear(hdim, 1)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act], dim=-1)
        h = self.backbone(x, return_features=False)
        return self.q(h).squeeze(-1)

    @torch.no_grad()
    def get_features(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act], dim=-1)
        _, feat = self.backbone(x, return_features=True)
        return feat


@torch.no_grad()
def build_linear_name_map(module: nn.Module) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for name, m in module.named_modules():
        if isinstance(m, nn.Linear):
            out[id(m)] = name
    return out


@torch.no_grad()
def probe_backbone_postact_activations(backbone: nn.Module, name_map: Dict[int, str], x_b: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Returns dict: linear_name -> post-activation outputs [B,H]
    Assumes MLPBackbone from networks.py (has fcs, lns, act).
    """
    acts: Dict[str, torch.Tensor] = {}
    h = x_b
    fcs = getattr(backbone, "fcs", None)
    lns = getattr(backbone, "lns", None)
    act = getattr(backbone, "act", None)
    if fcs is None or lns is None or act is None:
        raise ValueError("Backbone must expose .fcs, .lns, .act for probing.")
    for fc, ln in zip(fcs, lns):
        h = act(ln(fc(h)))
        lname = name_map.get(id(fc), None) or f"linear_{len(acts)}"
        acts[lname] = h.detach()
    return acts


def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for p, tp in zip(source.parameters(), target.parameters()):
            tp.data.mul_(1.0 - tau).add_(p.data, alpha=tau)


def train(cfg: DictConfig, envs: gym.vector.VectorEnv, device: torch.device, logger: RunLogger) -> None:
    sac_cfg: SACConfig = __import__("src.rl.sac", fromlist=["load_sac_config"]).load_sac_config(cfg)  # type: ignore

    obs_dim = infer_obs_dim(envs.single_observation_space)
    act_kind, act_dim = infer_act_dims(envs.single_action_space)
    if act_kind != "continuous":
        raise ValueError("SAC only supports continuous (Box) action spaces.")

    num_envs = envs.num_envs
    total_env_steps = int(cfg.run.total_env_steps)

    actor = SquashedGaussianActor(obs_dim, act_dim, sac_cfg.hidden_dims, sac_cfg.activation, sac_cfg.layernorm).to(device)
    q1 = CriticQ(obs_dim, act_dim, sac_cfg.hidden_dims, sac_cfg.activation, sac_cfg.layernorm, name_prefix="q1").to(device)
    q2 = CriticQ(obs_dim, act_dim, sac_cfg.hidden_dims, sac_cfg.activation, sac_cfg.layernorm, name_prefix="q2").to(device)

    q1_t = CriticQ(obs_dim, act_dim, sac_cfg.hidden_dims, sac_cfg.activation, sac_cfg.layernorm, name_prefix="q1t").to(device)
    q2_t = CriticQ(obs_dim, act_dim, sac_cfg.hidden_dims, sac_cfg.activation, sac_cfg.layernorm, name_prefix="q2t").to(device)
    q1_t.load_state_dict(q1.state_dict())
    q2_t.load_state_dict(q2.state_dict())
    q1_t.eval()
    q2_t.eval()

    actor_opt = optim.Adam(actor.parameters(), lr=sac_cfg.lr, eps=1e-5)
    q_opt = optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=sac_cfg.lr, eps=1e-5)

    # Entropy temperature
    if sac_cfg.autotune:
        log_alpha = torch.zeros((), device=device, requires_grad=True)
        alpha_opt = optim.Adam([log_alpha], lr=sac_cfg.lr, eps=1e-5)
        if sac_cfg.target_entropy is None:
            target_entropy = -float(act_dim)
        else:
            target_entropy = float(sac_cfg.target_entropy)
    else:
        log_alpha = None
        alpha_opt = None
        target_entropy = None

    rb = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=sac_cfg.buffer_size)

    # Instrumentation
    instr = cfg.instrumentation
    instr_enabled = bool(instr.enabled)
    metric_every_updates = int(instr.metric_every_updates)
    heavy_every_updates = int(instr.heavy_metric_every_updates)
    probe_bs = int(instr.probe_batch_size)

    tau_act = float(instr.tau)
    grad_q = float(instr.grad_quantile)
    eps = float(instr.eps)

    cka_ref_actor = None
    cka_ref_q1 = None
    cka_ref_q2 = None
    if bool(instr.repr_metrics.linear_cka):
        mode = str(instr.repr_metrics.cka_reference)
        beta = float(instr.repr_metrics.cka_ema_beta)
        cka_ref_actor = CKAReference(mode=mode, beta=beta)
        cka_ref_q1 = CKAReference(mode=mode, beta=beta)
        cka_ref_q2 = CKAReference(mode=mode, beta=beta)

    prev_act_masks_actor: Dict[str, torch.Tensor] = {}
    prev_act_masks_q1: Dict[str, torch.Tensor] = {}
    prev_act_masks_q2: Dict[str, torch.Tensor] = {}
    prev_grad_masks_actor: Dict[str, torch.Tensor] = {}
    prev_grad_masks_q1: Dict[str, torch.Tensor] = {}
    prev_grad_masks_q2: Dict[str, torch.Tensor] = {}

    # Layer name maps for probing
    actor_name_map = build_linear_name_map(actor.backbone)
    q1_name_map = build_linear_name_map(q1.backbone)
    q2_name_map = build_linear_name_map(q2.backbone)

    actor_linear_names = [n for n, _ in list_named_linears(actor.backbone)]
    q1_linear_names = [n for n, _ in list_named_linears(q1.backbone)]
    q2_linear_names = [n for n, _ in list_named_linears(q2.backbone)]

    # ReDo
    redo_cfg = cfg.redo
    redo_enabled = bool(redo_cfg.enabled)
    scheduler = ReDoScheduler(
        mode=str(redo_cfg.mode),
        redo_every_updates=int(redo_cfg.redo_every_updates),
        target_dormant_frac=float(redo_cfg.target_dormant_frac),
        patience=int(redo_cfg.patience),
    )

    # Env init
    obs, _ = envs.reset(seed=int(cfg.seed))
    global_step = 0
    update_step = 0
    start_time = time.time()

    while global_step < total_env_steps:
        logger.set_env_step(global_step)

        # Collect action (random until learning_starts)
        if global_step < sac_cfg.learning_starts:
            actions = np.stack([envs.single_action_space.sample() for _ in range(num_envs)], axis=0).astype(np.float32)
        else:
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
                a, _logp = actor.sample(obs_t)
                actions = a.cpu().numpy().astype(np.float32)

        next_obs, rewards, terminated, truncated, infos = envs.step(actions)
        dones = np.logical_or(terminated, truncated)

        # Log episodic returns
        if "final_info" in infos:
            for fi in infos["final_info"]:
                if fi is None:
                    continue
                if "episode" in fi:
                    ep = fi["episode"]
                    logger.log_scalar("charts/episodic_return", ep["r"], step=global_step)
                    logger.log_scalar("charts/episodic_length", ep["l"], step=global_step)

        # Store transitions
        rb.add_batch(
            obs=obs.astype(np.float32),
            actions=actions.astype(np.float32),
            rewards=rewards.astype(np.float32),
            next_obs=next_obs.astype(np.float32),
            dones=dones.astype(np.float32),
        )
        obs = next_obs
        global_step += num_envs
        logger.set_env_step(global_step)

        # Train
        if global_step >= sac_cfg.learning_starts and (global_step // num_envs) % sac_cfg.train_frequency == 0:
            for g in range(sac_cfg.gradient_steps):
                update_step += 1
                logger.set_update_step(update_step)

                batch = rb.sample(sac_cfg.batch_size)
                b_obs = torch.tensor(batch["obs"], dtype=torch.float32, device=device)
                b_actions = torch.tensor(batch["actions"], dtype=torch.float32, device=device)
                b_rewards = torch.tensor(batch["rewards"], dtype=torch.float32, device=device)
                b_next_obs = torch.tensor(batch["next_obs"], dtype=torch.float32, device=device)
                b_dones = torch.tensor(batch["dones"], dtype=torch.float32, device=device)

                # Alpha
                if sac_cfg.autotune:
                    alpha = log_alpha.exp()
                else:
                    alpha = torch.tensor(sac_cfg.alpha, device=device)

                #  - Critic update  -
                with torch.no_grad():
                    next_a, next_logp = actor.sample(b_next_obs)
                    q1_next = q1_t(b_next_obs, next_a)
                    q2_next = q2_t(b_next_obs, next_a)
                    q_next = torch.min(q1_next, q2_next) - alpha * next_logp
                    target_q = b_rewards + sac_cfg.gamma * (1.0 - b_dones) * q_next

                q1_pred = q1(b_obs, b_actions)
                q2_pred = q2(b_obs, b_actions)
                q1_loss = nn.functional.mse_loss(q1_pred, target_q)
                q2_loss = nn.functional.mse_loss(q2_pred, target_q)
                q_loss = q1_loss + q2_loss

                q_opt.zero_grad(set_to_none=True)
                q_loss.backward()

                # Gradient dormancy measurement BEFORE step (when scheduled)
                do_metrics = instr_enabled and (update_step % metric_every_updates == 0)
                grad_actor = None
                grad_q1 = None
                grad_q2 = None
                if do_metrics:
                    grad_q1 = linear_row_grad_norms(q1.backbone, q1_linear_names)
                    grad_q2 = linear_row_grad_norms(q2.backbone, q2_linear_names)

                q_opt.step()

                #  - Actor update (possibly delayed)  -
                if update_step % sac_cfg.policy_delay == 0:
                    # Freeze critics for actor update
                    for p in q1.parameters():
                        p.requires_grad_(False)
                    for p in q2.parameters():
                        p.requires_grad_(False)

                    a, logp = actor.sample(b_obs)
                    q1_pi = q1(b_obs, a)
                    q2_pi = q2(b_obs, a)
                    q_pi = torch.min(q1_pi, q2_pi)
                    actor_loss = (alpha * logp - q_pi).mean()

                    actor_opt.zero_grad(set_to_none=True)
                    actor_loss.backward()

                    if do_metrics:
                        grad_actor = linear_row_grad_norms(actor.backbone, actor_linear_names)

                    actor_opt.step()

                    # Unfreeze critics
                    for p in q1.parameters():
                        p.requires_grad_(True)
                    for p in q2.parameters():
                        p.requires_grad_(True)
                else:
                    actor_loss = torch.tensor(0.0)

                #  - Alpha autotune  -
                if sac_cfg.autotune and (update_step % sac_cfg.policy_delay == 0):
                    # logp from actor update batch
                    alpha_loss = -(log_alpha * (logp.detach() + target_entropy)).mean()
                    alpha_opt.zero_grad(set_to_none=True)
                    alpha_loss.backward()
                    alpha_opt.step()
                else:
                    alpha_loss = torch.tensor(0.0)

                #  - Target update  -
                soft_update(q1, q1_t, sac_cfg.tau)
                soft_update(q2, q2_t, sac_cfg.tau)

                # Logging
                if update_step % max(1, metric_every_updates // 5) == 0:
                    logger.log_dict(
                        {
                            "losses/q_loss": float(q_loss.item()),
                            "losses/q1_loss": float(q1_loss.item()),
                            "losses/q2_loss": float(q2_loss.item()),
                            "losses/actor_loss": float(actor_loss.item()),
                            "losses/alpha_loss": float(alpha_loss.item()),
                            "charts/alpha": float(alpha.item()),
                            "charts/sps": int(global_step / (time.time() - start_time + 1e-8)),
                        },
                        step=global_step,
                    )

                #  - Instrumentation: dormancy + repr  -
                if do_metrics:
                    with torch.no_grad():
                        # Probe batch: deterministic sample size-limited
                        probe = rb.sample(min(probe_bs, len(rb)))
                        p_obs = torch.tensor(probe["obs"], dtype=torch.float32, device=device)
                        p_act = torch.tensor(probe["actions"], dtype=torch.float32, device=device)
                        p_x_q = torch.cat([p_obs, p_act], dim=-1)

                        # Activation dormancy
                        act_actor = probe_backbone_postact_activations(actor.backbone, actor_name_map, p_obs)
                        act_q1 = probe_backbone_postact_activations(q1.backbone, q1_name_map, p_x_q)
                        act_q2 = probe_backbone_postact_activations(q2.backbone, q2_name_map, p_x_q)

                        rep_actor = compute_activation_dormancy(act_actor, tau=tau_act, eps=eps, prev_masks=prev_act_masks_actor)
                        prev_act_masks_actor = {k: v.clone() for k, v in rep_actor.layer_masks.items()}
                        for lname, st in rep_actor.layer_stats.items():
                            logger.log_scalar(f"dormancy/activation/actor/layer_frac/{lname}", st.frac_dormant, step=global_step)

                        rep_q1 = compute_activation_dormancy(act_q1, tau=tau_act, eps=eps, prev_masks=prev_act_masks_q1)
                        prev_act_masks_q1 = {k: v.clone() for k, v in rep_q1.layer_masks.items()}
                        for lname, st in rep_q1.layer_stats.items():
                            logger.log_scalar(f"dormancy/activation/q1/layer_frac/{lname}", st.frac_dormant, step=global_step)

                        rep_q2 = compute_activation_dormancy(act_q2, tau=tau_act, eps=eps, prev_masks=prev_act_masks_q2)
                        prev_act_masks_q2 = {k: v.clone() for k, v in rep_q2.layer_masks.items()}
                        for lname, st in rep_q2.layer_stats.items():
                            logger.log_scalar(f"dormancy/activation/q2/layer_frac/{lname}", st.frac_dormant, step=global_step)

                        # Gradient dormancy
                        if grad_actor is not None:
                            g_actor = compute_gradient_dormancy(grad_actor, q=grad_q, prev_masks=prev_grad_masks_actor)
                            prev_grad_masks_actor = {k: v.clone() for k, v in g_actor.layer_masks.items()}
                            for lname, st in g_actor.layer_stats.items():
                                logger.log_scalar(f"dormancy/gradient/actor/layer_frac/{lname}", st.frac_dormant, step=global_step)

                        if grad_q1 is not None:
                            g_q1 = compute_gradient_dormancy(grad_q1, q=grad_q, prev_masks=prev_grad_masks_q1)
                            prev_grad_masks_q1 = {k: v.clone() for k, v in g_q1.layer_masks.items()}
                            for lname, st in g_q1.layer_stats.items():
                                logger.log_scalar(f"dormancy/gradient/q1/layer_frac/{lname}", st.frac_dormant, step=global_step)

                        if grad_q2 is not None:
                            g_q2 = compute_gradient_dormancy(grad_q2, q=grad_q, prev_masks=prev_grad_masks_q2)
                            prev_grad_masks_q2 = {k: v.clone() for k, v in g_q2.layer_masks.items()}
                            for lname, st in g_q2.layer_stats.items():
                                logger.log_scalar(f"dormancy/gradient/q2/layer_frac/{lname}", st.frac_dormant, step=global_step)

                        # Representation metrics
                        Z_actor = actor.get_features(p_obs)
                        Z_q1 = q1.get_features(p_obs, p_act)
                        Z_q2 = q2.get_features(p_obs, p_act)

                        rm_actor = compute_repr_metrics(
                            Z_actor,
                            do_effective_rank=bool(instr.repr_metrics.effective_rank),
                            do_cosine_div=bool(instr.repr_metrics.cosine_diversity),
                            svd_topk=int(instr.repr_metrics.svd_topk),
                            cka_ref=cka_ref_actor,
                        )
                        rm_q1 = compute_repr_metrics(
                            Z_q1,
                            do_effective_rank=bool(instr.repr_metrics.effective_rank),
                            do_cosine_div=bool(instr.repr_metrics.cosine_diversity),
                            svd_topk=int(instr.repr_metrics.svd_topk),
                            cka_ref=cka_ref_q1,
                        )
                        rm_q2 = compute_repr_metrics(
                            Z_q2,
                            do_effective_rank=bool(instr.repr_metrics.effective_rank),
                            do_cosine_div=bool(instr.repr_metrics.cosine_diversity),
                            svd_topk=int(instr.repr_metrics.svd_topk),
                            cka_ref=cka_ref_q2,
                        )

                        for k, v in rm_actor.items():
                            tag = f"actor_{k}"
                            if isinstance(v, (int, float)):
                                logger.log_scalar(tag, v, step=global_step)
                            else:
                                logger.log_text(tag, str(v), step=global_step)

                        for k, v in rm_q1.items():
                            tag = f"q1_{k}"
                            if isinstance(v, (int, float)):
                                logger.log_scalar(tag, v, step=global_step)
                            else:
                                logger.log_text(tag, str(v), step=global_step)

                        for k, v in rm_q2.items():
                            tag = f"q2_{k}"
                            if isinstance(v, (int, float)):
                                logger.log_scalar(tag, v, step=global_step)
                            else:
                                logger.log_text(tag, str(v), step=global_step)

                #  - ReDo integration  -
                if redo_enabled:
                    selection = str(redo_cfg.selection).lower()
                    scope = str(redo_cfg.scope).lower()
                    init_mode = str(redo_cfg.init_mode)
                    outgoing = str(redo_cfg.outgoing)
                    reset_bias = bool(redo_cfg.reset_bias)
                    max_frac = float(redo_cfg.max_recycled_frac_per_layer)

                    # Determine global dormant frac (actor + critics)
                    def _mean_frac(masks: Dict[str, torch.Tensor]) -> Optional[float]:
                        if not masks:
                            return None
                        return float(np.mean([m.float().mean().item() for m in masks.values()]))

                    if selection == "activation":
                        frac_actor = _mean_frac(prev_act_masks_actor)
                        frac_q1 = _mean_frac(prev_act_masks_q1)
                        frac_q2 = _mean_frac(prev_act_masks_q2)
                        fracs = [f for f in [frac_actor, frac_q1, frac_q2] if f is not None]
                        dormant_frac_global = float(np.mean(fracs)) if fracs else None
                    else:
                        frac_actor = _mean_frac(prev_grad_masks_actor)
                        frac_q1 = _mean_frac(prev_grad_masks_q1)
                        frac_q2 = _mean_frac(prev_grad_masks_q2)
                        fracs = [f for f in [frac_actor, frac_q1, frac_q2] if f is not None]
                        dormant_frac_global = float(np.mean(fracs)) if fracs else None

                    if scheduler.should_redo(update_step=update_step, dormant_frac_global=dormant_frac_global):
                        # Apply per-scope; use appropriate optimizer for state reset
                        def _opt(use_actor: bool):
                            if not bool(redo_cfg.reset_optimizer_state):
                                return None
                            return actor_opt if use_actor else q_opt

                        if selection == "activation":
                            masks_actor = prev_act_masks_actor
                            masks_q1 = prev_act_masks_q1
                            masks_q2 = prev_act_masks_q2
                        else:
                            masks_actor = prev_grad_masks_actor
                            masks_q1 = prev_grad_masks_q1
                            masks_q2 = prev_grad_masks_q2

                        total_recycled = 0

                        if scope in ("policy_only", "all", "shared_trunk"):
                            if masks_actor:
                                res = redo_apply_on_sequential_linears(
                                    model=actor.backbone,
                                    dormant_masks=masks_actor,
                                    optimizer=_opt(True),
                                    init_mode=init_mode,
                                    reset_bias=reset_bias,
                                    outgoing=outgoing,
                                    max_frac=max_frac,
                                )
                                logger.log_scalar("redo/actor_total_recycled", res.total_recycled, step=global_step)
                                total_recycled += res.total_recycled

                        if scope in ("value_only", "all", "shared_trunk"):
                            if masks_q1:
                                res1 = redo_apply_on_sequential_linears(
                                    model=q1.backbone,
                                    dormant_masks=masks_q1,
                                    optimizer=_opt(False),
                                    init_mode=init_mode,
                                    reset_bias=reset_bias,
                                    outgoing=outgoing,
                                    max_frac=max_frac,
                                )
                                logger.log_scalar("redo/q1_total_recycled", res1.total_recycled, step=global_step)
                                total_recycled += res1.total_recycled
                            if masks_q2:
                                res2 = redo_apply_on_sequential_linears(
                                    model=q2.backbone,
                                    dormant_masks=masks_q2,
                                    optimizer=_opt(False),
                                    init_mode=init_mode,
                                    reset_bias=reset_bias,
                                    outgoing=outgoing,
                                    max_frac=max_frac,
                                )
                                logger.log_scalar("redo/q2_total_recycled", res2.total_recycled, step=global_step)
                                total_recycled += res2.total_recycled

                        logger.log_scalar("redo/total_recycled", total_recycled, step=global_step)

                        if str(redo_cfg.mode).lower() == "conditioned":
                            scheduler.reset()

                logger.flush()

    envs.close()
