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
class DQNConfig:
    # Replay / training
    buffer_size: int = 1_000_000
    learning_starts: int = 80_000
    batch_size: int = 256
    train_frequency: int = 4          # env steps between gradient steps
    gradient_steps: int = 1           # how many updates per train event
    target_network_frequency: int = 1_000
    tau: float = 1.0                  # soft update; 1.0 means hard copy
    # Optimization
    lr: float = 2.5e-4
    gamma: float = 0.99
    max_grad_norm: float = 10.0
    # Exploration
    exploration_fraction: float = 0.1
    eps_start: float = 1.0
    eps_end: float = 0.05
    # Network
    hidden_dims: Tuple[int, ...] = (256, 256)
    activation: str = "relu"
    layernorm: bool = False
    # Misc
    seed: int = 1


def _get(cfg: DictConfig, path: str, default: Any) -> Any:
    cur: Any = cfg
    for key in path.split("."):
        if not hasattr(cur, key):
            return default
        cur = getattr(cur, key)
    return cur if cur is not None else default


def load_dqn_config(cfg: DictConfig) -> DQNConfig:
    hidden = _get(cfg, "algo.hidden_dims", list(DQNConfig.hidden_dims))
    return DQNConfig(
        buffer_size=int(_get(cfg, "algo.buffer_size", DQNConfig.buffer_size)),
        learning_starts=int(_get(cfg, "algo.learning_starts", DQNConfig.learning_starts)),
        batch_size=int(_get(cfg, "algo.batch_size", DQNConfig.batch_size)),
        train_frequency=int(_get(cfg, "algo.train_frequency", DQNConfig.train_frequency)),
        gradient_steps=int(_get(cfg, "algo.gradient_steps", DQNConfig.gradient_steps)),
        target_network_frequency=int(_get(cfg, "algo.target_network_frequency", DQNConfig.target_network_frequency)),
        tau=float(_get(cfg, "algo.tau", DQNConfig.tau)),
        lr=float(_get(cfg, "algo.lr", DQNConfig.lr)),
        gamma=float(_get(cfg, "algo.gamma", DQNConfig.gamma)),
        max_grad_norm=float(_get(cfg, "algo.max_grad_norm", DQNConfig.max_grad_norm)),
        exploration_fraction=float(_get(cfg, "algo.exploration_fraction", DQNConfig.exploration_fraction)),
        eps_start=float(_get(cfg, "algo.eps_start", DQNConfig.eps_start)),
        eps_end=float(_get(cfg, "algo.eps_end", DQNConfig.eps_end)),
        hidden_dims=tuple(int(x) for x in hidden),
        activation=str(_get(cfg, "algo.activation", DQNConfig.activation)),
        layernorm=bool(_get(cfg, "algo.layernorm", DQNConfig.layernorm)),
        seed=int(_get(cfg, "seed", DQNConfig.seed)),
    )


#    
# Replay buffer (vector-env aware)
#    

class ReplayBuffer:
    def __init__(self, obs_dim: int, size: int):
        self.size = int(size)
        self.obs = np.zeros((self.size, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.size,), dtype=np.int64)
        self.rewards = np.zeros((self.size,), dtype=np.float32)
        self.dones = np.zeros((self.size,), dtype=np.float32)

        self._idx = 0
        self._full = False

    def __len__(self) -> int:
        return self.size if self._full else self._idx

    def add_batch(
        self,
        obs: np.ndarray,          # [N, obs_dim]
        actions: np.ndarray,      # [N]
        rewards: np.ndarray,      # [N]
        next_obs: np.ndarray,     # [N, obs_dim]
        dones: np.ndarray,        # [N] bool/0-1
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
# Network
#    

class DQNNet(nn.Module):
    """
    Q(s) MLP with a backbone (named linears) + linear head.
    We expose:
      - backbone (for dormancy probing)
      - features (penultimate representation)
    """
    def __init__(self, obs_dim: int, n_actions: int, hidden_dims: Tuple[int, ...], activation: str, layernorm: bool):
        super().__init__()
        self.backbone = build_mlp_backbone(
            in_dim=obs_dim,
            hidden_dims=list(hidden_dims),
            activation=activation,
            layernorm=layernorm,
            name_prefix="q",
        )
        self.q_head = nn.Linear(hidden_dims[-1], n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.backbone(obs, return_features=False)
        return self.q_head(h)

    @torch.no_grad()
    def get_features(self, obs: torch.Tensor) -> torch.Tensor:
        _, feat = self.backbone(obs, return_features=True)
        return feat


@torch.no_grad()
def build_linear_name_map(module: nn.Module) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for name, m in module.named_modules():
        if isinstance(m, nn.Linear):
            out[id(m)] = name
    return out


@torch.no_grad()
def probe_backbone_postact_activations(backbone: nn.Module, name_map: Dict[int, str], obs_b: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Returns dict: linear_name -> post-activation outputs [B,H]
    Assumes backbone exposes .fcs, .lns, .act (MLPBackbone from networks.py).
    """
    acts: Dict[str, torch.Tensor] = {}
    h = obs_b
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


def _linear_schedule(start: float, end: float, duration: int, t: int) -> float:
    if duration <= 0:
        return end
    frac = min(1.0, max(0.0, t / duration))
    return start + frac * (end - start)


#    
# Training
#    

def train(cfg: DictConfig, envs: gym.vector.VectorEnv, device: torch.device, logger: RunLogger) -> None:
    dqn_cfg = load_dqn_config(cfg)

    obs_dim = infer_obs_dim(envs.single_observation_space)
    act_kind, act_dim = infer_act_dims(envs.single_action_space)
    if act_kind != "discrete":
        raise ValueError("DQN only supports discrete action spaces.")

    num_envs = envs.num_envs
    total_env_steps = int(cfg.run.total_env_steps)

    q_net = DQNNet(obs_dim, act_dim, dqn_cfg.hidden_dims, dqn_cfg.activation, dqn_cfg.layernorm).to(device)
    q_target = DQNNet(obs_dim, act_dim, dqn_cfg.hidden_dims, dqn_cfg.activation, dqn_cfg.layernorm).to(device)
    q_target.load_state_dict(q_net.state_dict())
    q_target.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=dqn_cfg.lr, eps=1e-5)
    rb = ReplayBuffer(obs_dim=obs_dim, size=dqn_cfg.buffer_size)

    # Instrumentation
    instr = cfg.instrumentation
    instr_enabled = bool(instr.enabled)
    metric_every_updates = int(instr.metric_every_updates)
    heavy_every_updates = int(instr.heavy_metric_every_updates)
    probe_bs = int(instr.probe_batch_size)
    tau_act = float(instr.tau)
    grad_q = float(instr.grad_quantile)
    eps = float(instr.eps)

    cka_ref = None
    if bool(instr.repr_metrics.linear_cka):
        cka_ref = CKAReference(mode=str(instr.repr_metrics.cka_reference), beta=float(instr.repr_metrics.cka_ema_beta))

    prev_act_masks: Dict[str, torch.Tensor] = {}
    prev_grad_masks: Dict[str, torch.Tensor] = {}

    backbone_name_map = build_linear_name_map(q_net.backbone)
    linear_layer_names = [n for n, _ in list_named_linears(q_net.backbone)]

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

    # Epsilon schedule duration in env steps
    eps_duration = int(dqn_cfg.exploration_fraction * total_env_steps)

    while global_step < total_env_steps:
        logger.set_env_step(global_step)

        # Epsilon-greedy action selection (vectorized)
        eps_now = _linear_schedule(dqn_cfg.eps_start, dqn_cfg.eps_end, eps_duration, global_step)
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)  # [N, obs_dim]
            q_values = q_net(obs_t)  # [N, A]
            greedy = torch.argmax(q_values, dim=1).cpu().numpy()  # [N]

        random_actions = np.random.randint(0, act_dim, size=(num_envs,))
        do_random = np.random.rand(num_envs) < eps_now
        actions = np.where(do_random, random_actions, greedy)

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

        # Store transitions (one per env)
        rb.add_batch(
            obs=obs,
            actions=actions.astype(np.int64),
            rewards=rewards.astype(np.float32),
            next_obs=next_obs.astype(np.float32),
            dones=dones.astype(np.float32),
        )

        obs = next_obs
        global_step += num_envs
        logger.set_env_step(global_step)

        # Training step(s)
        if global_step >= dqn_cfg.learning_starts and (global_step // num_envs) % dqn_cfg.train_frequency == 0:
            for _ in range(dqn_cfg.gradient_steps):
                update_step += 1
                logger.set_update_step(update_step)

                batch = rb.sample(dqn_cfg.batch_size)
                b_obs = torch.tensor(batch["obs"], dtype=torch.float32, device=device)
                b_actions = torch.tensor(batch["actions"], dtype=torch.int64, device=device)  # [B]
                b_rewards = torch.tensor(batch["rewards"], dtype=torch.float32, device=device)  # [B]
                b_next_obs = torch.tensor(batch["next_obs"], dtype=torch.float32, device=device)
                b_dones = torch.tensor(batch["dones"], dtype=torch.float32, device=device)

                # Compute TD target
                with torch.no_grad():
                    next_q = q_target(b_next_obs)          # [B, A]
                    next_max = next_q.max(dim=1).values    # [B]
                    target = b_rewards + dqn_cfg.gamma * (1.0 - b_dones) * next_max

                # Current Q
                q = q_net(b_obs)                           # [B, A]
                q_a = q.gather(1, b_actions.view(-1, 1)).squeeze(1)

                loss = nn.functional.smooth_l1_loss(q_a, target)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                # Gradient dormancy measurement BEFORE step (when scheduled)
                do_metrics = instr_enabled and (update_step % metric_every_updates == 0)
                grad_norms = None
                if do_metrics:
                    grad_norms = linear_row_grad_norms(q_net.backbone, linear_layer_names)

                nn.utils.clip_grad_norm_(q_net.parameters(), dqn_cfg.max_grad_norm)
                optimizer.step()

                # Target net update
                if update_step % dqn_cfg.target_network_frequency == 0:
                    if dqn_cfg.tau >= 1.0:
                        q_target.load_state_dict(q_net.state_dict())
                    else:
                        # Polyak
                        for p, tp in zip(q_net.parameters(), q_target.parameters()):
                            tp.data.mul_(1.0 - dqn_cfg.tau).add_(p.data, alpha=dqn_cfg.tau)

                # Logging (loss + epsilon + SPS)
                if update_step % max(1, metric_every_updates // 5) == 0:
                    logger.log_scalar("losses/td_loss", float(loss.item()), step=global_step)
                    logger.log_scalar("charts/epsilon", float(eps_now), step=global_step)
                    logger.log_scalar("charts/sps", int(global_step / (time.time() - start_time + 1e-8)), step=global_step)

                #  - Instrumentation: dormancy + repr  -
                if do_metrics:
                    with torch.no_grad():
                        # Probe batch from replay buffer (deterministic: sample once)
                        probe_np = rb.sample(min(probe_bs, len(rb)))["obs"]
                        probe_obs = torch.tensor(probe_np, dtype=torch.float32, device=device)

                        # Activation-based dormancy
                        act_map = probe_backbone_postact_activations(q_net.backbone, backbone_name_map, probe_obs)
                        act_report = compute_activation_dormancy(act_map, tau=tau_act, eps=eps, prev_masks=prev_act_masks)
                        prev_act_masks = {k: v.clone() for k, v in act_report.layer_masks.items()}

                        for lname, st in act_report.layer_stats.items():
                            logger.log_scalar(f"dormancy/activation/layer_frac/{lname}", st.frac_dormant, step=global_step)
                            if st.death_rate is not None:
                                logger.log_scalar(f"dormancy/activation/death_rate/{lname}", st.death_rate, step=global_step)
                            if st.revival_rate is not None:
                                logger.log_scalar(f"dormancy/activation/revival_rate/{lname}", st.revival_rate, step=global_step)
                            if st.overlap_prev is not None:
                                logger.log_scalar(f"dormancy/activation/overlap_prev/{lname}", st.overlap_prev, step=global_step)

                        if act_report.layer_stats:
                            global_frac = float(np.mean([s.frac_dormant for s in act_report.layer_stats.values()]))
                            logger.log_scalar("dormancy/activation/global_frac", global_frac, step=global_step)

                        # Gradient-based dormancy
                        if grad_norms is not None:
                            grad_report = compute_gradient_dormancy(grad_norms, q=grad_q, prev_masks=prev_grad_masks)
                            prev_grad_masks = {k: v.clone() for k, v in grad_report.layer_masks.items()}

                            for lname, st in grad_report.layer_stats.items():
                                logger.log_scalar(f"dormancy/gradient/layer_frac/{lname}", st.frac_dormant, step=global_step)
                                if st.death_rate is not None:
                                    logger.log_scalar(f"dormancy/gradient/death_rate/{lname}", st.death_rate, step=global_step)
                                if st.revival_rate is not None:
                                    logger.log_scalar(f"dormancy/gradient/revival_rate/{lname}", st.revival_rate, step=global_step)
                                if st.overlap_prev is not None:
                                    logger.log_scalar(f"dormancy/gradient/overlap_prev/{lname}", st.overlap_prev, step=global_step)

                            if grad_report.layer_stats:
                                gfrac = float(np.mean([s.frac_dormant for s in grad_report.layer_stats.values()]))
                                logger.log_scalar("dormancy/gradient/global_frac", gfrac, step=global_step)

                        # Representation metrics (penultimate features)
                        Z = q_net.get_features(probe_obs)
                        rm = compute_repr_metrics(
                            Z,
                            do_effective_rank=bool(instr.repr_metrics.effective_rank),
                            do_cosine_div=bool(instr.repr_metrics.cosine_diversity),
                            svd_topk=int(instr.repr_metrics.svd_topk),
                            cka_ref=cka_ref,
                        )
                        for k, v in rm.items():
                            if isinstance(v, (int, float)):
                                logger.log_scalar(k, v, step=global_step)
                            else:
                                logger.log_text(k, str(v), step=global_step)

                #  - ReDo integration  -
                if redo_enabled:
                    selection = str(redo_cfg.selection).lower()
                    masks = prev_act_masks if selection == "activation" else prev_grad_masks

                    dormant_frac_global = None
                    if masks:
                        dormant_frac_global = float(np.mean([m.float().mean().item() for m in masks.values()]))

                    if scheduler.should_redo(update_step=update_step, dormant_frac_global=dormant_frac_global):
                        if masks:
                            opt_for_reset = optimizer if bool(redo_cfg.reset_optimizer_state) else None
                            res = redo_apply_on_sequential_linears(
                                model=q_net.backbone,
                                dormant_masks=masks,
                                optimizer=opt_for_reset,
                                init_mode=str(redo_cfg.init_mode),
                                reset_bias=bool(redo_cfg.reset_bias),
                                outgoing=str(redo_cfg.outgoing),
                                max_frac=float(redo_cfg.max_recycled_frac_per_layer),
                                allowed_layers=list(redo_cfg.layers) if getattr(redo_cfg, "layers", None) else None
                            )
                            logger.log_scalar("redo/total_recycled", res.total_recycled, step=global_step)
                            for lname, k in res.recycled_by_layer.items():
                                logger.log_scalar(f"redo/recycled/{lname}", k, step=global_step)

                        if str(redo_cfg.mode).lower() == "conditioned":
                            scheduler.reset()

                logger.flush()

    envs.close()

