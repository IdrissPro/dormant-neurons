# src/main.py
from __future__ import annotations

import os
import json
import time
import random
import platform
import subprocess
from dataclasses import asdict
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv

import hydra
from omegaconf import DictConfig, OmegaConf

from src.logging.logger import RunLogger

def _git_commit_hash() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return None


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_determinism(cfg: DictConfig) -> None:
    if not cfg.determinism.enabled:
        return
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Note: full determinism can still be broken by some ops / envs.
    # We keep this minimal and log the setting.


def select_device(cfg: DictConfig) -> torch.device:
    if cfg.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_env_thunk(
    env_id: str,
    env_kwargs: Dict[str, Any],
    seed: int,
    idx: int,
    capture_video: bool,
    video_dir: str,
) -> Callable[[], gym.Env]:
    def thunk() -> gym.Env:
        env = gym.make(env_id, **(env_kwargs or {}))
        env = gym.wrappers.RecordEpisodeStatistics(env)

        # Seed handling: Gymnasium recommends seeding reset().
        env.reset(seed=seed + idx)
        env.action_space.seed(seed + idx)
        env.observation_space.seed(seed + idx)

        if capture_video and idx == 0:
            os.makedirs(video_dir, exist_ok=True)
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=video_dir,
                episode_trigger=lambda ep: True,  # record all episodes for env0
                disable_logger=True,
            )
        return env

    return thunk


def build_vector_env(cfg: DictConfig, run_dir: str) -> SyncVectorEnv:
    env_id = cfg.env.id
    env_kwargs = OmegaConf.to_container(cfg.env.kwargs, resolve=True) if cfg.env.kwargs is not None else {}
    num_envs = int(cfg.env.num_envs)

    video_dir = os.path.join(run_dir, cfg.env.video_dir)
    thunks = [
        make_env_thunk(
            env_id=env_id,
            env_kwargs=env_kwargs,
            seed=int(cfg.seed),
            idx=i,
            capture_video=bool(cfg.env.capture_video),
            video_dir=video_dir,
        )
        for i in range(num_envs)
    ]
    return SyncVectorEnv(thunks)


def system_info(device: torch.device) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "git_commit": _git_commit_hash(),
    }
    if torch.cuda.is_available():
        info.update(
            {
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
                "gpu_name": torch.cuda.get_device_name(0),
            }
        )
    return info


def dump_resolved_config(cfg: DictConfig, run_dir: str) -> None:
    path = os.path.join(run_dir, "config_resolved.yaml")
    with open(path, "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))


def dispatch_train(cfg: DictConfig, envs: gym.vector.VectorEnv, device: torch.device, logger: RunLogger) -> None:
    algo = str(cfg.algo.name).lower()
    if algo == "ppo":
        from src.rl.ppo import train as train_fn
    elif algo == "sac":
        from src.rl.sac import train as train_fn
    elif algo == "dqn":
        from src.rl.dqn import train as train_fn
    else:
        raise ValueError(f"Unknown algo.name: {cfg.algo.name}")

    train_fn(cfg=cfg, envs=envs, device=device, logger=logger)


@hydra.main(version_base="1.3", config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    # Hydra sets working directory to run dir if hydra.job.chdir=true
    run_dir = os.getcwd()

    # Resolve config once for deterministic behavior
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    # Seeding + determinism
    set_global_seeds(int(cfg.seed))
    configure_determinism(cfg)
    device = select_device(cfg)

    # Optional torch.compile is enabled per algo file if supported
    # (we keep it off by default for easier debugging).

    # Logging setup
    logger = RunLogger(cfg=cfg, run_dir=run_dir)
    dump_resolved_config(cfg, run_dir)

    # System info
    logger.log_text("system/info", json.dumps(system_info(device), indent=2))
    logger.log_config(cfg)

    # Build envs
    envs = build_vector_env(cfg, run_dir)

    # Dispatch training
    try:
        dispatch_train(cfg, envs, device, logger)
    finally:
        envs.close()
        logger.close()


if __name__ == "__main__":
    main()

