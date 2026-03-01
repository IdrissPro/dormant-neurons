#!/usr/bin/env bash
set -euo pipefail

# Quick sanity checks to ensure training loops run end-to-end.

echo "=== PPO CartPole short run ==="
python -m src.main +experiment=ppo_cartpole_baseline run.total_env_steps=20000 env.num_envs=4 instrumentation.metric_every_updates=50

echo "=== DQN CartPole short run ==="
python -m src.main +experiment=dqn_cartpole_baseline run.total_env_steps=20000 env.num_envs=4 algo.learning_starts=1000 instrumentation.metric_every_updates=50

echo "=== SAC HalfCheetah short run ==="
python -m src.main +experiment=sac_halfcheetah_baseline run.total_env_steps=20000 env.num_envs=2 algo.learning_starts=1000 instrumentation.metric_every_updates=50

echo "Smoke tests done."
