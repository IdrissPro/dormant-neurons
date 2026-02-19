#!/usr/bin/env bash
set -euo pipefail

# scripts/run_sweep.sh
# Usage:
#   bash scripts/run_sweep.sh
#
# It will run a few experiments across multiple seeds.
# You can edit EXPERIMENTS and SEEDS arrays.

SEEDS=(1 2 3 4 5)

EXPERIMENTS=(
  "ppo_cartpole_baseline"
  "ppo_minigrid_redo"
  "sac_halfcheetah_baseline"
  "sac_halfcheetah_redo"
  "dqn_cartpole_baseline"
  "dqn_cartpole_redo"
)

for EXP in "${EXPERIMENTS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    echo "=== Running ${EXP} seed=${SEED} ==="
    python -m src.main \
      +experiment="${EXP}" \
      seed="${SEED}" \
      hydra.run.dir="runs/${EXP}/seed_${SEED}"
  done
done

echo "All sweeps finished."
