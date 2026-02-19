# redo-rl

Research scaffold for ReDo-style neuron recycling + RL algorithms (PPO/SAC/DQN),
with instrumentation for dormancy, representation metrics, and event tracking.

## Quickstart
```bash
pip install -e .
python -m src.main --config configs/base.yaml
```

## Layout
- `configs/`: YAML configs (base + algo/env/redo overrides)
- `src/`: source code
- `scripts/`: shell helpers for sweeps and plotting
