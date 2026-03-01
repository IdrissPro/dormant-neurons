PYTHON ?= python
PIP ?= pip
VENV ?= .venv_redo

# Default variables (override on CLI, e.g. make run EXP=ppo_cartpole_baseline SEED=2)
EXP ?= ppo_cartpole_baseline
SEED ?= 1
STEPS ?=
DEVICE ?=
RUNS_DIR ?= runs
OUT_DIR ?= out
FIGS_DIR ?= figs
VENV ?= .venv_redo
# Optional extra Hydra overrides
OVERRIDES ?=

.PHONY: help install install-dev smoke test lint format run sweep tensorboard aggregate plot report clean clean-runs clean-artifacts

help:
	@echo "Available targets:"
	@echo "  make install                 Install project dependencies"
	@echo "  make install-dev             Install dependencies + dev tools"
	@echo "  make smoke                   Run short smoke tests"
	@echo "  make test                    Run unit tests"
	@echo "  make lint                    Run ruff checks"
	@echo "  make format                  Format code with ruff/black"
	@echo "  make run EXP=<name> SEED=1   Run one experiment"
	@echo "  make sweep                   Run scripts/run_sweep.sh"
	@echo "  make tensorboard             Launch TensorBoard on runs/"
	@echo "  make aggregate               Aggregate metrics into CSV"
	@echo "  make plot                    Generate figures from aggregated metrics"
	@echo "  make report                  Aggregate + plot"
	@echo "  make clean                   Remove Python cache files"
	@echo "  make clean-runs              Remove runs/"
	@echo "  make clean-artifacts         Remove out/ figs/ videos/"
	@echo ""
	@echo "Examples:"
	@echo "  make run EXP=sac_halfcheetah_redo SEED=3"
	@echo "  make run EXP=ppo_minigrid_redo SEED=1 STEPS=500000"
	@echo "  make run EXP=dqn_cartpole_redo SEED=2 OVERRIDES='redo.tau=0.01 redo.redo_every_updates=1000'"
	@echo "  make aggregate RUNS_DIR=runs"
	@echo "  make plot OUT_DIR=out FIGS_DIR=figs"

venv:
	$(PYTHON) -m venv $(VENV)
	@echo "Activate with: source $(VENV)/bin/activate"

install:
	$(PIP) install -r requirements.txt

install-dev:
	$(PIP) install -r requirements.txt
	$(PIP) install pytest ruff black

smoke:
	$(PYTHON) -m src.main --config-name experiment\ppo_cartpole_baseline run.total_env_steps=20000 env.num_envs=4 instrumentation.metric_every_updates=50
	$(PYTHON) -m src.main --config-name experiment\dqn_cartpole_baseline run.total_env_steps=20000 env.num_envs=4 algo.learning_starts=1000 instrumentation.metric_every_updates=50
	$(PYTHON) -m src.main --config-name experiment\sac_halfcheetah_baseline run.total_env_steps=20000 env.num_envs=2 algo.learning_starts=1000 instrumentation.metric_every_updates=50

test:
	pytest

lint:
	ruff check src tests

format:
	ruff format src tests
	black src tests

run:
	$(PYTHON) -m src.main --config-name experiment\$(EXP) seed=$(SEED) \
		$(if $(STEPS),run.total_env_steps=$(STEPS),) \
		$(if $(DEVICE),device=$(DEVICE),) \
		$(OVERRIDES)

sweep:
	$(MAKE) run EXP=ppo_cartpole_baseline SEED=1
	$(MAKE) run EXP=ppo_cartpole_baseline SEED=2
	$(MAKE) run EXP=ppo_cartpole_baseline SEED=3
	$(MAKE) run EXP=ppo_cartpole_baseline SEED=4
	$(MAKE) run EXP=ppo_cartpole_baseline SEED=5

	$(MAKE) run EXP=ppo_minigrid_redo SEED=1
	$(MAKE) run EXP=ppo_minigrid_redo SEED=2
	$(MAKE) run EXP=ppo_minigrid_redo SEED=3
	$(MAKE) run EXP=ppo_minigrid_redo SEED=4
	$(MAKE) run EXP=ppo_minigrid_redo SEED=5

	$(MAKE) run EXP=sac_halfcheetah_baseline SEED=1
	$(MAKE) run EXP=sac_halfcheetah_baseline SEED=2
	$(MAKE) run EXP=sac_halfcheetah_baseline SEED=3
	$(MAKE) run EXP=sac_halfcheetah_baseline SEED=4
	$(MAKE) run EXP=sac_halfcheetah_baseline SEED=5

	$(MAKE) run EXP=sac_halfcheetah_redo SEED=1
	$(MAKE) run EXP=sac_halfcheetah_redo SEED=2
	$(MAKE) run EXP=sac_halfcheetah_redo SEED=3
	$(MAKE) run EXP=sac_halfcheetah_redo SEED=4
	$(MAKE) run EXP=sac_halfcheetah_redo SEED=5

	$(MAKE) run EXP=dqn_cartpole_baseline SEED=1
	$(MAKE) run EXP=dqn_cartpole_baseline SEED=2
	$(MAKE) run EXP=dqn_cartpole_baseline SEED=3
	$(MAKE) run EXP=dqn_cartpole_baseline SEED=4
	$(MAKE) run EXP=dqn_cartpole_baseline SEED=5

	$(MAKE) run EXP=dqn_cartpole_redo SEED=1
	$(MAKE) run EXP=dqn_cartpole_redo SEED=2
	$(MAKE) run EXP=dqn_cartpole_redo SEED=3
	$(MAKE) run EXP=dqn_cartpole_redo SEED=4
	$(MAKE) run EXP=dqn_cartpole_redo SEED=5

tensorboard:
	tensorboard --logdir $(RUNS_DIR)

aggregate:
	$(PYTHON) -m src.analysis.aggregate --runs_dir $(RUNS_DIR) --out_dir $(OUT_DIR) --include_text --text_prefix repr/svd_topk

plot:
	$(PYTHON) -m src.analysis.plots --in_dir $(OUT_DIR) --out_dir $(FIGS_DIR) --group_by env algo

report: aggregate plot

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

clean-runs:
	if exist runs rmdir /s /q runs

clean-artifacts:
	if exist out rmdir /s /q out
	if exist figs rmdir /s /q figs
	if exist videos rmdir /s /q videos

