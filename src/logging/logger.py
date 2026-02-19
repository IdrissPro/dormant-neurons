from __future__ import annotations
import os
import io
import json
import time
from typing import Any, Dict, Optional, Union

from omegaconf import DictConfig, OmegaConf

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception as e:  # pragma: no cover
    SummaryWriter = None  # type: ignore


Number = Union[int, float]


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, bool):
            return float(int(x))
        if isinstance(x, (int, float)):
            return float(x)
        # numpy scalar
        if hasattr(x, "item"):
            return float(x.item())
        return float(x)
    except Exception:
        return None


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


class RunLogger:
    """
    Minimal, consistent logger for RL experiments.

    Outputs:
      - TensorBoard: <run_dir>/events/  (if enabled)
      - JSONL:       <run_dir>/metrics.jsonl (if enabled)
      - Stdout:      prints key metrics (if enabled)
    """

    def __init__(self, cfg: DictConfig, run_dir: str):
        self.cfg = cfg
        self.run_dir = run_dir

        self._tb_enabled = bool(cfg.logging.tb)
        self._jsonl_enabled = bool(cfg.logging.jsonl)
        self._stdout_enabled = bool(cfg.logging.stdout)

        self._tb: Optional[SummaryWriter] = None
        self._jsonl_fh: Optional[io.TextIOWrapper] = None

        if self._tb_enabled:
            if SummaryWriter is None:
                raise RuntimeError("TensorBoard SummaryWriter not available. Install tensorboard.")
            tb_dir = os.path.join(run_dir, "events")
            os.makedirs(tb_dir, exist_ok=True)
            self._tb = SummaryWriter(log_dir=tb_dir)

        if self._jsonl_enabled:
            path = os.path.join(run_dir, "metrics.jsonl")
            self._jsonl_fh = open(path, "a", encoding="utf-8")

        # Step counters (you can use both; many logs use env_steps as x-axis)
        self.env_step: int = 0
        self.update_step: int = 0

    #   config / text  

    def log_config(self, cfg: Optional[DictConfig] = None) -> None:
        cfg = cfg or self.cfg
        data = OmegaConf.to_container(cfg, resolve=True)
        self._write_jsonl({"time": _now_iso(), "type": "config", "data": data})
        if self._tb is not None:
            # TensorBoard doesn't store structured YAML; store as text.
            self._tb.add_text("config/resolved", OmegaConf.to_yaml(cfg, resolve=True), global_step=0)

    def log_text(self, tag: str, text: str, step: Optional[int] = None) -> None:
        step = int(self.env_step if step is None else step)
        self._write_jsonl({"time": _now_iso(), "type": "text", "tag": tag, "step": step, "text": text})
        if self._tb is not None:
            self._tb.add_text(tag, text, global_step=step)
        if self._stdout_enabled:
            print(f"[text] {tag} (step={step})")

    #   scalars  

    def log_scalar(self, tag: str, value: Number, step: Optional[int] = None) -> None:
        step = int(self.env_step if step is None else step)
        v = _safe_float(value)
        if v is None:
            return
        self._write_jsonl({"time": _now_iso(), "type": "scalar", "tag": tag, "step": step, "value": v})
        if self._tb is not None:
            self._tb.add_scalar(tag, v, global_step=step)
        if self._stdout_enabled and (tag.startswith("charts/") or tag.startswith("losses/")):
            print(f"{tag}: {v:.6f} (step={step})")

    def log_dict(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        prefix: str = "",
        only_numeric: bool = True,
    ) -> None:
        step = int(self.env_step if step is None else step)
        flat: Dict[str, float] = {}

        for k, v in metrics.items():
            tag = f"{prefix}{k}" if prefix else k
            fv = _safe_float(v)
            if fv is None:
                if not only_numeric:
                    self._write_jsonl(
                        {"time": _now_iso(), "type": "kv", "tag": tag, "step": step, "value": str(v)}
                    )
                continue
            flat[tag] = fv

        if self._jsonl_enabled:
            # store as one record for efficiency
            self._write_jsonl({"time": _now_iso(), "type": "scalars", "step": step, "values": flat})

        if self._tb is not None:
            for tag, fv in flat.items():
                self._tb.add_scalar(tag, fv, global_step=step)

        if self._stdout_enabled and "charts/episodic_return" in flat:
            print(f"return={flat['charts/episodic_return']:.2f} step={step}")

    #   housekeeping  

    def set_env_step(self, env_step: int) -> None:
        self.env_step = int(env_step)

    def set_update_step(self, update_step: int) -> None:
        self.update_step = int(update_step)

    def flush(self) -> None:
        if self._tb is not None:
            self._tb.flush()
        if self._jsonl_fh is not None:
            self._jsonl_fh.flush()

    def close(self) -> None:
        self.flush()
        if self._tb is not None:
            self._tb.close()
            self._tb = None
        if self._jsonl_fh is not None:
            self._jsonl_fh.close()
            self._jsonl_fh = None

    #   internal  

    def _write_jsonl(self, record: Dict[str, Any]) -> None:
        if not self._jsonl_enabled or self._jsonl_fh is None:
            return
        self._jsonl_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
