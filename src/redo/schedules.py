from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class ConditionedTriggerState:
    """
    Tracks consecutive times a layer (or global) dormant fraction exceeded a threshold.
    """
    consecutive_hits: int = 0


@dataclass
class ReDoScheduler:
    """
    Decide when to apply ReDo recycling.

    Modes:
      - scheduled: every redo_every_updates updates
      - conditioned: if dormant fraction exceeds target for patience checks

    The scheduler operates in "update steps" (gradient updates), not env steps.
    """
    mode: str = "scheduled"  # "scheduled" | "conditioned"
    redo_every_updates: int = 5000

    # conditioned mode
    target_dormant_frac: float = 0.25
    patience: int = 3

    # state
    _state_by_key: Dict[str, ConditionedTriggerState] = field(default_factory=dict)

    def reset(self) -> None:
        self._state_by_key.clear()

    def should_redo_scheduled(self, update_step: int) -> bool:
        if self.redo_every_updates <= 0:
            return False
        return (update_step % self.redo_every_updates) == 0

    def should_redo_conditioned(self, key: str, dormant_frac: float) -> bool:
        """
        key: typically "global" or a layer-group key.
        dormant_frac: fraction dormant at current measurement.
        """
        st = self._state_by_key.get(key)
        if st is None:
            st = ConditionedTriggerState()
            self._state_by_key[key] = st

        if dormant_frac >= self.target_dormant_frac:
            st.consecutive_hits += 1
        else:
            st.consecutive_hits = 0

        return st.consecutive_hits >= self.patience

    def should_redo(
        self,
        update_step: int,
        dormant_frac_global: Optional[float] = None,
        key: str = "global",
    ) -> bool:
        mode = self.mode.lower()
        if mode == "scheduled":
            return self.should_redo_scheduled(update_step)
        if mode == "conditioned":
            if dormant_frac_global is None:
                return False
            return self.should_redo_conditioned(key=key, dormant_frac=float(dormant_frac_global))
        raise ValueError(f"Unknown ReDoScheduler mode: {self.mode}")
