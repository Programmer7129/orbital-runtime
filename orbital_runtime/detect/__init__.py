"""Detection: three tiers behind one `observe()` call.

Tiers escalate in cost (research doc SS3):

  1. `GuardTier`   ~0%    -- scalars the loop already computed
  2. `AbftTier`    <10%   -- sampled checksums around GEMMs
  3. `WatcherTier` ~0%    -- driver error reports (synthetic in sim)

`Detector` runs them cheapest-first and returns the first trigger, so a run
that is already visibly NaN never pays for a checksum to confirm it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .abft import AbftStats, AbftTier
from .guards import GuardTier, grads_are_finite
from .verdict import (
    NO_DETECTION,
    REASON_ABFT_MISMATCH,
    REASON_GRAD_NORM_ZSCORE,
    REASON_LOSS_SPIKE,
    REASON_NONFINITE_GRAD,
    REASON_NONFINITE_LOSS,
    REASON_SEFI,
    REASON_XID_FATAL,
    TIER_ABFT,
    TIER_GUARD,
    TIER_WATCHER,
    Verdict,
)
from .watcher import DcgmXidSource, SimulatedXidSource, WatcherTier, XidSource

__all__ = [
    "AbftStats",
    "AbftTier",
    "DcgmXidSource",
    "Detector",
    "GuardTier",
    "SimulatedXidSource",
    "Verdict",
    "WatcherTier",
    "XidSource",
    "NO_DETECTION",
    "REASON_ABFT_MISMATCH",
    "REASON_GRAD_NORM_ZSCORE",
    "REASON_LOSS_SPIKE",
    "REASON_NONFINITE_GRAD",
    "REASON_NONFINITE_LOSS",
    "REASON_SEFI",
    "REASON_XID_FATAL",
    "TIER_ABFT",
    "TIER_GUARD",
    "TIER_WATCHER",
    "grads_are_finite",
]


@dataclass
class Detector:
    """Composite of the enabled tiers."""

    guards: GuardTier | None = None
    abft: AbftTier | None = None
    watcher: WatcherTier | None = None

    detections: int = 0
    history: list[Verdict] = field(default_factory=list)
    per_tier: dict[str, int] = field(default_factory=dict)

    def before_step(self, *, t_sim: float, in_saa: bool) -> None:
        """Tell the tiers where we are in the orbit, and arm them."""
        if self.abft is not None:
            self.abft.set_position(t_sim=t_sim, in_saa=in_saa)
            self.abft.arm()

    def observe(
        self,
        *,
        step: int,
        loss: float,
        grad_norm: float,
        model: torch.nn.Module | None = None,
    ) -> Verdict:
        """Run the tiers cheapest-first; return the first trigger."""
        if self.abft is not None:
            self.abft.disarm()

        for tier in (self.guards, self.watcher, self.abft):
            if tier is None:
                continue
            verdict = tier.observe(
                step=step, loss=loss, grad_norm=grad_norm, model=model
            )
            if verdict.triggered:
                self.detections += 1
                self.history.append(verdict)
                self.per_tier[verdict.tier] = self.per_tier.get(verdict.tier, 0) + 1
                return verdict
        return NO_DETECTION

    def reset(self) -> None:
        """After a rollback, the baselines describe a state that no longer
        exists -- keeping them would compare replayed steps against a
        corrupted history."""
        for tier in (self.guards, self.abft, self.watcher):
            if tier is not None:
                tier.reset()

    def stats(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "detections": self.detections,
            "detections_by_tier": dict(self.per_tier),
        }
        if self.abft is not None:
            out.update(self.abft.stats.as_dict())
        return out
