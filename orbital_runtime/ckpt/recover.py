"""The recovery loop: detect -> restore last VERIFIED checkpoint -> replay.

Research doc SS4. This is the piece that turns detection into survival.

Choosing WHICH checkpoint (the subtle part)
-------------------------------------------
"Last verified checkpoint" is not the same as "newest checkpoint". Detection
lags the fault: M2 measured a median latency of 2 steps with ABFT, and up to
24 with guards alone. So the newest checkpoint may have been taken AFTER the
corruption landed, and restoring it would faithfully reinstate the fault --
recovery would become a way of preserving the corruption rather than undoing
it, and the run would loop: restore, re-detect, restore, forever.

So a rollback target must be older than the corruption, not merely older
than the detection. We do not know when the fault landed, only when it was
seen, so we subtract a safety margin from the detection step and require the
checkpoint to predate that.

**The margin depends on WHICH TIER spoke**, and this is where M2's latency
measurement turns into M3 behaviour:

* An **ABFT mismatch** compares this step's GEMM against a checksum taken
  one step ago. A mismatch at step D therefore means the fault landed
  between D-1's update and D's forward -- the fault is localised to one
  step, and any checkpoint at or before D-1 is safe. Margin: 1.
* A **fatal Xid** is timestamped by the driver. Margin: 1.
* A **NaN loss** or a **z-score** says only "something is wrong now". The
  corruption may have been quietly compounding for many steps -- M2 measured
  guard latencies up to 24 steps -- so the margin must cover that.

Using the pessimistic margin for everything (the first version of this file
did) throws away good checkpoints and forces needlessly deep rollbacks: it
replayed 37 steps where 1 would have done, then ran out of eligible
checkpoints and declared a recoverable run unrecoverable. Detection latency
is not a statistic on a slide -- it is exactly how much work a rollback
burns.

The double buffer earns its keep here: if the newest eligible checkpoint
fails its checksum (corrupted at rest, or a torn write), we fall back to the
older slot rather than losing the run.

Why replay is not free
----------------------
Rolling back to step C and replaying to step D costs (D - C) steps of real
work, during which the satellite keeps flying and fresh radiation keeps
arriving (see injector.py -- the clock counts executed steps). That is the
honest cost of protection and it is exactly what the overhead number must
include. It is also why detection LATENCY matters as much as recall: latency
sets how far back we must rewind, and therefore how much work burns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..detect.verdict import (
    REASON_ABFT_MISMATCH,
    REASON_XID_FATAL,
    Verdict,
)
from ..inject.injector import RadiationEnvironment
from ..telemetry import EVENT_CHECKPOINT, EVENT_ROLLBACK, Telemetry
from .policy import CheckpointPolicy
from .saver import Checkpoint, CheckpointSaver

# Steps subtracted from the detection step when choosing a rollback target,
# per detection reason. See the module docstring: this is the detector's
# latency, and it sets how much work a rollback burns.
#
# 1 for the tiers that localise the fault to a single step (ABFT's trusted
# snapshot is one step old; a driver Xid is timestamped).
LAG_LOCALISED = 1

# For the tiers that only say "something is wrong now". M2 measured guard
# latencies with a median of 24 steps; 25 covers the measured worst case.
LAG_UNLOCALISED = 25

DETECTION_LAG_BY_REASON: dict[str, int] = {
    REASON_ABFT_MISMATCH: LAG_LOCALISED,
    REASON_XID_FATAL: LAG_LOCALISED,
}

# Consecutive failed recoveries before giving up. A run that cannot recover
# is a run to report honestly, not to retry forever.
DEFAULT_MAX_CONSECUTIVE_FAILURES = 3


class RecoveryExhausted(RuntimeError):
    """No usable checkpoint remains. The run is genuinely lost."""


@dataclass
class RecoveryStats:
    rollbacks: int = 0
    replayed_steps: int = 0
    failed_restores: int = 0
    deepest_rollback: int = 0
    # Rollbacks to a checkpoint we could not PROVE predates the corruption.
    # Tracked separately: a run that recovered only via best-effort rollbacks
    # recovered on a guess, and the demo must not claim otherwise.
    best_effort_rollbacks: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "rollbacks": self.rollbacks,
            "replayed_steps": self.replayed_steps,
            "failed_restores": self.failed_restores,
            "deepest_rollback": self.deepest_rollback,
            "best_effort_rollbacks": self.best_effort_rollbacks,
        }


@dataclass
class RecoveryOrchestrator:
    """Owns checkpoint cadence and the rollback decision."""

    saver: CheckpointSaver
    policy: CheckpointPolicy
    env: RadiationEnvironment | None = None
    telemetry: Telemetry | None = None
    detector: Any = None  # reset after rollback
    unlocalised_lag_steps: int = LAG_UNLOCALISED
    max_consecutive_failures: int = DEFAULT_MAX_CONSECUTIVE_FAILURES

    stats: RecoveryStats = field(default_factory=RecoveryStats)
    _consecutive_failures: int = field(default=0, init=False)
    _pending_save_reason: str = field(default="", init=False)

    @property
    def rollbacks(self) -> int:
        return self.stats.rollbacks

    # ------------------------------------------------------------------ #
    # Checkpoint cadence
    # ------------------------------------------------------------------ #

    def before_step(self, *, step: int, t_sim: float, in_saa: bool) -> None:
        """Decide whether to checkpoint. Called before the forward pass.

        The decision is made here but the SAVE happens in `after_step`,
        after `optimizer.step()`. Saving here would snapshot state that
        this step's radiation has already landed on -- persisting the fault
        into the very checkpoint meant to escape it.
        """
        seconds_per_step = (
            self.env.seconds_per_step if self.env is not None else 1.0
        )
        save, reason = self.policy.should_save(
            step=step,
            t_sim=t_sim,
            in_saa=in_saa,
            seconds_per_step=seconds_per_step,
        )
        self._pending_save_reason = reason if save else ""

    def after_step(self, *, step: int, t_sim: float) -> None:
        """Take the checkpoint decided in `before_step`, at a trusted moment."""
        if not self._pending_save_reason:
            return
        reason, self._pending_save_reason = self._pending_save_reason, ""

        ck = self.saver.save(step=step, t_sim=t_sim)
        self.policy.record_save(step)
        if self.telemetry:
            self.telemetry.emit(
                EVENT_CHECKPOINT,
                step=step,
                t_sim=t_sim,
                reason=reason,
                slot=ck.slot,
                wall_s=round(ck.wall_s, 5),
            )

    # ------------------------------------------------------------------ #
    # Rollback
    # ------------------------------------------------------------------ #

    def lag_for(self, verdict: Verdict) -> int:
        """How many steps back the corruption might reach, given who spoke."""
        return DETECTION_LAG_BY_REASON.get(verdict.reason, self.unlocalised_lag_steps)

    def on_detection(self, *, step: int, verdict: Verdict) -> int:
        """Restore and return the step to resume from.

        Raises RecoveryExhausted when nothing usable is left -- reporting a
        lost run honestly beats looping forever pretending to recover.
        """
        lag = self.lag_for(verdict)
        safe_before = step - lag + 1
        candidates = self.saver.candidates(before_step=safe_before)

        if not candidates:
            # Nothing predates the corruption. Falling back to the oldest
            # checkpoint we still hold is a real option -- it may predate the
            # fault even if we cannot prove it -- so we try it rather than
            # abandoning a possibly-recoverable run. But if that keeps
            # failing we are looping on a checkpoint that already contains
            # the corruption, and the honest move is to stop.
            oldest = self.saver.candidates()
            if oldest and self._consecutive_failures < self.max_consecutive_failures:
                self._consecutive_failures += 1
                target = oldest[-1]
                if self.saver.restore(target):
                    return self._finish_rollback(
                        step, target, verdict, best_effort=True
                    )
                self.stats.failed_restores += 1

            self._emit_failed_rollback(step, verdict, "no_checkpoint_predates_corruption")
            raise RecoveryExhausted(
                f"detection at step {step} ({verdict.reason}, lag {lag}); no verified "
                f"checkpoint older than step {safe_before} "
                f"after {self._consecutive_failures} best-effort attempts"
            )

        for ck in candidates:  # newest first; fall back through the buffer
            if self.saver.restore(ck):
                return self._finish_rollback(step, ck, verdict)
            self.stats.failed_restores += 1

        self._consecutive_failures += 1
        self._emit_failed_rollback(step, verdict, "all_checkpoints_failed_verification")
        raise RecoveryExhausted(
            f"detection at step {step}: all {len(candidates)} candidate checkpoints "
            "failed verification"
        )

    def _finish_rollback(
        self, step: int, ck: Checkpoint, verdict: Verdict, *, best_effort: bool = False
    ) -> int:
        # The checkpoint holds POST-step-`ck.step` state (it was saved right
        # after that step's optimizer.step()), so training resumes at the NEXT
        # step. Returning ck.step would re-execute ck.step against weights that
        # already contain its update -- applying it twice and desynchronising
        # the replay from the original trajectory. (Off-by-one found by hostile
        # review, item 6; fixed.) resume = ck.step + 1.
        resume = ck.step + 1
        replayed = step - ck.step
        self.stats.rollbacks += 1
        self.stats.replayed_steps += replayed
        self.stats.deepest_rollback = max(self.stats.deepest_rollback, replayed)
        if best_effort:
            self.stats.best_effort_rollbacks += 1
        else:
            # Only a PROVEN-safe rollback clears the failure streak. A
            # best-effort one might have restored the corruption itself, so
            # it must not reset the counter that stops us looping on it.
            self._consecutive_failures = 0

        # The detector's baselines describe a state that no longer exists,
        # and the ABFT trust anchor now points at pre-rollback weights.
        if self.detector is not None:
            self.detector.reset()
            # Re-anchor ABFT's trusted checksums on the RESTORED (known-good)
            # weights right now, before the next step's env.advance() can land
            # fresh radiation. reset() clears _trusted; without re-anchoring
            # here the first replayed forward would lazily snapshot a weight
            # that this step's radiation may have already corrupted -- a
            # one-step blind window after every rollback. (Hostile review,
            # item 7; fixed.) The lazy "no radiation yet" fallback in
            # _trusted_checksum is only sound for step 0 of a run.
            abft = getattr(self.detector, "abft", None)
            if abft is not None:
                abft.refresh_checksums()
        # Same for the cadence: the step counter just moved backwards, and a
        # policy that thinks it is overdue would save on the very next step.
        self.policy.reset(ck.step)

        if self.telemetry:
            self.telemetry.emit(
                EVENT_ROLLBACK,
                step=step,
                t_sim=self.env.now if self.env else 0.0,
                restored_to=ck.step,
                resume_step=resume,
                replayed_steps=replayed,
                slot=ck.slot,
                trigger=verdict.reason,
                trigger_tier=verdict.tier,
                certain=verdict.certain,
                best_effort=best_effort,
                lag=self.lag_for(verdict),
            )
        return resume

    def _emit_failed_rollback(self, step: int, verdict: Verdict, why: str) -> None:
        if self.telemetry:
            self.telemetry.emit(
                EVENT_ROLLBACK,
                step=step,
                t_sim=self.env.now if self.env else 0.0,
                restored_to=None,
                failed=True,
                why=why,
                trigger=verdict.reason,
            )

    def stats_dict(self) -> dict[str, Any]:
        return {
            **self.stats.as_dict(),
            **self.saver.stats(),
            **self.policy.stats(),
        }
