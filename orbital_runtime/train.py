"""The training loop the runtime protects.

One loop serves both demo runs. `--protect off` is not a different code
path with the injector bolted on -- it is this same loop with detection and
recovery disabled. That matters: if the two runs used different loops, any
overhead or divergence we measured could be an artifact of the loop rather
than of protection, and the headline number would be meaningless.

Structured for M2/M3 to slot into: `detector` and `recovery` are optional
collaborators, absent in M1.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import torch

from .inject.injector import RadiationEnvironment
from .inject.sefi import SefiCrash
from .telemetry import EVENT_RUN_END, EVENT_RUN_START, EVENT_STEP, Telemetry
from .workload import Workload

# Why a run stopped.
DEATH_NONE = None
DEATH_NAN = "nan_loss"  # loss went non-finite: failure mode (a)
DEATH_SEFI = "sefi"  # simulated crash/hang
DEATH_EXPLODED = "loss_exploded"  # finite but absurd -- diverged past recovery


# A loss above this is treated as a dead run even while still finite.
# Rationale: cross-entropy over a 65-char vocab starts at ln(65) ~ 4.17; a
# loss in the hundreds is not "training badly", it is a corrupted model that
# has not yet produced a literal NaN. Without this bound an exponent-bit
# strike can spend thousands of steps producing 1e30 losses before finally
# overflowing, which is neither realistic nor watchable in a 90-second demo.
LOSS_EXPLOSION_THRESHOLD = 1e4


@dataclass
class StepRecord:
    step: int
    loss: float
    t_sim: float
    in_saa: bool
    grad_norm: float


@dataclass
class TrainResult:
    """Everything the demo banner and the tests need."""

    steps_completed: int
    steps_requested: int
    final_loss: float
    final_val_loss: float
    died: bool
    death_reason: str | None
    death_step: int | None
    history: list[StepRecord] = field(default_factory=list)
    wall_s: float = 0.0
    injected: int = 0
    detected: int = 0
    recovered: int = 0
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def completed(self) -> bool:
        return not self.died and self.steps_completed >= self.steps_requested

    @property
    def losses(self) -> list[float]:
        return [r.loss for r in self.history]

    def summary(self) -> str:
        status = "COMPLETED" if self.completed else f"DIED ({self.death_reason})"
        return (
            f"{status} at step {self.steps_completed}/{self.steps_requested} | "
            f"train loss {self.final_loss:.4f} | val loss {self.final_val_loss:.4f} | "
            f"injected {self.injected} detected {self.detected} recovered {self.recovered} | "
            f"{self.wall_s:.1f}s"
        )


@dataclass
class TrainConfig:
    steps: int = 400
    grad_clip: float = 1.0  # nanoGPT default; also the honest baseline (see below)
    eval_every: int = 0  # 0 = only at the end
    eval_batches: int = 8
    log_every: int = 10


def train(
    workload: Workload,
    *,
    cfg: TrainConfig,
    env: RadiationEnvironment | None = None,
    telemetry: Telemetry | None = None,
    detector: Any = None,  # M2
    recovery: Any = None,  # M3
) -> TrainResult:
    """Run the loop. `env=None` is a clean baseline (no radiation)."""
    model, optimizer = workload.model, workload.optimizer
    model.train()

    history: list[StepRecord] = []
    died, death_reason, death_step = False, DEATH_NONE, None
    t_start = time.perf_counter()
    step = 0

    if telemetry:
        telemetry.emit(
            EVENT_RUN_START,
            step=0,
            t_sim=0.0,
            steps=cfg.steps,
            protected=recovery is not None,
            irradiated=env is not None,
            scheduled_upsets=env.scheduled_upsets if env else 0,
            device=str(workload.device),
        )

    try:
        while step < cfg.steps:
            t_sim = env.t_sim(step) if env else 0.0
            in_saa = env.in_saa(step) if env else False

            # --- radiation lands before the forward pass ---
            # Ordering is load-bearing: the ABFT checksums snapshotted after
            # the PREVIOUS step's optimizer.step() predate this radiation, so
            # a flip landing here is compared against pre-flip truth.
            if env is not None:
                env.advance_to(step)  # may raise SefiCrash

            # --- M2: tell the tiers where we are, and arm them ---
            if detector is not None:
                detector.before_step(t_sim=t_sim, in_saa=in_saa)

            # --- M3: checkpoint policy decides before we compute ---
            if recovery is not None:
                recovery.before_step(step=step, t_sim=t_sim, in_saa=in_saa)

            loss = workload.loss_for_step(step)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if env is not None:
                env.collect_activation_hits(step)

            # grad_clip returns the PRE-clip norm: the quantity the free-tier
            # z-score detector keys on (research doc SS3).
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            )
            optimizer.step()

            # The weights just changed legitimately, and no radiation has
            # landed since. This instant -- and only this instant -- is when
            # ABFT's trusted checksums may be taken. Snapshotting anywhere
            # later would anchor trust to already-corrupted weights.
            if detector is not None and detector.abft is not None:
                detector.abft.refresh_checksums()

            loss_val = float(loss.item())
            record = StepRecord(step, loss_val, t_sim, in_saa, grad_norm)
            history.append(record)

            if telemetry and (cfg.log_every and step % cfg.log_every == 0):
                telemetry.emit(
                    EVENT_STEP,
                    step=step,
                    t_sim=t_sim,
                    loss=loss_val,
                    grad_norm=grad_norm,
                    in_saa=in_saa,
                )

            # --- M2: detection ---
            verdict = None
            if detector is not None:
                verdict = detector.observe(
                    step=step, loss=loss_val, grad_norm=grad_norm, model=model
                )

            # --- M3: recovery on detection ---
            if verdict is not None and verdict.triggered and recovery is not None:
                step = recovery.on_detection(step=step, verdict=verdict)
                continue

            # --- unprotected death conditions ---
            if not math.isfinite(loss_val):
                died, death_reason, death_step = True, DEATH_NAN, step
                break
            if abs(loss_val) > LOSS_EXPLOSION_THRESHOLD:
                died, death_reason, death_step = True, DEATH_EXPLODED, step
                break

            if cfg.eval_every and step and step % cfg.eval_every == 0:
                workload.evaluate(cfg.eval_batches)

            step += 1

    except SefiCrash as e:
        died, death_reason, death_step = True, DEATH_SEFI, step
        if telemetry:
            telemetry.emit(EVENT_STEP, step=step, t_sim=e.t_sim, sefi=True)

    wall = time.perf_counter() - t_start

    # A dead model produces a meaningless val loss; report it anyway rather
    # than hiding it -- "inf" is the honest number for a NaN'd model.
    try:
        val_loss = workload.evaluate(cfg.eval_batches)
    except Exception:
        val_loss = float("inf")
    if not math.isfinite(val_loss):
        val_loss = float("inf")

    result = TrainResult(
        steps_completed=len(history),
        steps_requested=cfg.steps,
        final_loss=history[-1].loss if history else float("inf"),
        final_val_loss=val_loss,
        died=died,
        death_reason=death_reason,
        death_step=death_step,
        history=history,
        wall_s=wall,
        injected=(env.stats.flips + env.stats.activation_hits) if env else 0,
        detected=detector.detections if detector is not None else 0,
        recovered=recovery.rollbacks if recovery is not None else 0,
        stats=env.stats.as_dict() if env else {},
    )

    if telemetry:
        telemetry.emit(
            EVENT_RUN_END,
            step=result.steps_completed,
            t_sim=env.now if env else 0.0,
            died=died,
            death_reason=death_reason,
            final_loss=result.final_loss,
            final_val_loss=result.final_val_loss,
            wall_s=round(wall, 3),
            **result.stats,
        )
    return result
