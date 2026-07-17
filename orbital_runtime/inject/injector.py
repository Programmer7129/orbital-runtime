"""The radiation environment, wired to a training loop.

Owns the whole fault timeline for a run and dispatches it as simulation
time advances.

Time compression
----------------
The demo is "90 minutes in orbit, 90 seconds on screen" (PLAN.md). Sim time
and wall time are decoupled: a run of `n_steps` steps covering `orbits`
orbits maps step -> sim time linearly,

    t_sim(step) = (step / n_steps) * orbits * period_s

so the satellite crosses the SAA at a fixed FRACTION of the run regardless
of how fast the hardware is. That is what makes the demo reproducible on a
laptop and on an A100 alike, and it is why the schedule is drawn in sim
time up front rather than sampled per step.

Determinism (PLAN.md design rule 3)
-----------------------------------
The ENTIRE fault timeline -- flip times, SEFI times -- is drawn before step
0 from named RNG streams. Nothing about the fault schedule depends on the
workload's behaviour, on wall-clock timing, or on whether protection is
enabled. This is what makes `--protect on` and `--protect off` a controlled
experiment: both runs face a bit-identical bombardment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..orbit.flux import FluxModel, UpsetEvent
from ..rng import STREAM_COMPUTE, STREAM_MEMORY, STREAM_SEFI, STREAM_XID, stream
from ..telemetry import (
    EVENT_ACTIVATION,
    EVENT_FLIP,
    EVENT_SEFI,
    EVENT_XID,
    Telemetry,
)
from .compute import ComputeInjector
from .memory import MemoryInjector
from .sefi import SefiCrash, SefiEvent, SefiInjector
from .xid import XidSimulator


@dataclass
class InjectionStats:
    """Running totals, reported in the demo banner."""

    flips: int = 0
    flips_in_saa: int = 0
    flips_nonfinite: int = 0
    activation_hits: int = 0
    sefis: int = 0
    xids: int = 0
    bit_histogram: dict[int, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "flips": self.flips,
            "flips_in_saa": self.flips_in_saa,
            "flips_nonfinite": self.flips_nonfinite,
            "activation_hits": self.activation_hits,
            "sefis": self.sefis,
            "xids": self.xids,
        }


class RadiationEnvironment:
    """Dispatches a pre-drawn fault schedule into a live training run."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        flux: FluxModel,
        seed: int,
        n_steps: int,
        orbits: float,
        telemetry: Telemetry | None = None,
        sefi: SefiInjector | None = None,
        xid: XidSimulator | None = None,
        inject_activations: bool = False,
        activation_share: float = 0.0,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.flux = flux
        self.seed = seed
        self.n_steps = max(1, n_steps)
        self.orbits = orbits
        self.telemetry = telemetry
        self.stats = InjectionStats()

        self.duration_s = orbits * flux.track.period_s

        self.memory = MemoryInjector(model, optimizer)
        self.compute = ComputeInjector(model) if inject_activations else None
        self.sefi = sefi or SefiInjector(flux.track)
        self.xid = xid or XidSimulator(ecc_on=False)

        self.activation_share = activation_share if inject_activations else 0.0

        # --- draw the entire fault timeline up front (determinism) ---
        self.upsets: list[UpsetEvent] = flux.sample(0.0, self.duration_s, seed)
        self.sefi_events: list[SefiEvent] = self.sefi.schedule(
            0.0, self.duration_s, stream(seed, STREAM_SEFI)
        )

        self._rng_mem = stream(seed, STREAM_MEMORY)
        self._rng_compute = stream(seed, STREAM_COMPUTE)
        self._rng_xid = stream(seed, STREAM_XID)

        self._upset_cursor = 0
        self._sefi_cursor = 0
        self._t_sim = 0.0

        if self.compute is not None:
            self.compute.attach()

    # ------------------------------------------------------------------ #
    # Clock
    # ------------------------------------------------------------------ #

    def t_sim(self, step: int) -> float:
        """Simulation time at a training step (the time-compression map)."""
        return (step / self.n_steps) * self.duration_s

    @property
    def now(self) -> float:
        return self._t_sim

    def in_saa(self, step: int) -> bool:
        return self.flux.track.in_saa(self.t_sim(step))

    @property
    def scheduled_upsets(self) -> int:
        return len(self.upsets)

    # ------------------------------------------------------------------ #
    # Dispatch
    # ------------------------------------------------------------------ #

    def advance_to(self, step: int) -> None:
        """Fire every fault scheduled at or before `step`'s sim time.

        Called once per training step, BEFORE the forward pass, so that
        activation hits armed here land in this step's forward.
        """
        t = self.t_sim(step)
        self._t_sim = t

        while (
            self._upset_cursor < len(self.upsets)
            and self.upsets[self._upset_cursor].t <= t
        ):
            self._fire_upset(self.upsets[self._upset_cursor], step)
            self._upset_cursor += 1

        while (
            self._sefi_cursor < len(self.sefi_events)
            and self.sefi_events[self._sefi_cursor].t <= t
        ):
            ev = self.sefi_events[self._sefi_cursor]
            self._sefi_cursor += 1
            self._fire_sefi(ev, step)  # raises

    def _fire_upset(self, ev: UpsetEvent, step: int) -> None:
        # Route to the activation channel or the memory channel.
        if (
            self.compute is not None
            and self.activation_share > 0.0
            and self._rng_compute.random() < self.activation_share
        ):
            self.compute.arm(self._rng_compute)
            self.stats.activation_hits += 1
            if self.telemetry:
                self.telemetry.emit(
                    EVENT_ACTIVATION,
                    step=step,
                    t_sim=ev.t,
                    in_saa=ev.in_saa,
                    armed=True,
                )
            return

        flip = self.memory.inject(self._rng_mem)
        if flip is None:
            return

        self.stats.flips += 1
        self.stats.flips_in_saa += int(ev.in_saa)
        self.stats.flips_nonfinite += int(flip.became_nonfinite)
        self.stats.bit_histogram[flip.bit] = self.stats.bit_histogram.get(flip.bit, 0) + 1

        if self.telemetry:
            self.telemetry.emit(
                EVENT_FLIP,
                step=step,
                t_sim=ev.t,
                in_saa=ev.in_saa,
                **flip.as_record(),
            )

        xid_ev = self.xid.on_flip(ev.t, self._rng_xid)
        if xid_ev is not None:
            self.stats.xids += 1
            if self.telemetry:
                self.telemetry.emit(EVENT_XID, step=step, **xid_ev.as_record())

    def _fire_sefi(self, ev: SefiEvent, step: int) -> None:
        self.stats.sefis += 1
        xid_ev = self.xid.on_sefi(ev.t, self._rng_xid)
        self.stats.xids += 1
        if self.telemetry:
            self.telemetry.emit(EVENT_SEFI, step=step, **ev.as_record())
            self.telemetry.emit(EVENT_XID, step=step, **xid_ev.as_record())
        raise SefiCrash(ev.t, ev.orbit, ev.flavour)

    # ------------------------------------------------------------------ #
    # Post-step bookkeeping
    # ------------------------------------------------------------------ #

    def collect_activation_hits(self, step: int) -> None:
        """Log activation corruptions that actually fired this forward."""
        if self.compute is None:
            return
        for hit in self.compute.drain_hits():
            if self.telemetry:
                self.telemetry.emit(
                    EVENT_ACTIVATION,
                    step=step,
                    t_sim=self._t_sim,
                    fired=True,
                    **hit.as_record(),
                )

    def close(self) -> None:
        if self.compute is not None:
            self.compute.detach()
