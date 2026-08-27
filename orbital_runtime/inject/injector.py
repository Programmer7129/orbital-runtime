"""The radiation environment, wired to a training loop.

Owns the whole fault timeline for a run and dispatches it as simulation
time advances.

Time compression
----------------
The demo is "90 minutes in orbit, 90 seconds on screen" (PLAN.md). Sim time
and wall time are decoupled: a run covering `orbits` orbits in `n_steps`
steps maps executed work -> sim time linearly,

    t_sim = (executed_steps / n_steps) * orbits * period_s

so the satellite crosses the SAA at a fixed FRACTION of the run regardless
of how fast the hardware is. That is what makes the demo reproducible on a
laptop and on an A100 alike, and it is why the schedule is drawn in sim
time up front rather than sampled per step.

The clock counts EXECUTED steps, not the training step index
------------------------------------------------------------
These differ the moment M3 rolls back, and the distinction is the
difference between an honest recovery demo and a rigged one.

A rollback rewinds the training step counter. The satellite does not fly
backwards. If sim time were keyed to the training step, a rollback would
rewind the orbit too -- the run would re-enter the SAA it just escaped, meet
the very same scheduled upsets again, and be unable to make progress. Worse,
replay would be FREE in mission time: the protected run would dodge
radiation by rewinding the universe, and the demo would be measuring a
physical impossibility.

Keying to executed steps makes replay cost exactly what it should: real time
passes, the orbit advances, and fresh radiation arrives while the run redoes
work it had already done. That cost is the honest price of protection, and
it is what the overhead number is supposed to include.

Consequence: a protected run that replays reaches sim times beyond
`orbits * period`, so the schedule is drawn over a longer horizon (see
SCHEDULE_HEADROOM). Both runs consume the same schedule PREFIX, so they
still face identical radiation for identical work -- the controlled
experiment survives.

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

from ..orbit.flux import ECC_DUE_SHARE, FluxModel, UpsetEvent
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
from .sefi import SEFI_DUE, SefiCrash, SefiEvent, SefiInjector
from .xid import XidSimulator


# How much further than the nominal mission the fault schedule is drawn.
#
# A protected run that rolls back executes more steps than it trains, so it
# reaches sim times past `orbits * period`. Without headroom the schedule
# would simply run out and the protected run would finish its last stretch
# in a radiation-free universe -- silently flattering exactly the run we are
# trying to prove. 3x covers heavy replay; exceeding it is reported rather
# than passed over (see `schedule_exhausted`).
SCHEDULE_HEADROOM = 3.0


@dataclass
class InjectionStats:
    """Running totals, reported in the demo banner."""

    flips: int = 0  # upset EVENTS delivered (one per scheduled arrival)
    flips_in_saa: int = 0
    flips_nonfinite: int = 0
    bit_flips: int = 0  # total BITS flipped (>= flips once MBUs land)
    multi_bit_events: int = 0  # events that flipped more than one bit (MBU)
    activation_hits: int = 0
    sefis: int = 0
    dues: int = 0  # ECC-on detected-uncorrectable functional interrupts
    xids: int = 0
    bit_histogram: dict[int, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "flips": self.flips,
            "flips_in_saa": self.flips_in_saa,
            "flips_nonfinite": self.flips_nonfinite,
            "bit_flips": self.bit_flips,
            "multi_bit_events": self.multi_bit_events,
            "activation_hits": self.activation_hits,
            "sefis": self.sefis,
            "dues": self.dues,
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
        legacy_fault_model: bool = False,
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
        # SEFI is ON by default (M4c), calibrated from the flux model against
        # Suncatcher's SEFI cross-section (see inject/sefi.py::from_flux). A
        # caller that wants an isolated memory-fault stream passes an explicit
        # SefiInjector(track) (p=0) -- detect_eval does exactly this.
        self.sefi = sefi if sefi is not None else SefiInjector.from_flux(flux)
        self.xid = xid or XidSimulator(ecc_on=False)

        self.activation_share = activation_share if inject_activations else 0.0
        self.legacy_fault_model = legacy_fault_model

        # --- draw the entire fault timeline up front (determinism) ---
        # Drawn over the headroom horizon, not the nominal mission, so a
        # replaying run keeps meeting radiation past its nominal end.
        self.horizon_s = self.duration_s * SCHEDULE_HEADROOM
        self.upsets: list[UpsetEvent] = flux.sample(0.0, self.horizon_s, seed)
        self.sefi_events: list[SefiEvent] = self.sefi.schedule(
            0.0, self.horizon_s, stream(seed, STREAM_SEFI)
        )

        self._rng_mem = stream(seed, STREAM_MEMORY)
        self._rng_compute = stream(seed, STREAM_COMPUTE)
        self._rng_xid = stream(seed, STREAM_XID)

        self._upset_cursor = 0
        self._sefi_cursor = 0
        self._executed = 0

        if self.compute is not None:
            self.compute.attach()

    # ------------------------------------------------------------------ #
    # Clock -- driven by executed work, never rewound
    # ------------------------------------------------------------------ #

    @property
    def executed(self) -> int:
        """Steps of work actually performed, including replayed ones."""
        return self._executed

    def t_sim_for(self, executed: int) -> float:
        """The time-compression map: executed work -> orbital time."""
        return (executed / self.n_steps) * self.duration_s

    @property
    def now(self) -> float:
        """Current mission time."""
        return self.t_sim_for(self._executed)

    @property
    def in_saa(self) -> bool:
        return self.flux.track.in_saa(self.now)

    @property
    def seconds_per_step(self) -> float:
        """Sim seconds bought by one step of work (for lookahead)."""
        return self.duration_s / self.n_steps

    @property
    def scheduled_upsets(self) -> int:
        """Upsets in the whole drawn horizon (most are past the mission end)."""
        return len(self.upsets)

    @property
    def scheduled_within_mission(self) -> int:
        """Upsets inside the nominal mission -- the number the demo quotes."""
        return sum(1 for e in self.upsets if e.t <= self.duration_s)

    @property
    def schedule_exhausted(self) -> bool:
        """True if the run outlived its drawn radiation.

        Must never be True in a reported result: past this point the run is
        flying through an empty universe, which would silently flatter it.
        """
        return self.now > self.horizon_s

    def tick(self) -> None:
        """One step of work executed. Mission time advances, never rewinds."""
        self._executed += 1

    # ------------------------------------------------------------------ #
    # Dispatch
    # ------------------------------------------------------------------ #

    def advance(self, step: int) -> None:
        """Fire every fault scheduled at or before the current mission time.

        Called once per step ATTEMPT, before the forward pass, so that
        activation hits armed here land in this step's forward.
        """
        t = self.now

        while (
            self._upset_cursor < len(self.upsets)
            and self.upsets[self._upset_cursor].t <= t
        ):
            # Advance the cursor BEFORE firing: an ECC-on DUE raises a
            # SefiCrash here, and if the cursor had not moved the same event
            # would re-fire on every replay attempt (an infinite loop). Same
            # discipline as the SEFI loop below.
            ev = self.upsets[self._upset_cursor]
            self._upset_cursor += 1
            self._fire_upset(ev, step)  # may raise SefiCrash (ECC-on DUE)

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

        # --- ECC-on redistribution: SDC -> DUE (NSREC'21) ---
        # Under ECC, every leaked event is multi-bit (single-bit is corrected).
        # NSREC'21 finds DUE dominant: most leaked events are detected-
        # uncorrectable functional interrupts (a crash we recover from by
        # process-restart), a minority miscorrected silent corruption (SDC,
        # injected). The flux model already scaled the event rate down to the
        # leaked (MBU) share; here we split each leaked event DUE vs SDC.
        if self.xid.ecc_on and self._rng_xid.random() < ECC_DUE_SHARE:
            self._fire_due(ev, step)  # emits fatal DUE Xid, then raises
            return

        # GPU-calibrated fault classes by default (Tung et al. 2026). The
        # legacy MICRO'21 memory-only path stays reachable for A/B comparison
        # of results produced before the fault model was corrected.
        cluster = (
            self.memory.inject_event(self._rng_mem)
            if self.legacy_fault_model
            else self.memory.inject_gpu_event(self._rng_mem)
        )
        if cluster is None:
            return

        self.stats.flips += 1  # one upset EVENT
        self.stats.flips_in_saa += int(ev.in_saa)
        self.stats.flips_nonfinite += int(cluster.became_nonfinite)
        self.stats.bit_flips += cluster.size
        self.stats.multi_bit_events += int(cluster.multi_bit)
        for b in cluster.bit_positions:
            self.stats.bit_histogram[b] = self.stats.bit_histogram.get(b, 0) + 1

        if self.telemetry:
            self.telemetry.emit(
                EVENT_FLIP,
                step=step,
                t_sim=ev.t,
                in_saa=ev.in_saa,
                **cluster.as_record(),
            )

        # Xid stream. Under ecc_off a multi-bit event would be uncorrectable,
        # but the demo default reports nothing at all (report_prob 0 -- the
        # silent-corruption regime). Under ecc_on, reaching HERE means this
        # leaked event was MISCORRECTED into a silent SDC (the DUE half already
        # crashed above), so it must NOT surface as a fatal Xid -- a fatal Xid
        # is a DETECTED error, and an SDC is by definition undetected. Hence
        # multi_bit is forced False on the ecc_on SDC path.
        xid_ev = self.xid.on_flip(
            ev.t, self._rng_xid, multi_bit=(cluster.multi_bit and not self.xid.ecc_on)
        )
        if xid_ev is not None:
            self.stats.xids += 1
            if self.telemetry:
                self.telemetry.emit(EVENT_XID, step=step, **xid_ev.as_record())

    def _fire_due(self, ev: UpsetEvent, step: int) -> None:
        """An ECC-on detected-uncorrectable error: a functional interrupt.

        The device's ECC logic caught a multi-bit error it could not fix and
        halts the job (Xid 48/95 class). Surfaced as a fatal Xid and raised as
        a SefiCrash(DUE) so it takes the same process-restart recovery path as
        a SEFI. This is the DUE half of the SDC->DUE redistribution.
        """
        self.stats.dues += 1
        orbit = self.flux.track.orbit_index(ev.t)
        xid_ev = self.xid.on_sefi(ev.t, self._rng_xid)  # always fatal (detected)
        self.stats.xids += 1
        if self.telemetry:
            self.telemetry.emit(EVENT_SEFI, step=step, t_sim=ev.t, orbit=orbit, flavour=SEFI_DUE)
            self.telemetry.emit(EVENT_XID, step=step, **xid_ev.as_record())
        raise SefiCrash(ev.t, orbit, SEFI_DUE)

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
                    t_sim=self.now,
                    fired=True,
                    **hit.as_record(),
                )

    def close(self) -> None:
        if self.compute is not None:
            self.compute.detach()
