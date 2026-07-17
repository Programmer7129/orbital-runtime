"""SEFIs: simulated single-event functional interrupts (hangs / crashes).

Why this channel exists at all
------------------------------
Research doc SS1: "SEFIs: simulated process hang/crash (Jetson RADECS 2024
finding: reboot cross-section EXCEEDS bit-error cross-section -- crashes
matter as much as flips)."

That finding is the reason a bit-flip-only story is incomplete: on real
flight-adjacent hardware the device is at least as likely to fall over
entirely as it is to hand you a quietly wrong number. A runtime that only
catches silent corruption would still lose the job.

Calibration honesty
-------------------
The research doc gives the QUALITATIVE ordering (reboot cross-section >
bit-error cross-section) but no per-transit reboot probability, and the two
cross-sections are quoted in different units (per-device vs per-bit), so the
ordering CANNOT be converted into a rate here without inventing a number.
We refuse to invent one (PLAN.md acceptance criterion 3).

Therefore: SEFI probability is an explicit, swept parameter, DEFAULT 0.0
(off). Nothing in the headline demo depends on an uncited SEFI rate. When a
real per-transit probability is available (M4, beam-data or vendor Xid
statistics), set `p_per_transit` and the channel is already wired.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..orbit.track import OrbitTrack

# SEFI flavours we simulate.
SEFI_CRASH = "crash"  # process dies outright
SEFI_HANG = "hang"  # process stops making progress

# Default per-SAA-transit SEFI probability. 0.0 = channel off.
# NOT a cited number -- see module docstring. Deliberately off by default so
# that no headline figure can silently depend on an invented rate.
DEFAULT_P_PER_TRANSIT = 0.0

# Split between crash and hang, given that a SEFI occurs.
# Also an assumption; only reachable when the channel is explicitly enabled.
DEFAULT_CRASH_SHARE = 0.5


class SefiCrash(RuntimeError):
    """Raised to simulate a radiation-induced process death.

    Deliberately a hard exception rather than a return code: the
    unprotected run must die the way a real one does, and the protected
    run's recovery path must prove it can catch and resume from it (M3).
    """

    def __init__(self, t_sim: float, orbit: int, flavour: str = SEFI_CRASH) -> None:
        self.t_sim = t_sim
        self.orbit = orbit
        self.flavour = flavour
        super().__init__(
            f"SEFI ({flavour}) at t_sim={t_sim:.1f}s during orbit {orbit} SAA transit"
        )


@dataclass(frozen=True)
class SefiEvent:
    """A scheduled SEFI."""

    t: float
    orbit: int
    flavour: str

    def as_record(self) -> dict:
        return {"t_sim": self.t, "orbit": self.orbit, "flavour": self.flavour}


class SefiInjector:
    """Draws SEFIs as a per-SAA-transit Bernoulli process.

    Per-transit (not per-second) because the RADECS finding is reported as a
    cross-section against the SAA proton environment: the risk is
    concentrated in the transit, which is also what makes the demo's
    "adaptive vigilance" idea (checkpoint before SAA entry) pay off.
    """

    def __init__(
        self,
        track: OrbitTrack,
        *,
        p_per_transit: float = DEFAULT_P_PER_TRANSIT,
        crash_share: float = DEFAULT_CRASH_SHARE,
    ) -> None:
        if not 0.0 <= p_per_transit <= 1.0:
            raise ValueError(f"p_per_transit must be in [0,1], got {p_per_transit}")
        if not 0.0 <= crash_share <= 1.0:
            raise ValueError(f"crash_share must be in [0,1], got {crash_share}")
        self.track = track
        self.p_per_transit = p_per_transit
        self.crash_share = crash_share

    @property
    def enabled(self) -> bool:
        return self.p_per_transit > 0.0

    def schedule(self, t0: float, t1: float, rng: np.random.Generator) -> list[SefiEvent]:
        """Draw the SEFI schedule over [t0, t1]. Deterministic given `rng`.

        Drawn up front like the flux schedule, so the whole run's fault
        timeline is fixed before step 0 (PLAN.md design rule 3).
        """
        if not self.enabled:
            return []
        events: list[SefiEvent] = []
        for lo, hi in self.track.saa_windows(t0, t1):
            # Scale by the fraction of the transit actually inside [t0,t1]:
            # a clipped half-transit carries half the risk.
            frac = (hi - lo) / self.track.saa_duration_s
            if rng.random() >= self.p_per_transit * frac:
                continue
            t = float(rng.uniform(lo, hi))
            flavour = SEFI_CRASH if rng.random() < self.crash_share else SEFI_HANG
            events.append(SefiEvent(t, self.track.orbit_index(t), flavour))
        return events
