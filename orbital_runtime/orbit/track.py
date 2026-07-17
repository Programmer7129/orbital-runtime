"""Parametric LEO orbit with a South Atlantic Anomaly (SAA) transit window.

Design notes (honesty over fidelity):

* The orbit is a simple parametric model, not an SGP4 propagation. For the
  purpose of driving a time-varying Poisson upset intensity, what matters is
  (a) the orbital period and (b) the duration + placement of the SAA transit
  per orbit — both of which are set from published flight-data summaries
  (see docs/research/technical-foundations.md §2).

* SAA membership is gated by ORBIT PHASE (a fixed window of the orbital
  period), not by a lat/lon polygon test. This keeps the SAA duration exactly
  at the calibrated value every orbit and makes seeded runs bit-reproducible.
  The ground track (lat/lon) is computed for telemetry/dashboard display only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Sidereal day, seconds. Used only for ground-track longitude regression
# (display). Standard value: 86164.0905 s.
_SIDEREAL_DAY_S = 86164.0905


@dataclass(frozen=True)
class OrbitTrack:
    """A repeating parametric LEO orbit.

    Attributes carry their calibration sources per PLAN.md design rule 2.
    """

    # ~95-minute orbital period: ISS-class LEO (400-550 km) periods are
    # 92-96 min; PLAN.md and research doc §2 specify 95 min.
    period_s: float = 95.0 * 60.0

    # SAA transit duration per orbit: flight data shows SAA transits of
    # <20 min/orbit, modeled as ~10 min per ~95-min orbit
    # (research doc §2: "50-100x multiplier ~10 min per ~95-min orbit").
    saa_duration_s: float = 10.0 * 60.0

    # Where in the orbit phase [0,1) the SAA window opens. Arbitrary
    # placement (the physics constrains duration and multiplier, not phase);
    # fixed for determinism.
    saa_start_phase: float = 0.35

    # Ground-track inclination, display only. 51.6 deg = ISS-class LEO.
    inclination_deg: float = 51.6

    # ------------------------------------------------------------------ #
    # Phase / SAA geometry (drives the flux model)
    # ------------------------------------------------------------------ #

    def phase(self, t: float) -> float:
        """Orbit phase in [0, 1) at simulation time t (seconds)."""
        return (t / self.period_s) % 1.0

    def orbit_index(self, t: float) -> int:
        """Which orbit (0-based) simulation time t falls in."""
        return int(t // self.period_s)

    @property
    def saa_fraction(self) -> float:
        """Fraction of each orbit spent inside the SAA window."""
        return self.saa_duration_s / self.period_s

    def in_saa(self, t: float) -> bool:
        """True if simulation time t lies inside the SAA transit window."""
        p = self.phase(t)
        return self.saa_start_phase <= p < self.saa_start_phase + self.saa_fraction

    def saa_entry_time(self, orbit: int) -> float:
        """Absolute time of SAA entry for a given orbit index."""
        return orbit * self.period_s + self.saa_start_phase * self.period_s

    def saa_windows(self, t0: float, t1: float) -> list[tuple[float, float]]:
        """All SAA windows overlapping [t0, t1], clipped to the interval."""
        if t1 <= t0:
            return []
        windows: list[tuple[float, float]] = []
        first_orbit = self.orbit_index(t0) - 1  # -1 to catch a window straddling t0
        last_orbit = self.orbit_index(t1) + 1
        for k in range(first_orbit, last_orbit + 1):
            start = self.saa_entry_time(k)
            end = start + self.saa_duration_s
            lo, hi = max(start, t0), min(end, t1)
            if hi > lo:
                windows.append((lo, hi))
        return windows

    def saa_seconds(self, t0: float, t1: float) -> float:
        """Total seconds of [t0, t1] spent inside SAA windows."""
        return sum(hi - lo for lo, hi in self.saa_windows(t0, t1))

    # ------------------------------------------------------------------ #
    # Ground track (telemetry / dashboard display only)
    # ------------------------------------------------------------------ #

    def ground_track(self, t: float) -> tuple[float, float]:
        """(lat_deg, lon_deg) of the sub-satellite point at time t.

        Circular-orbit approximation: latitude oscillates sinusoidally up to
        the inclination; longitude advances with the orbit and regresses with
        Earth rotation. Display only — SAA gating uses phase, not this.
        """
        p = self.phase(t)
        lat = self.inclination_deg * math.sin(2.0 * math.pi * p)
        lon = (360.0 * (t / self.period_s) - 360.0 * (t / _SIDEREAL_DAY_S)) % 360.0
        if lon >= 180.0:
            lon -= 360.0
        return lat, lon
