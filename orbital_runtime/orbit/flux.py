"""Time-varying Poisson upset intensity lambda(t) for a LEO orbit.

PLAN.md design rule 2 (Calibration is sacred): every constant in this module
carries a comment citing its source. Sources are section references into
`docs/research/technical-foundations.md`, which in turn cites the primary
literature. The demo must be able to say "these rates come from NASA NEPP /
CREME96 / Suncatcher" and have that be literally true.

Model
-----
    lambda(t) = quiescent * saa(t)

    quiescent = (base_rate * bits_resident * storm * ecc_leak / 86400) / A
    A         = saa_fraction * saa_multiplier + (1 - saa_fraction)

where `base_rate` is in upsets/bit-day, `saa(t)` is 1 outside the SAA and
`saa_multiplier` inside it, `storm` is 1 unless storm mode is enabled, and
`ecc_leak` is 1 with ECC off.

Why the `A` normalization (this matters -- it is the difference between a
calibrated model and a plausible-looking one)
--------------------------------------------------------------------------
Research doc SS2 pins down two independent numbers that must BOTH come out
of this model:

  (a) an H100-class 80 GB HBM sees 640-64,000 flips/day over the
      1e-9..1e-7 sweep, and
  (b) 80-97% of LEO upsets arrive inside SAA transits.

Published flight rates like "1e-9 upsets/bit-day (Flying Laptop, 600 km
SSO)" are **orbit-averaged**: an observed total upset count divided by bits
divided by days, with the satellite's SAA passes already inside that total.
So the SAA multiplier must REDISTRIBUTE upsets in time, not manufacture new
ones. Applying a raw 75x multiplier on top of the published rate would
inflate the daily total by A = 8.79x -- reporting ~5,600 flips/day at 1e-9
where the cited source says 640, i.e. quietly contradicting our own
citation while looking superficially reasonable.

Dividing by the time-average multiplier `A` makes the orbit-averaged rate
equal the published `base_rate` exactly, while leaving the SAA share
untouched at saa_fraction*M / A. Both anchors hold simultaneously; see
tests/test_flux.py::test_daily_total_matches_research_doc_anchor and
::test_saa_share_matches_flight_data.

lambda(t) is **piecewise constant** -- it only changes at SAA boundaries.
That lets us sample arrivals exactly, per segment, rather than by thinning:
draw N ~ Poisson(lambda * dt), then place N uniform times in the segment.
Exact sampling means the statistical tests check the model, not the
convergence of a rejection sampler.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..rng import STREAM_FLUX, stream
from .track import OrbitTrack

SECONDS_PER_DAY = 86400.0

# H100-class 80 GB HBM, in bits: 80 * 2^30 bytes * 8 bits/byte = 6.87e11.
# Research doc SS2 quotes 6.4e11 ("80 GB HBM = 6.4e11 bits"), i.e. decimal
# GB (80e9 * 8 = 6.4e11). We keep the research doc's figure so the demo's
# arithmetic matches the cited text exactly.
H100_HBM_BITS = 6.4e11

# Default base rate, upsets/bit-day. Sweepable 1e-9 -> 1e-7 (research doc
# SS2). Bounds: NASA NEPP design criteria are 1e-5..1e-7; modern
# deep-submicron flight data sits at the low end (~1e-9..1e-8, Flying Laptop
# 600 km SSO). Methodology to cite: CREME96. We default to the geometric
# middle of the sweep band.
DEFAULT_BASE_RATE_UPSETS_PER_BIT_DAY = 1e-8

# SAA intensity multiplier. Research doc SS2: flight data shows 80-97% of all
# LEO SEUs occur inside SAA transits (<20 min/orbit), modeled as a 50-100x
# multiplier over ~10 min of a ~95-min orbit.
#
# We default to 75x, the midpoint of the published 50-100x band. With the
# 10/95-min SAA window this yields an 89.8% SAA share of all upsets -- which
# lands on Proba-II's observed ~90% (research doc SS2). The band endpoints
# bracket the flight-data band correctly too: 50x -> 85.5%, 100x -> 92.2%,
# both inside 80-97%. See tests/test_flux.py::test_saa_share_matches_flight_data.
DEFAULT_SAA_MULTIPLIER = 75.0

# Storm multiplier, applied only when storm mode is enabled (off by default,
# per PLAN.md architecture note). Research doc SS2: +10-100x transient, from
# the May 2024 Gannon storm (Wu 2025, Space Weather). We default to the
# conservative low end of that band.
DEFAULT_STORM_MULTIPLIER = 10.0

# Fraction of upsets that leak through SEC-DED ECC in `ecc_on` mode.
#
# !! ENGINEERING ASSUMPTION -- NOT YET TRACEABLE TO A CITED NUMBER !!
#
# Research doc SS2 states the qualitative behaviour ("ECC-on mode: only
# multi-bit residuals + logic faults + SEFIs leak through") but gives no
# quantitative multi-bit-upset fraction, and we will not invent one: PLAN.md
# acceptance criterion 3 requires every physics constant to be traceable to
# the research doc, and a fabricated MBU fraction in a YC demo is a
# credibility risk we are not taking.
#
# 0.02 is a placeholder chosen to be small-but-nonzero so the ecc_on code
# path is exercised and testable. Any headline number produced in `ecc_on`
# mode MUST cite a real MBU fraction first -- see STATUS.md (M4 needs).
# `ecc_off` (the default, and the mode the headline demo runs in) is fully
# calibrated and unaffected by this constant.
DEFAULT_ECC_LEAK_FRACTION = 0.02

MODE_ECC_OFF = "ecc_off"
MODE_ECC_ON = "ecc_on"
VALID_MODES = (MODE_ECC_OFF, MODE_ECC_ON)


@dataclass(frozen=True)
class UpsetEvent:
    """A single scheduled single-event upset."""

    t: float  # simulation time, seconds
    in_saa: bool  # whether it arrived during an SAA transit


@dataclass(frozen=True)
class Segment:
    """A maximal interval over which lambda(t) is constant."""

    t0: float
    t1: float
    rate_per_s: float
    in_saa: bool

    @property
    def duration(self) -> float:
        return self.t1 - self.t0

    @property
    def expected(self) -> float:
        return self.rate_per_s * self.duration


@dataclass(frozen=True)
class FluxModel:
    """Time-varying Poisson intensity over an `OrbitTrack`."""

    bits_resident: float
    track: OrbitTrack = field(default_factory=OrbitTrack)
    base_rate_upsets_per_bit_day: float = DEFAULT_BASE_RATE_UPSETS_PER_BIT_DAY
    saa_multiplier: float = DEFAULT_SAA_MULTIPLIER
    storm_enabled: bool = False
    storm_multiplier: float = DEFAULT_STORM_MULTIPLIER
    mode: str = MODE_ECC_OFF
    ecc_leak_fraction: float = DEFAULT_ECC_LEAK_FRACTION

    def __post_init__(self) -> None:
        if self.mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {self.mode!r}")
        if self.bits_resident < 0:
            raise ValueError(f"bits_resident must be >= 0, got {self.bits_resident}")
        if self.base_rate_upsets_per_bit_day < 0:
            raise ValueError(
                f"base_rate must be >= 0, got {self.base_rate_upsets_per_bit_day}"
            )
        if self.saa_multiplier < 1.0:
            raise ValueError(
                f"saa_multiplier must be >= 1, got {self.saa_multiplier}"
            )
        if not 0.0 <= self.ecc_leak_fraction <= 1.0:
            raise ValueError(
                f"ecc_leak_fraction must be in [0,1], got {self.ecc_leak_fraction}"
            )

    # ------------------------------------------------------------------ #
    # Intensity
    # ------------------------------------------------------------------ #

    @property
    def time_average_multiplier(self) -> float:
        """`A`: the orbit-time-average of saa(t).

        Dividing by this makes the orbit-averaged intensity equal the
        published (orbit-averaged) base rate. See module docstring.
        """
        f = self.track.saa_fraction
        return f * self.saa_multiplier + (1.0 - f)

    @property
    def mean_rate_per_s(self) -> float:
        """Orbit-averaged lambda, i.e. the published rate in per-second form."""
        per_day = self.base_rate_upsets_per_bit_day * self.bits_resident
        if self.storm_enabled:
            per_day *= self.storm_multiplier
        if self.mode == MODE_ECC_ON:
            per_day *= self.ecc_leak_fraction
        return per_day / SECONDS_PER_DAY

    @property
    def quiescent_rate_per_s(self) -> float:
        """lambda outside the SAA."""
        return self.mean_rate_per_s / self.time_average_multiplier

    @property
    def saa_rate_per_s(self) -> float:
        """lambda inside an SAA transit."""
        return self.quiescent_rate_per_s * self.saa_multiplier

    def intensity(self, t: float) -> float:
        """lambda(t) in upsets/second."""
        rate = self.quiescent_rate_per_s
        if self.track.in_saa(t):
            rate *= self.saa_multiplier
        return rate

    def segments(self, t0: float, t1: float) -> list[Segment]:
        """Partition [t0, t1] into maximal constant-lambda segments.

        Boundaries are exactly the SAA window edges, so the returned
        segments tile [t0, t1] with no gaps or overlaps.
        """
        if t1 <= t0:
            return []
        quiescent = self.quiescent_rate_per_s
        saa_rate = self.saa_rate_per_s

        segs: list[Segment] = []
        cursor = t0
        for lo, hi in self.track.saa_windows(t0, t1):
            if lo > cursor:  # quiescent stretch before this SAA window
                segs.append(Segment(cursor, lo, quiescent, in_saa=False))
            segs.append(Segment(lo, hi, saa_rate, in_saa=True))
            cursor = hi
        if cursor < t1:  # trailing quiescent stretch
            segs.append(Segment(cursor, t1, quiescent, in_saa=False))
        return segs

    def expected_upsets(self, t0: float, t1: float) -> float:
        """Integral of lambda(t) over [t0, t1] -- the Poisson mean."""
        return sum(s.expected for s in self.segments(t0, t1))

    def expected_upsets_per_day(self) -> float:
        """Upsets/day, orbit-averaged.

        The headline calibration figure: research doc SS2 says an H100-class
        80 GB HBM sees 640-64,000 flips/day across the 1e-9..1e-7 sweep.

        Computed by scaling ONE WHOLE ORBIT up to a day, rather than
        integrating lambda(t) over a literal 86400 s window. A day is 15.16
        orbits, and that trailing 0.16 orbit does not contain a proportional
        slice of an SAA transit (the window sits at phase 0.35-0.46), so a
        literal integration reports ~0.9% fewer upsets than the cited rate --
        an artifact of where the day boundary falls in the orbit, not
        physics. The whole-orbit average is the quantity flight data
        actually reports.
        """
        per_orbit = self.expected_upsets(0.0, self.track.period_s)
        return per_orbit * (SECONDS_PER_DAY / self.track.period_s)

    def expected_upsets_in_saa_per_orbit(self) -> float:
        """Expected upsets inside the one SAA transit of a single orbit.

        The SEFI channel is calibrated against this (see
        `inject/sefi.py::SefiInjector.from_flux`): a SEFI cross-section is
        measured against the same SAA proton environment that drives these
        upsets, so the per-transit SEFI probability scales with the number of
        upset (SDC-class) events the transit delivers.
        """
        return self.expected_upsets(0.0, self.track.period_s) * self.saa_share()

    def saa_share(self) -> float:
        """Fraction of expected upsets that arrive inside SAA transits.

        Closed form over one orbit; compare against the 80-97% flight-data
        band (research doc SS2).
        """
        segs = self.segments(0.0, self.track.period_s)
        total = sum(s.expected for s in segs)
        if total == 0.0:
            return 0.0
        return sum(s.expected for s in segs if s.in_saa) / total

    # ------------------------------------------------------------------ #
    # Sampling
    # ------------------------------------------------------------------ #

    def sample(self, t0: float, t1: float, seed: int) -> list[UpsetEvent]:
        """Draw the upset schedule over [t0, t1]; deterministic in `seed`.

        Exact piecewise sampling: within each constant-lambda segment the
        process is homogeneous Poisson, so the count is Poisson(lambda*dt)
        and the arrival times are i.i.d. uniform given the count.
        """
        rng = stream(seed, STREAM_FLUX)
        return self._sample_with(rng, t0, t1)

    def _sample_with(
        self, rng: np.random.Generator, t0: float, t1: float
    ) -> list[UpsetEvent]:
        events: list[UpsetEvent] = []
        for seg in self.segments(t0, t1):
            if seg.rate_per_s <= 0.0:
                continue
            n = int(rng.poisson(seg.expected))
            if n == 0:
                continue
            times = rng.uniform(seg.t0, seg.t1, size=n)
            times.sort()
            events.extend(UpsetEvent(float(t), seg.in_saa) for t in times)
        # Segments are already in time order and non-overlapping, so the
        # concatenation is sorted; sort anyway to keep the contract explicit.
        events.sort(key=lambda e: e.t)
        return events
