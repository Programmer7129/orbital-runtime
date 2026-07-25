"""Calibration and statistical tests for the time-varying Poisson engine.

PLAN.md M0 requires: "flip counts over N simulated orbits match lambda(t)
expectations; SAA share lands in the 80-97% band from flight data."

Statistical tests use fixed seeds, so they are deterministic -- they cannot
flake -- but tolerances are still derived from Poisson sigma rather than
hand-tuned to the observed value, so they would genuinely fail if the model
drifted.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from orbital_runtime.orbit import OrbitTrack
from orbital_runtime.orbit.flux import (
    DEFAULT_ECC_LEAK_FRACTION,
    DEFAULT_SAA_MULTIPLIER,
    ECC_DUE_SHARE,
    ECC_MBU_SHARE,
    ECC_SDC_SHARE,
    H100_HBM_BITS,
    MODE_ECC_OFF,
    MODE_ECC_ON,
    SECONDS_PER_DAY,
    FluxModel,
)

SEED = 1337


def h100_flux(**kw) -> FluxModel:
    return FluxModel(bits_resident=H100_HBM_BITS, **kw)


# --------------------------------------------------------------------- #
# Calibration anchors (research doc SS2)
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "base_rate, expected_per_day",
    [
        (1e-9, 640.0),  # research doc SS2: "640-64,000 flips/day"
        (1e-7, 64_000.0),
    ],
)
def test_daily_total_matches_research_doc_anchor(base_rate, expected_per_day):
    """Orbit-averaged daily total must equal the published rate x bits.

    This is what the `A` normalization in flux.py buys: the SAA multiplier
    redistributes upsets in time without inflating the cited daily total.
    """
    flux = h100_flux(base_rate_upsets_per_bit_day=base_rate)
    assert flux.expected_upsets_per_day() == pytest.approx(expected_per_day, rel=1e-9)


def test_daily_total_is_invariant_to_saa_multiplier():
    """Changing the SAA multiplier moves upsets in time, not in total."""
    totals = {
        m: h100_flux(saa_multiplier=m).expected_upsets_per_day()
        for m in (1.0, 50.0, 75.0, 100.0)
    }
    assert len(set(round(v, 6) for v in totals.values())) == 1


def test_low_rate_anchor_is_demo_friendly():
    """Research doc SS2: at 1e-9, ~1 flip every 2 minutes (H100-class)."""
    flux = h100_flux(base_rate_upsets_per_bit_day=1e-9)
    mean_interval_s = 1.0 / flux.mean_rate_per_s
    assert 100 < mean_interval_s < 160  # ~135 s


@pytest.mark.parametrize("saa_multiplier", [50.0, 75.0, 100.0])
def test_saa_share_matches_flight_data(saa_multiplier):
    """80-97% of LEO SEUs occur inside SAA transits (research doc SS2)."""
    flux = h100_flux(saa_multiplier=saa_multiplier)
    assert 0.80 <= flux.saa_share() <= 0.97


def test_default_saa_share_matches_proba_ii():
    """Default 75x is chosen to reproduce Proba-II's observed ~90% share."""
    assert h100_flux().saa_share() == pytest.approx(0.898, abs=0.005)


def test_saa_share_closed_form():
    """Share is exactly f*M/A -- independent of base rate and bits."""
    track = OrbitTrack()
    f = track.saa_fraction
    m = DEFAULT_SAA_MULTIPLIER
    expected = (f * m) / (f * m + (1 - f))
    for base_rate in (1e-9, 1e-8, 1e-7):
        flux = h100_flux(base_rate_upsets_per_bit_day=base_rate)
        assert flux.saa_share() == pytest.approx(expected)


def test_saa_multiplier_of_one_gives_time_proportional_share():
    """Degenerate check: no amplification => share is just the time fraction."""
    flux = h100_flux(saa_multiplier=1.0)
    assert flux.saa_share() == pytest.approx(flux.track.saa_fraction)


# --------------------------------------------------------------------- #
# Intensity / segment structure
# --------------------------------------------------------------------- #


def test_intensity_steps_up_inside_saa():
    flux = h100_flux()
    track = flux.track
    start = track.saa_start_phase * track.period_s
    outside = flux.intensity(start - 1.0)
    inside = flux.intensity(start + 1.0)
    assert inside == pytest.approx(outside * DEFAULT_SAA_MULTIPLIER)
    assert inside == pytest.approx(flux.saa_rate_per_s)
    assert outside == pytest.approx(flux.quiescent_rate_per_s)


def test_segments_tile_the_interval_without_gaps():
    flux = h100_flux()
    t1 = 3 * flux.track.period_s
    segs = flux.segments(0.0, t1)
    assert segs[0].t0 == pytest.approx(0.0)
    assert segs[-1].t1 == pytest.approx(t1)
    for a, b in zip(segs, segs[1:]):
        assert a.t1 == pytest.approx(b.t0)  # contiguous
        assert a.in_saa != b.in_saa  # maximal: neighbours must differ
    assert sum(s.duration for s in segs) == pytest.approx(t1)


def test_segments_saa_duration_matches_track():
    flux = h100_flux()
    n = 4
    segs = flux.segments(0.0, n * flux.track.period_s)
    saa_time = sum(s.duration for s in segs if s.in_saa)
    assert saa_time == pytest.approx(n * flux.track.saa_duration_s)


def test_expected_upsets_is_additive_over_subintervals():
    flux = h100_flux(base_rate_upsets_per_bit_day=1e-7)
    p = flux.track.period_s
    whole = flux.expected_upsets(0.0, 2 * p)
    parts = sum(flux.expected_upsets(a, b) for a, b in [(0.0, 0.3 * p), (0.3 * p, 1.7 * p), (1.7 * p, 2 * p)])
    assert whole == pytest.approx(parts)


def test_empty_and_degenerate_intervals():
    flux = h100_flux()
    assert flux.segments(10.0, 10.0) == []
    assert flux.expected_upsets(10.0, 5.0) == 0.0
    assert flux.sample(10.0, 10.0, seed=SEED) == []


# --------------------------------------------------------------------- #
# Storm / ECC modes
# --------------------------------------------------------------------- #


def test_storm_mode_is_off_by_default_and_scales_total_when_on():
    """Storm is a real transient increase in flux (research doc SS2)."""
    base = h100_flux()
    assert base.storm_enabled is False
    storm = h100_flux(storm_enabled=True, storm_multiplier=10.0)
    assert storm.expected_upsets_per_day() == pytest.approx(
        base.expected_upsets_per_day() * 10.0
    )
    # A storm redistributes nothing -- the SAA share is unchanged.
    assert storm.saa_share() == pytest.approx(base.saa_share())


def test_ecc_on_leaks_only_a_fraction():
    off = h100_flux(mode=MODE_ECC_OFF)
    on = h100_flux(mode=MODE_ECC_ON, ecc_leak_fraction=0.02)
    assert on.expected_upsets_per_day() == pytest.approx(
        off.expected_upsets_per_day() * 0.02
    )


def test_ecc_leak_fraction_defaults_to_cited_mbu_share():
    """M4c: the retired placeholder (0.02, uncited) is replaced by MICRO'21's
    31.5% multi-bit share -- SEC-DED corrects single-bit but not multi-bit, so
    the MBU share IS the ecc-on leak fraction. Cited, not invented."""
    assert DEFAULT_ECC_LEAK_FRACTION == ECC_MBU_SHARE == pytest.approx(0.315)
    # The share is duplicated in inject.memory (import-cycle avoidance); the two
    # copies trace to the same MICRO'21 anchor and must never drift apart.
    from orbital_runtime.inject.memory import MBU_SHARE

    assert ECC_MBU_SHARE == MBU_SHARE


def test_ecc_due_sdc_split_is_nsrec21_and_due_dominant():
    """NSREC'21: with ECC on, DUE exceeds SDC by 2.2-2.7x (we take 2.3x). The
    leaked (multi-bit) events split into a DUE-dominant crash channel and an
    SDC miscorrection channel; the shares partition the leak."""
    assert ECC_DUE_SHARE + ECC_SDC_SHARE == pytest.approx(1.0)
    assert ECC_DUE_SHARE > ECC_SDC_SHARE  # DUE dominant
    assert ECC_DUE_SHARE / ECC_SDC_SHARE == pytest.approx(2.3)  # NSREC'21 midpoint


def test_ecc_on_default_leaks_the_cited_share():
    """The default ecc_on flux now scales by the cited MBU share, no override."""
    off = h100_flux(mode=MODE_ECC_OFF)
    on = h100_flux(mode=MODE_ECC_ON)  # ecc_leak_fraction defaults to 0.315
    assert on.expected_upsets_per_day() == pytest.approx(
        off.expected_upsets_per_day() * ECC_MBU_SHARE
    )


def test_invalid_configs_rejected():
    with pytest.raises(ValueError, match="mode"):
        h100_flux(mode="ecc_maybe")
    with pytest.raises(ValueError, match="bits_resident"):
        FluxModel(bits_resident=-1)
    with pytest.raises(ValueError, match="base_rate"):
        h100_flux(base_rate_upsets_per_bit_day=-1e-9)
    with pytest.raises(ValueError, match="saa_multiplier"):
        h100_flux(saa_multiplier=0.5)
    with pytest.raises(ValueError, match="ecc_leak_fraction"):
        h100_flux(ecc_leak_fraction=1.5)


# --------------------------------------------------------------------- #
# Statistical validation of the sampler (M0 acceptance)
# --------------------------------------------------------------------- #


def test_sampled_count_matches_expectation_over_many_orbits():
    """Counts over N orbits match the lambda(t) integral within Poisson noise."""
    flux = h100_flux(base_rate_upsets_per_bit_day=1e-7)
    n_orbits = 15
    t1 = n_orbits * flux.track.period_s
    expected = flux.expected_upsets(0.0, t1)
    assert expected > 500  # enough events for the sigma bound to be tight

    events = flux.sample(0.0, t1, seed=SEED)
    sigma = math.sqrt(expected)
    # 4 sigma: deterministic under a fixed seed, but still a real constraint
    # (a model that drifted by >4 sigma would fail).
    assert abs(len(events) - expected) < 4 * sigma


def test_sampled_mean_converges_across_independent_seeds():
    """Average count over many seeds converges to the analytic mean."""
    flux = h100_flux(base_rate_upsets_per_bit_day=1e-7)
    t1 = 2 * flux.track.period_s
    expected = flux.expected_upsets(0.0, t1)

    n_seeds = 200
    counts = [len(flux.sample(0.0, t1, seed=s)) for s in range(n_seeds)]
    mean = float(np.mean(counts))
    # Standard error of the mean over n_seeds draws.
    sem = math.sqrt(expected / n_seeds)
    assert abs(mean - expected) < 4 * sem
    # Poisson: variance == mean. Loose bound; catches a broken sampler.
    assert 0.5 < float(np.var(counts)) / expected < 2.0


def test_sampled_saa_share_lands_in_flight_data_band():
    """M0 acceptance: empirical SAA share inside the 80-97% flight band."""
    flux = h100_flux(base_rate_upsets_per_bit_day=1e-7)
    t1 = 20 * flux.track.period_s
    events = flux.sample(0.0, t1, seed=SEED)
    assert len(events) > 1000

    share = sum(1 for e in events if e.in_saa) / len(events)
    assert 0.80 <= share <= 0.97
    assert share == pytest.approx(flux.saa_share(), abs=0.03)


def test_event_in_saa_flag_agrees_with_track_geometry():
    """The flag on each event must match an independent geometry query."""
    flux = h100_flux(base_rate_upsets_per_bit_day=1e-7)
    events = flux.sample(0.0, 5 * flux.track.period_s, seed=SEED)
    assert events
    for e in events:
        assert e.in_saa == flux.track.in_saa(e.t)


def test_events_are_sorted_and_within_bounds():
    flux = h100_flux(base_rate_upsets_per_bit_day=1e-7)
    t0, t1 = 100.0, 100.0 + 3 * flux.track.period_s
    events = flux.sample(t0, t1, seed=SEED)
    assert events
    times = [e.t for e in events]
    assert times == sorted(times)
    assert all(t0 <= t < t1 for t in times)


def test_zero_rate_yields_no_events():
    flux = h100_flux(base_rate_upsets_per_bit_day=0.0)
    assert flux.sample(0.0, 10 * flux.track.period_s, seed=SEED) == []


def test_interarrival_times_are_exponential_within_a_quiescent_stretch():
    """Homogeneous Poisson within a constant-lambda segment.

    Checks the sampler produces a real Poisson process, not just the right
    count: inside one quiescent stretch, gaps are Exponential(lambda), whose
    mean is 1/lambda and whose median is ln(2)/lambda.
    """
    track = OrbitTrack()
    flux = FluxModel(bits_resident=H100_HBM_BITS, base_rate_upsets_per_bit_day=1e-6)
    # A window strictly before the SAA opens: lambda is constant here.
    t0, t1 = 0.0, track.saa_start_phase * track.period_s

    gaps: list[float] = []
    for seed in range(60):
        times = [e.t for e in flux.sample(t0, t1, seed=seed)]
        gaps.extend(np.diff(times).tolist())
    assert len(gaps) > 2000

    lam = flux.quiescent_rate_per_s
    assert float(np.mean(gaps)) == pytest.approx(1.0 / lam, rel=0.10)
    assert float(np.median(gaps)) == pytest.approx(math.log(2) / lam, rel=0.12)


# --------------------------------------------------------------------- #
# Determinism (PLAN.md design rule 3)
# --------------------------------------------------------------------- #


def test_same_seed_reproduces_schedule_exactly():
    flux = h100_flux(base_rate_upsets_per_bit_day=1e-7)
    t1 = 3 * flux.track.period_s
    a = flux.sample(0.0, t1, seed=SEED)
    b = flux.sample(0.0, t1, seed=SEED)
    assert a == b  # frozen dataclass equality: exact float match


def test_different_seeds_give_different_schedules():
    flux = h100_flux(base_rate_upsets_per_bit_day=1e-7)
    t1 = 3 * flux.track.period_s
    a = flux.sample(0.0, t1, seed=1)
    b = flux.sample(0.0, t1, seed=2)
    assert [e.t for e in a] != [e.t for e in b]
