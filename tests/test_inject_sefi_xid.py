"""SEFI and synthetic Xid channels."""

from __future__ import annotations

import numpy as np
import pytest

from orbital_runtime.inject.sefi import (
    SEFI_CRASH,
    SEFI_HANG,
    SefiCrash,
    SefiInjector,
)
from orbital_runtime.inject.xid import (
    FATAL_XIDS,
    XID_MEANING,
    XidSimulator,
)
from orbital_runtime.orbit.track import OrbitTrack
from orbital_runtime.rng import STREAM_SEFI, STREAM_XID, stream


# --------------------------------------------------------------------- #
# SEFI
# --------------------------------------------------------------------- #


def test_sefi_is_off_by_default():
    """No headline number may depend on an uncited SEFI rate."""
    inj = SefiInjector(OrbitTrack())
    assert inj.p_per_transit == 0.0
    assert not inj.enabled
    assert inj.schedule(0.0, 100 * OrbitTrack().period_s, stream(0, STREAM_SEFI)) == []


def test_from_flux_calibrates_a_nonzero_on_by_default_probability():
    """M4c: SEFI is calibrated ON from the flux model, not invented/off.

    `from_flux` rides the SDC-class upset stream at Suncatcher's SEFI/SDC
    cross-section ratio, so a real irradiated device gets a nonzero, cited
    per-transit SEFI probability.
    """
    from orbital_runtime.inject.sefi import SEFI_PER_SDC_EVENT
    from orbital_runtime.orbit.flux import FluxModel, H100_HBM_BITS

    flux = FluxModel(bits_resident=H100_HBM_BITS, base_rate_upsets_per_bit_day=1e-8)
    inj = SefiInjector.from_flux(flux)
    assert inj.enabled
    assert 0.0 < inj.p_per_transit <= 1.0

    # It equals 1 - exp(-r * mu) for r = sigma_SEFI/sigma_SDC and mu the
    # expected SDC-class upsets in one SAA transit.
    mu = flux.expected_upsets_in_saa_per_orbit()
    expected = 1.0 - np.exp(-SEFI_PER_SDC_EVENT * mu)
    assert inj.p_per_transit == pytest.approx(expected)


def test_from_flux_probability_grows_with_exposure():
    """More upsets per transit -> more SEFIs per transit (same beam ratio)."""
    from orbital_runtime.orbit.flux import FluxModel, H100_HBM_BITS

    lo = SefiInjector.from_flux(
        FluxModel(bits_resident=H100_HBM_BITS, base_rate_upsets_per_bit_day=1e-9)
    )
    hi = SefiInjector.from_flux(
        FluxModel(bits_resident=H100_HBM_BITS, base_rate_upsets_per_bit_day=1e-7)
    )
    assert hi.p_per_transit > lo.p_per_transit


def test_suncatcher_cross_section_cross_check():
    """The two Suncatcher numbers reproduce its stated '~1 SEFI per 5 krad'."""
    from orbital_runtime.inject.sefi import (
        DOSE_FLUENCE_P_PER_CM2_PER_RAD,
        SUNCATCHER_SEFI_SIGMA_CM2,
    )

    fluence_5krad = 5000.0 * DOSE_FLUENCE_P_PER_CM2_PER_RAD
    expected_sefis = SUNCATCHER_SEFI_SIGMA_CM2 * fluence_5krad
    assert expected_sefis == pytest.approx(1.0, abs=0.3)  # 0.79 ~= 1


def test_sefi_rate_matches_per_transit_probability():
    """Bernoulli per SAA transit, at the configured probability."""
    track = OrbitTrack()
    p = 0.25
    inj = SefiInjector(track, p_per_transit=p)
    n_orbits = 400
    events = inj.schedule(0.0, n_orbits * track.period_s, stream(3, STREAM_SEFI))

    expected = n_orbits * p
    sigma = np.sqrt(n_orbits * p * (1 - p))
    assert abs(len(events) - expected) < 4 * sigma


def test_sefi_always_lands_inside_an_saa_transit():
    """The RADECS cross-section is against the SAA proton environment."""
    track = OrbitTrack()
    inj = SefiInjector(track, p_per_transit=0.9)
    events = inj.schedule(0.0, 60 * track.period_s, stream(4, STREAM_SEFI))
    assert events
    for e in events:
        assert track.in_saa(e.t)
        assert e.orbit == track.orbit_index(e.t)


def test_sefi_at_probability_one_fires_every_transit():
    track = OrbitTrack()
    inj = SefiInjector(track, p_per_transit=1.0)
    n = 12
    events = inj.schedule(0.0, n * track.period_s, stream(5, STREAM_SEFI))
    assert len(events) == n


def test_clipped_transit_carries_proportional_risk():
    """Half a transit inside the window is half the risk, not full risk."""
    track = OrbitTrack()
    inj = SefiInjector(track, p_per_transit=1.0)
    start = track.saa_entry_time(0)
    # Only the last 25% of transit 0 lies in the window.
    quarter = start + 0.75 * track.saa_duration_s
    events = inj.schedule(quarter, start + track.saa_duration_s, stream(6, STREAM_SEFI))
    # p * frac = 1.0 * 0.25 -> fires sometimes, not always.
    fires = sum(
        len(inj.schedule(quarter, start + track.saa_duration_s, stream(s, STREAM_SEFI)))
        for s in range(200)
    )
    assert 30 < fires < 70  # ~25% of 200
    assert isinstance(events, list)


def test_sefi_flavours_split_by_crash_share():
    track = OrbitTrack()
    inj = SefiInjector(track, p_per_transit=1.0, crash_share=1.0)
    events = inj.schedule(0.0, 20 * track.period_s, stream(7, STREAM_SEFI))
    assert {e.flavour for e in events} == {SEFI_CRASH}

    inj = SefiInjector(track, p_per_transit=1.0, crash_share=0.0)
    events = inj.schedule(0.0, 20 * track.period_s, stream(7, STREAM_SEFI))
    assert {e.flavour for e in events} == {SEFI_HANG}


def test_sefi_schedule_is_deterministic():
    track = OrbitTrack()
    inj = SefiInjector(track, p_per_transit=0.5)
    a = inj.schedule(0.0, 30 * track.period_s, stream(11, STREAM_SEFI))
    b = inj.schedule(0.0, 30 * track.period_s, stream(11, STREAM_SEFI))
    assert a == b


def test_sefi_crash_carries_context():
    e = SefiCrash(1234.5, 3, SEFI_CRASH)
    assert e.t_sim == 1234.5 and e.orbit == 3
    assert "SEFI" in str(e) and "orbit 3" in str(e)


def test_invalid_sefi_config_rejected():
    with pytest.raises(ValueError, match="p_per_transit"):
        SefiInjector(OrbitTrack(), p_per_transit=1.5)
    with pytest.raises(ValueError, match="crash_share"):
        SefiInjector(OrbitTrack(), crash_share=-0.1)


# --------------------------------------------------------------------- #
# Xid
# --------------------------------------------------------------------- #


def test_xid_stream_is_silent_with_ecc_off():
    """The heart of the pitch, encoded as a test.

    With ECC off, a single-bit flip produces NO driver report. The job gets
    zero warning from the hardware -- which is exactly why detection has to
    live in the application layer, and why this product exists.
    """
    sim = XidSimulator(ecc_on=False)
    assert sim.silent
    rng = stream(0, STREAM_XID)
    assert all(sim.on_flip(float(i), rng) is None for i in range(500))
    assert sim.drain() == []


def test_xid_reports_appear_with_ecc_on():
    sim = XidSimulator(ecc_on=True, report_prob=1.0)
    rng = stream(1, STREAM_XID)
    events = [sim.on_flip(float(i), rng) for i in range(50)]
    assert all(e is not None for e in events)
    assert all(e.code in XID_MEANING for e in events)


def test_multi_bit_upsets_are_fatal_single_bit_are_not():
    """SEC-DED corrects single-bit errors; multi-bit defeats it."""
    rng = stream(2, STREAM_XID)
    sim = XidSimulator(ecc_on=True, report_prob=1.0)

    single = [sim.on_flip(float(i), rng, multi_bit=False) for i in range(40)]
    assert not any(e.fatal for e in single)

    multi = [sim.on_flip(float(i), rng, multi_bit=True) for i in range(40)]
    assert all(e.fatal for e in multi)
    assert all(e.code in FATAL_XIDS for e in multi)


def test_sefi_always_reports_even_when_flips_are_silent():
    """A device that fell over cannot hide it, ECC or no ECC."""
    sim = XidSimulator(ecc_on=False)  # silent for flips
    ev = sim.on_sefi(10.0, stream(3, STREAM_XID))
    assert ev is not None and ev.fatal
    assert len(sim.drain()) == 1


def test_report_probability_is_honoured():
    sim = XidSimulator(ecc_on=True, report_prob=0.5)
    rng = stream(4, STREAM_XID)
    n = 2000
    reported = sum(sim.on_flip(float(i), rng) is not None for i in range(n))
    assert abs(reported - n * 0.5) < 4 * np.sqrt(n * 0.25)


def test_drain_is_destructive():
    sim = XidSimulator(ecc_on=True, report_prob=1.0)
    sim.on_flip(0.0, stream(5, STREAM_XID))
    assert len(sim.drain()) == 1
    assert sim.drain() == []


def test_xid_record_is_dashboard_ready():
    sim = XidSimulator(ecc_on=True, report_prob=1.0)
    rec = sim.on_flip(7.5, stream(6, STREAM_XID)).as_record()
    assert rec["t_sim"] == 7.5
    assert set(rec) >= {"xid", "detail", "fatal", "meaning"}
    assert rec["meaning"] != "unknown"


def test_invalid_report_prob_rejected():
    with pytest.raises(ValueError, match="report_prob"):
        XidSimulator(report_prob=2.0)
