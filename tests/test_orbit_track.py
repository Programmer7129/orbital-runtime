"""Geometry tests for the parametric LEO orbit."""

from __future__ import annotations

import numpy as np
import pytest

from orbital_runtime.orbit import OrbitTrack


def test_period_and_saa_duration_match_calibration():
    """Research doc SS2: ~10 min SAA transit per ~95-min orbit."""
    track = OrbitTrack()
    assert track.period_s == pytest.approx(95 * 60)
    assert track.saa_duration_s == pytest.approx(10 * 60)
    assert track.saa_fraction == pytest.approx(10 / 95)
    # Flight data bounds the transit at <20 min/orbit.
    assert track.saa_duration_s < 20 * 60


def test_phase_wraps_and_orbit_index_advances():
    track = OrbitTrack()
    p = track.period_s
    assert track.phase(0.0) == pytest.approx(0.0)
    assert track.phase(p / 4) == pytest.approx(0.25)
    assert track.phase(p) == pytest.approx(0.0)  # wrapped
    assert track.phase(2.5 * p) == pytest.approx(0.5)
    assert track.orbit_index(0.0) == 0
    assert track.orbit_index(p - 1) == 0
    assert track.orbit_index(p) == 1
    assert track.orbit_index(3.7 * p) == 3


def test_in_saa_matches_window_edges():
    track = OrbitTrack()
    p = track.period_s
    start = track.saa_start_phase * p
    end = start + track.saa_duration_s

    assert not track.in_saa(start - 1)
    assert track.in_saa(start)  # half-open [start, end)
    assert track.in_saa((start + end) / 2)
    assert not track.in_saa(end)

    # Repeats every orbit. Probed 1 us inside the edge rather than exactly on
    # it: saa_start_phase * period_s is not exactly representable, so an
    # exact-edge query many orbits out is ambiguous by ~1e-12 s. The SAA has
    # no nanosecond-sharp boundary, so requiring exactness there would be
    # testing float arithmetic, not the orbit model.
    us = 1e-6
    for k in (1, 5, 50):
        assert track.in_saa(start + k * p + us)
        assert track.in_saa(end + k * p - us)
        assert not track.in_saa(start + k * p - us)
        assert not track.in_saa(end + k * p + us)


def test_saa_seconds_over_n_orbits_is_exactly_n_transits():
    """The phase-gated window gives exactly the calibrated duration/orbit."""
    track = OrbitTrack()
    n = 7
    total = track.saa_seconds(0.0, n * track.period_s)
    assert total == pytest.approx(n * track.saa_duration_s)


def test_saa_windows_clip_to_interval_and_do_not_overlap():
    track = OrbitTrack()
    p = track.period_s
    windows = track.saa_windows(0.0, 3 * p)
    assert len(windows) == 3
    for lo, hi in windows:
        assert 0.0 <= lo < hi <= 3 * p
    # Strictly increasing, non-overlapping.
    for (_, hi), (lo_next, _) in zip(windows, windows[1:]):
        assert hi <= lo_next

    # A window straddling t0 is clipped, not dropped.
    start = track.saa_start_phase * p
    mid = start + track.saa_duration_s / 2
    clipped = track.saa_windows(mid, mid + 60)
    assert clipped[0][0] == pytest.approx(mid)


def test_saa_windows_empty_for_degenerate_interval():
    track = OrbitTrack()
    assert track.saa_windows(100.0, 100.0) == []
    assert track.saa_windows(100.0, 50.0) == []
    assert track.saa_seconds(100.0, 50.0) == 0.0


def test_ground_track_stays_in_bounds_and_respects_inclination():
    track = OrbitTrack()
    times = np.linspace(0, 5 * track.period_s, 2000)
    lats, lons = zip(*(track.ground_track(t) for t in times))
    assert max(abs(x) for x in lats) <= track.inclination_deg + 1e-9
    assert all(-180.0 <= x < 180.0 for x in lons)
    # Latitude peaks at the inclination a quarter-orbit in.
    lat_peak, _ = track.ground_track(track.period_s * 0.25)
    assert lat_peak == pytest.approx(track.inclination_deg)
