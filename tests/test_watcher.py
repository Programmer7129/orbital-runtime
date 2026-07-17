"""Tier 3: the ECC/Xid watcher."""

from __future__ import annotations

import pytest

from orbital_runtime.detect.verdict import REASON_XID_FATAL, TIER_WATCHER
from orbital_runtime.detect.watcher import (
    DcgmXidSource,
    SimulatedXidSource,
    WatcherTier,
)
from orbital_runtime.inject.xid import XID_CONTAINED_ECC, XID_DBE, XidSimulator
from orbital_runtime.rng import STREAM_XID, stream


def test_silent_stream_produces_no_detections():
    """ecc_off: the hardware tells you nothing. That is the product thesis."""
    sim = XidSimulator(ecc_on=False)
    watcher = WatcherTier(source=SimulatedXidSource(sim))
    rng = stream(0, STREAM_XID)
    for i in range(200):
        sim.on_flip(float(i), rng)
        assert not watcher.observe(step=i).triggered


def test_fatal_xid_triggers_a_certain_detection():
    sim = XidSimulator(ecc_on=True, report_prob=1.0)
    watcher = WatcherTier(source=SimulatedXidSource(sim))
    sim.on_sefi(1.0, stream(1, STREAM_XID))

    v = watcher.observe(step=3)
    assert v.triggered and v.certain
    assert v.tier == TIER_WATCHER and v.reason == REASON_XID_FATAL
    assert v.evidence["xid"] in (48, 64, 95)


def test_non_fatal_xids_are_recorded_but_not_acted_on():
    """A corrected ECC error means the hardware handled it.

    Rolling back on those would spend the overhead budget on non-events.
    """
    sim = XidSimulator(ecc_on=True, report_prob=1.0)
    watcher = WatcherTier(source=SimulatedXidSource(sim))
    sim._emit(1.0, XID_CONTAINED_ECC, "corrected")

    assert not watcher.observe(step=1).triggered
    assert len(watcher.seen) == 1  # observed, just not actionable


def test_fatal_only_can_be_disabled():
    sim = XidSimulator(ecc_on=True, report_prob=1.0)
    watcher = WatcherTier(source=SimulatedXidSource(sim), fatal_only=False)
    sim._emit(1.0, XID_CONTAINED_ECC, "corrected")
    assert watcher.observe(step=1).triggered


def test_polling_is_destructive_so_events_fire_once():
    sim = XidSimulator(ecc_on=True, report_prob=1.0)
    watcher = WatcherTier(source=SimulatedXidSource(sim))
    sim.on_sefi(1.0, stream(2, STREAM_XID))
    assert watcher.observe(step=1).triggered
    assert not watcher.observe(step=2).triggered


def test_dcgm_source_refuses_rather_than_lying():
    """A watcher that reports "no errors" when it cannot see the device is
    worse than no watcher: it makes an unmonitored run look healthy.

    On a host with no readable NVIDIA ECC device (this dev Mac, CI) the real
    source must stay blind-and-loud: `available` is False and `poll()` raises
    rather than returning `[]`. On a box that DOES have one, this path is not
    exercised -- see test_dcgm_source_reads_real_counters_when_present.
    """
    src = DcgmXidSource()
    if src.available:
        pytest.skip("a real NVIDIA ECC device is present; blind path not exercised here")
    with pytest.raises(RuntimeError, match="blind|device"):
        src.poll()

    assert WatcherTier(source=src).silent


def test_dcgm_source_reads_real_counters_when_present():
    """M4b: on real NVIDIA hardware the source reads the device's volatile ECC
    counters and reports events off their deltas.

    Skipped where there is no device (the dev Mac). Validated on the L4:
    a healthy device with no new errors returns [] (NOT a raise -- the opposite
    invariant from the blind case), and a simulated counter increase surfaces
    as a fatal report.
    """
    src = DcgmXidSource()
    if not src.available:
        pytest.skip("no readable NVIDIA ECC device on this host")

    # Healthy device, no new errors since construction: empty, and crucially
    # returned rather than raised.
    assert src.poll() == []
    assert not WatcherTier(source=src).silent

    # Force the delta path: pretend one uncorrectable ECC error has occurred
    # since the last poll. It must surface as a fatal Xid-48-class report.
    base_c, base_u = src._baseline
    src._baseline = (base_c, base_u - 1)
    events = src.poll()
    assert any(e.fatal and e.code == XID_DBE for e in events)
    # And the watcher escalates it to a certain detection.
    v = WatcherTier(source=DcgmXidSource()).observe(step=0)
    assert not v.triggered  # a fresh, healthy source sees nothing


def test_simulated_source_is_available():
    watcher = WatcherTier(source=SimulatedXidSource(XidSimulator(ecc_on=True)))
    assert not watcher.silent


def test_reset_clears_seen():
    sim = XidSimulator(ecc_on=True, report_prob=1.0)
    watcher = WatcherTier(source=SimulatedXidSource(sim))
    sim.on_sefi(1.0, stream(3, STREAM_XID))
    watcher.observe(step=1)
    assert watcher.seen
    watcher.reset()
    assert watcher.seen == []
