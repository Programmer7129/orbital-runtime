"""The composite detector, and the evaluation harness's oracle."""

from __future__ import annotations

import pytest

from bench.detect_eval import evaluate_seed, first_divergence, summarise
from orbital_runtime.detect import Detector, GuardTier
from orbital_runtime.detect.abft import AbftTier
from orbital_runtime.detect.watcher import SimulatedXidSource, WatcherTier
from orbital_runtime.detect.verdict import (
    REASON_ABFT_MISMATCH,
    REASON_NONFINITE_LOSS,
    TIER_ABFT,
    TIER_GUARD,
)
from orbital_runtime.inject.injector import RadiationEnvironment
from orbital_runtime.inject.memory import MemoryInjector, flip_bit
from orbital_runtime.inject.xid import XidSimulator
from orbital_runtime.orbit.flux import FluxModel
from orbital_runtime.rng import STREAM_XID, stream
from orbital_runtime.train import TrainConfig, train

TINY_KW = dict(n_layer=1, n_head=2, n_embd=32, block_size=32, batch_size=8)


def same_losses(a: list[float], b: list[float]) -> bool:
    """Exact equality, but NaN-aware.

    A plain `a == b` is wrong for any run that died: NaN != NaN, so two
    bit-identical loss curves that both end in NaN would compare unequal
    and the test would "fail" on a working system. Determinism still means
    EXACT equality everywhere else -- no tolerance is used here.
    """
    import math

    if len(a) != len(b):
        return False
    return all(
        (math.isnan(x) and math.isnan(y)) or x == y for x, y in zip(a, b)
    )


# --------------------------------------------------------------------- #
# Composite behaviour
# --------------------------------------------------------------------- #


def test_empty_detector_never_fires(tiny_workload):
    det = Detector()
    assert not det.observe(step=0, loss=2.0, grad_norm=1.0).triggered
    assert det.detections == 0


def test_cheapest_tier_wins_the_report(tiny_workload):
    """A visibly NaN run must not pay for a checksum to confirm it."""
    w = tiny_workload()
    abft = AbftTier(w.model, base_sample_rate=1.0, adaptive=False).attach()
    det = Detector(guards=GuardTier(), abft=abft)

    v = det.observe(step=0, loss=float("nan"), grad_norm=1.0, model=w.model)
    assert v.triggered
    assert v.tier == TIER_GUARD and v.reason == REASON_NONFINITE_LOSS


def test_abft_reports_when_guards_see_nothing(tiny_workload):
    """The silent-corruption case: the loss looks fine, the weights do not."""
    w = tiny_workload()
    abft = AbftTier(w.model, base_sample_rate=1.0, adaptive=False).attach()
    det = Detector(guards=GuardTier(), abft=abft)

    det.before_step(t_sim=0.0, in_saa=False)
    w.loss_for_step(0)
    abft.refresh_checksums()

    # Corrupt a weight after the trusted snapshot.
    linear = w.model.blocks[0].mlp.c_fc
    flip_bit(linear.weight.data, 10, 30)

    det.before_step(t_sim=1.0, in_saa=False)
    w.loss_for_step(1)
    v = det.observe(step=1, loss=2.0, grad_norm=1.0, model=w.model)

    assert v.triggered
    assert v.tier == TIER_ABFT and v.reason == REASON_ABFT_MISMATCH
    assert det.per_tier[TIER_ABFT] == 1


def test_detections_and_history_accumulate(tiny_workload):
    det = Detector(guards=GuardTier())
    for i in range(3):
        det.observe(step=i, loss=float("nan"), grad_norm=1.0)
    assert det.detections == 3
    assert len(det.history) == 3
    assert det.per_tier[TIER_GUARD] == 3


def test_reset_clears_every_tier(tiny_workload):
    w = tiny_workload()
    abft = AbftTier(w.model, base_sample_rate=1.0, adaptive=False).attach()
    watcher = WatcherTier(source=SimulatedXidSource(XidSimulator(ecc_on=True, report_prob=1.0)))
    guards = GuardTier()
    det = Detector(guards=guards, abft=abft, watcher=watcher)

    for i in range(60):
        det.observe(step=i, loss=2.0, grad_norm=1.0)
    assert guards.warm

    det.reset()
    assert not guards.warm
    assert abft._trusted == {}
    assert watcher.seen == []


def test_stats_include_abft_sampling(tiny_workload):
    w = tiny_workload()
    abft = AbftTier(w.model, base_sample_rate=1.0, adaptive=False).attach()
    det = Detector(guards=GuardTier(), abft=abft)
    det.before_step(t_sim=0.0, in_saa=False)
    w.loss_for_step(0)
    det.observe(step=0, loss=2.0, grad_norm=1.0)

    s = det.stats()
    assert "detections" in s and "abft_gemms_seen" in s
    assert s["abft_gemms_seen"] > 0


def test_before_step_propagates_orbital_position(tiny_workload):
    w = tiny_workload()
    abft = AbftTier(w.model, base_sample_rate=0.1, saa_sample_rate=1.0, adaptive=True)
    det = Detector(abft=abft)

    det.before_step(t_sim=100.0, in_saa=True)
    assert abft.sample_rate() == 1.0
    det.before_step(t_sim=200.0, in_saa=False)
    assert abft.sample_rate() == 0.1


# --------------------------------------------------------------------- #
# Detection must not perturb the run it is measuring
# --------------------------------------------------------------------- #


def test_detection_does_not_change_the_run(tiny_workload):
    """Load-bearing for every number in bench/.

    If attaching a detector altered the trajectory, the overhead benchmark
    would be comparing two different computations and the precision/recall
    oracle would be comparing a run against a clean run it no longer
    corresponds to.
    """
    w1 = tiny_workload(seed=4)
    plain = train(w1, cfg=TrainConfig(steps=60))

    w2 = tiny_workload(seed=4)
    abft = AbftTier(w2.model, base_sample_rate=1.0, adaptive=False).attach()
    watched = train(
        w2, cfg=TrainConfig(steps=60), detector=Detector(guards=GuardTier(), abft=abft)
    )

    assert plain.losses == watched.losses  # exact


def test_detection_does_not_change_an_irradiated_run(tiny_workload):
    def run(with_detector: bool):
        w = tiny_workload(seed=1)
        bits = MemoryInjector(w.model, w.optimizer).static_resident_bits()
        flux = FluxModel(bits_resident=bits, base_rate_upsets_per_bit_day=5e-4)
        env = RadiationEnvironment(
            w.model, w.optimizer, flux=flux, seed=1, n_steps=80, orbits=2.0
        )
        det = None
        if with_detector:
            abft = AbftTier(w.model, base_sample_rate=1.0, adaptive=False).attach()
            det = Detector(guards=GuardTier(), abft=abft)
        return train(w, cfg=TrainConfig(steps=80), env=env, detector=det).losses

    assert same_losses(run(False), run(True))


# --------------------------------------------------------------------- #
# The evaluation oracle
# --------------------------------------------------------------------- #


def test_first_divergence_is_exact():
    assert first_divergence([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) is None
    assert first_divergence([1.0, 2.0, 3.0], [1.0, 2.5, 3.0]) == 1
    assert first_divergence([1.0, 2.0], [1.0, 2.0, 3.0]) == 2  # died early
    # Even the last representable bit counts as divergence: determinism
    # means clean runs match exactly, so any difference is a real fault.
    assert first_divergence([1.0], [1.0 + 2**-52]) == 0
    # NaN-aware: two identically-dead curves have not "diverged" at the NaN.
    nan = float("nan")
    assert first_divergence([1.0, nan], [1.0, nan]) is None
    assert first_divergence([1.0, 2.0], [1.0, nan]) == 1


def test_oracle_finds_no_corruption_in_a_clean_run():
    out = evaluate_seed(
        seed=1,
        rate=0.0,
        steps=40,
        orbits=2.0,
        device="cpu",
        tiers="guards+abft",
        workload_kw=TINY_KW,
    )
    assert not out.corrupted
    assert out.corruption_step is None
    assert not out.detected  # and no false positive


def test_oracle_finds_corruption_in_an_irradiated_run():
    out = evaluate_seed(
        seed=1,
        rate=5e-4,
        steps=120,
        orbits=2.0,
        device="cpu",
        tiers="guards+abft",
        workload_kw=TINY_KW,
    )
    assert out.corrupted
    assert out.corruption_step is not None
    assert out.flips > 0
    assert out.detected
    assert out.latency is not None and out.latency >= 0


def test_abft_detects_sooner_than_guards_alone():
    """The measured case for tier 2, as an assertion.

    Guards can only see corruption once it has grown enough to move the
    loss. ABFT sees the corrupted weight on the next forward pass. Lower
    latency is not cosmetic: it is exactly the number of steps M3 must
    replay after a rollback.
    """
    kw = dict(seed=1, rate=5e-4, steps=120, orbits=2.0, device="cpu", workload_kw=TINY_KW)
    guards_only = evaluate_seed(tiers="guards", **kw)
    with_abft = evaluate_seed(tiers="guards+abft", **kw)

    assert guards_only.corrupted and with_abft.corrupted
    # Same run, same corruption -- only the detector differs.
    assert guards_only.corruption_step == with_abft.corruption_step
    assert with_abft.latency is not None
    assert guards_only.latency is not None
    assert with_abft.latency <= guards_only.latency


def test_summarise_computes_precision_and_recall():
    from bench.detect_eval import RunOutcome

    outcomes = [
        RunOutcome(1, True, 10, True, 12, 2, False, 5),  # TP
        RunOutcome(2, True, 10, False, None, None, False, 5),  # FN
        RunOutcome(3, False, None, True, 4, None, False, 0),  # FP
        RunOutcome(4, False, None, False, None, None, False, 0),  # TN
    ]
    s = summarise(outcomes, "x")
    assert (s["tp"], s["fn"], s["fp"], s["tn"]) == (1, 1, 1, 1)
    assert s["precision"] == pytest.approx(0.5)
    assert s["recall"] == pytest.approx(0.5)
    assert s["median_latency_steps"] == 2
