"""M3 acceptance: detect -> restore -> replay, end to end."""

from __future__ import annotations

import math

import pytest

from orbital_runtime.ckpt.policy import CheckpointPolicy
from orbital_runtime.ckpt.recover import (
    LAG_LOCALISED,
    LAG_UNLOCALISED,
    RecoveryExhausted,
    RecoveryOrchestrator,
)
from orbital_runtime.ckpt.saver import CheckpointSaver
from orbital_runtime.detect import Detector, GuardTier
from orbital_runtime.detect.abft import AbftTier
from orbital_runtime.detect.verdict import (
    REASON_ABFT_MISMATCH,
    REASON_LOSS_SPIKE,
    REASON_NONFINITE_LOSS,
    REASON_XID_FATAL,
    TIER_ABFT,
    TIER_GUARD,
    Verdict,
)
from orbital_runtime.inject.injector import RadiationEnvironment
from orbital_runtime.inject.memory import MemoryInjector
from orbital_runtime.orbit.flux import FluxModel
from orbital_runtime.orbit.track import OrbitTrack
from orbital_runtime.rng import STREAM_ABFT, stream
from orbital_runtime.train import TrainConfig, train

RATE = 5e-4


def build_protected(w, tmp_path, *, seed: int, steps: int, orbits: float = 2.0, rate=RATE):
    """The shipped protected configuration, as `--protect on` builds it.

    The environment exists even at rate=0: it is the orbital CLOCK, not just
    the radiation source, and the checkpoint policy needs it to know where
    in the orbit it is.
    """
    bits = MemoryInjector(w.model, w.optimizer).static_resident_bits()
    flux = FluxModel(
        bits_resident=bits, track=OrbitTrack(), base_rate_upsets_per_bit_day=rate
    )
    env = RadiationEnvironment(
        w.model, w.optimizer, flux=flux, seed=seed, n_steps=steps, orbits=orbits
    )

    abft = AbftTier(
        w.model,
        base_sample_rate=0.1,
        saa_sample_rate=1.0,
        adaptive=True,
        rng=stream(seed, STREAM_ABFT),
    ).attach()
    detector = Detector(guards=GuardTier(), abft=abft)
    recovery = RecoveryOrchestrator(
        saver=CheckpointSaver(
            w.model, w.optimizer, directory=tmp_path / f"ck{seed}", use_async=False
        ),
        policy=CheckpointPolicy(track=OrbitTrack(), base_interval=25, saa_interval=8),
        env=env,
        detector=detector,
    )
    return env, detector, recovery


# --------------------------------------------------------------------- #
# THE M3 acceptance test
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", [1, 2, 5])
def test_protected_run_survives_what_kills_the_unprotected_one(
    tiny_workload, tmp_path, seed
):
    """PLAN.md M3: "end-to-end protected run completes a multi-orbit session
    with injected faults".

    Same seed, same workload, same bit-identical bombardment. The only
    difference is protection.
    """
    # --- unprotected: dies ---
    w_off = tiny_workload(seed=seed)
    bits = MemoryInjector(w_off.model, w_off.optimizer).static_resident_bits()
    flux = FluxModel(
        bits_resident=bits, track=OrbitTrack(), base_rate_upsets_per_bit_day=RATE
    )
    env_off = RadiationEnvironment(
        w_off.model, w_off.optimizer, flux=flux, seed=seed, n_steps=120, orbits=2.0
    )
    off = train(w_off, cfg=TrainConfig(steps=120), env=env_off)
    assert off.died, f"seed {seed}: unprotected run was supposed to die"

    # --- protected: survives ---
    w_on = tiny_workload(seed=seed)
    env_on, detector, recovery = build_protected(w_on, tmp_path, seed=seed, steps=120)
    on = train(
        w_on, cfg=TrainConfig(steps=120), env=env_on, detector=detector, recovery=recovery
    )

    assert on.completed, f"seed {seed}: protected run died ({on.death_reason})"
    assert on.steps_completed == 120
    assert math.isfinite(on.final_loss) and math.isfinite(on.final_val_loss)
    assert on.injected > 0  # it really was irradiated
    assert on.detected > 0 and on.recovered > 0
    assert not env_on.schedule_exhausted  # and never flew out of its radiation


def test_protected_run_trains_a_healthy_model(tiny_workload, tmp_path):
    """Surviving is not enough -- the model has to be worth keeping.

    A run that "completes" with a wrecked model has recovered nothing.
    """
    clean = train(tiny_workload(seed=1), cfg=TrainConfig(steps=120))

    w = tiny_workload(seed=1)
    env, detector, recovery = build_protected(w, tmp_path, seed=1, steps=120)
    on = train(w, cfg=TrainConfig(steps=120), env=env, detector=detector, recovery=recovery)

    assert on.completed
    # Within striking distance of a run that was never irradiated at all.
    assert on.final_val_loss < clean.final_val_loss + 0.5, (
        f"protected val {on.final_val_loss:.3f} vs clean {clean.final_val_loss:.3f}"
    )


def test_replay_cost_is_accounted_not_hidden(tiny_workload, tmp_path):
    """Protection is not free, and the result must say so.

    steps_completed is training progress; steps_executed is work done. A
    rollback makes them diverge, and conflating them would let the protected
    run take credit for redoing work it had already done.
    """
    w = tiny_workload(seed=1)
    env, detector, recovery = build_protected(w, tmp_path, seed=1, steps=120)
    on = train(w, cfg=TrainConfig(steps=120), env=env, detector=detector, recovery=recovery)

    assert on.recovered > 0
    assert on.steps_executed > on.steps_completed
    assert on.replayed_steps == on.steps_executed - on.steps_completed
    assert on.replayed_steps > 0
    assert recovery.stats.replayed_steps > 0


def test_replay_costs_orbit_time(tiny_workload, tmp_path):
    """The satellite keeps flying while the job redoes work.

    A protected run that replays must experience MORE radiation than the
    nominal mission holds -- if it did not, replay would be a way of dodging
    physics.
    """
    w = tiny_workload(seed=1)
    env, detector, recovery = build_protected(w, tmp_path, seed=1, steps=120)
    train(w, cfg=TrainConfig(steps=120), env=env, detector=detector, recovery=recovery)

    assert recovery.stats.rollbacks > 0
    assert env.executed > 120
    assert env.now > env.duration_s  # flew past the nominal mission
    assert env.stats.flips > env.scheduled_within_mission


# --------------------------------------------------------------------- #
# Choosing the rollback target
# --------------------------------------------------------------------- #


def test_rollback_margin_depends_on_which_tier_spoke():
    """M2's latency measurement, turned into M3 behaviour.

    ABFT localises a fault to one step, so it can roll back to the newest
    checkpoint. A NaN could have been brewing for many steps, so it must
    rewind further. Using the pessimistic margin for both throws away good
    checkpoints and replays work needlessly.
    """
    rec = RecoveryOrchestrator(saver=None, policy=None)  # type: ignore[arg-type]
    abft = Verdict(True, 50, TIER_ABFT, REASON_ABFT_MISMATCH)
    nan = Verdict(True, 50, TIER_GUARD, REASON_NONFINITE_LOSS)
    spike = Verdict(True, 50, TIER_GUARD, REASON_LOSS_SPIKE)
    xid = Verdict(True, 50, "watcher", REASON_XID_FATAL)

    assert rec.lag_for(abft) == LAG_LOCALISED
    assert rec.lag_for(xid) == LAG_LOCALISED
    assert rec.lag_for(nan) == LAG_UNLOCALISED
    assert rec.lag_for(spike) == LAG_UNLOCALISED
    assert LAG_LOCALISED < LAG_UNLOCALISED


def test_abft_detection_rolls_back_shallower_than_a_nan(tiny_workload, tmp_path):
    """Low latency buys shallow rollbacks -- the point of paying for ABFT."""
    w = tiny_workload(seed=1)
    saver = CheckpointSaver(w.model, w.optimizer, directory=tmp_path / "ck", use_async=False)
    for s in (10, 40):
        saver.save(step=s)

    rec = RecoveryOrchestrator(
        saver=saver, policy=CheckpointPolicy(track=OrbitTrack()), detector=None
    )
    # ABFT at step 45 -> newest checkpoint (40) is provably safe.
    assert rec.on_detection(step=45, verdict=Verdict(True, 45, TIER_ABFT, REASON_ABFT_MISMATCH)) == 40

    # A NaN at 45 must rewind past the 25-step margin -> back to 10.
    assert rec.on_detection(step=45, verdict=Verdict(True, 45, TIER_GUARD, REASON_NONFINITE_LOSS)) == 10


def test_rollback_never_targets_a_checkpoint_inside_the_suspect_window(
    tiny_workload, tmp_path
):
    """Restoring a checkpoint that already contains the fault would make
    recovery a way of PRESERVING corruption, and the run would loop."""
    w = tiny_workload(seed=1)
    saver = CheckpointSaver(w.model, w.optimizer, directory=tmp_path / "ck", use_async=False)
    saver.save(step=10)
    saver.save(step=44)  # inside the 25-step suspect window of a NaN at 45

    rec = RecoveryOrchestrator(
        saver=saver, policy=CheckpointPolicy(track=OrbitTrack()), detector=None
    )
    got = rec.on_detection(step=45, verdict=Verdict(True, 45, TIER_GUARD, REASON_NONFINITE_LOSS))
    assert got == 10  # not 44


def test_recovery_resets_the_detector(tiny_workload, tmp_path):
    """Post-rollback, the baselines describe a state that no longer exists,
    and ABFT's trust anchor points at pre-rollback weights."""
    w = tiny_workload(seed=1)
    saver = CheckpointSaver(w.model, w.optimizer, directory=tmp_path / "ck", use_async=False)
    saver.save(step=0)

    guards = GuardTier()
    abft = AbftTier(w.model, base_sample_rate=1.0, adaptive=False).attach()
    detector = Detector(guards=guards, abft=abft)
    for i in range(60):
        detector.observe(step=i, loss=2.0, grad_norm=1.0)
    abft.refresh_checksums()
    assert guards.warm and abft._trusted

    rec = RecoveryOrchestrator(
        saver=saver, policy=CheckpointPolicy(track=OrbitTrack()), detector=detector
    )
    rec.on_detection(step=1, verdict=Verdict(True, 1, TIER_ABFT, REASON_ABFT_MISMATCH))

    assert not guards.warm
    assert abft._trusted == {}


def test_unrecoverable_run_says_so_rather_than_looping(tiny_workload, tmp_path):
    """A run that cannot recover is a run to report, not to retry forever."""
    w = tiny_workload(seed=1)
    saver = CheckpointSaver(w.model, w.optimizer, directory=tmp_path / "ck", use_async=False)
    rec = RecoveryOrchestrator(
        saver=saver, policy=CheckpointPolicy(track=OrbitTrack()), detector=None
    )
    # No checkpoints at all.
    with pytest.raises(RecoveryExhausted, match="no verified checkpoint"):
        rec.on_detection(step=5, verdict=Verdict(True, 5, TIER_GUARD, REASON_NONFINITE_LOSS))


def test_best_effort_rollbacks_are_counted_separately(tiny_workload, tmp_path):
    """A recovery made on a guess must not be reported as a proven one."""
    w = tiny_workload(seed=1)
    saver = CheckpointSaver(w.model, w.optimizer, directory=tmp_path / "ck", use_async=False)
    saver.save(step=40)  # only checkpoint, inside the suspect window

    rec = RecoveryOrchestrator(
        saver=saver, policy=CheckpointPolicy(track=OrbitTrack()), detector=None
    )
    got = rec.on_detection(step=45, verdict=Verdict(True, 45, TIER_GUARD, REASON_NONFINITE_LOSS))
    assert got == 40
    assert rec.stats.best_effort_rollbacks == 1
    assert rec.stats.rollbacks == 1


def test_train_reports_an_unrecoverable_run_honestly(tiny_workload, tmp_path):
    """The loop must surface RecoveryExhausted, not crash on it."""
    from orbital_runtime.train import DEATH_UNRECOVERABLE

    w = tiny_workload(seed=5)
    bits = MemoryInjector(w.model, w.optimizer).static_resident_bits()
    flux = FluxModel(bits_resident=bits, base_rate_upsets_per_bit_day=5e-3)  # brutal
    env = RadiationEnvironment(
        w.model, w.optimizer, flux=flux, seed=5, n_steps=60, orbits=2.0
    )
    detector = Detector(guards=GuardTier())
    recovery = RecoveryOrchestrator(
        saver=CheckpointSaver(
            w.model, w.optimizer, directory=tmp_path / "ck", use_async=False
        ),
        # Never checkpoints: nothing to fall back to.
        policy=CheckpointPolicy(track=OrbitTrack(), base_interval=10**6, adaptive=False),
        env=env,
        detector=detector,
    )
    recovery.policy.record_save(0)  # suppress even the initial save

    result = train(
        w, cfg=TrainConfig(steps=60), env=env, detector=detector, recovery=recovery
    )
    assert result.died
    assert result.death_reason == DEATH_UNRECOVERABLE


# --------------------------------------------------------------------- #
# Checkpoint cadence in a live run
# --------------------------------------------------------------------- #


def test_checkpoints_land_before_saa_entry_in_a_real_run(tiny_workload, tmp_path):
    """Adaptive vigilance, end to end rather than in a unit test."""
    w = tiny_workload(seed=4)
    env, detector, recovery = build_protected(w, tmp_path, seed=4, steps=120, rate=0.0)
    # rate=0 so the cadence is observed without rollbacks perturbing it.
    train(w, cfg=TrainConfig(steps=120), env=env, detector=detector, recovery=recovery)

    assert recovery.policy.pre_saa_saves >= 1
    assert recovery.saver.saves >= 2


def test_protected_run_works_on_every_device(tiny_workload, tmp_path, device):
    """PLAN.md design rule 1, on the path that broke it.

    The checkpoint tests were CPU-only, and the checksum accumulates in
    float64 -- which MPS does not support. `make demo` defaults to
    --device auto, so the whole protected demo raised on the very machine it
    is developed on. Staging to CPU fixes it; this test is why it stays
    fixed.
    """
    w = tiny_workload(seed=1, device=device)
    env, detector, recovery = build_protected(w, tmp_path / device, seed=1, steps=40)
    result = train(
        w, cfg=TrainConfig(steps=40), env=env, detector=detector, recovery=recovery
    )
    assert result.steps_executed > 0
    assert recovery.saver.saves > 0  # it really did checkpoint


def test_protection_is_deterministic(tiny_workload, tmp_path):
    """A demo that never flakes on stage (design rule 3)."""

    def run(tag: int):
        w = tiny_workload(seed=1)
        env, detector, recovery = build_protected(
            w, tmp_path / f"r{tag}", seed=1, steps=100
        )
        r = train(
            w, cfg=TrainConfig(steps=100), env=env, detector=detector, recovery=recovery
        )
        return r.steps_executed, r.recovered, r.detected, round(r.final_loss, 10)

    assert run(1) == run(2)
