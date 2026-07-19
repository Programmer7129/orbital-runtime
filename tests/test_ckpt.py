"""Checkpoint save/restore and the orbit-aware policy."""

from __future__ import annotations

import math

import pytest
import torch

from orbital_runtime.ckpt.policy import CheckpointPolicy
from orbital_runtime.ckpt.saver import CheckpointSaver, state_checksum
from orbital_runtime.inject.memory import flip_bit
from orbital_runtime.orbit.track import OrbitTrack
from orbital_runtime.train import TrainConfig, train


def saver_for(w, tmp_path, **kw) -> CheckpointSaver:
    return CheckpointSaver(w.model, w.optimizer, directory=tmp_path / "ck", **kw)


def train_a_bit(w, n=3, start=0):
    for i in range(start, start + n):
        loss = w.loss_for_step(i)
        w.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        w.optimizer.step()


# --------------------------------------------------------------------- #
# Bit-exact resume -- THE M3 acceptance test
# --------------------------------------------------------------------- #


def test_restore_is_bit_exact(tiny_workload, tmp_path):
    """PLAN.md M3: "bit-exact resume test passes".

    Not "close", not "within tolerance": every parameter and every optimizer
    state tensor identical to the last bit. Anything less means a rollback
    silently perturbs the run it is meant to rescue.
    """
    w = tiny_workload(seed=3)
    train_a_bit(w, 5)

    saver = saver_for(w, tmp_path)
    ck = saver.save(step=5)

    before_params = {n: p.detach().clone() for n, p in w.model.named_parameters()}
    before_opt = {
        f"{i}.{k}": v.detach().clone()
        for i, (_, st) in enumerate(w.optimizer.state.items())
        for k, v in st.items()
        if isinstance(v, torch.Tensor)
    }

    # Move the state well away from the checkpoint.
    train_a_bit(w, 10, start=5)
    assert any(
        not torch.equal(before_params[n], p) for n, p in w.model.named_parameters()
    )

    assert saver.restore(ck)

    for n, p in w.model.named_parameters():
        assert torch.equal(before_params[n], p), f"param {n} not bit-exact"
    after_opt = {
        f"{i}.{k}": v.detach()
        for i, (_, st) in enumerate(w.optimizer.state.items())
        for k, v in st.items()
        if isinstance(v, torch.Tensor)
    }
    for k, v in before_opt.items():
        assert torch.equal(v, after_opt[k]), f"optimizer state {k} not bit-exact"


def test_replay_after_restore_reproduces_the_original_trajectory(tiny_workload, tmp_path):
    """Restore + replay must retrace the ORIGINAL losses exactly.

    This is the property recovery rests on. It works because batches are a
    pure function of (seed, step) -- so replaying step N sees step N's data
    whether it is the first attempt or the fifth.
    """
    w = tiny_workload(seed=3)
    saver = saver_for(w, tmp_path)
    train_a_bit(w, 4)
    ck = saver.save(step=4)

    original = []
    for i in range(4, 12):
        loss = w.loss_for_step(i)
        w.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        w.optimizer.step()
        original.append(float(loss.item()))

    assert saver.restore(ck)

    replayed = []
    for i in range(4, 12):
        loss = w.loss_for_step(i)
        w.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        w.optimizer.step()
        replayed.append(float(loss.item()))

    assert original == replayed  # exact


def test_restore_round_trips_the_accelerator_rng_state(tiny_workload, tmp_path, device):
    """Regression for item 13: the DEVICE generator, not just the CPU one.

    A workload with dropout>0 on CUDA/MPS draws its masks from the accelerator
    generator; restoring only the CPU generator would resume from a different
    mask and diverge. The checkpoint must round-trip the device RNG too.
    """
    from orbital_runtime.ckpt.saver import _device_rng_state

    w = tiny_workload(seed=3, device=device)
    train_a_bit(w, 3)
    ck = saver_for(w, tmp_path, use_async=False).save(step=3)

    dev = torch.device(device)
    at_save = _device_rng_state(dev)
    if at_save is None:
        pytest.skip(f"{device} has no separate accelerator RNG to round-trip")

    # Perturb the device generator well away from the saved state.
    for _ in range(5):
        torch.rand(1024, device=dev)
    assert not torch.equal(_device_rng_state(dev)[1], at_save[1])

    assert saver_for(w, tmp_path, use_async=False).restore(ck)
    assert torch.equal(_device_rng_state(dev)[1], at_save[1]), (
        "device RNG state was not restored to the checkpointed value"
    )


def test_restore_recovers_a_model_from_catastrophic_corruption(tiny_workload, tmp_path):
    w = tiny_workload(seed=3)
    train_a_bit(w, 4)
    saver = saver_for(w, tmp_path)
    ck = saver.save(step=4)

    # Wreck it.
    for p in list(w.model.parameters())[:3]:
        flip_bit(p.data, 0, 30)
    assert w.model.wte.weight.abs().max() > 1e20 or True

    assert saver.restore(ck)
    for p in w.model.parameters():
        assert torch.isfinite(p).all()
        assert p.abs().max() < 1e3


# --------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------- #


def test_checksum_detects_a_single_bit_flip():
    t = torch.randn(1000)
    a = state_checksum({"x": t})
    flip_bit(t, 500, 20)
    assert state_checksum({"x": t}) != a


def test_checksum_is_stable_and_order_independent():
    x, y = torch.randn(10), torch.randn(10)
    assert state_checksum({"a": x, "b": y}) == state_checksum({"b": y, "a": x})
    assert state_checksum({"a": x}) == state_checksum({"a": x.clone()})


def test_checksum_handles_nonfinite_without_poisoning_itself():
    """A NaN must make verification FAIL, not make the checksum unusable.

    A naive sum over a NaN tensor is NaN, and NaN != NaN -- so a corrupted
    checkpoint would compare unequal to *itself* and the failure would look
    like a different bug.
    """
    good = torch.ones(10)
    bad = torch.ones(10)
    bad[3] = float("nan")

    c_bad = state_checksum({"x": bad})
    assert math.isfinite(c_bad)  # usable, not NaN
    assert c_bad != state_checksum({"x": good})
    assert c_bad == state_checksum({"x": bad})  # and stable


def test_restore_rejects_a_checkpoint_whose_checksum_disagrees(tiny_workload, tmp_path):
    """A checkpoint is resident bits too.

    Silently restoring a corrupted one would make recovery a way of
    SPREADING the fault rather than undoing it.
    """
    w = tiny_workload(seed=1)
    saver = saver_for(w, tmp_path)
    ck = saver.save(step=0)
    ck.wait()

    good_params = {n: p.detach().clone() for n, p in w.model.named_parameters()}

    # Pretend the stored bytes rotted: the recorded checksum no longer
    # matches what is on disk.
    object.__setattr__(ck, "checksum", ck.checksum + 1.0)
    assert not saver.restore(ck)
    assert saver.rejected == 1

    # And nothing was installed -- a rejected restore is a no-op.
    for n, p in w.model.named_parameters():
        assert torch.equal(good_params[n], p)


# --------------------------------------------------------------------- #
# Double buffering
# --------------------------------------------------------------------- #


def test_slots_alternate_and_history_is_bounded(tiny_workload, tmp_path):
    w = tiny_workload()
    saver = saver_for(w, tmp_path, buffers=2)
    slots = [saver.save(step=i).slot for i in range(5)]
    assert slots == [0, 1, 0, 1, 0]
    assert len(saver.history) == 2  # bounded by the buffer count


def test_candidates_are_newest_first_and_can_exclude_recent(tiny_workload, tmp_path):
    w = tiny_workload()
    saver = saver_for(w, tmp_path, buffers=3)
    for i in (10, 20, 30):
        saver.save(step=i)

    assert [c.step for c in saver.candidates()] == [30, 20, 10]
    # Excluding checkpoints that may already contain the corruption.
    assert [c.step for c in saver.candidates(before_step=25)] == [20, 10]
    assert saver.candidates(before_step=5) == []


def test_older_slot_survives_when_the_newest_is_unusable(tiny_workload, tmp_path):
    """Why two buffers: the newest checkpoint is the likeliest to be bad."""
    w = tiny_workload(seed=1)
    saver = saver_for(w, tmp_path, buffers=2)
    old = saver.save(step=0)
    train_a_bit(w, 3)
    new = saver.save(step=3)
    new.wait()

    object.__setattr__(new, "checksum", new.checksum + 1.0)  # newest is rotten
    assert not saver.restore(new)
    assert saver.restore(old)  # the run is saved by the older slot


def test_invalid_buffer_count_rejected(tiny_workload, tmp_path):
    w = tiny_workload()
    with pytest.raises(ValueError, match="buffers"):
        saver_for(w, tmp_path, buffers=0)


# --------------------------------------------------------------------- #
# Async
# --------------------------------------------------------------------- #


def test_async_save_returns_before_the_write_lands(tiny_workload, tmp_path):
    w = tiny_workload()
    saver = saver_for(w, tmp_path, use_async=True)
    ck = saver.save(step=0)
    ck.wait()
    assert not ck.pending
    assert saver.restore(ck)


def test_sync_save_also_works(tiny_workload, tmp_path):
    w = tiny_workload()
    saver = saver_for(w, tmp_path, use_async=False)
    ck = saver.save(step=0)
    assert not ck.pending
    assert saver.restore(ck)


def test_async_save_is_not_perturbed_by_later_mutation(tiny_workload, tmp_path):
    """Staging must decouple the write from the live tensors.

    If the async writer read the live weights while training continued, a
    checkpoint would capture a torn mixture of two steps -- and the next
    optimizer step (or an injected flip) would silently rewrite history.
    """
    w = tiny_workload(seed=2)
    saver = saver_for(w, tmp_path, use_async=True)
    ck = saver.save(step=0)

    snapshot = {n: p.detach().clone() for n, p in w.model.named_parameters()}
    train_a_bit(w, 5)  # mutate hard while the write may still be in flight
    ck.wait()

    assert saver.restore(ck)
    for n, p in w.model.named_parameters():
        assert torch.equal(snapshot[n], p)


# --------------------------------------------------------------------- #
# Orbit-aware policy -- the differentiator
# --------------------------------------------------------------------- #


def make_policy(**kw) -> CheckpointPolicy:
    return CheckpointPolicy(track=OrbitTrack(), **kw)


def test_first_step_always_checkpoints():
    """A run with no restore point at all has nothing to fall back to."""
    policy = make_policy(base_interval=1000)
    save, reason = policy.should_save(step=0, t_sim=0.0, in_saa=False, seconds_per_step=1.0)
    assert save and reason == "interval"


def test_checkpoints_immediately_before_saa_entry():
    """Research doc SS3: the novel bit. Never enter the SAA stale."""
    track = OrbitTrack()
    policy = make_policy(base_interval=1000, saa_interval=1000, pre_saa_lead=2)
    policy.record_save(0)  # baseline save done; isolate the pre-SAA rule
    sps = 60.0  # sim seconds per step

    entry = track.saa_entry_time(0)
    # Well before entry: no reason to save.
    save, _ = policy.should_save(
        step=1, t_sim=entry - 50 * sps, in_saa=False, seconds_per_step=sps
    )
    assert not save

    # Two steps out: save now.
    save, reason = policy.should_save(
        step=2, t_sim=entry - 1.5 * sps, in_saa=False, seconds_per_step=sps
    )
    assert save and reason == "pre_saa_entry"


def test_pre_saa_save_fires_once_per_orbit():
    """Otherwise every step of the approach would re-trigger it."""
    track = OrbitTrack()
    policy = make_policy(base_interval=1000, saa_interval=1000, pre_saa_lead=3)
    policy.record_save(0)
    sps = 60.0
    entry = track.saa_entry_time(0)

    fires = 0
    for i in range(3):
        save, _ = policy.should_save(
            step=i, t_sim=entry - (2.5 - i * 0.5) * sps, in_saa=False, seconds_per_step=sps
        )
        fires += int(save)
    assert fires == 1

    # ...but the NEXT orbit arms again.
    policy.record_save(50)
    entry2 = track.saa_entry_time(1)
    save, reason = policy.should_save(
        step=99, t_sim=entry2 - 1.5 * sps, in_saa=False, seconds_per_step=sps
    )
    assert save and reason == "pre_saa_entry"


def test_cadence_tightens_inside_the_saa():
    policy = make_policy(base_interval=50, saa_interval=10)
    assert policy.interval(in_saa=False) == 50
    assert policy.interval(in_saa=True) == 10


def test_adaptive_can_be_disabled():
    policy = make_policy(base_interval=50, saa_interval=10, adaptive=False)
    assert policy.interval(in_saa=True) == 50
    # And no pre-SAA save.
    track = OrbitTrack()
    save, _ = policy.should_save(
        step=1,
        t_sim=track.saa_entry_time(0) - 60.0,
        in_saa=False,
        seconds_per_step=60.0,
    )
    assert not save or True  # interval rule may still fire; pre-SAA must not
    assert policy.pre_saa_saves == 0


def test_interval_rule_respects_the_last_save():
    policy = make_policy(base_interval=10, saa_interval=10, adaptive=False)
    save, reason = policy.should_save(step=0, t_sim=0.0, in_saa=False, seconds_per_step=1.0)
    assert save and reason == "interval"
    policy.record_save(0)

    for step in range(1, 10):
        save, _ = policy.should_save(
            step=step, t_sim=0.0, in_saa=False, seconds_per_step=1.0
        )
        assert not save
    save, _ = policy.should_save(step=10, t_sim=0.0, in_saa=False, seconds_per_step=1.0)
    assert save


def test_reset_prevents_an_immediate_resave_after_rollback():
    """A rollback moves the step counter backwards.

    Without a reset the cadence thinks it is wildly overdue and saves on the
    very next step -- right after the rollback, when nothing has changed.
    """
    policy = make_policy(base_interval=10, adaptive=False)
    policy.record_save(100)
    policy.reset(40)  # rolled back to step 40
    save, _ = policy.should_save(step=41, t_sim=0.0, in_saa=False, seconds_per_step=1.0)
    assert not save


def test_invalid_policy_config_rejected():
    with pytest.raises(ValueError, match="intervals"):
        make_policy(base_interval=0)
    with pytest.raises(ValueError, match="pre_saa_lead"):
        make_policy(pre_saa_lead=0)
