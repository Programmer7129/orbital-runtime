"""Integrity tier: exact checksums over the state ABFT cannot see."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from orbital_runtime.detect.abft import AbftTier
from orbital_runtime.detect.integrity import (
    REASON_INTEGRITY_MISMATCH,
    TIER_INTEGRITY,
    IntegrityTier,
    checksum,
    checksum_device,
)
from orbital_runtime.inject.memory import flip_bit


def _model_and_opt(seed: int = 0):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 8))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    # One step so Adam materialises exp_avg / exp_avg_sq. Before this the
    # optimizer state does not exist and the tier has nothing to track.
    model(torch.randn(4, 32)).sum().backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    return model, opt


# --------------------------------------------------------------------------- #
# checksum primitive
# --------------------------------------------------------------------------- #


def test_checksum_is_exact_for_every_bit():
    """A flip of bit k must move the checksum by exactly +/- 2^k."""
    t = torch.randn(256, dtype=torch.float32)
    for bit in range(32):
        before = checksum(t)
        flip_bit(t, 7, bit)
        after = checksum(t)
        assert abs(after - before) == 2**bit, f"bit {bit} moved sum by {after - before}"
        flip_bit(t, 7, bit)  # restore
        assert checksum(t) == before


def test_checksum_stable_under_no_change():
    t = torch.randn(1000)
    assert checksum(t) == checksum(t)


def test_checksum_accumulator_does_not_overflow():
    """int32 views summed in int32 would wrap; the int64 accumulator must not.

    All-ones float32 bit patterns give the largest per-element magnitude, so
    this is the worst case for the accumulator width.
    """
    t = torch.full((1_000_000,), -1.0, dtype=torch.float32)
    got = checksum(t)
    expected = int(t.view(torch.int32)[0].item()) * 1_000_000
    assert got == expected


def test_checksum_device_returns_device_tensor_without_sync():
    t = torch.randn(64)
    out = checksum_device(t)
    assert isinstance(out, torch.Tensor)
    assert out.dtype == torch.int64
    assert out.ndim == 0


def test_checksum_handles_non_contiguous():
    base = torch.randn(16, 16)
    view = base.t()  # non-contiguous
    assert not view.is_contiguous()
    checksum(view)  # must not raise


def test_checksum_rejects_unsupported_dtype():
    with pytest.raises(TypeError):
        checksum(torch.ones(4, dtype=torch.int8))


# --------------------------------------------------------------------------- #
# coverage
# --------------------------------------------------------------------------- #


def test_tracks_optimizer_state_which_is_the_bulk_of_the_target():
    """Adam state is 2x the parameter count and is the largest strikeable area."""
    model, opt = _model_and_opt()
    tier = IntegrityTier(model, optimizer=opt)
    names = [n for n, _ in tier.targets()]

    assert any(n.startswith("opt:") and n.endswith("exp_avg") for n in names)
    assert any(n.startswith("opt:") and n.endswith("exp_avg_sq") for n in names)

    param_bits = sum(p.numel() * p.element_size() * 8 for p in model.parameters())
    tier.refresh()
    # params + 2 Adam moments => at least 3x the parameter bits.
    assert tier.stats.bits_covered >= 3 * param_bits * 0.99


def test_covers_every_strikeable_bit():
    """Coverage must be total: what this tier misses, nothing else checks."""
    model, opt = _model_and_opt()
    tier = IntegrityTier(model, optimizer=opt)
    tier.refresh()
    tracked = {id(t) for _, t in tier.targets()}

    for p in model.parameters():
        assert id(p) in tracked, "a parameter is unprotected"
    for group in opt.param_groups:
        for p in group["params"]:
            for _, st in opt.state.get(p, {}).items():
                if torch.is_tensor(st) and st.dtype.is_floating_point:
                    assert id(st) in tracked, "an optimizer state tensor is unprotected"


def test_optimizer_state_appearing_late_is_picked_up():
    """Targets resolve lazily; state created after construction must be tracked."""
    torch.manual_seed(0)
    model = nn.Linear(8, 8)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    tier = IntegrityTier(model, optimizer=opt)
    tier.refresh()
    before = tier.stats.tensors_tracked

    model(torch.randn(2, 8)).sum().backward()
    opt.step()
    tier.refresh()

    assert tier.stats.tensors_tracked > before


# --------------------------------------------------------------------------- #
# detection
# --------------------------------------------------------------------------- #


def test_clean_state_never_fires():
    model, opt = _model_and_opt()
    tier = IntegrityTier(model, optimizer=opt, repair=False)
    tier.refresh()
    for step in range(20):
        assert not tier.check_now(step).triggered


def test_detects_flip_in_optimizer_state():
    """The fault class ABFT is structurally blind to."""
    model, opt = _model_and_opt()
    tier = IntegrityTier(model, optimizer=opt, repair=False)
    tier.refresh()

    p = model[0].weight
    flip_bit(opt.state[p]["exp_avg"], 3, 30)

    v = tier.check_now(1)
    assert v.triggered
    assert v.tier == TIER_INTEGRITY
    assert v.reason == REASON_INTEGRITY_MISMATCH
    assert v.certain, "an exact checksum admits no benign explanation"
    assert "exp_avg" in v.evidence["tensor"]
    assert v.evidence["bit"] == 30


def test_detects_flip_in_every_bit_position():
    """No bit is too low-order to SEE. This is the whole point of exactness.

    Detection is unconditional. Escalation is not -- see the severity tests
    below. Separating the two is deliberate: a tier that only reports what it
    intends to act on cannot be audited for what it chose to absorb.
    """
    model, opt = _model_and_opt()
    p = model[0].weight
    for bit in range(32):
        tier = IntegrityTier(model, optimizer=opt, repair=False)
        tier.refresh()
        flip_bit(opt.state[p]["exp_avg"], 5, bit)
        findings = tier.verify()
        assert len(findings) == 1, f"missed bit {bit}"
        assert findings[0]["bit"] == bit
        flip_bit(opt.state[p]["exp_avg"], 5, bit)  # restore


# --------------------------------------------------------------------------- #
# Repair: locate and invert, instead of rolling back
# --------------------------------------------------------------------------- #


def test_repairs_single_element_flip_exactly():
    """The tensor must come back bit-identical, with nothing escalated."""
    model, opt = _model_and_opt()
    target = opt.state[model[0].weight]["exp_avg"]
    tier = IntegrityTier(model, optimizer=opt)  # repair on (default)
    tier.refresh()

    pristine = target.clone()
    flip_bit(target, 11, 27)
    assert not torch.equal(target, pristine), "the flip must really have landed"

    v = tier.check_now(1)
    assert not v.triggered, "a repaired fault must not trigger a rollback"
    assert tier.stats.repaired == 1
    assert torch.equal(target, pristine), "repair was not bit-exact"


def test_repairs_every_bit_position_in_every_tracked_tensor_kind():
    model, opt = _model_and_opt()
    for tensor in (
        opt.state[model[0].weight]["exp_avg"],
        opt.state[model[2].weight]["exp_avg_sq"],
        model[0].weight,
        model[0].bias,
    ):
        for bit in (0, 7, 15, 22, 23, 30, 31):
            tier = IntegrityTier(model, optimizer=opt)
            tier.refresh()
            pristine = tensor.clone()
            flip_bit(tensor, 3, bit)
            tier.check_now(1)
            assert torch.equal(tensor, pristine), f"bit {bit} not repaired"


def test_repairs_multi_bit_upset_within_one_element():
    """An MBU cluster stays in one element, so the index is still determined."""
    model, opt = _model_and_opt()
    target = opt.state[model[0].weight]["exp_avg"]
    tier = IntegrityTier(model, optimizer=opt)
    tier.refresh()
    pristine = target.clone()
    for bit in (4, 5, 6):
        flip_bit(target, 20, bit)
    tier.check_now(1)
    assert tier.stats.repaired == 1
    assert torch.equal(target, pristine)


def test_multi_element_corruption_is_not_repaired_and_escalates():
    """Inversion is underdetermined across elements, so fall back to rollback.

    This is the nullification-tile and warp-aligned-track case. Silently
    "repairing" it would corrupt the tensor further, so the tier must decline.
    """
    model, opt = _model_and_opt()
    target = opt.state[model[0].weight]["exp_avg"]
    tier = IntegrityTier(model, optimizer=opt)
    tier.refresh()
    with torch.no_grad():
        target.reshape(-1)[5] = 0.0
        target.reshape(-1)[9] = 0.0

    v = tier.check_now(1)
    assert tier.stats.repaired == 0, "must not claim to repair a multi-element fault"
    assert v.triggered, "unrepairable corruption must escalate"


def test_repair_leaves_no_residual_for_the_next_check():
    """After a repair the baseline still holds, so the next step is clean."""
    model, opt = _model_and_opt()
    tier = IntegrityTier(model, optimizer=opt)
    tier.refresh()
    flip_bit(opt.state[model[0].weight]["exp_avg"], 2, 19)
    tier.check_now(1)
    assert not tier.check_now(2).triggered
    assert tier.stats.repaired == 1


def test_repair_survives_a_real_training_loop():
    model, opt = _model_and_opt()
    tier = IntegrityTier(model, optimizer=opt)
    tier.refresh()
    for step in range(25):
        model(torch.randn(4, 32)).sum().backward()
        if step % 5 == 0:
            flip_bit(opt.state[model[0].weight]["exp_avg"], step, 18)
        assert not tier.check_now(step).triggered, f"escalated at step {step}"
        opt.step()
        opt.zero_grad(set_to_none=True)
        tier.refresh()
    assert tier.stats.repaired == 5


# --------------------------------------------------------------------------- #
# Severity policy: detect everything, escalate what matters
# --------------------------------------------------------------------------- #


def test_low_mantissa_flip_in_optimizer_state_is_absorbed_not_escalated():
    """A 5e-7 relative change in an Adam moment must not cost a rollback.

    Adam's moments are running averages; a small perturbation decays over
    ~1/(1-beta) steps on its own. Rolling back costs a checkpoint restore plus
    replayed steps, during which more radiation lands. Escalating on these
    produced a rollback treadmill that killed a 300-step run at step 112.
    """
    model, opt = _model_and_opt()
    tier = IntegrityTier(model, optimizer=opt, repair=False)
    tier.refresh()
    flip_bit(opt.state[model[0].weight]["exp_avg"], 5, 2)

    v = tier.check_now(1)
    assert not v.triggered, "escalated on a negligible optimizer perturbation"
    assert tier.stats.benign == 1, "absorbed corruption must still be counted"
    assert tier.stats.mismatches == 1, "and must still be recorded as detected"


def test_exponent_flip_in_optimizer_state_is_escalated():
    """A sign or exponent strike is not absorbable: sqrt() and the update rule
    cannot recover from a negative or 2^128-scaled second moment."""
    model, opt = _model_and_opt()
    tier = IntegrityTier(model, optimizer=opt, repair=False)
    tier.refresh()
    flip_bit(opt.state[model[0].weight]["exp_avg_sq"], 5, 31)  # sign bit
    assert tier.check_now(1).triggered


def test_parameters_escalate_at_a_far_lower_threshold_than_optimizer_state():
    """Params compound; moments decay. The policy must reflect that asymmetry.

    Bit 15 is ~0.4% of the value: absorbed in an Adam moment, escalated in a
    weight, because the weight error is re-applied every step and rides into
    the next checkpoint.
    """
    model, opt = _model_and_opt()

    tier = IntegrityTier(model, optimizer=opt, repair=False)
    tier.refresh()
    flip_bit(opt.state[model[0].weight]["exp_avg"], 7, 15)
    assert not tier.check_now(1).triggered, "optimizer: should absorb"

    tier2 = IntegrityTier(model, optimizer=opt, repair=False)
    tier2.refresh()
    flip_bit(model[0].weight, 7, 15)
    assert tier2.check_now(1).triggered, "parameter: should escalate"


def test_datapath_fault_always_escalates():
    """Nullification overwrites the value outright; there is no small version."""
    model, opt = _model_and_opt()
    tier = IntegrityTier(model, optimizer=opt, repair=False)
    tier.refresh()
    with torch.no_grad():
        opt.state[model[0].weight]["exp_avg"].reshape(-1)[3] = 0.0
    v = tier.check_now(1)
    # A zeroed element gives a checksum delta that is not a clean power of two,
    # so `bit` is None -> treated as a datapath fault -> always severe.
    assert v.triggered


def test_detects_flip_in_non_linear_param():
    """Embeddings and norms are unprotected by ABFT too."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Embedding(50, 16), nn.LayerNorm(16))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model(torch.zeros(2, dtype=torch.long)).sum().backward()
    opt.step()

    tier = IntegrityTier(model, optimizer=opt, repair=False)
    tier.refresh()
    flip_bit(model[0].weight, 11, 20)
    assert tier.check_now(1).triggered


def test_detects_multi_bit_upset_in_one_element():
    """A cluster in one element cannot cancel: deltas are distinct powers of two."""
    model, opt = _model_and_opt()
    p = model[0].weight
    tier = IntegrityTier(model, optimizer=opt, repair=False)
    tier.refresh()
    for bit in (12, 13, 14):
        flip_bit(opt.state[p]["exp_avg"], 9, bit)
    assert tier.check_now(1).triggered


def test_reports_all_mismatching_tensors():
    model, opt = _model_and_opt()
    tier = IntegrityTier(model, optimizer=opt, repair=False)
    tier.refresh()
    flip_bit(opt.state[model[0].weight]["exp_avg"], 1, 22)
    flip_bit(opt.state[model[2].weight]["exp_avg_sq"], 1, 22)
    v = tier.check_now(1)
    assert v.evidence["n_mismatched_tensors"] == 2


def test_persistent_corruption_does_not_refire_forever():
    """Re-trust after reporting, so the NEXT distinct fault stays visible."""
    model, opt = _model_and_opt()
    tier = IntegrityTier(model, optimizer=opt, repair=False)
    tier.refresh()
    flip_bit(opt.state[model[0].weight]["exp_avg"], 2, 25)
    assert tier.check_now(1).triggered
    assert not tier.check_now(2).triggered, "same fault re-fired"


def test_abft_is_blind_to_what_this_tier_catches():
    """Regression guard on the gap that motivated this tier.

    If a future change makes ABFT cover optimizer state, this test failing is
    the signal to re-derive the coverage split rather than to delete the tier.
    """
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 8))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(4, 32)
    model(x).sum().backward()
    opt.step()
    opt.zero_grad(set_to_none=True)

    abft = AbftTier(
        model,
        rng=np.random.default_rng(0),
        base_sample_rate=1.0,
        saa_sample_rate=1.0,
        adaptive=False,
    ).attach()
    abft.refresh_checksums()
    abft.arm()

    flip_bit(opt.state[model[0].weight]["exp_avg"], 4, 29)

    model(x).sum().backward()
    abft.disarm()
    assert not abft.observe(step=1).triggered, "ABFT unexpectedly saw optimizer state"

    tier = IntegrityTier(model, optimizer=opt, repair=False)
    tier._base = {}
    tier.refresh()
    flip_bit(opt.state[model[0].weight]["exp_avg"], 4, 28)
    assert tier.check_now(2).triggered


# --------------------------------------------------------------------------- #
# lifecycle
# --------------------------------------------------------------------------- #


def test_reset_clears_snapshots():
    model, opt = _model_and_opt()
    tier = IntegrityTier(model, optimizer=opt)
    tier.refresh()
    assert tier._base
    tier.reset()
    assert not tier._base


def test_refresh_after_step_absorbs_the_legitimate_update():
    """An optimizer step changes state legitimately and must not fire."""
    model, opt = _model_and_opt()
    tier = IntegrityTier(model, optimizer=opt, repair=False)
    tier.refresh()
    for step in range(5):
        model(torch.randn(4, 32)).sum().backward()
        assert not tier.check_now(step).triggered
        opt.step()
        opt.zero_grad(set_to_none=True)
        tier.refresh()


def test_disabled_optimizer_tracking_shrinks_coverage():
    model, opt = _model_and_opt()
    full = IntegrityTier(model, optimizer=opt)
    full.refresh()
    params_only = IntegrityTier(model, optimizer=opt, track_optimizer_state=False)
    params_only.refresh()
    assert params_only.stats.bits_covered < full.stats.bits_covered


# --------------------------------------------------------------------------- #
# Loop-order regression. This is the bug the first wiring shipped with.
# --------------------------------------------------------------------------- #


def test_checking_after_optimizer_step_would_false_positive():
    """Documents WHY check_now() must precede optimizer.step().

    The optimizer legitimately rewrites params and both Adam moments. Verifying
    against a pre-step snapshot afterwards mismatches on every step. The first
    wiring did exactly this and produced 4 detections in 20 steps with zero
    upsets delivered.
    """
    model, opt = _model_and_opt()
    tier = IntegrityTier(model, optimizer=opt, repair=False)
    tier.refresh()

    model(torch.randn(4, 32)).sum().backward()
    opt.step()  # <- the wrong place to check is right here, after this

    assert tier.check_now(1).triggered, (
        "expected the post-step check to false-positive; if this stops firing, "
        "the ordering constraint has changed and the train loop must be re-derived"
    )


def test_real_loop_order_is_clean_over_many_steps():
    """backward -> check_now -> step -> refresh must never fire without a fault."""
    model, opt = _model_and_opt()
    tier = IntegrityTier(model, optimizer=opt, repair=False)
    tier.refresh()
    for step in range(30):
        model(torch.randn(4, 32)).sum().backward()
        assert not tier.check_now(step).triggered, f"false positive at step {step}"
        opt.step()
        opt.zero_grad(set_to_none=True)
        tier.refresh()


def test_observe_surfaces_the_held_verdict_exactly_once():
    model, opt = _model_and_opt()
    tier = IntegrityTier(model, optimizer=opt, repair=False)
    tier.refresh()
    flip_bit(opt.state[model[0].weight]["exp_avg"], 6, 27)

    assert tier.check_now(1).triggered
    assert tier.observe(step=1).triggered, "verdict did not reach observe()"
    assert not tier.observe(step=1).triggered, "verdict surfaced twice"


def test_sync_handles_hundreds_of_tensors():
    """Regression: combining many device scalars must not crash the backend.

    The first implementation used `torch.stack`, which lowers to `torch.cat`.
    On MPS (torch 2.13.0) `cat` binds every source buffer to one Metal compute
    encoder and overruns Metal's per-encoder resource limit at a few hundred
    inputs -- SIGSEGV, not an exception, killing the interpreter mid-run. A real
    model tracks ~208 tensors, so this is the operating point, not an edge case.

    Runs on whatever device the suite is on; the count is what matters.
    """
    from orbital_runtime.detect.integrity import _sync

    vals = [torch.tensor(i, dtype=torch.int64) for i in range(512)]
    out = _sync(vals)
    assert out == list(range(512))


def test_sync_empty_is_safe():
    from orbital_runtime.detect.integrity import _sync

    assert _sync([]) == []
