"""Tests for the outcome campaign (`bench/sdc_campaign.py`).

The campaign publishes a headline number, so the harness itself has to be
held to the standard the number is quoted at. Three things must hold or the
result is worthless:

  1. The classifier maps each observation to the right bucket, including the
     awkward cases (a NaN output must be DETECTED, never MASKED and never
     SDC; two identical NaN outputs must not be called "different").
  2. Injection is exactly reverted, so trial N+1 starts from the same state
     as trial 1 and the campaign cannot drift.
  3. Uninjected trials come back masked -- the A/A control that licenses the
     bitwise golden comparison.
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import pytest
import torch

from bench.sdc_campaign import (
    ARM_DEFENDED,
    ARM_UNDEFENDED,
    CAUGHT,
    CRASH,
    MASKED,
    NONFINITE,
    REPAIRED,
    SDC,
    Campaign,
    Observable,
    Site,
    bit_field,
    bit_width,
    classify,
    classify_defended,
    format_name,
    observe,
    optimizer_sites,
    pick_site,
    rel_delta,
    summarise,
    weight_sites,
)
from orbital_runtime.inject.memory import flip_bit


def _obs(loss: float, tokens: list[int], finite: bool = True) -> Observable:
    return Observable(
        loss=loss,
        tokens=np.array(tokens, dtype=np.int32).tobytes(),
        finite=finite,
    )


# --------------------------------------------------------------------------- #
# Bit fields
# --------------------------------------------------------------------------- #


def test_bit_field_matches_ieee754_fp32_layout():
    assert [bit_field(b) for b in (0, 22)] == ["mantissa", "mantissa"]
    assert [bit_field(b) for b in (23, 30)] == ["exponent", "exponent"]
    assert bit_field(31) == "sign"


def test_every_fp32_bit_is_classified():
    fields = [bit_field(b) for b in range(32)]
    assert fields.count("mantissa") == 23
    assert fields.count("exponent") == 8
    assert fields.count("sign") == 1


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #


def test_identical_output_is_masked():
    g = _obs(1.5, [1, 2, 3])
    assert classify(g, _obs(1.5, [1, 2, 3]), None) == (MASKED, False)


def test_exception_is_crash_not_sdc():
    g = _obs(1.5, [1, 2, 3])
    outcome, critical = classify(g, None, RuntimeError("boom"))
    assert outcome == CRASH
    assert critical


def test_nonfinite_output_is_detected_never_sdc():
    """The whole taxonomy collapses if a NaN run is counted as silent."""
    g = _obs(1.5, [1, 2, 3])
    outcome, _ = classify(g, _obs(math.nan, [1, 2, 3], finite=False), None)
    assert outcome == NONFINITE
    assert outcome != SDC


def test_nonfinite_wins_over_a_changed_output():
    # A NaN run whose tokens also changed is still DETECTED. Ordering matters:
    # classifying it as SDC would move a loud failure into the silent bucket
    # and inflate the headline.
    g = _obs(1.5, [1, 2, 3])
    outcome, _ = classify(g, _obs(math.inf, [9, 9, 9], finite=False), None)
    assert outcome == NONFINITE


def test_changed_loss_alone_is_sdc_but_not_critical():
    g = _obs(1.5, [1, 2, 3])
    outcome, critical = classify(g, _obs(1.5000001, [1, 2, 3]), None)
    assert outcome == SDC
    assert not critical


def test_changed_tokens_is_critical_sdc():
    g = _obs(1.5, [1, 2, 3])
    outcome, critical = classify(g, _obs(1.5000001, [1, 7, 3]), None)
    assert outcome == SDC
    assert critical


def test_last_bit_of_the_loss_counts_as_a_difference():
    """No tolerance anywhere: one ulp is a different answer."""
    a = 1.5
    b = math.nextafter(1.5, 2.0)
    assert classify(_obs(a, [1]), _obs(b, [1]), None)[0] == SDC


def test_identical_nan_outputs_compare_equal():
    # `==` says NaN != NaN, which would call a reproducible NaN output
    # "changed". The comparison goes through the bit pattern instead.
    n = _obs(math.nan, [1, 2, 3], finite=True)
    assert n.identical_to(_obs(math.nan, [1, 2, 3], finite=True))


def test_signed_zero_is_a_difference():
    assert not _obs(0.0, [1]).identical_to(_obs(-0.0, [1]))


def test_tokens_changed_counts_positions():
    assert _obs(1.0, [1, 2, 3, 4]).tokens_changed(_obs(1.0, [1, 9, 3, 9])) == 2


# --------------------------------------------------------------------------- #
# Site selection
# --------------------------------------------------------------------------- #


def test_pick_site_is_weighted_by_element_count():
    """A 1-element tensor must not be as likely a target as a 10,000-element one.

    Uniform-over-tensors would over-weight layer-norm gains by four orders of
    magnitude and misdescribe what a particle actually hits.
    """
    small = Site("small", "weight", torch.zeros(1))
    big = Site("big", "weight", torch.zeros(9999))
    rng = np.random.default_rng(0)
    picks = [pick_site([small, big], rng)[0].name for _ in range(2000)]
    assert picks.count("small") < 10
    assert picks.count("big") > 1900


def test_pick_site_offset_is_in_range():
    sites = [
        Site("a", "weight", torch.zeros(5)),
        Site("b", "weight", torch.zeros(7)),
    ]
    rng = np.random.default_rng(1)
    for _ in range(500):
        site, index = pick_site(sites, rng)
        assert 0 <= index < site.tensor.numel()


def test_pick_site_reaches_the_last_element():
    """An off-by-one in the cumulative search would make the final element
    unstrikeable and silently shrink the target surface."""
    sites = [Site("a", "weight", torch.zeros(4))]
    rng = np.random.default_rng(2)
    seen = {pick_site(sites, rng)[1] for _ in range(200)}
    assert seen == {0, 1, 2, 3}


def test_weight_sites_counts_a_tied_matrix_once(tiny_workload):
    w = tiny_workload()
    names = [s.name for s in weight_sites(w.model)]
    assert len(names) == len(set(names))
    # nanoGPT ties lm_head.weight to wte.weight.
    assert w.model.lm_head.weight is w.model.wte.weight
    assert sum(1 for n in names if n.endswith("lm_head.weight")) == 0


def test_optimizer_sites_finds_both_adam_moments(stepped_workload):
    names = [s.name for s in optimizer_sites(stepped_workload.model, stepped_workload.optimizer)]
    assert any(n.endswith(".exp_avg") for n in names)
    assert any(n.endswith(".exp_avg_sq") for n in names)


# --------------------------------------------------------------------------- #
# Reversibility -- the property that makes trials independent
# --------------------------------------------------------------------------- #


def test_flip_is_its_own_inverse_for_every_bit():
    t = torch.tensor([0.3, -2.5, 1e-8, 7.0], dtype=torch.float32)
    original = t.clone()
    for bit in range(32):
        for idx in range(t.numel()):
            flip_bit(t, idx, bit)
            flip_bit(t, idx, bit)
    assert torch.equal(t.view(torch.int32), original.view(torch.int32))


# --------------------------------------------------------------------------- #
# rel_delta
# --------------------------------------------------------------------------- #


def test_rel_delta_reports_inf_for_a_blowup():
    assert rel_delta(1.0, math.inf) == math.inf
    assert rel_delta(1.0, math.nan) == math.inf


def test_rel_delta_of_a_nullified_value():
    assert rel_delta(4.0, 0.0) == 1.0


def test_rel_delta_from_zero_is_inf_only_when_it_moved():
    assert rel_delta(0.0, 0.0) == 0.0
    assert rel_delta(0.0, 1.0) == math.inf


# --------------------------------------------------------------------------- #
# Summary arithmetic
# --------------------------------------------------------------------------- #


def _trial(outcome, critical=False, loss_delta=0.0, bit=0):
    from bench.sdc_campaign import Trial

    return Trial(
        outcome=outcome,
        bit=bit,
        field=bit_field(bit),
        target="weight",
        site="x",
        value_before=1.0,
        value_after=2.0,
        rel_delta=1.0,
        tokens_changed=1 if critical else 0,
        loss_delta=loss_delta,
        critical=critical,
    )


def test_summarise_buckets_are_exhaustive():
    trials = [
        _trial(MASKED),
        _trial(CRASH),
        _trial(NONFINITE),
        _trial(SDC),
        _trial(SDC, critical=True),
    ]
    s = summarise(trials)
    assert s["masked"] + s["detected"] + s["sdc"] == s["trials"] == 5
    assert s["detected"] == 2
    assert s["sdc"] == 2
    assert s["sdc_critical"] == 1
    assert s["sdc_critical_pct"] == pytest.approx(20.0)


def test_summarise_magnitude_ignores_the_detected_trials():
    """Detected trials carry an infinite delta; letting them into the median
    would report `inf` and hide the quiet magnitudes entirely."""
    trials = [
        _trial(NONFINITE, loss_delta=math.inf),
        _trial(SDC, loss_delta=1e-6),
        _trial(SDC, loss_delta=1e-4),
        _trial(SDC, loss_delta=1e-2),
    ]
    s = summarise(trials)
    assert math.isfinite(s["sdc_loss_delta_median"])
    assert s["sdc_loss_delta_max"] == pytest.approx(1e-2)


def test_summarise_of_nothing_does_not_divide_by_zero():
    assert summarise([])["sdc_pct"] == 0.0


# --------------------------------------------------------------------------- #
# End to end, on a real (tiny) workload
# --------------------------------------------------------------------------- #


def _args(**kw) -> argparse.Namespace:
    base = dict(
        mode="inference",
        train_steps=3,
        warmup=5,
        eval_batches=2,
        device="cpu",
        seed=1337,
        n_layer=1,
        n_head=2,
        n_embd=32,
        batch_size=4,
        block_size=32,
        target="weight",
        arm="undefended",
        dtype="float32",
        noise_band=0,
    )
    base.update(kw)
    return argparse.Namespace(**base)


@pytest.fixture
def campaign(corpus_exists):
    return Campaign(_args())


def test_control_arm_is_all_masked(campaign):
    """The A/A control. If this fails the bitwise golden comparison is not
    licensed and no SDC number from this harness means anything."""
    assert campaign.control(5) == [MASKED] * 5


def test_a_trial_leaves_the_model_bit_identical(campaign):
    before = {k: v.clone() for k, v in campaign.model.state_dict().items()}
    rng = np.random.default_rng(7)
    for bit in (0, 15, 23, 30, 31):
        campaign.trial("weight", bit, rng)
    for k, v in campaign.model.state_dict().items():
        if v.dtype.is_floating_point:
            assert torch.equal(v.view(torch.int32), before[k].view(torch.int32)), k


def test_golden_is_stable_after_a_hundred_trials(campaign):
    """Drift is the failure mode that would quietly turn masked trials into
    SDCs as a campaign runs."""
    rng = np.random.default_rng(11)
    for _ in range(100):
        campaign.trial("weight", int(rng.integers(0, 32)), rng)
    assert observe(campaign.model, campaign.batches).identical_to(campaign.golden)


def test_low_mantissa_flips_are_mostly_masked(campaign):
    """The physical claim, at the quiet end: bit 0 moves a weight by 2^-23
    relative, which the forward pass rounds away."""
    rng = np.random.default_rng(3)
    outcomes = [campaign.trial("weight", 0, rng).outcome for _ in range(30)]
    assert outcomes.count(MASKED) > 15
    assert NONFINITE not in outcomes
    assert CRASH not in outcomes


def test_high_exponent_flips_are_never_masked(campaign):
    """And at the loud end: bit 30 multiplies a weight by ~2^128."""
    rng = np.random.default_rng(4)
    outcomes = [campaign.trial("weight", 30, rng).outcome for _ in range(20)]
    assert MASKED not in outcomes


def test_every_trial_records_the_bit_it_was_asked_for(campaign):
    rng = np.random.default_rng(5)
    for bit in range(0, 32, 7):
        t = campaign.trial("weight", bit, rng)
        assert t.bit == bit
        assert t.field == bit_field(bit)


def test_activation_injection_fires_and_is_recorded(campaign):
    rng = np.random.default_rng(6)
    t = campaign.trial("activation", 30, rng)
    assert t.site != "<never fired>"
    assert t.target == "activation"


def test_train_mode_reaches_optimizer_state(corpus_exists):
    """Adam's moments are invisible to a forward pass, so this is the only
    mode in which striking them can produce anything but `masked`."""
    camp = Campaign(_args(mode="train", train_steps=3, target="optimizer"))
    rng = np.random.default_rng(9)
    outcomes = [camp.trial("optimizer", 30, rng).outcome for _ in range(15)]
    assert set(outcomes) != {MASKED}


def test_train_mode_restores_optimizer_state(corpus_exists):
    camp = Campaign(_args(mode="train", train_steps=2, target="optimizer"))
    rng = np.random.default_rng(10)
    for _ in range(10):
        camp.trial("optimizer", int(rng.integers(0, 32)), rng)
    assert camp.control(2) == [MASKED, MASKED]


def test_gradient_injection_is_recorded(corpus_exists):
    camp = Campaign(_args(mode="train", train_steps=2, target="gradient"))
    rng = np.random.default_rng(12)
    t = camp.trial("gradient", 30, rng)
    assert t.target == "gradient"
    assert t.site != "<no grad>"


# --------------------------------------------------------------------------- #
# Float formats
# --------------------------------------------------------------------------- #


def test_bf16_keeps_all_eight_exponent_bits_in_half_the_word():
    """The commercially relevant fact: bf16 buys fp32's dynamic range by
    spending mantissa, so half its word is exponent or sign."""
    fields = [bit_field(b, torch.bfloat16) for b in range(16)]
    assert fields.count("mantissa") == 7
    assert fields.count("exponent") == 8
    assert fields.count("sign") == 1
    assert [bit_field(b, torch.float32) for b in range(32)].count("exponent") == 8


def test_fp16_trades_the_other_way():
    fields = [bit_field(b, torch.float16) for b in range(16)]
    assert fields.count("mantissa") == 10
    assert fields.count("exponent") == 5


def test_sign_bit_is_always_the_top_bit_of_the_format():
    for dt in (torch.float32, torch.bfloat16, torch.float16):
        assert bit_field(bit_width(dt) - 1, dt) == "sign"


def test_format_names():
    assert format_name(torch.float32) == "fp32"
    assert format_name(torch.bfloat16) == "bf16"


def test_bf16_campaign_runs_and_sweeps_sixteen_bits(corpus_exists):
    camp = Campaign(_args(dtype="bfloat16"))
    assert camp.width == 16
    assert camp.fmt == "bf16"
    rng = np.random.default_rng(21)
    t = camp.trial("weight", 14, rng)
    assert t.fmt == "bf16"
    assert t.field == "exponent"


# --------------------------------------------------------------------------- #
# Defended classification
# --------------------------------------------------------------------------- #


def _g():
    return _obs(1.5, [1, 2, 3])


def test_repair_requires_the_output_to_actually_come_back():
    """A 'repair' that leaves the answer wrong is not a repair."""
    assert classify_defended(_g(), _obs(1.5, [1, 2, 3]), None, True, True)[0] == REPAIRED
    assert classify_defended(_g(), _obs(9.9, [7, 7, 7]), None, True, True)[0] == CAUGHT


def test_a_tier_firing_does_not_relabel_a_loud_failure():
    """Crediting the detector for a NaN the free screen already catches would
    inflate the delta between the arms with work it did not do."""
    nan = _obs(math.nan, [1, 2, 3], finite=False)
    assert classify_defended(_g(), nan, None, True, True)[0] == NONFINITE
    assert classify_defended(_g(), None, RuntimeError("x"), True, True)[0] == CRASH


def test_silent_and_wrong_is_still_sdc_when_no_tier_fires():
    assert classify_defended(_g(), _obs(1.6, [1, 2, 3]), None, False, False)[0] == SDC


def test_masked_stays_masked_when_no_tier_fires():
    assert classify_defended(_g(), _obs(1.5, [1, 2, 3]), None, False, False)[0] == MASKED


# --------------------------------------------------------------------------- #
# The paired arms, end to end
# --------------------------------------------------------------------------- #


@pytest.fixture
def paired(corpus_exists):
    return Campaign(_args(arm="paired"))


def test_integrity_tier_repairs_single_bit_flips_in_stored_state(paired):
    """An exact integer checksum with locate-and-repair cannot miss a
    single-bit flip in a single element. If this regresses, the defended arm
    is misconfigured."""
    rng = np.random.default_rng(31)
    outcomes = [paired.trial("weight", 20, rng, ARM_DEFENDED).outcome for _ in range(12)]
    assert outcomes.count(REPAIRED) >= 10
    assert SDC not in outcomes


def test_the_scan_is_not_skipped_after_the_first_trial(paired):
    """IntegrityTier.check_now() rate-limits against the last step it scanned,
    and reset() does not clear that. Passing a constant step made every trial
    after the first skip its scan and report a false 0% detection rate."""
    rng = np.random.default_rng(32)
    first = paired.trial("weight", 20, rng, ARM_DEFENDED)
    rest = [paired.trial("weight", 20, rng, ARM_DEFENDED) for _ in range(6)]
    assert first.outcome == REPAIRED
    assert all(t.outcome == REPAIRED for t in rest)


def test_abft_sees_the_corrupted_gemm_output(paired):
    """Forward hooks fire in registration order, so the injector must attach
    before ABFT. Attaching ABFT first makes it read the clean tensor and pass
    every time, which looks like 'ABFT detects nothing'."""
    rng = np.random.default_rng(33)
    outcomes = [
        paired.trial("activation", 31, rng, ARM_DEFENDED).outcome for _ in range(10)
    ]
    assert outcomes.count(CAUGHT) >= 5


def test_undefended_arm_runs_no_detector(paired):
    """The headline arm must be genuinely undefended: no tier may fire in it."""
    rng = np.random.default_rng(34)
    trials = [paired.trial("weight", 25, rng, ARM_UNDEFENDED) for _ in range(10)]
    assert all(t.fired_tier == "" for t in trials)
    assert all(not t.repaired for t in trials)
    assert REPAIRED not in [t.outcome for t in trials]
    assert CAUGHT not in [t.outcome for t in trials]


def test_pairing_hits_the_same_site_when_the_rng_is_rewound(paired):
    """This is what makes the two arms comparable at all."""
    rng = np.random.default_rng(35)
    for bit in (5, 20, 31):
        state = rng.bit_generator.state
        a = paired.trial("weight", bit, rng, ARM_UNDEFENDED)
        rng.bit_generator.state = state
        b = paired.trial("weight", bit, rng, ARM_DEFENDED)
        assert a.site == b.site
        assert a.value_before == b.value_before


def test_defended_arm_leaves_the_model_bit_identical(paired):
    """Repair writes into the tensor, so the restore path matters more here."""
    before = {k: v.clone() for k, v in paired.model.state_dict().items()}
    rng = np.random.default_rng(36)
    for bit in (0, 20, 30, 31):
        paired.trial("weight", bit, rng, ARM_DEFENDED)
    for k, v in paired.model.state_dict().items():
        if v.dtype.is_floating_point:
            assert torch.equal(v.view(torch.int32), before[k].view(torch.int32)), k
