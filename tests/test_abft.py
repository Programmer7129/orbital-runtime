"""Tier 2: ABFT checksum verification."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from orbital_runtime.detect.abft import AbftTier, _tolerance
from orbital_runtime.inject.memory import flip_bit
from orbital_runtime.orbit.flux import FluxModel
from orbital_runtime.orbit.track import OrbitTrack


class Net(nn.Module):
    def __init__(self, bias: bool = True) -> None:
        super().__init__()
        self.fc1 = nn.Linear(32, 64, bias=bias)
        self.fc2 = nn.Linear(64, 16, bias=bias)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def armed(model, **kw) -> AbftTier:
    tier = AbftTier(model, base_sample_rate=1.0, adaptive=False, **kw).attach()
    tier.refresh_checksums()
    tier.arm()
    return tier


def check(tier: AbftTier, net, x, step: int = 0):
    """One armed forward + observe -- the sequence the real loop performs.

    `observe()` is where the tier syncs and resolves its queued checks, so
    stats are only meaningful after it. Tests go through this helper rather
    than reading stats straight after a forward, so they exercise the same
    ordering `train()` does.
    """
    tier.arm()
    net(x)
    return tier.observe(step=step)


# --------------------------------------------------------------------- #
# No false positives on healthy arithmetic
# --------------------------------------------------------------------- #


def test_clean_forward_raises_no_mismatch():
    """A tier that cried wolf every step would be worthless."""
    torch.manual_seed(0)
    net, x = Net(), torch.randn(8, 32)
    tier = armed(net)
    for i in range(20):
        assert not check(tier, net, x, step=i).triggered
    assert tier.stats.checks == 40  # 2 linears x 20 forwards
    assert tier.stats.mismatches == 0


def test_clean_residual_sits_far_below_tolerance():
    """Justifies DEFAULT_SAFETY_FACTOR rather than leaving it a magic number.

    Measures the real rounding-noise residual of a clean GEMM against the
    theoretical bound, and requires healthy headroom.
    """
    torch.manual_seed(0)
    net = Net()
    ratios = []
    for _ in range(30):
        x = torch.randn(8, 32)
        with torch.no_grad():
            y = net.fc1(x)
            s = net.fc1.weight.sum(dim=0)
            lhs = (y - net.fc1.bias).sum(dim=-1)
            rhs = x @ s
            residual = float((lhs - rhs).abs().max())
            scale = float(max(lhs.abs().max(), rhs.abs().max(), 1e-8))
            tol = _tolerance(torch.float32, 32, scale, safety=1.0)
            ratios.append(residual / tol)
    # Clean residuals must be well inside even the unsafetied bound.
    assert max(ratios) < 1.0, f"clean residual exceeds raw bound: {max(ratios):.2f}"


def test_no_false_positive_across_a_real_training_run(tiny_workload):
    """The tier must survive weights that legitimately change every step."""
    from orbital_runtime.detect import Detector, GuardTier
    from orbital_runtime.train import TrainConfig, train

    w = tiny_workload(seed=2)
    tier = AbftTier(w.model, base_sample_rate=1.0, adaptive=False).attach()
    det = Detector(guards=GuardTier(), abft=tier)
    result = train(w, cfg=TrainConfig(steps=80), detector=det)

    assert result.completed
    assert tier.stats.checks > 0
    assert tier.stats.mismatches == 0, "ABFT fired on a clean run"
    assert det.detections == 0


def test_bias_free_linear_is_handled():
    torch.manual_seed(0)
    net, x = Net(bias=False), torch.randn(8, 32)
    tier = armed(net)
    check(tier, net, x)
    assert tier.stats.mismatches == 0


# --------------------------------------------------------------------- #
# It catches what it claims to catch
# --------------------------------------------------------------------- #


def test_catches_a_corrupted_weight():
    """THE test for the trusted-snapshot ordering.

    If checksums were derived from the live (corrupted) weights, both sides
    of the identity would contain the same bad value, agree perfectly, and
    this would pass silently.
    """
    torch.manual_seed(0)
    net, x = Net(), torch.randn(8, 32)
    tier = armed(net)

    assert not check(tier, net, x).triggered  # trust established

    # Radiation lands AFTER the trusted snapshot.
    flip_bit(net.fc1.weight.data, 100, 30)  # exponent MSB

    v = check(tier, net, x, step=1)
    assert tier.stats.mismatches >= 1
    assert v.triggered and v.certain
    assert v.evidence["module"] == "fc1"
    assert v.evidence["ratio"] > 1.0


def test_a_stale_or_live_checksum_would_miss_it():
    """Demonstrates the failure mode the ordering prevents.

    Deriving `s` from the corrupted weight makes the check agree with
    itself. Encoded so that a future refactor back to lazy/`_version`
    caching fails loudly here.
    """
    torch.manual_seed(0)
    net, x = Net(), torch.randn(8, 32)
    flip_bit(net.fc1.weight.data, 100, 30)

    tier = AbftTier(net, base_sample_rate=1.0, adaptive=False).attach()
    tier.refresh_checksums()  # trust taken AFTER corruption -- the wrong order
    assert not check(tier, net, x).triggered
    assert tier.stats.mismatches == 0  # blind, exactly as predicted


def test_detects_mid_mantissa_corruption_invisible_to_the_loss():
    """The gap tier 1 cannot cover.

    A mid-mantissa strike moves the loss by far less than its own noise, so
    no z-score will ever see it -- that is the silent-divergence regime.
    ABFT compares the same computation two ways, so it sees a discrepancy
    that is tiny in absolute terms but large against rounding noise.
    """
    torch.manual_seed(0)
    net, x = Net(), torch.randn(8, 32)
    tier = armed(net)
    check(tier, net, x)

    before = float(net.fc1.weight.data.reshape(-1)[100].item())
    flip_bit(net.fc1.weight.data, 100, 15)
    after = float(net.fc1.weight.data.reshape(-1)[100].item())
    assert abs(after - before) / abs(before) < 0.01  # a 0.3% perturbation

    assert check(tier, net, x, step=1).triggered
    assert tier.stats.mismatches >= 1


@pytest.mark.parametrize("bit", [31, 30, 29, 25, 22, 20, 18, 15])
def test_detects_every_bit_down_to_the_noise_floor(bit):
    torch.manual_seed(0)
    net, x = Net(), torch.randn(8, 32)
    tier = armed(net)
    check(tier, net, x)
    flip_bit(net.fc1.weight.data, 100, bit)
    check(tier, net, x, step=1)
    assert tier.stats.mismatches >= 1, f"missed a bit-{bit} strike"


@pytest.mark.parametrize("bit", [12, 10, 8, 5, 0])
def test_low_mantissa_strikes_are_below_the_noise_floor_and_missed(bit):
    """Honesty: the sensitivity limit, asserted rather than hidden.

    Measured floor for fp32 Linear weights: ABFT catches bits >= 15
    (relative perturbation >= ~3e-3) and misses bits <= 12 (<= ~4e-4). So
    it covers 17 of 32 bit positions.

    This is not a defect to be tuned away. Below the floor the perturbation
    is indistinguishable from the GEMM's own rounding noise, and claiming to
    catch it would mean claiming to beat floating-point arithmetic. Lowering
    the threshold to reach these bits would simply trade them for false
    positives on every clean step. The saving grace is that the same tiny
    magnitude that hides them also makes them nearly harmless -- and what
    accumulation of them CAN do (silent drift) is what tier 1's z-score and
    M3's verified checkpoints are for.
    """
    torch.manual_seed(0)
    net, x = Net(), torch.randn(8, 32)
    tier = armed(net)
    check(tier, net, x)
    flip_bit(net.fc1.weight.data, 100, bit)
    check(tier, net, x, step=1)
    assert tier.stats.mismatches == 0


def test_read_only_never_perturbs_the_computation():
    """Detection must not change the number the model produces."""
    torch.manual_seed(0)
    net, x = Net(), torch.randn(8, 32)
    clean = net(x).clone()

    tier = armed(net)
    checked = net(x)
    tier.observe(step=0)
    assert torch.equal(checked, clean)
    assert tier.stats.checks > 0


def test_gradients_still_flow_with_the_tier_attached():
    torch.manual_seed(0)
    net, x = Net(), torch.randn(8, 32)
    tier = armed(net)
    net(x).sum().backward()
    tier.observe(step=0)
    assert all(p.grad is not None for p in net.parameters())
    assert tier.stats.checks > 0


# --------------------------------------------------------------------- #
# Adaptive vigilance -- the differentiator
# --------------------------------------------------------------------- #


def test_sample_rate_cranks_up_inside_the_saa():
    """Research doc SS3: position-aware protection scheduling."""
    net = Net()
    tier = AbftTier(net, base_sample_rate=0.1, saa_sample_rate=1.0, adaptive=True)

    tier.set_position(t_sim=0.0, in_saa=False)
    assert tier.sample_rate() == 0.1
    tier.set_position(t_sim=2000.0, in_saa=True)
    assert tier.sample_rate() == 1.0


def test_adaptive_can_be_disabled():
    tier = AbftTier(Net(), base_sample_rate=0.1, saa_sample_rate=1.0, adaptive=False)
    tier.set_position(t_sim=0.0, in_saa=True)
    assert tier.sample_rate() == 0.1


def test_sampling_rate_is_honoured_statistically():
    torch.manual_seed(0)
    net, x = Net(), torch.randn(8, 32)
    tier = AbftTier(
        net,
        base_sample_rate=0.25,
        adaptive=False,
        rng=np.random.default_rng(7),
    ).attach()
    tier.refresh_checksums()

    for i in range(200):
        check(tier, net, x, step=i)

    seen = tier.stats.gemms_seen
    assert seen == 400  # 2 linears x 200 forwards
    rate = tier.stats.gemms_verified / seen
    assert abs(rate - 0.25) < 4 * np.sqrt(0.25 * 0.75 / seen)


def test_adaptive_vigilance_concentrates_checks_where_the_upsets_are():
    """The economic argument, measured.

    ~90% of upsets arrive in ~10.5% of the orbit. Full sampling inside the
    SAA and 10% outside covers most of the risk for ~19% average sampling.
    """
    net = Net()
    track = OrbitTrack()
    flux = FluxModel(bits_resident=1e9, track=track)
    tier = AbftTier(
        net, flux=flux, base_sample_rate=0.1, saa_sample_rate=1.0, adaptive=True
    )

    f = track.saa_fraction
    mean_rate = f * tier.saa_sample_rate + (1 - f) * tier.base_sample_rate
    assert mean_rate == pytest.approx(0.19, abs=0.01)
    # ...while covering the SAA, where ~90% of upsets land, completely.
    assert tier.saa_sample_rate == 1.0
    assert flux.saa_share() > 0.85


def test_invalid_sample_rates_rejected():
    with pytest.raises(ValueError, match="base_sample_rate"):
        AbftTier(Net(), base_sample_rate=1.5)
    with pytest.raises(ValueError, match="saa_sample_rate"):
        AbftTier(Net(), saa_sample_rate=-0.1)


# --------------------------------------------------------------------- #
# Variance-aware tolerance (V-ABFT)
# --------------------------------------------------------------------- #


def test_tolerance_scales_with_dtype_epsilon():
    """Why a fixed threshold cannot work across formats.

    bf16's epsilon is ~4000x fp32's, so any fixed threshold is either deaf
    in bf16 or hysterical in fp32.
    """
    fp32 = _tolerance(torch.float32, 100, 1.0, 1.0)
    bf16 = _tolerance(torch.bfloat16, 100, 1.0, 1.0)
    fp16 = _tolerance(torch.float16, 100, 1.0, 1.0)
    assert bf16 > fp16 > fp32
    assert bf16 / fp32 > 1000


def test_tolerance_grows_with_reduction_length_and_magnitude():
    assert _tolerance(torch.float32, 400, 1.0, 1.0) == pytest.approx(
        2 * _tolerance(torch.float32, 100, 1.0, 1.0)
    )  # sqrt(K)
    assert _tolerance(torch.float32, 100, 10.0, 1.0) == pytest.approx(
        10 * _tolerance(torch.float32, 100, 1.0, 1.0)
    )  # linear in magnitude


def test_bf16_clean_forward_does_not_false_positive():
    """The format the variance-aware threshold exists for."""
    torch.manual_seed(0)
    net = Net().to(torch.bfloat16)
    x = torch.randn(8, 32).to(torch.bfloat16)
    tier = armed(net)
    for i in range(10):
        assert not check(tier, net, x, step=i).triggered
    assert tier.stats.checks > 0
    assert tier.stats.mismatches == 0


def test_bf16_still_catches_a_real_fault():
    """Loose tolerance must not mean blind."""
    torch.manual_seed(0)
    net = Net().to(torch.bfloat16)
    x = torch.randn(8, 32).to(torch.bfloat16)
    tier = armed(net)
    check(tier, net, x)
    flip_bit(net.fc1.weight.data, 100, 14)  # bf16 exponent MSB
    assert check(tier, net, x, step=1).triggered
    assert tier.stats.mismatches >= 1


def test_stats_report_actual_sampling():
    torch.manual_seed(0)
    net, x = Net(), torch.randn(8, 32)
    tier = armed(net)
    check(tier, net, x)
    d = tier.stats.as_dict()
    assert d["abft_gemms_seen"] == 2
    assert d["abft_sample_rate_actual"] == 1.0


# --------------------------------------------------------------------- #
# Running-error (L1) scale -- the M4c fix for the 768-dim false positives
# --------------------------------------------------------------------- #


def test_tolerance_is_keyed_to_l1_not_post_reduction_magnitude():
    """THE regression test for the catastrophic-cancellation false positive.

    A wide reduction whose terms nearly cancel has |result| << sum|term|. The
    old tolerance keyed to |result| understated the true rounding noise and
    tripped on clean steps (3/6 of the 768-dim runs, STATUS M4b). The fix keys
    the tolerance to the L1 term-magnitude, which is cancellation-invariant.

    White-box: the ratio the tier stores must match the L1-scaled formula and
    be strictly, materially below the |result|-scaled one -- so a refactor back
    to `max(|lhs|, |rhs|)` fails here loudly, not silently in production.
    """
    torch.manual_seed(0)
    k, out = 512, 64
    lin = nn.Linear(k, out, bias=False)
    with torch.no_grad():
        w = torch.randn(out, k) * 5.0
        w[0::2] += 8.0  # even output cols biased +, odd biased - => the
        w[1::2] -= 8.0  # out-dim reduction sums large ~cancelling terms
        lin.weight.copy_(w)
    x = torch.randn(4, 16, k)

    tier = AbftTier(lin, base_sample_rate=1.0, adaptive=False).attach()
    tier.refresh_checksums()
    tier.arm()
    lin(x)
    assert len(tier._pending) == 1
    stored_ratio = float(tier._pending[0][1])  # the device scalar the tier kept

    # Recompute both candidate ratios from the same arithmetic.
    with torch.no_grad():
        s = lin.weight.sum(dim=0).to(torch.float32)
        lhs = lin(x).to(torch.float32).sum(dim=-1)
        rhs = x.to(torch.float32) @ s
        residual = (lhs - rhs).abs()
        coeff = _tolerance(torch.float32, k, 1.0, tier.safety_factor)

        value_scale = torch.maximum(
            torch.maximum(lhs.abs().max(), rhs.abs().max()), torch.tensor(1e-8)
        )
        ratio_value = float((residual.max() / (value_scale * coeff)))

        l1 = torch.maximum(
            lin(x).to(torch.float32).abs().sum(dim=-1),
            x.to(torch.float32).abs() @ s.abs(),
        )
        ratio_l1 = float((residual / (torch.maximum(l1, torch.tensor(1e-8)) * coeff)).max())

    assert stored_ratio == pytest.approx(ratio_l1, rel=1e-4)
    # L1 is the looser (correct) bound: strictly below the |result| ratio, and
    # by a wide margin here (the terms cancel), which is exactly what kills the
    # false positive without touching recall.
    assert ratio_l1 < 0.5 * ratio_value

    tier.observe(step=0)
    assert tier.stats.mismatches == 0  # clean cancelling reduction: no FP
