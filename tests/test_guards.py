"""Tier 1: the free tier."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from orbital_runtime.detect.guards import (
    DEFAULT_WARMUP_STEPS,
    GuardTier,
    grads_are_finite,
)
from orbital_runtime.detect.verdict import (
    REASON_GRAD_NORM_ZSCORE,
    REASON_LOSS_SPIKE,
    REASON_NONFINITE_GRAD,
    REASON_NONFINITE_LOSS,
    TIER_GUARD,
)


def warm(tier: GuardTier, *, loss: float = 2.0, grad: float = 1.0, n: int | None = None):
    """Feed steady healthy steps until the statistical guards are live."""
    n = n if n is not None else DEFAULT_WARMUP_STEPS + 20
    for i in range(n):
        # A little noise, or the variance estimate is degenerate.
        tier.observe(
            step=i,
            loss=loss + 0.01 * math.sin(i),
            grad_norm=grad + 0.01 * math.cos(i),
        )
    return tier


# --------------------------------------------------------------------- #
# Proof-grade checks: no warmup, no threshold
# --------------------------------------------------------------------- #


def test_nan_loss_detected_immediately_without_warmup():
    """A NaN needs no baseline to be recognised as wrong."""
    tier = GuardTier()
    v = tier.observe(step=0, loss=float("nan"), grad_norm=1.0)
    assert v.triggered and v.certain
    assert v.tier == TIER_GUARD and v.reason == REASON_NONFINITE_LOSS


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_every_nonfinite_loss_is_caught(bad):
    assert GuardTier().observe(step=0, loss=bad, grad_norm=1.0).triggered


def test_nonfinite_grad_norm_detected():
    v = GuardTier().observe(step=0, loss=2.0, grad_norm=float("inf"))
    assert v.triggered and v.certain
    assert v.reason == REASON_NONFINITE_GRAD


def test_nonfinite_samples_never_poison_the_baseline():
    """A NaN must not become part of the mean it is judged against."""
    tier = warm(GuardTier())
    mean_before = tier._loss.mean
    tier.observe(step=999, loss=float("nan"), grad_norm=1.0)
    assert tier._loss.mean == mean_before
    assert math.isfinite(tier._loss.mean)


# --------------------------------------------------------------------- #
# Statistical checks
# --------------------------------------------------------------------- #


def test_no_detection_on_a_steady_healthy_stream():
    tier = GuardTier()
    triggers = [
        tier.observe(step=i, loss=2.0 + 0.01 * math.sin(i), grad_norm=1.0 + 0.01 * math.cos(i))
        for i in range(300)
    ]
    assert not any(v.triggered for v in triggers)


def test_statistical_guards_stay_silent_during_warmup():
    """Early training is violently non-stationary.

    The loss legitimately falls from ln(65)~4.17 to ~3 in the first few
    dozen steps. A z-score over a handful of samples would read that healthy
    descent as an anomaly, so the tier refuses to judge until it has a
    baseline worth the name.
    """
    tier = GuardTier(warmup_steps=40)
    fired = False
    for i in range(40):
        # A steep, entirely healthy descent.
        v = tier.observe(step=i, loss=4.17 - i * 0.03, grad_norm=1.0 + i * 0.05)
        fired = fired or v.triggered
    assert not fired
    assert not tier.warm or tier._observed >= 40


def test_grad_norm_spike_detected_after_warmup():
    tier = warm(GuardTier())
    v = tier.observe(step=500, loss=2.0, grad_norm=50.0)
    assert v.triggered
    assert v.reason == REASON_GRAD_NORM_ZSCORE
    assert not v.certain  # an inference, not proof
    assert v.evidence["z"] > 6.0


def test_loss_spike_detected_after_warmup():
    tier = warm(GuardTier())
    v = tier.observe(step=500, loss=40.0, grad_norm=1.0)
    assert v.triggered
    assert v.reason == REASON_LOSS_SPIKE
    assert not v.certain


def test_guards_are_one_sided():
    """A loss that drops sharply is the model learning, not a fault."""
    tier = warm(GuardTier())
    assert not tier.observe(step=500, loss=0.001, grad_norm=1.0).triggered
    assert not tier.observe(step=501, loss=2.0, grad_norm=0.0001).triggered


def test_a_sample_is_judged_before_it_joins_the_baseline():
    """Otherwise a spike partially hides inside its own baseline."""
    tier = warm(GuardTier())
    mean_before = tier._loss.mean
    v = tier.observe(step=500, loss=40.0, grad_norm=1.0)
    assert v.triggered
    assert v.evidence["baseline_mean"] == pytest.approx(mean_before, abs=1e-6)


def test_reset_clears_history():
    tier = warm(GuardTier())
    assert tier.warm
    tier.reset()
    assert not tier.warm
    assert tier._observed == 0
    # And a spike no longer fires, because there is no baseline to spike against.
    assert not tier.observe(step=0, loss=40.0, grad_norm=1.0).triggered


def test_zero_variance_baseline_does_not_divide_by_zero():
    """A perfectly flat loss must not make every step infinitely anomalous."""
    tier = GuardTier(warmup_steps=5)
    for i in range(50):
        assert not tier.observe(step=i, loss=2.0, grad_norm=1.0).triggered
    # Still no explosion when something finally differs.
    v = tier.observe(step=51, loss=2.0000001, grad_norm=1.0)
    assert not v.triggered


def test_ewma_tracks_a_drifting_healthy_baseline():
    """A slowly-falling loss is normal training and must not accumulate z."""
    tier = GuardTier(warmup_steps=20)
    fired = False
    loss = 4.0
    for i in range(400):
        loss *= 0.995  # steady healthy decay
        v = tier.observe(step=i, loss=loss + 0.005 * math.sin(i), grad_norm=1.0)
        fired = fired or v.triggered
    assert not fired


# --------------------------------------------------------------------- #
# grads_are_finite
# --------------------------------------------------------------------- #


def test_grads_are_finite_helper():
    net = nn.Linear(4, 4)
    net(torch.randn(2, 4)).sum().backward()
    assert grads_are_finite(net)
    net.weight.grad[0, 0] = float("nan")
    assert not grads_are_finite(net)


def test_grads_are_finite_with_no_grads():
    assert grads_are_finite(nn.Linear(4, 4))
