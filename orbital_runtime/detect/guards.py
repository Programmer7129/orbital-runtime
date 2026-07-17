"""Tier 1 -- the free tier (~0% overhead).

Research doc SS3: "`torch.isfinite` on loss/grads per step; gradient-norm
z-score + loss-spike detection (AWS 'SDC in LLM training', arXiv 2502.12340:
real SDCs cause loss spikes AND silent convergence drift)."

Free because every quantity here is already computed by the training loop:
the loss is the loss, and the gradient norm is what `clip_grad_norm_`
returns anyway. The tier adds arithmetic on two scalars per step, not a
pass over any tensor. Nothing here touches the model.

Two qualitatively different signals live in this tier:

* **isfinite** -- proof. A non-finite loss cannot happen in a healthy run of
  this workload, so a trigger is certain and needs no threshold.
* **z-score / loss-spike** -- inference. Healthy training is noisy and
  occasionally spikes on its own, so these trade precision for the ability
  to catch corruption BEFORE it becomes a NaN. Their thresholds are the
  only tunable knobs in the tier, and `bench/detect_eval.py` measures what
  they cost in false positives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .verdict import (
    NO_DETECTION,
    REASON_GRAD_NORM_ZSCORE,
    REASON_LOSS_SPIKE,
    REASON_NONFINITE_GRAD,
    REASON_NONFINITE_LOSS,
    TIER_GUARD,
    Verdict,
)

# Steps of history required before the statistical guards may fire.
# Early training is violently non-stationary -- the loss drops from ln(65)
# to ~3 within a few dozen steps -- so a z-score computed over 5 samples
# would flag the healthy initial descent as corruption.
DEFAULT_WARMUP_STEPS = 40

# EWMA horizon for the running mean/variance of the monitored scalars.
# Long enough to be stable, short enough to track the genuine downward drift
# of a healthy loss curve.
DEFAULT_EWMA_ALPHA = 0.05

# Gradient-norm z-score above which we call it. 6 sigma is deliberately
# conservative: the free tier runs every step of every run, so a
# false-positive rate that looks tiny per step is not tiny per run.
DEFAULT_GRAD_Z_THRESHOLD = 6.0

# Loss-spike z-score threshold. Higher than the gradient threshold because
# the loss is the noisier of the two.
DEFAULT_LOSS_Z_THRESHOLD = 8.0


@dataclass
class _EWMA:
    """Exponentially-weighted running mean and variance (West's method)."""

    alpha: float
    mean: float = 0.0
    var: float = 0.0
    n: int = 0

    def update(self, x: float) -> None:
        if not math.isfinite(x):
            return  # never poison the baseline with a corrupted sample
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.var = 0.0
            return
        delta = x - self.mean
        self.mean += self.alpha * delta
        self.var = (1 - self.alpha) * (self.var + self.alpha * delta * delta)

    @property
    def std(self) -> float:
        return math.sqrt(max(self.var, 0.0))

    def z(self, x: float) -> float:
        """Signed z-score. 0 when there is no spread to speak of."""
        s = self.std
        if s <= 1e-12:
            return 0.0
        return (x - self.mean) / s


@dataclass
class GuardTier:
    """Free-tier detector."""

    warmup_steps: int = DEFAULT_WARMUP_STEPS
    ewma_alpha: float = DEFAULT_EWMA_ALPHA
    grad_z_threshold: float = DEFAULT_GRAD_Z_THRESHOLD
    loss_z_threshold: float = DEFAULT_LOSS_Z_THRESHOLD
    check_grads: bool = True

    _loss: _EWMA = field(init=False, repr=False)
    _grad: _EWMA = field(init=False, repr=False)
    _observed: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._loss = _EWMA(self.ewma_alpha)
        self._grad = _EWMA(self.ewma_alpha)

    @property
    def warm(self) -> bool:
        return self._observed >= self.warmup_steps

    def observe(
        self,
        *,
        step: int,
        loss: float,
        grad_norm: float,
        model: torch.nn.Module | None = None,
    ) -> Verdict:
        """Inspect one step. Returns a triggered Verdict or NO_DETECTION."""
        # --- proof-grade checks: no thresholds, no warmup ---
        if not math.isfinite(loss):
            return Verdict(
                True,
                step,
                TIER_GUARD,
                REASON_NONFINITE_LOSS,
                {"loss": loss},
            )
        if not math.isfinite(grad_norm):
            return Verdict(
                True,
                step,
                TIER_GUARD,
                REASON_NONFINITE_GRAD,
                {"grad_norm": grad_norm},
            )

        # --- statistical checks: only once the baseline means something ---
        verdict = NO_DETECTION
        if self.warm:
            gz = self._grad.z(grad_norm)
            lz = self._loss.z(loss)
            # One-sided: corruption inflates these. A loss that drops
            # sharply is the model learning, not a fault.
            if gz > self.grad_z_threshold:
                verdict = Verdict(
                    True,
                    step,
                    TIER_GUARD,
                    REASON_GRAD_NORM_ZSCORE,
                    {
                        "grad_norm": grad_norm,
                        "z": round(gz, 3),
                        "baseline_mean": round(self._grad.mean, 6),
                        "baseline_std": round(self._grad.std, 6),
                    },
                )
            elif lz > self.loss_z_threshold:
                verdict = Verdict(
                    True,
                    step,
                    TIER_GUARD,
                    REASON_LOSS_SPIKE,
                    {
                        "loss": loss,
                        "z": round(lz, 3),
                        "baseline_mean": round(self._loss.mean, 6),
                        "baseline_std": round(self._loss.std, 6),
                    },
                )

        # Update the baseline AFTER judging, so a step is never compared
        # against a baseline that already contains it.
        #
        # A triggering sample is deliberately still folded in: if the run is
        # genuinely corrupted, M3 rolls back and this detector's state is
        # rebuilt anyway; if it was a false alarm, refusing to learn from
        # legitimate spikes would keep the variance estimate too small and
        # cause the SAME false alarm every time.
        self._loss.update(loss)
        if self.check_grads:
            self._grad.update(grad_norm)
        self._observed += 1
        return verdict

    def reset(self) -> None:
        """Drop all history (M3 calls this after a rollback)."""
        self._loss = _EWMA(self.ewma_alpha)
        self._grad = _EWMA(self.ewma_alpha)
        self._observed = 0


def grads_are_finite(model: torch.nn.Module) -> bool:
    """Explicit isfinite sweep over gradients.

    Not used on the hot path -- `clip_grad_norm_` already returns a norm
    that is non-finite iff any gradient is, so the loop gets this signal for
    free. Provided for workloads that do not clip.
    """
    for p in model.parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            return False
    return True
