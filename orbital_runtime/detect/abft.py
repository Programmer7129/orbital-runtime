"""Tier 2 -- ABFT: sampled checksum verification around nn.Linear GEMMs.

Research doc SS3: "checksum verification around nn.Linear GEMMs on a
sampling schedule. Literature: FT-CNN 4-8%; V-ABFT ~12% w/ variance-based
thresholds (solves fp16/bf16 rounding-noise-vs-fault discrimination, arXiv
2602.08043); ApproxABFT cuts exact-ABFT overhead ~43%."

The check
---------
`nn.Linear` computes `Y = X @ W.T + b`. Summing a row of the bias-free
product over the output dimension:

    sum_j (X @ W.T)_ij  =  sum_j sum_k X_ik W_jk  =  sum_k X_ik * s_k

where `s = W.sum(dim=0)` is a vector over the INPUT dimension. So

    rowsum(Y - b)  ==  X @ s

must hold exactly in exact arithmetic. The left side is a reduction over an
output we already have; the right side is a matrix-vector product costing
O(B*T*in) against the GEMM's O(B*T*in*out) -- i.e. 1/out of the work.

WHEN the checksum is taken is the whole ballgame
-----------------------------------------------
`s` must be captured at a moment the weights are TRUSTED -- immediately
after `optimizer.step()` -- and the next forward compared against that
stored value.

Computing `s` lazily from the current weights instead would make this tier
blind to exactly the faults it exists to catch. If `W` is corrupted and we
then derive `s` from the corrupted `W`, both sides of the identity contain
the same bad weight, they agree perfectly, and the check passes. (Caching
keyed on `weight._version` has the same flaw with extra steps: an injected
flip bumps the version, so the cache would "helpfully" refresh itself
against the corruption.) Such a tier would only ever catch transient
compute faults, while silently reporting a clean bill of health for every
memory SEU in a Linear weight.

With the trusted-snapshot ordering, a flip landing between step N-1's
update and step N's forward is compared against pre-flip truth, and shows
up as a discrepancy of `X[i,k] * delta`.

Sensitivity, honestly
---------------------
That discrepancy is only visible if it exceeds rounding noise. A bit-30
strike (delta ~ 1e38) is caught instantly; a low-mantissa strike
(delta ~ 1e-9 * |W|) is NOT detectable -- it is buried in the same noise
floor that makes it nearly harmless. The tier's real sensitivity curve is
measured, not asserted: see `bench/detect_eval.py`.

Why a threshold is unavoidable (and why V-ABFT exists)
-----------------------------------------------------
In floating point the identity does NOT hold exactly: the two sides sum the
same terms in different orders, so they differ by rounding noise. Comparing
with `==` would fire every step; comparing with a fixed epsilon either
misses real faults on large activations or drowns in false positives on
small ones.

So the tolerance is **scaled to the arithmetic**, which is the "variance-
aware threshold" idea: rounding error in a K-term dot product accumulates
as a random walk, growing like eps*sqrt(K), and scales with the magnitude
of the terms being summed. `_tolerance()` below builds exactly that bound.
This is what makes the tier usable in bf16, whose eps is ~1e-2 -- 4000x
looser than fp32's -- where a fixed threshold is hopeless.

What it catches that tier 1 cannot
----------------------------------
A weight corrupted in a low mantissa bit changes the GEMM result far too
little to move the loss out of its noise band, so no z-score will ever see
it -- that is the silent-divergence regime. But the checksum compares the
SAME computation two ways, so it sees a discrepancy that is tiny in
absolute terms yet enormous relative to rounding noise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor, nn

from ..orbit.flux import FluxModel
from .verdict import NO_DETECTION, REASON_ABFT_MISMATCH, TIER_ABFT, Verdict

# Multiplier on the theoretical rounding-error bound. The bound is an
# order-of-magnitude estimate (it assumes errors accumulate as a random
# walk), so a safety factor absorbs its slack. 16 was chosen by measuring
# the observed clean-run residual distribution against the bound -- see
# tests/test_abft.py::test_clean_residual_sits_far_below_tolerance.
DEFAULT_SAFETY_FACTOR = 16.0

# Fraction of eligible GEMMs verified per step when NOT in the SAA.
DEFAULT_BASE_SAMPLE_RATE = 0.1

# Fraction verified inside an SAA transit -- "adaptive vigilance".
DEFAULT_SAA_SAMPLE_RATE = 1.0


def _tolerance(dtype: torch.dtype, k: int, scale: float, safety: float) -> float:
    """Rounding-noise bound for a K-term reduction at a given magnitude.

    eps * sqrt(K) * scale is the standard random-walk estimate of
    accumulated rounding error; `safety` covers the estimate's slack.
    """
    eps = float(torch.finfo(dtype).eps)
    return safety * eps * math.sqrt(max(k, 1)) * scale


@dataclass
class AbftStats:
    checks: int = 0
    mismatches: int = 0
    gemms_seen: int = 0
    gemms_verified: int = 0

    def as_dict(self) -> dict:
        return {
            "abft_checks": self.checks,
            "abft_mismatches": self.mismatches,
            "abft_gemms_seen": self.gemms_seen,
            "abft_gemms_verified": self.gemms_verified,
            "abft_sample_rate_actual": (
                self.gemms_verified / self.gemms_seen if self.gemms_seen else 0.0
            ),
        }


class AbftTier:
    """Checksum-verifies a sampled subset of nn.Linear forward passes.

    Attaches forward hooks like the compute injector, but read-only: it
    never modifies an output.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        flux: FluxModel | None = None,
        base_sample_rate: float = DEFAULT_BASE_SAMPLE_RATE,
        saa_sample_rate: float = DEFAULT_SAA_SAMPLE_RATE,
        safety_factor: float = DEFAULT_SAFETY_FACTOR,
        adaptive: bool = True,
        rng: np.random.Generator | None = None,
    ) -> None:
        if not 0.0 <= base_sample_rate <= 1.0:
            raise ValueError(f"base_sample_rate must be in [0,1], got {base_sample_rate}")
        if not 0.0 <= saa_sample_rate <= 1.0:
            raise ValueError(f"saa_sample_rate must be in [0,1], got {saa_sample_rate}")
        self.model = model
        self.flux = flux
        self.base_sample_rate = base_sample_rate
        self.saa_sample_rate = saa_sample_rate
        self.safety_factor = safety_factor
        self.adaptive = adaptive
        self.rng = rng if rng is not None else np.random.default_rng(0)

        self.stats = AbftStats()
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        # name -> trusted `W.sum(dim=0)`, snapshotted at a known-good moment.
        self._trusted: dict[str, Tensor] = {}
        self._t_sim: float = 0.0
        self._in_saa: bool = False
        self._findings: list[dict] = []
        # Queued device-side checks, resolved once per step. See _verify.
        self._pending: list[tuple[str, Tensor, Tensor, int, str]] = []
        self._enabled = False

    # ------------------------------------------------------------------ #
    # Adaptive vigilance
    # ------------------------------------------------------------------ #

    def sample_rate(self) -> float:
        """Verification intensity, keyed to orbital position.

        The differentiator (research doc SS3): "detection intensity and
        checkpoint cadence keyed to orbital position (crank ABFT sampling +
        checkpoint immediately before SAA entry). Novel; nothing in
        literature does position-aware protection scheduling."

        The economics: ~90% of upsets arrive in ~10% of the orbit, so
        spending the verification budget uniformly wastes most of it. Paying
        full price inside the SAA and a tenth of it outside buys most of the
        coverage for a fraction of the average overhead.
        """
        if not self.adaptive:
            return self.base_sample_rate
        return self.saa_sample_rate if self._in_saa else self.base_sample_rate

    def set_position(self, *, t_sim: float, in_saa: bool) -> None:
        self._t_sim = t_sim
        self._in_saa = in_saa

    # ------------------------------------------------------------------ #
    # Hooks
    # ------------------------------------------------------------------ #

    def attach(self) -> AbftTier:
        if self._handles:
            return self
        for name, mod in self.model.named_modules():
            if isinstance(mod, nn.Linear):
                self._handles.append(mod.register_forward_hook(self._make_hook(name)))
        return self

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self) -> AbftTier:
        return self.attach()

    def __exit__(self, *exc: object) -> None:
        self.detach()

    @torch.no_grad()
    def refresh_checksums(self) -> None:
        """Snapshot `s = W.sum(dim=0)` for every Linear, as ground truth.

        MUST be called only when the weights are trusted -- i.e. straight
        after `optimizer.step()`, before any radiation can land. Everything
        this tier detects rests on this snapshot predating the fault.

        Cheap: O(in*out) per layer against the GEMM's O(B*T*in*out), so
        ~1/(B*T) of a forward pass, and it is the only unsampled work the
        tier does per step.
        """
        for name, mod in self.model.named_modules():
            if isinstance(mod, nn.Linear):
                self._trusted[name] = mod.weight.detach().sum(dim=0).to(torch.float32)

    def _trusted_checksum(self, name: str, weight: Tensor) -> Tensor:
        """The stored snapshot, or take one now if this is the first sight.

        Falling back to the live weight on the very first forward is safe:
        no radiation has been dispatched yet at that point.
        """
        s = self._trusted.get(name)
        if s is None:
            s = weight.detach().sum(dim=0).to(torch.float32)
            self._trusted[name] = s
        return s

    def _make_hook(self, name: str):
        def hook(module: nn.Linear, inputs: tuple, output: Tensor):
            if not self._enabled:
                return output
            self.stats.gemms_seen += 1
            if self.rng.random() >= self.sample_rate():
                return output
            self.stats.gemms_verified += 1
            self._verify(name, module, inputs[0], output)
            return output  # read-only: never modifies the computation

        return hook

    @torch.no_grad()
    def _verify(self, name: str, module: nn.Linear, x: Tensor, y: Tensor) -> None:
        """Queue a check. Performs NO host synchronisation.

        Every value here stays a device tensor. Reading any of them with
        `.item()`/`float()`/`int()` would block until the GPU drains, and
        this runs once per sampled GEMM -- several times per step. Measured
        on MPS, a `.item()`-per-check implementation of this exact math cost
        +203% at full sampling, almost none of it arithmetic: the tier was
        sync-bound, stalling the pipeline it was supposed to be quietly
        watching. The comparison is deferred to `observe()`, which syncs
        ONCE per step for all checks at once.
        """
        self.stats.checks += 1

        x_ = x.detach()
        y_ = y.detach()
        if module.bias is not None:
            y_ = y_ - module.bias

        s = self._trusted_checksum(name, module.weight)

        # Reduce in fp32 regardless of the working dtype: a bf16 reduction
        # would inject more rounding noise than the fault we are hunting.
        lhs = y_.to(torch.float32).sum(dim=-1)
        rhs = x_.to(torch.float32) @ s

        residual = (lhs - rhs).abs().max()
        k = x_.shape[-1]  # reduction length

        # Magnitude floor of 1e-8: an all-zero GEMM has no scale to key to.
        scale = torch.maximum(
            torch.maximum(lhs.abs().max(), rhs.abs().max()),
            torch.tensor(1e-8, device=lhs.device, dtype=lhs.dtype),
        )
        # Tolerance coefficient from the single source of truth (`_tolerance`),
        # evaluated at unit scale so we can multiply the device-tensor `scale`
        # without an early host sync. Previously this line inlined the same
        # eps*sqrt(K)*safety formula -- a duplicate the hostile review flagged
        # (item 17). One formula now, exercised by tests/test_abft.py.
        tol = scale * _tolerance(x_.dtype, k, 1.0, self.safety_factor)

        self._pending.append((name, residual, tol, k, str(x_.dtype).replace("torch.", "")))

    @torch.no_grad()
    def _resolve_pending(self) -> None:
        """Bring the queued checks to the host in ONE synchronisation."""
        if not self._pending:
            return
        ratios = torch.stack(
            [r / t.clamp_min(1e-38) for _, r, t, _, _ in self._pending]
        ).tolist()  # <- the single sync point

        for (name, _, _, k, dtype), ratio in zip(self._pending, ratios):
            if ratio > 1.0:
                self.stats.mismatches += 1
                self._findings.append(
                    {"module": name, "ratio": ratio, "k": k, "dtype": dtype}
                )
        self._pending = []

    # ------------------------------------------------------------------ #
    # Loop interface
    # ------------------------------------------------------------------ #

    def arm(self) -> None:
        """Verify during the next forward pass."""
        self._enabled = True

    def disarm(self) -> None:
        self._enabled = False

    def observe(self, *, step: int, **_: object) -> Verdict:
        """Report whatever the hooks found during this step's forward.

        This is the tier's single synchronisation point per step.
        """
        self._resolve_pending()
        if not self._findings:
            return NO_DETECTION
        worst = max(self._findings, key=lambda f: f["ratio"])
        n = len(self._findings)
        self._findings = []
        return Verdict(
            True,
            step,
            TIER_ABFT,
            REASON_ABFT_MISMATCH,
            {"mismatches_this_step": n, **worst},
        )

    def reset(self) -> None:
        self._findings = []
        self._pending = []
        self._trusted.clear()  # restored weights need a fresh trust anchor
