"""Memory SEUs: single-bit flips in resident tensors.

Research doc SS1 ("MVP approach"): Poisson process over resident bytes;
`tensor.view(torch.int32) ^= (1 << k)` on random bits of params / optimizer
state / activations.

Targeting model
---------------
Every resident bit is equally likely to be struck. So a flip is placed by:

  1. choose a tensor with probability proportional to its BIT COUNT,
  2. choose a uniform element within it,
  3. choose a uniform bit within that element.

This composition is exactly uniform over all resident bits, which is the
physically correct model for an ionizing particle hitting a memory array --
and it is why the flip distribution over bit positions is uniform rather
than biased toward the exponent. That uniformity is what produces BOTH demo
failure modes for free: a strike on a high exponent bit (fp32 bits 30..23)
explodes the value into NaN/Inf, while a strike on a low mantissa bit
perturbs it imperceptibly and drives silent divergence (PLAN.md M1).

Device-agnostic (PLAN.md design rule 1): verified on CPU and MPS for fp32,
fp16 and bf16. No CUDA-only primitives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

# Float dtype -> same-width integer dtype for the bitwise view.
# torch.view(dtype) requires identical element size, so these must match
# exactly. fp8 is deliberately absent: torch has no 8-bit integer view that
# round-trips cleanly across CPU/MPS, and the MVP workload is fp32.
_INT_VIEW_DTYPE: dict[torch.dtype, torch.dtype] = {
    torch.float32: torch.int32,
    torch.float16: torch.int16,
    torch.bfloat16: torch.int16,
    torch.float64: torch.int64,
}

KIND_PARAM = "param"
KIND_OPTIMIZER = "optimizer"


def bits_of(t: Tensor) -> int:
    """Resident bits of a tensor."""
    return t.numel() * t.element_size() * 8


@dataclass(frozen=True)
class Target:
    """A tensor eligible to be struck."""

    name: str
    tensor: Tensor
    kind: str

    @property
    def bits(self) -> int:
        return bits_of(self.tensor)


@dataclass(frozen=True)
class Flip:
    """A record of one bit flip, sufficient to audit or undo it."""

    name: str
    kind: str
    index: int  # flat element index
    bit: int  # bit position within the element (0 = LSB)
    dtype: str
    value_before: float
    value_after: float

    @property
    def became_nonfinite(self) -> bool:
        return not math.isfinite(self.value_after)

    @property
    def relative_delta(self) -> float:
        """|after - before| / |before|; inf if the value blew up."""
        if not math.isfinite(self.value_after):
            return math.inf
        denom = abs(self.value_before)
        if denom == 0.0:
            return math.inf if self.value_after != 0.0 else 0.0
        return abs(self.value_after - self.value_before) / denom

    def as_record(self) -> dict:
        return {
            "name": self.name,
            "target_kind": self.kind,
            "index": self.index,
            "bit": self.bit,
            "dtype": self.dtype,
            "value_before": self.value_before,
            "value_after": self.value_after,
            "nonfinite": self.became_nonfinite,
        }


def flip_bit(t: Tensor, index: int, bit: int) -> tuple[float, float]:
    """XOR one bit of one element in-place. Returns (before, after).

    The tensor must be contiguous -- `view(dtype)` requires it, and a
    non-contiguous view would silently address the wrong memory.
    """
    int_dtype = _INT_VIEW_DTYPE.get(t.dtype)
    if int_dtype is None:
        raise TypeError(f"no integer view for dtype {t.dtype}")
    width = t.element_size() * 8
    if not 0 <= bit < width:
        raise ValueError(f"bit {bit} out of range for {width}-bit {t.dtype}")
    if index < 0 or index >= t.numel():
        raise IndexError(f"index {index} out of range for {t.numel()} elements")

    # Check contiguity on the ORIGINAL tensor, before any reshape.
    # `reshape(-1)` on a non-contiguous tensor silently returns a COPY, and a
    # flip into that copy would be discarded -- the model would train on
    # undamaged weights while telemetry happily reported a hit. Failing loudly
    # is the only acceptable behaviour. (`view(-1)` would also raise, but the
    # explicit check says why.)
    if not t.is_contiguous():
        raise ValueError(f"cannot flip bits of non-contiguous tensor {tuple(t.shape)}")

    with torch.no_grad():
        flat = t.detach().reshape(-1)
        before = float(flat[index].item())
        int_view = flat.view(int_dtype)
        # Build the mask in Python then cast: `1 << 31` overflows int32 if
        # constructed as a torch scalar of the view's dtype, and the sign bit
        # is exactly the bit we most want to be able to strike.
        mask = int(1) << bit
        signed_mask = mask - (1 << width) if mask >= (1 << (width - 1)) else mask
        int_view[index] ^= torch.tensor(signed_mask, dtype=int_dtype, device=t.device)
        after = float(flat[index].item())
    return before, after


class MemoryInjector:
    """Places bit flips across a set of target tensors.

    Targets are resolved LAZILY on every flip, because optimizer state does
    not exist until after the first `step()` -- resolving once at
    construction would make the entire optimizer state permanently immune,
    quietly halving the resident memory we claim to be modelling.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        *,
        target_optimizer_state: bool = True,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.target_optimizer_state = target_optimizer_state

    # ------------------------------------------------------------------ #
    # Target enumeration
    # ------------------------------------------------------------------ #

    def targets(self) -> list[Target]:
        """Currently resident, strikeable tensors."""
        out: list[Target] = []
        for name, p in self.model.named_parameters():
            if p.dtype in _INT_VIEW_DTYPE and p.numel() > 0:
                out.append(Target(name, p.data, KIND_PARAM))
        if self.target_optimizer_state and self.optimizer is not None:
            for tname, tensor in self._optimizer_tensors():
                out.append(Target(tname, tensor, KIND_OPTIMIZER))
        return out

    def _optimizer_tensors(self) -> list[tuple[str, Tensor]]:
        """Named optimizer state tensors (e.g. Adam's exp_avg/exp_avg_sq).

        Named by (param position, state key) so names are stable across
        runs -- dict iteration order over params is insertion order, which
        torch guarantees for param_groups.
        """
        out: list[tuple[str, Tensor]] = []
        assert self.optimizer is not None
        param_names = {id(p): n for n, p in self.model.named_parameters()}
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                state = self.optimizer.state.get(p)
                if not state:
                    continue
                pname = param_names.get(id(p), "unknown")
                for key, val in state.items():
                    if (
                        isinstance(val, Tensor)
                        and val.dtype in _INT_VIEW_DTYPE
                        and val.numel() > 0
                    ):
                        out.append((f"{pname}.{key}", val))
        return out

    def resident_bits(self) -> int:
        """Bits currently resident across all targets."""
        return sum(t.bits for t in self.targets())

    def static_resident_bits(self) -> int:
        """Bits resident once training is warm, known WITHOUT running.

        The flux schedule is drawn up front (for determinism), so lambda
        needs a bit count before step 0 -- but optimizer state does not
        exist yet at that point. We derive it analytically: parameter bits
        times (1 + states-per-param), where states-per-param comes from the
        optimizer type (AdamW: exp_avg + exp_avg_sq = 2).

        Using post-warmup residency is the honest choice: it is the state
        the run spends ~all of its time in.
        """
        param_bits = sum(
            bits_of(p) for _, p in self.model.named_parameters() if p.dtype in _INT_VIEW_DTYPE
        )
        if not self.target_optimizer_state or self.optimizer is None:
            return param_bits
        # If state exists, MEASURE it -- exact beats inferred.
        existing = self._optimizer_tensors()
        if existing:
            return param_bits + sum(bits_of(t) for _, t in existing)
        return param_bits * (1 + self._states_per_param())

    def _states_per_param(self) -> int:
        """Full-size state tensors the optimizer keeps per parameter.

        Inferred from the optimizer class, and used ONLY before any step has
        run (once state exists, `static_resident_bits` measures it instead).

        Counts state that scales with the parameter, not every state entry:
        Adam also keeps a 0-dim `step` scalar per param, and counting state
        *tensors* rather than *bits* would read AdamW as 3 states/param and
        set lambda 33% too high.
        """
        assert self.optimizer is not None
        opt = type(self.optimizer).__name__.lower()
        if opt in ("adam", "adamw", "nadam", "radam", "adamax"):
            return 2  # exp_avg, exp_avg_sq
        if opt in ("sgd",):
            # Momentum buffer only if momentum is enabled.
            return 1 if any(g.get("momentum", 0) for g in self.optimizer.param_groups) else 0
        if opt in ("rmsprop", "adagrad", "adadelta"):
            return 1
        return 0  # unknown optimizer: count params only, and do not pretend otherwise

    # ------------------------------------------------------------------ #
    # Injection
    # ------------------------------------------------------------------ #

    def inject(self, rng: np.random.Generator) -> Flip | None:
        """Strike one uniformly-random resident bit. None if no targets."""
        targets = self.targets()
        if not targets:
            return None

        weights = np.array([t.bits for t in targets], dtype=np.float64)
        total = weights.sum()
        if total <= 0:
            return None
        chosen = targets[int(rng.choice(len(targets), p=weights / total))]

        index = int(rng.integers(0, chosen.tensor.numel()))
        width = chosen.tensor.element_size() * 8
        bit = int(rng.integers(0, width))

        before, after = flip_bit(chosen.tensor, index, bit)
        return Flip(
            name=chosen.name,
            kind=chosen.kind,
            index=index,
            bit=bit,
            dtype=str(chosen.tensor.dtype).replace("torch.", ""),
            value_before=before,
            value_after=after,
        )
