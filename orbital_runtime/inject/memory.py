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

Multi-bit upsets (MBUs) -- M4c, cited
-------------------------------------
A pure independent-single-bit model UNDERSTATES correlated corruption and
overstates ECC effectiveness (an ionizing track deposits charge across
several adjacent cells, and SEC-DED corrects single-bit but not multi-bit
errors). The beam-data audit (docs/research/beam-calibration-audit.md) pins
this quantitatively from MICRO'21 (doi 10.1145/3466752.3480111; V100 HBM2,
ChipIR neutron beam):

    * 31.5% of upset EVENTS are multi-bit (MBU_SHARE)
    * ~75% of those are byte-contiguous (MBU_CONTIGUOUS_SHARE)
    * the broadest single event hit 5,359 memory entries

So an upset EVENT is no longer one bit -- it is a CLUSTER. We model the
common small-cluster regime: 68.5% single-bit, 31.5% multi-bit, of which 75%
flip a contiguous run of adjacent bits (one word) and 25% flip a scattered
set. The large-cluster tail (up to thousands of entries) is a disclosed,
UNMODELED extreme -- our cluster caps at `MBU_MAX_CLUSTER` bits within one
element. Conditions caveat: MICRO'21 is neutron / HBM2; our injector is
dtype-generic fp32, so the share is a cited anchor, not a device-matched
measurement. `inject()` remains a single-bit primitive (used by the
sensitivity tests); `inject_event()` is the physical, clustered event that
the radiation environment dispatches.

Device-agnostic (PLAN.md design rule 1): verified on CPU and MPS for fp32,
fp16 and bf16. No CUDA-only primitives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from .gpu_model import (
    CLASS_BITFLIP,
    CLASS_NULLIFICATION,
    CLASS_SPECIAL,
    PATH_MEMORY,
    sample_bit_position,
    sample_fault_class,
)

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

# --- Multi-bit-upset (MBU) cluster model, MICRO'21 (see module docstring) ---
# Fraction of upset EVENTS that flip more than one bit.
MBU_SHARE = 0.315
# Of the multi-bit events, the fraction that are byte-contiguous (a run of
# adjacent bits) rather than scattered.
MBU_CONTIGUOUS_SHARE = 0.75
# Geometric-tail parameter for the multi-bit cluster size: size = 1 +
# Geometric(p), so size >= 2 with mean 1 + 1/p. p=0.6 -> mean ~2.67 bits, a
# small-cluster regime. MICRO'21 gives the share + contiguity but not a full
# size histogram, so this is the deliberately-modest modeled distribution; the
# large-cluster tail is disclosed as unmodeled.
MBU_SIZE_GEOMETRIC_P = 0.6
# Cap on modeled cluster size (bits within one element). A byte is 8 bits;
# capping here keeps a contiguous cluster inside a single word, which is the
# regime we model. The thousands-of-entries tail is out of scope.
MBU_MAX_CLUSTER = 8


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


@dataclass(frozen=True)
class UpsetCluster:
    """One upset EVENT: a cluster of 1+ bit flips in a single element.

    A single-bit event has `size == 1`; a multi-bit (MBU) event flips several
    bits of the same word. All flips are on the same (name, index) element, so
    the net element value is `flips[-1].value_after` (flips are applied in
    order, each XOR compounding on the last).
    """

    flips: list[Flip]
    multi_bit: bool
    contiguous: bool
    # Which measured GPU mechanism produced this event (gpu_model.CLASS_*).
    # Empty for the legacy MICRO'21 memory-only path.
    fault_class: str = ""
    # Distinct ELEMENTS struck. >1 for nullification tiles and warp-aligned
    # tracks, where `flips` spans several elements rather than several bits of
    # one word.
    n_elements: int = 1
    # Element stride for a warp-aligned event; 1 = contiguous.
    stride: int = 1

    @property
    def size(self) -> int:
        return len(self.flips)

    @property
    def primary(self) -> Flip:
        return self.flips[0]

    @property
    def net_value_after(self) -> float:
        return self.flips[-1].value_after

    @property
    def became_nonfinite(self) -> bool:
        return not math.isfinite(self.net_value_after)

    @property
    def bit_positions(self) -> list[int]:
        return [f.bit for f in self.flips]

    def as_record(self) -> dict:
        p = self.flips[0]
        return {
            "name": p.name,
            "target_kind": p.kind,
            "index": p.index,
            "bit": p.bit,  # the primary (first) struck bit
            "dtype": p.dtype,
            "value_before": p.value_before,
            "value_after": self.net_value_after,
            "nonfinite": self.became_nonfinite,
            "cluster_size": self.size,
            "multi_bit": self.multi_bit,
            "contiguous": self.contiguous,
            "bits": self.bit_positions,
            "fault_class": self.fault_class,
            "n_elements": self.n_elements,
            "stride": self.stride,
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

    def _choose_bit(self, rng: np.random.Generator) -> tuple[Target, int, int] | None:
        """Pick a (target, element index, primary bit), uniform over all bits."""
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
        return chosen, index, bit

    def _flip_at(self, target: Target, index: int, bit: int) -> Flip:
        before, after = flip_bit(target.tensor, index, bit)
        return Flip(
            name=target.name,
            kind=target.kind,
            index=index,
            bit=bit,
            dtype=str(target.tensor.dtype).replace("torch.", ""),
            value_before=before,
            value_after=after,
        )

    def inject(self, rng: np.random.Generator) -> Flip | None:
        """Strike one uniformly-random resident bit. None if no targets.

        The single-bit primitive: used by the ABFT sensitivity tests and as
        the building block. The radiation environment dispatches
        `inject_event`, which applies the MBU cluster model on top of this.
        """
        pick = self._choose_bit(rng)
        if pick is None:
            return None
        return self._flip_at(*pick)

    # ------------------------------------------------------------------ #
    # GPU-calibrated fault classes (Tung et al. 2026, see inject/gpu_model.py)
    # ------------------------------------------------------------------ #

    def _element_span(
        self, numel: int, start: int, n: int, stride: int
    ) -> list[int]:
        """Indices for a contiguous or warp-aligned run, clipped to the tensor."""
        idx = [start + i * stride for i in range(n)]
        return [i for i in idx if 0 <= i < numel]

    def _write_at(self, target: Target, index: int, value: float) -> Flip:
        """Overwrite one element outright. Used by nullification and NaN forcing.

        Recorded as a `Flip` with `bit=-1`: this is not a bit toggle, it is a
        datapath fault that replaced the value. The sentinel keeps one record
        type across all mechanisms while staying honest about which occurred.
        """
        flat = target.tensor.detach().reshape(-1)
        before = float(flat[index].item())
        with torch.no_grad():
            flat[index] = value
        return Flip(
            name=target.name,
            kind=target.kind,
            index=index,
            bit=-1,
            dtype=str(target.tensor.dtype).replace("torch.", ""),
            value_before=before,
            value_after=float(value),
        )

    def inject_gpu_event(self, rng: np.random.Generator) -> UpsetCluster | None:
        """Strike one event drawn from the MEASURED GPU outcome distribution.

        Replaces the memory-only bit-flip model as the default dispatch path.
        The old model could not produce nullification at all, which is 50.68% of
        real GPU SDCs, so its recall numbers described a fault space that does
        not occur (see docs and `bench/fault_model_audit.py`).

        The event RATE still comes from the orbital flux model. Only the shape
        of a corruption is taken from NVIDIA's terrestrial characterization,
        because outcome is a property of the silicon and rate is a property of
        the environment.
        """
        targets = self.targets()
        if not targets:
            return None
        weights = np.array([t.bits for t in targets], dtype=np.float64)
        total = weights.sum()
        if total <= 0:
            return None
        target = targets[int(rng.choice(len(targets), p=weights / total))]
        numel = target.tensor.numel()
        width = target.tensor.element_size() * 8

        fc = sample_fault_class(rng, numel=numel, path=PATH_MEMORY)
        start = int(rng.integers(0, numel))

        if fc.label == CLASS_NULLIFICATION:
            indices = self._element_span(numel, start, fc.n_elements, fc.stride)
            flips = [self._write_at(target, i, 0.0) for i in indices]
            if not flips:
                return None
            return UpsetCluster(
                flips,
                multi_bit=False,
                contiguous=(fc.stride == 1),
                fault_class=fc.label,
                n_elements=len(flips),
                stride=fc.stride,
            )

        if fc.label == CLASS_SPECIAL:
            # Force the exponent to all-ones. A uniform bit draw reaches this
            # far too often; the measured share is 1.01%, so the mechanism is
            # selected explicitly rather than emerging from bit choice.
            value = math.inf if rng.random() < 0.5 else math.nan
            flip = self._write_at(target, start, value)
            return UpsetCluster(
                [flip],
                multi_bit=False,
                contiguous=True,
                fault_class=fc.label,
                n_elements=1,
                stride=1,
            )

        # CLASS_BITFLIP: non-special value change, possibly multi-bit and
        # possibly spanning a warp-aligned track.
        indices = self._element_span(numel, start, fc.n_elements, fc.stride)
        if not indices:
            return None
        flips: list[Flip] = []
        for idx in indices:
            bits = self._draw_bits(rng, width, fc.n_bits)
            for b in bits:
                flips.append(self._flip_at(target, idx, b))
        if not flips:
            return None
        return UpsetCluster(
            flips,
            multi_bit=(fc.n_bits > 1),
            contiguous=(fc.stride == 1),
            fault_class=fc.label,
            n_elements=len(indices),
            stride=fc.stride,
        )

    def _draw_bits(self, rng: np.random.Generator, width: int, n: int) -> list[int]:
        """Distinct bit positions, weighted LSB-heavy per Tung et al.

        The top exponent bits are deliberately reachable but rare: striking them
        produces Inf/NaN, and the measured special-value share is only 1.01%.
        """
        n = max(1, min(n, width))
        chosen: list[int] = []
        for _ in range(n * 4):  # bounded retry for distinctness
            b = sample_bit_position(rng, width)
            if b not in chosen:
                chosen.append(b)
            if len(chosen) == n:
                break
        return chosen or [0]

    def inject_event(self, rng: np.random.Generator) -> UpsetCluster | None:
        """Strike one upset EVENT -- a cluster of 1+ bits (MICRO'21 MBU model).

        68.5% of events are single-bit; 31.5% are multi-bit, of which 75% flip
        a contiguous run of adjacent bits in one word and 25% a scattered set.
        The primary bit is chosen uniformly (as `inject`), then the event type
        and cluster geometry are drawn. All flips land on the same element.
        """
        pick = self._choose_bit(rng)
        if pick is None:
            return None
        target, index, bit = pick
        width = target.tensor.element_size() * 8

        multi_bit = bool(rng.random() < MBU_SHARE)
        if not multi_bit:
            return UpsetCluster([self._flip_at(target, index, bit)], False, False)

        size = min(1 + int(rng.geometric(MBU_SIZE_GEOMETRIC_P)), MBU_MAX_CLUSTER, width)
        contiguous = bool(rng.random() < MBU_CONTIGUOUS_SHARE)
        if contiguous:
            # A run of `size` adjacent bits anchored at the primary bit, slid
            # to fit inside the word.
            start = min(max(bit, 0), width - size)
            bits = list(range(start, start + size))
        else:
            # `size` distinct bits scattered across the word (the primary bit
            # included, so the event is never smaller than intended).
            others = [b for b in range(width) if b != bit]
            extra = rng.choice(len(others), size=size - 1, replace=False)
            bits = sorted([bit] + [others[int(i)] for i in extra])

        flips = [self._flip_at(target, index, b) for b in bits]
        return UpsetCluster(flips, multi_bit=True, contiguous=contiguous)
