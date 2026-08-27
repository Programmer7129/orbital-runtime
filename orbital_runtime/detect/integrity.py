"""State integrity: exact checksums over the state ABFT cannot see.

Why this tier exists
--------------------
The ABFT tier verifies `nn.Linear` GEMMs against a trusted `W.sum(dim=0)`.
That covers Linear weights and nothing else. A coverage audit of the strikeable
resident state (`bench/coverage_audit.py`) found:

    linear_weight      32.7%   <- ABFT sees this
    optimizer_state    66.7%   <- invisible
    other_param         0.6%   <- invisible

So two thirds of everything radiation can strike had no detector at all. Adam
keeps `exp_avg` and `exp_avg_sq` per parameter, each the size of the parameter,
which makes optimizer state the LARGEST target in the process.

The failure this closes is worse than a miss. A flip in `exp_avg` perturbs the
next `optimizer.step()`, which writes a wrong weight, and `refresh_checksums()`
then snapshots that wrong weight as the new trusted baseline. The fault is
laundered into ground truth and every later ABFT check agrees with it.

Why not ABFT here
-----------------
ABFT works on a GEMM because the matrix product gives an independent second
path to the same answer. Optimizer state is not the output of anything -- there
is nothing to cross-check it against. The right instrument is an end-to-end
integrity check: snapshot after the state is written, verify before it is used
again. The OCP white paper "Silent Data Corruption in AI" (2025) lists exactly
this under "End-to-End Integrity Checks (Informational Redundancy)".

Why an integer checksum
-----------------------
A float sum would need a tolerance, and a tolerance is a hole: any flip too
small to move the sum past it passes silently. Summing the INTEGER VIEW in
int64 is exact. A flip of bit k changes the checksum by exactly +/- 2^k, so
every single-bit flip is caught with no threshold and no false positives from
rounding.

Multi-bit upsets inside one element are always caught: the delta is a sum of
distinct +/- 2^b terms, which cannot vanish. Cancellation needs two flips in
the same bit position of different elements with opposite direction, which the
cluster model does not produce.

Cost is one reduction per tensor per step, O(n) against the GEMM's O(B*T*n).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn

from .verdict import Verdict

# int32/int64 views for the float dtypes the runtime supports. bf16/fp16 view
# as int16; torch has no int16 sum accumulator wide enough to be safe, so they
# are promoted to int32 before the reduction.
_INT_VIEW: dict[torch.dtype, torch.dtype] = {
    torch.float32: torch.int32,
    torch.float64: torch.int64,
    torch.bfloat16: torch.int16,
    torch.float16: torch.int16,
}

@dataclass
class _Baseline:
    """The trusted snapshot of one tensor: row sums, column sums, and the tail.

    Row sums detect. Column sums locate, and are only consulted when a row has
    already flagged. Both are about sqrt(n) long, so a 1 GB model carries a few
    hundred KB of baseline.
    """

    rows: Tensor
    cols: Tensor | None
    tail: Tensor
    R: int
    C: int


TIER_INTEGRITY = "integrity"
REASON_INTEGRITY_MISMATCH = "integrity_mismatch"

# --- Severity policy ------------------------------------------------------- #
# Detection and RESPONSE are different decisions. This tier is exact, so it sees
# every flip including ones that do not matter -- and a rollback costs a
# checkpoint restore plus replayed steps, during which more radiation lands.
#
# Without a severity gate the runtime rolls back on a bit-2 flip in an Adam
# second moment (a 5e-7 relative change) and can enter a treadmill: roll back,
# replay, get struck during replay, roll back further, exhaust the checkpoint
# history, die "unrecoverable". Observed at 10.7M params: 5 upsets, 4
# detections, 3 rollbacks, 22 replayed steps, dead at step 112 of 300.
#
# The checksum delta is exactly +/-2^k, so the moved bit is known for free, and
# for an IEEE-754 float the bit position IS the severity: flipping mantissa bit
# m changes the value by about 2^(m-23). Bits 23+ are exponent or sign.
FP32_MANTISSA_BITS = 23

# Parameters persist and compound: a small error is re-applied at every step and
# rides into the next checkpoint. Escalate on anything above ~0.1%.
PARAM_ROLLBACK_RELATIVE = 2.0**-10

# Optimizer moments are running averages with beta ~0.9-0.999, so a perturbation
# decays over ~1/(1-beta) steps without intervention. Only escalate when the
# value is grossly wrong -- a sign or exponent strike, which sqrt() and the
# update rule cannot absorb. Below that, record it and let it wash out; paying a
# rollback would cost more than the fault.
OPTIMIZER_ROLLBACK_RELATIVE = 0.5


def _relative_change(bit: int) -> float:
    """Approximate relative value change implied by a flip of `bit`.

    Mantissa bit m moves the value by ~2^(m-23). Exponent and sign strikes are
    treated as total: they scale by a power of two or negate outright.
    """
    if bit < 0:
        return math.inf  # datapath fault: the value was overwritten
    if bit >= FP32_MANTISSA_BITS:
        return math.inf
    return 2.0 ** (bit - FP32_MANTISSA_BITS)


def checksum_device(t: Tensor) -> Tensor:
    """Exact int64 checksum of `t`, left ON THE DEVICE as a 0-dim tensor.

    Bitwise exact by construction: reinterpreting the float bits as integers
    and summing means a flip of bit k moves the result by exactly 2^k. No
    tolerance is involved, so there is no floating-point false-positive path
    and no minimum detectable perturbation.

    Returns a device tensor and never calls `.item()`. Reading it here would
    block until the accelerator drains, once per tracked tensor -- and this
    tier tracks a couple of hundred. A first implementation that synced per
    tensor cost +34% per step; batching every checksum into ONE sync (see
    `_sync`) brought the same work under 2%. This is the identical trap
    `AbftTier._verify` documents.
    """
    int_dtype = _INT_VIEW.get(t.dtype)
    if int_dtype is None:
        raise TypeError(f"no integer view for dtype {t.dtype}")
    flat = t.detach().reshape(-1) if t.is_contiguous() else t.detach().contiguous().reshape(-1)
    view = flat.view(int_dtype)
    # Accumulate in int64 via `dtype=`, never via `.to(torch.int64)`. The cast
    # would materialise a full int64 COPY of the tensor -- double the bytes of
    # the state being checked, allocated twice per step. Measured at +32% per
    # step on the default workload. `sum(dtype=...)` widens the accumulator
    # only, reads the tensor once, and allocates nothing.
    #
    # The width matters: int16 and int32 accumulators overflow on any realistic
    # tensor, and an overflowed checksum is a silent collision.
    return view.sum(dtype=torch.int64)


# Cache the small index vectors per (length, device, dtype). They are only
# ~sqrt(n) long, so the whole cache stays tiny even across hundreds of tensors,
# and many tensors in a model share a shape.
_IDX_CACHE: dict[tuple, Tensor] = {}


def _arange_cached(n: int, device: torch.device) -> Tensor:
    key = (n, str(device))
    got = _IDX_CACHE.get(key)
    if got is None:
        got = torch.arange(n, device=device, dtype=torch.int64)
        _IDX_CACHE[key] = got
    return got


def _grid(n: int) -> tuple[int, int]:
    """Split n into R x C with R*C <= n, both near sqrt(n).

    Any leftover tail is handled separately by the caller, so C does not have to
    divide n. Keeping both factors near sqrt(n) is what bounds the index vectors.
    """
    c = int(math.isqrt(n)) or 1
    return n // c, c


def _rowcol_eager(v: Tensor, R: int, C: int) -> tuple[Tensor, Tensor]:
    b = v[: R * C].view(R, C)
    return b.sum(dim=1, dtype=torch.int64), b.sum(dim=0, dtype=torch.int64)


# torch.compile fuses the two reductions into ONE kernel with two accumulators,
# so the data is read once instead of twice. Measured 3.06x on 10M elements, and
# bit-exact against eager. `.view(torch.int32)` is a same-width bitcast and
# lowers cleanly; the known inductor bitcast failures are different-width casts.
#
# dynamic=False specialises per shape. A model has on the order of tens of
# distinct tensor shapes, so that is tens of graphs compiled once, not hundreds.
_COMPILE_FAILED = False
try:  # pragma: no cover - depends on the installed torch build
    # A model has one distinct (R, C) grid per distinct tensor SIZE, and a real
    # model has more of those than dynamo's default recompile limit of 8. Past
    # the limit dynamo silently falls back to eager, which is how a first
    # attempt at this "compiled" and still ran at eager speed. Raise the limit
    # so every shape gets its own fused kernel; each is compiled once.
    import torch._dynamo

    if torch._dynamo.config.recompile_limit < 256:
        torch._dynamo.config.recompile_limit = 256
    _rowcol_compiled = torch.compile(_rowcol_eager, dynamic=False, fullgraph=True)
except Exception:  # inductor unavailable, unsupported backend, ...
    _rowcol_compiled = _rowcol_eager
    _COMPILE_FAILED = True


# Below this many elements, dynamo's guard checks cost more than the fused
# kernel saves. A real model is mostly SMALL tensors (bias vectors, layernorm
# gains) with a few large ones, and compiling everything is measurably WORSE
# than compiling nothing.
#
# Swept on an 85.2M-param model, 592 tracked tensors, per protected step:
#     compile nothing over 1M elems  141.2 ms
#     threshold 256K                 132.3 ms   <- best
#     compile everything             143.4 ms
#
# The gain from fusion here is modest (~7%) because this was measured on CPU,
# where the two reductions were already bandwidth-bound. The fusion should
# matter more on CUDA, where it also removes a kernel launch per tensor. That
# is unverified on real hardware -- see README.
_COMPILE_MIN_ELEMENTS = 1 << 18


def _rowcol(v: Tensor, R: int, C: int) -> tuple[Tensor, Tensor]:
    """Row and column sums in one pass, compiled when the backend allows it.

    Falls back to eager permanently on the first failure rather than retrying
    per tensor: a backend that cannot compile this will not start being able to,
    and paying the exception on every call would cost more than the fusion saves.
    """
    global _COMPILE_FAILED
    if not _COMPILE_FAILED and v.numel() >= _COMPILE_MIN_ELEMENTS:
        try:
            return _rowcol_compiled(v, R, C)
        except Exception:
            _COMPILE_FAILED = True
    return _rowcol_eager(v, R, C)


@torch.no_grad()
def row_sums(t: Tensor) -> tuple[Tensor, Tensor, int, int]:
    """Row sums of `t` reshaped to a near-square grid. Returns (rows, tail, R, C).

    The clean-path primitive. One reduction over the data, and the result IS the
    detector: comparing this vector against its snapshot finds any corruption,
    because a changed element changes its row's sum by exactly the delta.

    Column sums are deliberately NOT computed here. They are only needed to
    LOCATE a fault, and faults are rare -- at orbital rates roughly one per
    thousand chip-hours. Computing them every step would double the steady-state
    cost to buy nothing. This is the FT-BLAS pattern (Zhai et al., ICS '21):
    "If an error is detected, we continue to compute the reference column
    checksum ... to locate the erroneous row index. If there is no error
    detected when comparing the row checksum vectors, we do not need to verify
    the column checksum vectors."
    """
    int_dtype = _INT_VIEW.get(t.dtype)
    if int_dtype is None:
        raise TypeError(f"no integer view for dtype {t.dtype}")
    flat = (
        t.detach().reshape(-1)
        if t.is_contiguous()
        else t.detach().contiguous().reshape(-1)
    )
    v = flat.view(int_dtype)
    n = v.numel()
    if n == 0:
        z = torch.zeros(0, dtype=torch.int64, device=t.device)
        return z, z, 0, 0
    R, C = _grid(n)
    rows = v[: R * C].view(R, C).sum(dim=1, dtype=torch.int64)
    tail = v[R * C :].to(torch.int64)
    return rows, tail, R, C


@torch.no_grad()
def col_sums(t: Tensor, R: int, C: int) -> Tensor:
    """Column sums for the same grid. Error path only."""
    int_dtype = _INT_VIEW[t.dtype]
    flat = (
        t.detach().reshape(-1)
        if t.is_contiguous()
        else t.detach().contiguous().reshape(-1)
    )
    return flat.view(int_dtype)[: R * C].view(R, C).sum(dim=0, dtype=torch.int64)


@torch.no_grad()
def locate(
    t: Tensor, row_delta: Tensor, trusted_cols: Tensor, R: int, C: int
) -> tuple[int, int] | None:
    """Locate a single corrupted element by row/column intersection.

    Returns (flat_index, delta), or None when the fault is not a single element.

    Row `r` and the magnitude `delta` come from the row-sum comparison that
    already fired. One column reduction then gives `c`, and the corrupted
    element sits at their intersection: `index = r*C + c`.

    This is Huang & Abraham's full checksum matrix (1984). It is preferred over
    the `index = d2/d1` weighted form because there is no division and no
    rounding, so it cannot mis-locate when the corrupted value is INF or NaN --
    a failure mode ATTNChecker documents for the weighted construction.

    None when more than one row or column disagrees. That is the multi-element
    case (a nullification tile, a warp-aligned track), where the intersection is
    ambiguous and the honest answer is to escalate rather than guess.
    """
    bad_rows = torch.nonzero(row_delta, as_tuple=False).flatten()
    if bad_rows.numel() != 1:
        return None
    r = int(bad_rows[0].item())
    delta = int(row_delta[r].item())

    col_delta = col_sums(t, R, C) - trusted_cols
    bad_cols = torch.nonzero(col_delta, as_tuple=False).flatten()
    if bad_cols.numel() != 1:
        return None
    c = int(bad_cols[0].item())

    # Consistency check: a genuine single-element fault moves its row and its
    # column by the SAME amount. If they disagree, the intersection model does
    # not describe what happened and repairing would corrupt the tensor further.
    if int(col_delta[c].item()) != delta:
        return None
    return r * C + c, delta


@torch.no_grad()
def _checksum_pair(t: Tensor) -> tuple[Tensor, Tensor]:
    """Both checksums from ONE structured pass. Returns (plain, weighted).

    This is the function the whole tier is built on. See
    `weighted_checksum_device` for why it is a 2D decomposition rather than the
    obvious `sum(arange(n) * x)`.
    """
    int_dtype = _INT_VIEW.get(t.dtype)
    if int_dtype is None:
        raise TypeError(f"no integer view for dtype {t.dtype}")
    flat = (
        t.detach().reshape(-1)
        if t.is_contiguous()
        else t.detach().contiguous().reshape(-1)
    )
    v = flat.view(int_dtype)
    n = v.numel()
    if n == 0:
        z = torch.zeros((), dtype=torch.int64, device=t.device)
        return z, z.clone()

    R, C = _grid(n)
    body = v[: R * C].view(R, C)
    rows = body.sum(dim=1, dtype=torch.int64)  # R values
    cols = body.sum(dim=0, dtype=torch.int64)  # C values

    plain = rows.sum()
    weighted = C * (_arange_cached(R, t.device) * rows).sum() + (
        _arange_cached(C, t.device) * cols
    ).sum()

    # Tail elements that did not fit the grid. Short by construction (< C, so
    # ~sqrt(n)), so a direct weighted sum over them costs nothing.
    tail = v[R * C :]
    if tail.numel():
        tv = tail.to(torch.int64)
        tidx = _arange_cached(tail.numel(), t.device) + (R * C)
        plain = plain + tv.sum()
        weighted = weighted + (tidx * tv).sum()

    return plain, weighted


def weighted_checksum_device(t: Tensor) -> Tensor:
    """Index-weighted companion checksum: sum(i * x_int[i]), on device.

    This is what makes REPAIR possible instead of rollback. With two checksums
    over the same tensor, a single-element corruption is not just detected, it
    is LOCATED and INVERTED:

        d1 = sum(after) - sum(before)            = delta
        d2 = sum(i*after) - sum(i*before)        = index * delta
        index = d2 / d1

    Then subtracting `delta` from element `index` in integer space restores the
    tensor bit-for-bit. This is the classical algorithm-based fault tolerance
    result -- a checksum locates, a weighted checksum pinpoints -- applied to
    stored state rather than to a matrix product.

    The payoff is economic. Rollback costs a checkpoint restore plus every step
    since it, replayed under continuing radiation. On an L4 at 85M parameters
    that cost 76 replayed steps and the run still died "unrecoverable" at step
    33. Repair costs one subtraction.

    Falls back to rollback when the division does not yield a clean in-range
    integer, which is exactly the multi-element case where inversion is not
    determined.

    Implemented as a 2D decomposition, NOT as `sum(arange(n) * x)`.

    The naive form allocates three full-size int64 tensors per call -- the
    `.to(torch.int64)` cast, the index vector, and their product. At 85M
    elements that is ~2 GB of allocation and ~4.4 GB of memory traffic for a
    single checksum, which made this function 3x the cost of the plain one and
    the dominant term in protection overhead. It is the exact trap
    `checksum_device` documents, and the first version of this function walked
    straight into it.

    Reshaping to (R, C) removes the index vector entirely. A flat index is
    `i = r*C + c`, so:

        sum(i * x) = C * sum_r r*rowsum[r]  +  sum_c c*colsum[c]

    Row and column sums are ordinary reductions, and the two index vectors are
    only R and C long -- about sqrt(n) each, roughly 9,200 values instead of
    85,000,000. The plain checksum comes out of the same row sums for free.

    Overflow: int64 wraps here, and that is fine. Repair only ever uses
    DIFFERENCES of these sums, and two's-complement subtraction is exact
    modulo 2^64. For a single-element fault the true product stays far inside
    the range anyway (index < 2^27 and delta <= 2^31 give a product under 2^58).
    """
    s, w = _checksum_pair(t)
    return w


def _sync(values: list[Tensor]) -> list[int]:
    """Bring every checksum to the host in a SINGLE synchronisation.

    Writes each scalar into one preallocated buffer, then reads that buffer
    once. Deliberately avoids `torch.stack` AND `torch.cat`.

    Both crash the process on MPS (torch 2.13.0, macOS 26.5) when given a few
    hundred 0-dim tensors. Not an exception -- SIGSEGV inside
    `structured_cat_out_mps` -> `AGX::BlitDispatchContext::bindComputeResources`,
    i.e. Metal's per-encoder resource binding limit. `stack` is affected because
    it lowers to `cat`. A protected run died at step 1 with 208 tracked tensors.

    Index-assignment issues one small independent copy per scalar and binds no
    resource list, so it is safe on every backend. The single `.tolist()` at the
    end preserves the one-sync property that keeps this tier cheap.
    """
    if not values:
        return []
    buf = torch.empty(len(values), dtype=torch.int64, device=values[0].device)
    for i, v in enumerate(values):
        buf[i] = v
    return buf.tolist()


def checksum(t: Tensor) -> int:
    """Host-side convenience wrapper. Syncs -- do not call in a hot loop."""
    return int(checksum_device(t).item())


@dataclass
class IntegrityStats:
    tensors_tracked: int = 0
    checks: int = 0
    mismatches: int = 0
    bits_covered: int = 0
    # Corruptions detected and deliberately NOT escalated to a rollback
    # (below the severity threshold). Detected, logged, absorbed.
    benign: int = 0
    # Corruptions LOCATED and INVERTED exactly. Not misses, not rollbacks:
    # the tensor was restored bit-for-bit and the run continued.
    repaired: int = 0
    # Steps where the scrub cadence said no scan was due.
    steps_skipped: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "integrity_tensors_tracked": self.tensors_tracked,
            "integrity_checks": self.checks,
            "integrity_mismatches": self.mismatches,
            "integrity_benign": self.benign,
            "integrity_repaired": self.repaired,
            "integrity_steps_skipped": self.steps_skipped,
            "integrity_bits_covered": self.bits_covered,
        }


@dataclass
class IntegrityTier:
    """Verifies that resident state is bit-identical to its last snapshot.

    Lifecycle, and the order is load-bearing:

        optimizer.step()      state is written
        tier.refresh()        snapshot the NEW state as trusted
        ... radiation ...
        forward / backward
        tier.observe()        verify nothing moved since the snapshot

    `refresh()` must come immediately after `optimizer.step()`. Snapshotting
    any later would fold an already-landed fault into the baseline, which is
    the exact laundering failure this tier exists to prevent.
    """

    model: nn.Module
    optimizer: torch.optim.Optimizer | None = None
    track_optimizer_state: bool = True
    # Cover Linear weights too, despite ABFT also watching them. The two tiers
    # are complementary, not redundant:
    #   * ABFT samples (10% outside the SAA), so it leaves ~9% of weight
    #     corruption unseen even inside its own scope. This tier is exact and
    #     unsampled, so it closes that.
    #   * ABFT sees faults this tier cannot: a transient corruption of the GEMM
    #     ITSELF leaves the stored weights bit-identical, so a state checksum
    #     is blind to it while the checksum-vs-product comparison catches it.
    # Set True to skip them if the extra reduction ever shows up in a profile.
    skip_abft_covered: bool = False

    # --- Scrub cadence ------------------------------------------------- #
    # No production training system re-scans all state every step. Google,
    # ByteDance and Meta all detect on free scalars (loss, grad norm) and only
    # run checksums INSIDE a triggered replay; NVIDIA's cross-replica param hash
    # defaults to off and is documented for an interval of ~100 steps.
    #
    # We cannot adopt that trigger. It works terrestrially because SDC there is a
    # COMPUTE fault that perturbs metrics. In orbit the dominant threat is an SEU
    # in STORED state, and a mantissa-bit flip in an Adam moment moves no metric
    # at all -- that is what "silent" means, and catching it is the entire point
    # of this tier. A grad-norm trigger would systematically miss exactly the
    # fault class we exist to find.
    #
    # What DOES transfer is the cadence argument. Scanning every step when upsets
    # arrive every ~40 steps is ~40x more often than the physics requires. So the
    # scan interval is derived from the flux model, exactly as the checkpoint
    # policy is: scrub often where the upsets are, rarely where they are not.
    #
    # `flux` is optional. Without it the tier scans every step, which is the
    # conservative default and what every existing caller gets.
    flux: object | None = None
    seconds_per_step: float = 0.0
    # Target expected upsets between scrubs. Below 1 so a fault is typically
    # caught within one scrub interval of arriving.
    target_upsets_per_scan: float = 0.25
    max_scan_interval: int = 32
    # Track the index-weighted companion checksum so single-element faults
    # can be located and inverted instead of triggering a rollback. Costs
    # one extra reduction per tensor per step; saves a checkpoint restore
    # plus every step replayed since it.
    repair: bool = True

    stats: IntegrityStats = field(default_factory=IntegrityStats)
    _base: dict[str, '_Baseline'] = field(default_factory=dict)
    _in_saa: bool = False
    _last_scan_step: int = -(10**9)
    _steps_skipped: int = 0
    _last_repairs: list[dict] = field(default_factory=list)
    _last: list[dict] = field(default_factory=list)
    _pending: Verdict | None = None

    def __post_init__(self) -> None:
        self._abft_covered_ids = {
            id(m.weight) for m in self.model.modules() if isinstance(m, nn.Linear)
        }

    # ------------------------------------------------------------------ #

    def targets(self) -> list[tuple[str, Tensor]]:
        """Every tensor this tier is responsible for.

        Resolved lazily on each call, exactly as `MemoryInjector.targets()`
        does: Adam materialises `exp_avg`/`exp_avg_sq` only after the first
        `step()`, so a list captured at construction time would permanently
        exclude the largest target in the process.
        """
        out: list[tuple[str, Tensor]] = []
        for name, p in self.model.named_parameters():
            if self.skip_abft_covered and id(p) in self._abft_covered_ids:
                continue
            if p.dtype in _INT_VIEW:
                out.append((f"param:{name}", p))

        if self.track_optimizer_state and self.optimizer is not None:
            param_names = {id(p): n for n, p in self.model.named_parameters()}
            for group in self.optimizer.param_groups:
                for p in group["params"]:
                    pname = param_names.get(id(p), "?")
                    for key, tensor in self.optimizer.state.get(p, {}).items():
                        if torch.is_tensor(tensor) and tensor.dtype in _INT_VIEW:
                            out.append((f"opt:{pname}:{key}", tensor))
        return out

    @torch.no_grad()
    def refresh(self) -> None:
        """Snapshot every tracked tensor as trusted. Call after optimizer.step().

        Stores row sums (the detector) and column sums (the locator) for each
        tensor. Both are about sqrt(n) long, so the whole snapshot is a rounding
        error against the state it protects.
        """
        targets = self.targets()
        self._base = {}
        for name, t in targets:
            self._base[name] = self._snapshot(t)
        self.stats.tensors_tracked = len(targets)
        self.stats.bits_covered = sum(
            t.numel() * t.element_size() * 8 for _, t in targets
        )

    @torch.no_grad()
    def _snapshot(self, t: Tensor) -> "_Baseline":
        """Row and column sums for one tensor, from a single fused pass."""
        int_dtype = _INT_VIEW.get(t.dtype)
        if int_dtype is None:
            raise TypeError(f"no integer view for dtype {t.dtype}")
        flat = (
            t.detach().reshape(-1)
            if t.is_contiguous()
            else t.detach().contiguous().reshape(-1)
        )
        v = flat.view(int_dtype)
        n = v.numel()
        if n == 0:
            z = torch.zeros(0, dtype=torch.int64, device=t.device)
            return _Baseline(rows=z, cols=z.clone(), tail=z.clone(), R=0, C=0)
        R, C = _grid(n)
        rows, cols = _rowcol(v, R, C)
        return _Baseline(
            rows=rows,
            cols=cols if self.repair else None,
            tail=v[R * C :].to(torch.int64),
            R=R,
            C=C,
        )

    @torch.no_grad()
    def _repair_at(self, t: Tensor, idx: int, delta: int) -> None:
        """Subtract `delta` from element `idx` in integer space. Exact inverse."""
        int_dtype = _INT_VIEW[t.dtype]
        flat = (
            t.detach().reshape(-1)
            if t.is_contiguous()
            else t.detach().contiguous().reshape(-1)
        )
        view = flat.view(int_dtype)
        view[idx] = (view[idx].to(torch.int64) - delta).to(int_dtype)

    @torch.no_grad()
    def verify(self) -> list[dict]:
        """Recompute row sums, compare, and REPAIR what can be repaired.

        The clean path is ONE reduction per tensor plus a comparison over a
        sqrt(n)-length vector. Column sums -- needed only to locate a fault --
        are computed lazily, so a step with no corruption never pays for them.
        Faults are rare enough at orbital rates that this is nearly every step.

        Repair is attempted before severity is considered, because an exactly
        inverted fault leaves no residual error at all: strictly better than
        absorbing it and strictly cheaper than rolling back. Severity only
        decides what to do with what could NOT be repaired.
        """
        findings: list[dict] = []
        targets = self.targets()

        # Issue every row reduction, then sync ONCE on the small "did anything
        # move" flags. Only tensors that flag are examined further.
        computed = []
        flags = []
        for name, t in targets:
            base = self._base.get(name)
            rows, tail, R, C = row_sums(t)
            computed.append((name, t, rows, tail, R, C, base))
            if base is None:
                flags.append(torch.ones((), dtype=torch.int64, device=t.device))
                continue
            moved = (rows != base.rows).any().to(torch.int64)
            if tail.numel():
                moved = moved + (tail != base.tail).any().to(torch.int64)
            flags.append(moved)
        moved_flags = _sync(flags)  # the single sync on the clean path

        for (name, t, rows, tail, R, C, base), moved in zip(computed, moved_flags):
            self.stats.checks += 1
            if base is None:
                # First sight of a tensor -- optimizer state materialises after
                # the first step. Trust it now; nothing could have struck it
                # before it existed.
                self._base[name] = self._snapshot(t)
                continue
            if not moved:
                continue

            self.stats.mismatches += 1
            row_delta = rows - base.rows
            delta = int(row_delta.sum().item()) + (
                int((tail - base.tail).sum().item()) if tail.numel() else 0
            )
            bit = (
                (abs(delta).bit_length() - 1)
                if abs(delta) and (abs(delta) & (abs(delta) - 1)) == 0
                else None
            )

            found = None
            if self.repair and base.cols is not None and R and C:
                found = locate(t, row_delta, base.cols, R, C)

            if found is not None:
                idx, d = found
                self._repair_at(t, idx, d)
                self.stats.repaired += 1
                self._last_repairs.append(
                    {"tensor": name, "index": idx, "bit": bit, "delta": d}
                )
                continue  # baseline still valid: the tensor is bit-identical again

            findings.append(
                {
                    "tensor": name,
                    "delta": delta,
                    # A single flip moves the sum by exactly +/-2^k, so an
                    # exact power of two names the bit that moved.
                    "bit": bit,
                    "numel": t.numel(),
                    "dtype": str(t.dtype).replace("torch.", ""),
                }
            )
            # Re-trust so one persistent corruption does not re-fire every step.
            # Recovery rolls back; if it does not, we want the NEXT distinct
            # fault visible rather than buried under repeats of this one.
            self._base[name] = self._snapshot(t)
        self._last = findings
        return findings

    # ------------------------------------------------------------------ #

    def set_position(self, *, in_saa: bool) -> None:
        """Tell the tier where it is in the orbit. Mirrors AbftTier.set_position."""
        self._in_saa = in_saa

    def scan_interval(self) -> int:
        """Steps between full scrubs, derived from the upset rate.

        Returns 1 (scan every step) when no flux model is supplied, which keeps
        the conservative behaviour for callers that do not pass one.
        """
        if self.flux is None or self.seconds_per_step <= 0:
            return 1
        rate = (
            self.flux.saa_rate_per_s if self._in_saa else self.flux.quiescent_rate_per_s
        )
        if rate <= 0:
            return self.max_scan_interval
        per_step = rate * self.seconds_per_step
        if per_step <= 0:
            return self.max_scan_interval
        return max(1, min(int(self.target_upsets_per_scan / per_step), self.max_scan_interval))

    def check_now(self, step: int) -> Verdict:
        """Verify, and hold the verdict for the next `observe()`.

        MUST be called BEFORE `optimizer.step()`. The optimizer rewrites both
        the parameters and its own moment buffers, so a check placed after it
        compares legitimately-updated state against a stale snapshot and fires
        on every single step. That is not a theoretical risk: the first wiring
        of this tier ran after `optimizer.step()` and produced a 100% false
        positive rate (4 detections in 20 steps with zero upsets delivered).

        The protected window is therefore:

            optimizer.step()   state written
            refresh()          snapshot
            ... radiation ...
            forward / backward
            check_now()        <-- here, before the next step() moves anything
            optimizer.step()

        Anything landing between `check_now()` and `refresh()` is caught on the
        following iteration instead. No checksum scheme can close that window;
        it is one step wide and the rollback covers it.
        """
        if step - self._last_scan_step < self.scan_interval():
            # Between scrubs. A fault landing now is caught at the next one --
            # bounded by the interval, which is itself bounded by the upset rate.
            self.stats.steps_skipped += 1
            self._pending = None
            return Verdict(triggered=False, step=step)
        self._last_scan_step = step
        findings = self.verify()
        self._pending = self._verdict_from(findings, step)
        return self._pending

    def _is_severe(self, finding: dict) -> bool:
        """Does this corruption justify a rollback, or should it be logged only?

        See the severity policy at the top of this module. Detection is always
        recorded; only escalation is gated.
        """
        rel = _relative_change(finding.get("bit") if finding.get("bit") is not None else -1)
        is_optimizer = finding["tensor"].startswith("opt:")
        threshold = (
            OPTIMIZER_ROLLBACK_RELATIVE if is_optimizer else PARAM_ROLLBACK_RELATIVE
        )
        return rel >= threshold

    def _verdict_from(self, findings: list[dict], step: int) -> Verdict:
        if not findings:
            return Verdict(triggered=False, step=step)

        severe = [f for f in findings if self._is_severe(f)]
        self.stats.benign += len(findings) - len(severe)
        if not severe:
            # Real corruption, correctly detected, deliberately not escalated.
            # Recorded in `stats.benign` and in `_last` so a run can be audited
            # for what it chose to absorb -- this must never look like a miss.
            return Verdict(triggered=False, step=step)
        findings = severe
        worst = max(findings, key=lambda f: abs(f["delta"]))
        return Verdict(
            triggered=True,
            step=step,
            tier=TIER_INTEGRITY,
            reason=REASON_INTEGRITY_MISMATCH,
            evidence={
                "tensor": worst["tensor"],
                "bit": worst["bit"],
                "delta": worst["delta"],
                "dtype": worst["dtype"],
                "n_mismatched_tensors": len(findings),
            },
        )

    def observe(self, *, step: int, model: nn.Module | None = None, **_: object) -> Verdict:
        """Return the verdict from this step's `check_now()`, once.

        This tier does NOT verify here. Verification is time-critical and
        happens in `check_now()` before `optimizer.step()`; `observe()` only
        surfaces the result through the same path the other tiers use.
        """
        v, self._pending = self._pending, None
        return v if v is not None else Verdict(triggered=False, step=step)

    def reset(self) -> None:
        """After a rollback the snapshots describe state that no longer exists."""
        self._base.clear()
        self._last.clear()
        self._last_repairs.clear()
        self._pending = None
