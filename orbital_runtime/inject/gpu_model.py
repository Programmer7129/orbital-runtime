"""How a GPU actually corrupts: outcome distribution and fault geometry.

Source
------
Tung, Huang, Saxena, Shirvani, Hukerikar, Jain, Gongalore (NVIDIA), Tyagi
(Rochester), "The Anatomy of Silent Data Corruption: GPU Error Pattern Study
and Modeling Guidance", arXiv 2605.04213, DSN 2026. 3M+ simulator hours over 63
CUDA micro-benchmarks on a synthesized production-class datacenter GPU;
600 million corruptions across 25,000 SDC cases.

Why this module exists
----------------------
The injector modelled corruption as memory bit flips only. Against the measured
GPU distribution that is the minority case:

    nullification (value -> 0)   50.68%   <- a bit-flip model CANNOT produce it
    non-special bit flips        48.31%
    NaN / +-INF                   1.01%

A bit-flip in stored memory almost never yields exactly zero, so the single most
common GPU corruption outcome was unreachable by the old model. Worse, the
detection story led with the NaN guard, which addresses 1.01%.

Rate versus outcome -- the load-bearing distinction
---------------------------------------------------
This module governs WHAT a corruption looks like. It does NOT govern HOW OFTEN
one happens. Those are separate physics and must stay separate in the code:

  * OUTCOME is a property of the silicon -- how charge deposition or a logic
    fault propagates through the datapath. It is the same in orbit as on a
    bench, so NVIDIA's terrestrial measurement transfers.
  * RATE is a property of the environment. Ours stays calibrated to the orbital
    flux model in `orbit/flux.py` (Suncatcher: ~1 SDC per 17 rad, ~150 rad/yr
    in LEO). NVIDIA's terrestrial rate is defect-driven and does NOT transfer.

Do not let a terrestrial rate leak into the flux model, and do not let the
orbital rate change these shares.

Orbital applicability of the outcome shares
-------------------------------------------
Suncatcher (arXiv 2511.19468) irradiated a v6e TPU with 67 MeV protons and
found "Core logic and on-chip SRAM were the most SEE-sensitive components,
primarily manifesting as Silent Data Corruption", with "data mismatches
occurring without corresponding UECC flags". Logic and SRAM dominate in orbit
too, and those are the structures that produce nullification and tile-wide
corruption. So the mechanism mix transfers in kind. It is not measured for
orbit -- no public work reports an outcome histogram under a beam -- so treat
the shares as a cited terrestrial anchor, exactly as MICRO'21 is treated for
MBU clustering.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --- Outcome shares, Tung et al. Table/Section "corruption type distribution" #
# These are OUTCOMES OBSERVED, not mechanisms. The injector reproduces them by
# choosing a mechanism whose outcome falls in the right class (see below).
SHARE_NULLIFICATION = 0.5068
SHARE_NON_SPECIAL = 0.4831
SHARE_SPECIAL = 0.0101  # NaN / +-INF

# --- Bit-flip geometry, same source ------------------------------------------
# "Single-bit flips represent less than 40% of bit-flip events", against 72-98%
# in CPU studies. Multi-bit on GPUs is a concentrated secondary distribution,
# not a rare tail. The previous model used MICRO'21's 31.5% multi-bit share,
# measured on V100 HBM2 under neutrons -- a memory-array figure. This is the
# GPU-wide figure and is roughly double it.
GPU_SINGLE_BIT_SHARE = 0.40
GPU_MULTI_BIT_SHARE = 1.0 - GPU_SINGLE_BIT_SHARE

# "Flip probability decreases from least to most significant bits." Modelled as
# a geometric decay over bit position. The exact decay constant is not published,
# so this is a shape assumption calibrated to reproduce the ~1% special-value
# share on fp32 (only the top exponent bits produce NaN/Inf).
BIT_POSITION_DECAY = 0.92

# --- Spatial structure -------------------------------------------------------
# "warp-aligned spatial periodicity at multiple granularities (2, 4, 8,
# 16-element spacing)", following the 32-thread warp execution model.
# Corruption is structured, not independent per element.
WARP_SIZE = 32
WARP_STRIDES = (2, 4, 8, 16)
# Share of multi-element events that follow a warp-aligned stride rather than
# occupying a contiguous span. Not published as a number; the paper reports the
# periodicity exists at these granularities without a mixing ratio.
WARP_ALIGNED_SHARE = 0.5

# --- Control-logic (tile) faults ---------------------------------------------
# "Control logic faults cause catastrophic impacts, corrupting 20-75% of
# streaming multiprocessor output." One event, a large contiguous region.
TILE_FAULT_MIN_FRACTION = 0.20
TILE_FAULT_MAX_FRACTION = 0.75
# Share of nullification events that are tile-wide rather than a small span.
# Control logic is one of several nullification sources; data-buffer faults
# (L1, miss handlers) produce smaller footprints. Split evenly absent a
# published breakdown.
TILE_SHARE_OF_NULLIFICATION = 0.5
# Cap the modelled tile against the WHOLE tensor, since our targets are stored
# tensors rather than an SM output register file. Without a cap, a 75% tile on
# a 25M-element embedding is not a fault, it is deleting the model.
TILE_MAX_ELEMENTS = 4096

# Fault mechanism labels, recorded in telemetry so a run can be audited against
# the published distribution.
CLASS_NULLIFICATION = "nullification"
CLASS_BITFLIP = "bitflip"
CLASS_SPECIAL = "special"


@dataclass(frozen=True)
class FaultClass:
    """One drawn fault mechanism, with the geometry it implies."""

    label: str
    n_elements: int
    # Element stride: 1 = contiguous span, >1 = warp-aligned periodic.
    stride: int
    # For bitflip events only: how many bits within each struck element.
    n_bits: int
    # True when the mechanism must force a special value (NaN/Inf) rather than
    # letting the bit choice decide.
    force_special: bool = False


def sample_bit_position(rng: np.random.Generator, width: int) -> int:
    """Draw a bit position with probability decreasing LSB -> MSB.

    Tung et al.: "Flip probability decreases from least to most significant
    bits." A uniform draw over-weights the exponent, which is what made the old
    model produce far too many NaN/Inf outcomes relative to measurement.
    """
    weights = BIT_POSITION_DECAY ** np.arange(width, dtype=np.float64)
    weights /= weights.sum()
    return int(rng.choice(width, p=weights))


PATH_COMPUTE = "compute"
PATH_MEMORY = "memory"

# Nullification share on the MEMORY path. Tung et al.'s 50.68% is a COMPUTE
# measurement (fault injection into hardware units, observing SM output), so it
# does not transfer wholesale to stored state -- see `sample_fault_class`.
# A stored value can still be zeroed by a datapath fault on the write that
# produced it, which is a data-buffer effect rather than a control-logic one.
# Tung et al. give no stored-state split, so this is a modest modelled share and
# is flagged as an assumption, not a measurement.
MEMORY_NULLIFICATION_SHARE = 0.10
# Special values on the memory path come from bit choice, not from a forced
# mechanism: an exponent strike on a stored float genuinely does produce Inf.
MEMORY_SPECIAL_SHARE = 0.01


def sample_fault_class(
    rng: np.random.Generator, *, numel: int, path: str = PATH_COMPUTE
) -> FaultClass:
    """Draw a fault mechanism reproducing the measured outcome distribution.

    `numel` bounds the geometry: a span cannot exceed the tensor it lands in.

    `path` decides WHICH distribution applies, and the distinction is physical:

      * PATH_COMPUTE -- transient corruption of a value being computed
        (activations, GEMM output). This is what Tung et al. measured: they
        injected into control logic, data buffers and compute units and observed
        streaming-multiprocessor OUTPUT. The full distribution applies, tiles
        included, because a control-logic fault corrupts 20-75% of one SM's
        output in a single event.

      * PATH_MEMORY -- persistent corruption of stored state (parameters,
        optimizer moments). A control-logic tile fault CANNOT reach a tensor
        sitting in memory; it corrupts the datapath, not the array. Applying the
        compute distribution here produced a mean of ~597 zeroed elements per
        event, which is not a soft error, it is deleting the model. Stored state
        is governed by the memory-array literature (MICRO'21 MBU clustering on
        V100 HBM2 under neutrons), with a modest nullification share for
        write-path faults.

    Using the compute distribution for stored state was a category error in the
    first version of this module. Each source is now used for what it measured.
    """
    if path == PATH_MEMORY:
        return _sample_memory_class(rng, numel=numel)
    u = rng.random()

    # ---- nullification: 50.68% -------------------------------------------- #
    if u < SHARE_NULLIFICATION:
        if rng.random() < TILE_SHARE_OF_NULLIFICATION:
            frac = rng.uniform(TILE_FAULT_MIN_FRACTION, TILE_FAULT_MAX_FRACTION)
            n = int(min(max(1, frac * numel), TILE_MAX_ELEMENTS, numel))
            return FaultClass(CLASS_NULLIFICATION, n_elements=n, stride=1, n_bits=0)
        # Smaller data-buffer footprint: a warp-aligned or contiguous run.
        n = int(min(rng.integers(1, WARP_SIZE + 1), numel))
        stride = (
            int(rng.choice(WARP_STRIDES))
            if rng.random() < WARP_ALIGNED_SHARE
            else 1
        )
        return FaultClass(CLASS_NULLIFICATION, n_elements=n, stride=stride, n_bits=0)

    # ---- special values (NaN / +-Inf): 1.01% ------------------------------ #
    if u < SHARE_NULLIFICATION + SHARE_SPECIAL:
        return FaultClass(
            CLASS_SPECIAL, n_elements=1, stride=1, n_bits=1, force_special=True
        )

    # ---- non-special bit flips: 48.31% ------------------------------------ #
    multi = rng.random() >= GPU_SINGLE_BIT_SHARE
    n_bits = 1
    if multi:
        # Concentrated secondary distribution, not a long tail: 2-4 bits.
        n_bits = int(rng.integers(2, 5))
    # Bit flips can also span elements when a warp-aligned track is struck.
    if rng.random() < WARP_ALIGNED_SHARE:
        n = int(min(rng.integers(2, 9), numel))
        stride = int(rng.choice(WARP_STRIDES))
    else:
        n, stride = 1, 1
    return FaultClass(CLASS_BITFLIP, n_elements=n, stride=stride, n_bits=n_bits)


def _sample_memory_class(rng: np.random.Generator, *, numel: int) -> FaultClass:
    """Stored-state faults: bit-flip dominant, no control-logic tiles.

    Geometry follows the memory-array literature. An ionizing track deposits
    charge across adjacent CELLS, so a multi-bit event lands within one word or
    a short contiguous run of words -- never a warp-strided pattern, which is an
    artifact of how threads map onto compute units and has no meaning for a
    tensor at rest in DRAM.
    """
    u = rng.random()

    if u < MEMORY_NULLIFICATION_SHARE:
        # Write-path fault: a short contiguous run, capped well below a tile.
        n = int(min(rng.integers(1, 5), numel))
        return FaultClass(CLASS_NULLIFICATION, n_elements=n, stride=1, n_bits=0)

    if u < MEMORY_NULLIFICATION_SHARE + MEMORY_SPECIAL_SHARE:
        return FaultClass(
            CLASS_SPECIAL, n_elements=1, stride=1, n_bits=1, force_special=True
        )

    # MICRO'21 clustering: 31.5% of memory upset events are multi-bit. Kept at
    # the memory-measured value rather than the GPU-wide 60%, because that
    # figure aggregates compute-unit faults which do not apply here.
    multi = rng.random() < 0.315
    n_bits = 1 if not multi else int(min(1 + rng.geometric(0.6), 8))
    return FaultClass(CLASS_BITFLIP, n_elements=1, stride=1, n_bits=int(n_bits))


def expected_shares(path: str = PATH_COMPUTE) -> dict[str, float]:
    """The modelled distribution, for tests and for the audit report."""
    if path == PATH_MEMORY:
        return {
            CLASS_NULLIFICATION: MEMORY_NULLIFICATION_SHARE,
            CLASS_SPECIAL: MEMORY_SPECIAL_SHARE,
            CLASS_BITFLIP: 1.0 - MEMORY_NULLIFICATION_SHARE - MEMORY_SPECIAL_SHARE,
        }
    return {
        CLASS_NULLIFICATION: SHARE_NULLIFICATION,
        CLASS_BITFLIP: SHARE_NON_SPECIAL,
        CLASS_SPECIAL: SHARE_SPECIAL,
    }
