"""Memory SEU injector: bit-flip mechanics, targeting, determinism."""

from __future__ import annotations

import struct

import numpy as np
import pytest
import torch

from orbital_runtime.inject.memory import (
    KIND_OPTIMIZER,
    KIND_PARAM,
    MemoryInjector,
    bits_of,
    flip_bit,
)
from orbital_runtime.rng import STREAM_MEMORY, stream


# --------------------------------------------------------------------- #
# flip_bit mechanics -- the ~5 lines everything else rests on
# --------------------------------------------------------------------- #


def test_flip_is_a_single_bit_xor_in_ieee754(device):
    """The flipped value must equal the IEEE-754 bit pattern XOR (1<<k).

    Checked against `struct`, independently of torch, for every bit of an
    fp32 -- sign, all 8 exponent bits, all 23 mantissa bits.
    """
    original = 0.15625  # exactly representable; no rounding to confuse us
    for bit in range(32):
        t = torch.tensor([original], dtype=torch.float32, device=device)
        before, after = flip_bit(t, 0, bit)
        assert before == original

        raw = struct.unpack("<I", struct.pack("<f", original))[0]
        expected = struct.unpack("<f", struct.pack("<I", raw ^ (1 << bit)))[0]

        if np.isnan(expected):
            assert np.isnan(after)
        else:
            assert after == pytest.approx(expected, rel=0, abs=0)
        assert float(t[0].item()) == pytest.approx(after, rel=0, abs=0) or np.isnan(after)


def test_flip_is_in_place_on_the_real_tensor(device):
    """A flip must mutate the tensor itself, not a copy.

    The bug this guards against is silent and fatal: an injector that
    corrupts a detached copy delivers a beautiful telemetry stream while the
    model trains on undamaged weights, and the whole demo becomes a lie.
    """
    t = torch.ones(16, dtype=torch.float32, device=device)
    _, after = flip_bit(t, 5, 31)  # sign bit: 1.0 -> -1.0
    assert after == -1.0
    assert float(t[5].item()) == -1.0
    # Neighbours untouched.
    assert torch.all(t[torch.arange(16) != 5] == 1.0)


def test_sign_bit_flip_negates(device):
    t = torch.tensor([3.25, -7.5], dtype=torch.float32, device=device)
    _, a = flip_bit(t, 0, 31)
    assert a == -3.25
    _, b = flip_bit(t, 1, 31)
    assert b == 7.5


def test_exponent_msb_flip_explodes_a_small_weight(device):
    """Why failure mode (a) happens.

    fp32 bit 30 is the exponent MSB. A typical weight is < 2, so its
    exponent MSB is 0; setting it multiplies the value by 2**128. This is
    the single-bit event that turns a healthy weight into ~1e38 and takes
    the run out.
    """
    t = torch.tensor([0.01], dtype=torch.float32, device=device)
    before, after = flip_bit(t, 0, 30)
    assert abs(after) > 1e30
    assert np.isfinite(after)  # still FINITE -- the NaN arrives downstream
    assert after == pytest.approx(before * 2.0**128, rel=1e-6)


def test_low_mantissa_flip_is_nearly_invisible(device):
    """Why failure mode (b) happens: the same model, a different bit."""
    t = torch.tensor([0.01], dtype=torch.float32, device=device)
    before, after = flip_bit(t, 0, 0)  # mantissa LSB
    assert after != before
    assert abs(after - before) / abs(before) < 1e-6


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_flip_works_for_every_supported_dtype(device, dtype):
    """PLAN.md rule 1: device-agnostic, and format-agnostic across CPU/MPS."""
    t = torch.full((8,), 1.0, dtype=dtype, device=device)
    width = t.element_size() * 8
    _, after = flip_bit(t, 3, width - 1)  # sign bit
    assert after == -1.0


def test_flip_round_trips_to_the_original():
    """XOR is an involution: flipping twice restores the value exactly."""
    t = torch.tensor([0.123456], dtype=torch.float32)
    original = float(t[0].item())
    for bit in (0, 12, 23, 30, 31):
        flip_bit(t, 0, bit)
        flip_bit(t, 0, bit)
        assert float(t[0].item()) == original


def test_flip_rejects_bad_arguments():
    t = torch.ones(4, dtype=torch.float32)
    with pytest.raises(ValueError, match="out of range"):
        flip_bit(t, 0, 32)
    with pytest.raises(ValueError, match="out of range"):
        flip_bit(t, 0, -1)
    with pytest.raises(IndexError):
        flip_bit(t, 4, 0)
    with pytest.raises(TypeError, match="no integer view"):
        flip_bit(torch.ones(4, dtype=torch.int8), 0, 0)


def test_flip_rejects_non_contiguous():
    """Silently addressing the wrong memory would be worse than failing."""
    t = torch.ones(4, 4, dtype=torch.float32)
    with pytest.raises(ValueError, match="non-contiguous"):
        flip_bit(t.t(), 0, 0)


def test_bits_of_counts_resident_bits():
    assert bits_of(torch.ones(10, dtype=torch.float32)) == 320
    assert bits_of(torch.ones(10, dtype=torch.float16)) == 160
    assert bits_of(torch.ones(3, 4, dtype=torch.float64)) == 768


# --------------------------------------------------------------------- #
# Targeting
# --------------------------------------------------------------------- #


def test_targets_include_params_only_before_first_step(tiny_workload):
    w = tiny_workload()
    inj = MemoryInjector(w.model, w.optimizer)
    kinds = {t.kind for t in inj.targets()}
    assert kinds == {KIND_PARAM}


def test_targets_include_optimizer_state_after_first_step(stepped_workload):
    """Lazy target resolution: optimizer state must become strikeable.

    Resolving targets once at construction would make Adam's state -- two
    thirds of resident memory -- permanently immune.
    """
    w = stepped_workload
    inj = MemoryInjector(w.model, w.optimizer)
    targets = inj.targets()
    kinds = {t.kind for t in targets}
    assert kinds == {KIND_PARAM, KIND_OPTIMIZER}
    names = {t.name for t in targets if t.kind == KIND_OPTIMIZER}
    assert any(n.endswith("exp_avg") for n in names)
    assert any(n.endswith("exp_avg_sq") for n in names)


def test_static_resident_bits_measures_state_exactly_once_warm(stepped_workload):
    """Once state exists it is measured, not guessed."""
    w = stepped_workload
    inj = MemoryInjector(w.model, w.optimizer)
    assert inj.static_resident_bits() == inj.resident_bits()


def test_static_resident_bits_before_any_step_infers_adamw(tiny_workload, stepped_workload):
    """The pre-step estimate drives lambda, so it must be nearly exact.

    lambda is fixed before step 0 (determinism), when optimizer state does
    not exist yet -- so this inference IS the resident-bit count the physics
    uses. AdamW keeps exp_avg + exp_avg_sq: 3x parameter bits.

    The guard here is against counting state TENSORS instead of state BITS:
    Adam also keeps a 0-dim `step` scalar per param, which is 32 bits but
    would read as a third full state and set lambda 33% high.
    """
    cold = MemoryInjector(tiny_workload().model, tiny_workload().optimizer)
    param_bits = sum(bits_of(p) for p in cold.model.parameters())
    assert cold.static_resident_bits() == 3 * param_bits

    # And the cold inference must match warm reality to within the step
    # scalars it deliberately ignores (<0.1%).
    w = stepped_workload
    warm = MemoryInjector(w.model, w.optimizer).resident_bits()
    warm_params = sum(bits_of(p) for p in w.model.parameters())
    assert 3 * warm_params == pytest.approx(warm, rel=0.001)


def test_sgd_without_momentum_has_no_optimizer_state(tiny_workload):
    w = tiny_workload()
    sgd = torch.optim.SGD(w.model.parameters(), lr=0.1)
    inj = MemoryInjector(w.model, sgd)
    param_bits = sum(bits_of(p) for p in w.model.parameters())
    assert inj.static_resident_bits() == param_bits


def test_optimizer_state_can_be_excluded(stepped_workload):
    w = stepped_workload
    inj = MemoryInjector(w.model, w.optimizer, target_optimizer_state=False)
    assert {t.kind for t in inj.targets()} == {KIND_PARAM}


def test_targeting_is_proportional_to_bit_count(stepped_workload):
    """Every resident bit equally likely -- the physical model.

    A tensor holding 10% of the bits must take ~10% of the strikes.
    """
    w = stepped_workload
    inj = MemoryInjector(w.model, w.optimizer)
    targets = {t.name: t.bits for t in inj.targets()}
    total_bits = sum(targets.values())

    rng = stream(99, STREAM_MEMORY)
    counts: dict[str, int] = {}
    n = 6000
    for _ in range(n):
        f = inj.inject(rng)
        assert f is not None
        counts[f.name] = counts.get(f.name, 0) + 1

    # The largest tensor dominates; check it lands within Poisson noise.
    biggest = max(targets, key=lambda k: targets[k])
    expected = n * targets[biggest] / total_bits
    assert abs(counts[biggest] - expected) < 4 * np.sqrt(expected)


def test_bit_positions_are_uniform(stepped_workload):
    """Uniform over bits -- NOT biased toward the exponent.

    A biased injector would make the demo dishonest in either direction:
    exponent-biased exaggerates the failure rate, mantissa-biased hides it.
    """
    w = stepped_workload
    inj = MemoryInjector(w.model, w.optimizer)
    rng = stream(7, STREAM_MEMORY)

    n = 8000
    hist = np.zeros(32, dtype=int)
    for _ in range(n):
        f = inj.inject(rng)
        assert f is not None
        hist[f.bit] += 1

    expected = n / 32
    chi2 = float(((hist - expected) ** 2 / expected).sum())
    # 31 dof, 99.9th percentile ~ 62.5. Seeded, so this cannot flake.
    assert chi2 < 62.5, f"bit histogram not uniform: chi2={chi2}"
    assert hist.min() > 0


def test_inject_returns_none_with_no_targets():
    """An empty model has nothing to strike; must not raise."""
    empty = torch.nn.Module()
    inj = MemoryInjector(empty, None)
    assert inj.targets() == []
    assert inj.inject(stream(0, STREAM_MEMORY)) is None


# --------------------------------------------------------------------- #
# Determinism (PLAN.md design rule 3)
# --------------------------------------------------------------------- #


def test_same_seed_reproduces_the_same_flips(tiny_workload):
    def run() -> list[tuple]:
        w = tiny_workload()
        inj = MemoryInjector(w.model, w.optimizer)
        rng = stream(4242, STREAM_MEMORY)
        return [
            (f.name, f.index, f.bit, f.value_before, f.value_after)
            for f in (inj.inject(rng) for _ in range(40))
            if f is not None
        ]

    assert run() == run()


def test_flip_record_captures_before_and_after(stepped_workload):
    w = stepped_workload
    inj = MemoryInjector(w.model, w.optimizer)
    f = inj.inject(stream(1, STREAM_MEMORY))
    assert f is not None
    rec = f.as_record()
    assert rec["value_before"] != rec["value_after"]
    assert set(rec) >= {"name", "target_kind", "index", "bit", "dtype", "nonfinite"}
