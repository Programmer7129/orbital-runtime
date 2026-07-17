"""Determinism guarantees for named RNG streams (PLAN.md design rule 3)."""

from __future__ import annotations

import subprocess
import sys

from orbital_runtime.rng import STREAM_FLUX, STREAM_MEMORY, stream, torch_seed


def test_same_seed_and_name_reproduces_draws():
    a = stream(42, STREAM_FLUX).random(100)
    b = stream(42, STREAM_FLUX).random(100)
    assert (a == b).all()


def test_different_names_are_independent_streams():
    a = stream(42, STREAM_FLUX).random(100)
    b = stream(42, STREAM_MEMORY).random(100)
    assert not (a == b).any()


def test_different_seeds_diverge():
    a = stream(1, STREAM_FLUX).random(100)
    b = stream(2, STREAM_FLUX).random(100)
    assert not (a == b).any()


def test_stream_is_addressed_by_name_not_creation_order():
    """The property that makes protected-vs-unprotected comparable.

    Turning a subsystem on must not perturb another subsystem's draws --
    otherwise enabling ABFT would shift the flip schedule and the two demo
    runs would no longer face the same radiation.
    """
    baseline = stream(7, STREAM_FLUX).random(50)

    # Simulate another subsystem coming into existence first.
    _ = stream(7, "detect.abft").random(10)
    _ = stream(7, "some.new.subsystem").random(10)

    assert (stream(7, STREAM_FLUX).random(50) == baseline).all()


def test_stream_names_are_stable_across_processes():
    """Guards against PYTHONHASHSEED salting sneaking into key derivation.

    `hash()` differs per process; BLAKE2b does not. Run in a subprocess with
    a different hash seed and require identical output.
    """
    code = (
        "from orbital_runtime.rng import stream;"
        "print(stream(42, 'flux').integers(0, 2**31).item())"
    )

    def run(hashseed: str) -> str:
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env={"PYTHONHASHSEED": hashseed, "PATH": "/usr/bin:/bin"},
        )
        return out.stdout.strip()

    assert run("0") == run("12345")


def test_torch_seed_is_deterministic_and_in_range():
    a = torch_seed(9, "workload")
    b = torch_seed(9, "workload")
    assert a == b
    assert 0 <= a < 2**63
    assert torch_seed(9, "workload") != torch_seed(10, "workload")
