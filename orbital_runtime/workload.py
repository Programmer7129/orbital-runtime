"""Workload interface + registry.

A workload is whatever the runtime is keeping alive. The runtime must not
know or care what it is -- that is the product thesis (research doc SS5: "we
are a general-purpose runtime for unmodified PyTorch training + inference",
as opposed to RedNet's model-specific per-layer protection).

So the contract is deliberately tiny: give us a model, an optimizer, and a
way to compute a loss for a step index. Everything the runtime does --
injection targeting, detection, checkpointing, recovery -- works off that
alone.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class Workload(Protocol):
    """The surface the runtime needs from any workload."""

    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    device: torch.device

    def loss_for_step(self, step: int) -> torch.Tensor:
        """Compute the training loss for `step`.

        Must be deterministic in `step` (same seed => same batch), so that a
        rollback to step N and a replay from step N sees byte-identical
        data. Recovery correctness depends on this.
        """
        ...

    def evaluate(self, n_batches: int = 8) -> float:
        """Mean held-out loss. Used to detect silent divergence."""
        ...


def get_workload(name: str, **kwargs) -> Workload:
    """Build a workload by name."""
    if name == "nanogpt":
        from demo.workloads.nanogpt import build_nanogpt

        return build_nanogpt(**kwargs)
    raise ValueError(f"unknown workload {name!r} (known: nanogpt)")


def resolve_device(spec: str = "auto") -> torch.device:
    """Pick a device. CPU/MPS only -- no CUDA on the dev Mac.

    PLAN.md design rule 1: device-agnostic core. CUDA is selectable for the
    M4 cloud-GPU run but is never required.
    """
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
