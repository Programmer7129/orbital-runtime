"""Compute SEUs: forward hooks corrupting random activation elements.

Research doc SS1: "Compute SEUs: forward hooks corrupting random activation
elements." This is the PyTorchFI-style channel (approach vendored, not the
dependency -- PyTorchFI is semi-dormant, research doc SS1).

Distinct from `memory.py` in what it models: a memory SEU corrupts
PERSISTENT state (a weight stays wrong forever, and the error compounds
through every subsequent step), whereas a compute SEU corrupts a TRANSIENT
value mid-forward -- it pollutes one step's loss and gradients and is then
gone. Both appear in the demo; they fail differently, and the detectors see
them differently.

Off by default: PLAN.md lists activations as an optional target, and the
resident-bit accounting in `flux.py` covers params + optimizer state.
Enabling this channel adds a separate activation-upset rate rather than
re-slicing the memory rate, so the two are independently sweepable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

from .memory import _INT_VIEW_DTYPE, flip_bit


@dataclass(frozen=True)
class ActivationHit:
    """A record of one corrupted activation element."""

    module: str
    index: int
    bit: int
    dtype: str
    value_before: float
    value_after: float
    shape: tuple[int, ...]

    def as_record(self) -> dict:
        return {
            "module": self.module,
            "index": self.index,
            "bit": self.bit,
            "dtype": self.dtype,
            "value_before": self.value_before,
            "value_after": self.value_after,
            "shape": list(self.shape),
        }


class ComputeInjector:
    """Corrupts activations at the output of selected modules.

    Usage: `arm()` requests that the NEXT forward pass through a hooked
    module take a hit; the hook fires it and disarms. Arming is decoupled
    from firing because the upset schedule is in simulation time, but
    activations only exist during a forward pass -- so an upset that lands
    between steps waits for the next forward rather than being dropped.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        module_types: tuple[type, ...] = (nn.Linear,),
    ) -> None:
        self.model = model
        self.module_types = module_types
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._armed: int = 0
        self._rng: np.random.Generator | None = None
        self.hits: list[ActivationHit] = []
        self._hooked_names: list[str] = []

    # ------------------------------------------------------------------ #
    # Hook lifecycle
    # ------------------------------------------------------------------ #

    def attach(self) -> ComputeInjector:
        if self._handles:
            return self  # idempotent
        for name, mod in self.model.named_modules():
            if isinstance(mod, self.module_types):
                self._handles.append(
                    mod.register_forward_hook(self._make_hook(name))
                )
                self._hooked_names.append(name)
        return self

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._hooked_names.clear()

    @property
    def hooked_modules(self) -> list[str]:
        return list(self._hooked_names)

    def __enter__(self) -> ComputeInjector:
        return self.attach()

    def __exit__(self, *exc: object) -> None:
        self.detach()

    # ------------------------------------------------------------------ #
    # Arming / firing
    # ------------------------------------------------------------------ #

    def arm(self, rng: np.random.Generator, n: int = 1) -> None:
        """Request `n` corruptions on upcoming forward passes."""
        self._rng = rng
        self._armed += n

    @property
    def armed(self) -> int:
        return self._armed

    def _make_hook(self, name: str):
        def hook(_module: nn.Module, _inputs: tuple, output: Tensor):
            if self._armed <= 0 or self._rng is None:
                return output
            if not isinstance(output, Tensor):
                return output
            if output.dtype not in _INT_VIEW_DTYPE or output.numel() == 0:
                return output

            # Fire on this module with probability 1/remaining_modules would
            # bias toward early modules; instead corrupt the first eligible
            # forward. The module CHOICE is therefore execution-order driven,
            # which mirrors reality: a particle strikes whatever is in the
            # ALU/registers at that instant, not a uniformly-chosen layer.
            self._armed -= 1
            out = output if output.is_contiguous() else output.contiguous()
            index = int(self._rng.integers(0, out.numel()))
            width = out.element_size() * 8
            bit = int(self._rng.integers(0, width))

            # Detach: the corruption is a hardware event, not a differentiable
            # op. Autograd must see the corrupted VALUE flow forward, but must
            # not try to backprop through the XOR itself.
            corrupted = out.detach().clone()
            before, after = flip_bit(corrupted, index, bit)
            self.hits.append(
                ActivationHit(
                    module=name,
                    index=index,
                    bit=bit,
                    dtype=str(out.dtype).replace("torch.", ""),
                    value_before=before,
                    value_after=after,
                    shape=tuple(out.shape),
                )
            )
            # Re-attach to the graph: gradients flow through the delta as a
            # constant offset, so the corrupted value propagates to the loss
            # and to every downstream gradient, exactly as a real SEU would.
            return output + (corrupted - out.detach())

        return hook

    def drain_hits(self) -> list[ActivationHit]:
        hits, self.hits = self.hits, []
        return hits
