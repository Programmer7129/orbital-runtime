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

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

from .gpu_model import (
    CLASS_NULLIFICATION,
    CLASS_SPECIAL,
    PATH_COMPUTE,
    sample_bit_position,
    sample_fault_class,
)
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
    # Measured GPU mechanism (gpu_model.CLASS_*) and its geometry.
    fault_class: str = ""
    n_elements: int = 1
    stride: int = 1

    def as_record(self) -> dict:
        return {
            "module": self.module,
            "index": self.index,
            "bit": self.bit,
            "dtype": self.dtype,
            "value_before": self.value_before,
            "value_after": self.value_after,
            "shape": list(self.shape),
            "fault_class": self.fault_class,
            "n_elements": self.n_elements,
            "stride": self.stride,
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
        # Test hook: pin the drawn mechanism instead of sampling it, so a
        # test about masking or absorption is not also a lottery on the
        # fault-class distribution. None = sample normally.
        self.force_fault_class = None

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
            width = out.element_size() * 8
            numel = out.numel()

            # THIS is the path Tung et al. characterised: they injected into
            # control logic, data buffers and compute units, and observed
            # streaming-multiprocessor OUTPUT. An activation is exactly that --
            # a value in flight, not a tensor at rest -- so the full measured
            # distribution applies here, tiles and warp-aligned tracks included.
            # The stored-state path deliberately uses a different model; see
            # `gpu_model.sample_fault_class`.
            fc = self.force_fault_class or sample_fault_class(
                self._rng, numel=numel, path=PATH_COMPUTE
            )
            start = int(self._rng.integers(0, numel))
            indices = [
                start + i * fc.stride
                for i in range(fc.n_elements)
                if 0 <= start + i * fc.stride < numel
            ]
            if not indices:
                indices = [start]

            # Detach: the corruption is a hardware event, not a differentiable
            # op. Autograd must see the corrupted VALUE flow forward, but must
            # not try to backprop through the XOR itself.
            corrupted = out.detach().clone()
            flat = corrupted.reshape(-1)
            before = float(flat[indices[0]].item())

            if fc.label == CLASS_NULLIFICATION:
                # Nullification is 50.68% of measured GPU SDCs and was
                # unreachable by a bit-flip model: zeroing a float is not a
                # single-bit toggle.
                with torch.no_grad():
                    for i in indices:
                        flat[i] = 0.0
                bit = -1
            elif fc.label == CLASS_SPECIAL:
                with torch.no_grad():
                    flat[indices[0]] = (
                        math.inf if self._rng.random() < 0.5 else math.nan
                    )
                bit = -1
            else:
                bit = sample_bit_position(self._rng, width)
                for i in indices:
                    for _ in range(fc.n_bits):
                        b = sample_bit_position(self._rng, width)
                        flip_bit(corrupted, i, b)

            after = float(flat[indices[0]].item())
            self.hits.append(
                ActivationHit(
                    module=name,
                    index=indices[0],
                    bit=bit,
                    dtype=str(out.dtype).replace("torch.", ""),
                    value_before=before,
                    value_after=after,
                    shape=tuple(out.shape),
                    fault_class=fc.label,
                    n_elements=len(indices),
                    stride=fc.stride,
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
