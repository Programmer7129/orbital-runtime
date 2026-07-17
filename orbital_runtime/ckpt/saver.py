"""Checkpoint save/restore: model + optimizer + RNG + step.

Research doc SS4: "PyTorch Distributed Checkpoint (DCP) `async_save`
(PyTorch >=2.3), double-buffered to local NVMe: model + optimizer + RNG +
step. Overhead precedent: CheckFreq ~3.5% (FAST'21)."

Async, because a synchronous save stalls training for the whole write. DCP
stages the tensors and writes in the background, so the loop pays only the
staging copy. `async_save` works single-process (it warns that
`torch.distributed` is uninitialised and proceeds), which is what PLAN.md
wants -- no distributed support in the MVP.

Double buffering: two slots, alternating. The reason is not throughput, it
is that **the newest checkpoint is the one most likely to be untrustworthy**.
Corruption is detected some steps AFTER it lands, so the newest checkpoint
may already contain it; and a save interrupted by a crash leaves a torn
directory. Two slots mean there is always an older, independently verified
state to fall back to. One slot would mean a single bad save loses the run.

Verification
------------
Every checkpoint carries a checksum of its own tensors, computed at save
time from the staged values. `restore()` recomputes it after loading and
refuses to install a state that does not match. This catches corruption of
the checkpoint AT REST -- a checkpoint is resident bits like anything else,
and silently restoring a corrupted one would turn recovery into a way of
*spreading* the fault rather than undoing it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict_saver import async_save

# Number of checkpoint slots kept live. 2 = double-buffered (research doc SS4).
DEFAULT_BUFFERS = 2


def state_checksum(tensors: dict[str, torch.Tensor]) -> float:
    """A cheap order-independent checksum over tensor VALUES.

    Sum of fp64 sums, keyed by name. Not cryptographic -- it is here to
    catch bit flips, not adversaries. fp64 accumulation so that summing
    millions of fp32 values does not lose the low bits we are trying to
    protect, and so the sum of a corrupted exponent stands out hugely.

    NaN/Inf are mapped to sentinel values rather than poisoning the sum,
    so that a checkpoint containing them fails verification loudly instead
    of producing a NaN checksum that compares unequal to everything --
    including itself.

    Expects CPU tensors (see `_tensor_state`): float64 is unavailable on MPS.
    """
    total = 0.0
    for name, t in sorted(tensors.items()):
        if not t.is_floating_point():
            total += float(t.detach().to(torch.float64).sum().item())
            continue
        v = t.detach().to(torch.float64)
        finite = torch.isfinite(v)
        if not bool(finite.all()):
            # Distinct, deterministic contribution for a poisoned tensor.
            total += 1e300 + len(name)
            v = torch.where(finite, v, torch.zeros_like(v))
        total += float(v.sum().item())
    return total


@dataclass
class Checkpoint:
    """A saved, restorable training state."""

    step: int
    t_sim: float
    slot: int
    checksum: float
    path: Path
    # Exactly the tensor keys this checkpoint contains. Load must ask for
    # these and no others: a checkpoint taken at step 0 predates the
    # existence of Adam's state entirely, so a template built from the LIVE
    # (now warm) state would demand `opt.*` keys that were never written.
    keys: tuple[str, ...] = ()
    verified: bool = False
    wall_s: float = 0.0
    _future: Any = field(default=None, repr=False)

    def wait(self) -> Checkpoint:
        """Block until the async write has landed."""
        if self._future is not None:
            self._future.result()
            self._future = None
        return self

    @property
    def pending(self) -> bool:
        return self._future is not None


class CheckpointSaver:
    """Double-buffered async checkpointing for one training run."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        directory: Path | str,
        buffers: int = DEFAULT_BUFFERS,
        use_async: bool = True,
    ) -> None:
        if buffers < 1:
            raise ValueError(f"buffers must be >= 1, got {buffers}")
        self.model = model
        self.optimizer = optimizer
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.buffers = buffers
        self.use_async = use_async

        self.history: list[Checkpoint] = []
        self.saves = 0
        self.restores = 0
        self.rejected = 0
        self.save_wall_s = 0.0
        self._next_slot = 0

    # ------------------------------------------------------------------ #
    # State assembly
    # ------------------------------------------------------------------ #

    def _tensor_state(self) -> dict[str, torch.Tensor]:
        """Every tensor that must survive a rollback, flattened and named.

        Flat because DCP wants a tensor-valued mapping; named so the
        checksum is stable regardless of dict iteration order.

        **Staged to CPU.** Two reasons, one of them load-bearing:
        the checkpoint is bound for host storage anyway (research doc SS4:
        "double-buffered to local NVMe"), and `state_checksum` accumulates in
        float64 -- which MPS does not support at all. Keeping the staging
        buffer on-device would make checkpointing raise on the very machine
        this is developed on, and would violate PLAN.md design rule 1.
        Copying here also decouples the async write from the live tensors.
        """
        out: dict[str, torch.Tensor] = {}
        for name, p in self.model.named_parameters():
            out[f"model.{name}"] = p.detach().to("cpu", copy=True)
        for name, b in self.model.named_buffers():
            out[f"buffer.{name}"] = b.detach().to("cpu", copy=True)

        param_names = {id(p): n for n, p in self.model.named_parameters()}
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                st = self.optimizer.state.get(p)
                if not st:
                    continue
                pname = param_names.get(id(p), "unknown")
                for key, val in st.items():
                    if isinstance(val, torch.Tensor):
                        out[f"opt.{pname}.{key}"] = val.detach().to("cpu", copy=True)
        return out

    def _install_tensor_state(self, state: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                p.copy_(state[f"model.{name}"])
            for name, b in self.model.named_buffers():
                key = f"buffer.{name}"
                if key in state:
                    b.copy_(state[key])

            param_names = {id(p): n for n, p in self.model.named_parameters()}
            for group in self.optimizer.param_groups:
                for p in group["params"]:
                    st = self.optimizer.state.get(p)
                    if not st:
                        continue
                    pname = param_names.get(id(p), "unknown")
                    # Optimizer state the checkpoint predates must be
                    # DROPPED, not left behind. Restoring step-0 weights
                    # while keeping a warm (and possibly corrupted) Adam
                    # moment would resume from a state that never existed.
                    if not any(k.startswith(f"opt.{pname}.") for k in state):
                        self.optimizer.state.pop(p, None)
                        continue
                    for key, val in st.items():
                        if isinstance(val, torch.Tensor):
                            src = state.get(f"opt.{pname}.{key}")
                            if src is not None:
                                val.copy_(src)

    # ------------------------------------------------------------------ #
    # Save
    # ------------------------------------------------------------------ #

    def save(self, *, step: int, t_sim: float = 0.0) -> Checkpoint:
        """Snapshot the CURRENT state. Caller must ensure it is trusted.

        Like ABFT's checksum refresh, this must be called at a known-good
        moment (right after `optimizer.step()`, before radiation lands).
        Checkpointing corrupted state would persist the fault and make
        rollback useless -- it would restore the very thing we are escaping.
        """
        t0 = time.perf_counter()
        tensors = self._tensor_state()  # already a clone: staging is done
        checksum = state_checksum(tensors)

        slot = self._next_slot
        self._next_slot = (self._next_slot + 1) % self.buffers
        path = self.directory / f"slot{slot}"

        payload: dict[str, Any] = dict(tensors)
        payload["_meta.step"] = torch.tensor([step], dtype=torch.int64)
        payload["_meta.rng"] = torch.get_rng_state()

        future = None
        if self.use_async:
            future = async_save(payload, checkpoint_id=str(path))
        else:
            dcp.save(payload, checkpoint_id=str(path))

        ck = Checkpoint(
            step=step,
            t_sim=t_sim,
            slot=slot,
            checksum=checksum,
            path=path,
            keys=tuple(sorted(tensors)),
            verified=True,  # checksummed from staged values at a trusted moment
            wall_s=time.perf_counter() - t0,
            _future=future,
        )
        self.history.append(ck)
        if len(self.history) > self.buffers:
            self.history.pop(0)
        self.saves += 1
        self.save_wall_s += ck.wall_s
        return ck

    def wait_all(self) -> None:
        for ck in self.history:
            ck.wait()

    # ------------------------------------------------------------------ #
    # Restore
    # ------------------------------------------------------------------ #

    @property
    def latest(self) -> Checkpoint | None:
        return self.history[-1] if self.history else None

    def candidates(self, *, before_step: int | None = None) -> list[Checkpoint]:
        """Restorable checkpoints, newest first.

        `before_step` excludes checkpoints taken at or after the step where
        corruption is believed to have started -- those may already contain
        it, and restoring one would 'recover' straight back into the fault.
        """
        out = [
            c
            for c in self.history
            if before_step is None or c.step < before_step
        ]
        return sorted(out, key=lambda c: c.step, reverse=True)

    def restore(self, ck: Checkpoint) -> bool:
        """Load a checkpoint into the live model/optimizer.

        Returns False (installing nothing) if the checkpoint fails
        verification, so the caller can fall back to an older slot.
        """
        ck.wait()

        # Ask for exactly the keys this checkpoint holds -- no more (DCP
        # errors on a key it never wrote) and no fewer (a silently skipped
        # tensor would restore a torn state).
        live = self._tensor_state()
        missing = [k for k in ck.keys if k not in live]
        if missing:
            raise KeyError(
                f"checkpoint {ck.path} holds tensors absent from the live model: "
                f"{missing[:3]}"
            )
        template: dict[str, Any] = {k: live[k].clone() for k in ck.keys}
        template["_meta.step"] = torch.tensor([0], dtype=torch.int64)
        template["_meta.rng"] = torch.get_rng_state()
        dcp.load(template, checkpoint_id=str(ck.path))

        tensors = {k: v for k, v in template.items() if not k.startswith("_meta.")}
        if state_checksum(tensors) != ck.checksum:
            self.rejected += 1
            return False

        self._install_tensor_state(tensors)
        torch.set_rng_state(template["_meta.rng"].to(torch.uint8).cpu())
        self.restores += 1
        return True

    def stats(self) -> dict[str, Any]:
        return {
            "checkpoints_saved": self.saves,
            "checkpoints_restored": self.restores,
            "checkpoints_rejected": self.rejected,
            "checkpoint_wall_s": round(self.save_wall_s, 4),
        }
