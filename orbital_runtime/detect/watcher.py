"""Tier 3 -- the ECC/Xid watcher.

Research doc SS3: "ECC/Xid log watcher (DCGM counters real on NVIDIA,
synthetic in sim)". PLAN.md: "No real DCGM integration until the cloud-GPU
week (M4) -- simulated Xid until then."

This module is an INTERFACE with two implementations. The consumer must not
know which side it is talking to, so that M4's swap to real DCGM is a
constructor change and nothing else. `SimulatedXidSource` reads the
synthetic stream from `inject/xid.py`; `DcgmXidSource` is the M4 shape,
and it fails loudly rather than silently reporting "no errors" on a machine
where it cannot actually look.

Honesty note carried from M1
----------------------------
The synthetic stream is CORRELATED WITH the injected flips, not derived
from simulated ECC hardware. So this tier's recall in simulation measures
plumbing, not ECC physics, and must be reported as such. It is also silent
by default in `ecc_off` mode -- which is the point: with ECC off, the
hardware tells you nothing, and application-layer detection is all you have.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..inject.xid import XidEvent, XidSimulator
from .verdict import NO_DETECTION, REASON_XID_FATAL, TIER_WATCHER, Verdict


class XidSource(Protocol):
    """Anything that can hand us pending driver error reports."""

    def poll(self) -> list[XidEvent]:
        """Return reports since the last poll (destructive)."""
        ...

    @property
    def available(self) -> bool:
        """False when this source cannot observe the device at all."""
        ...


@dataclass
class SimulatedXidSource:
    """Reads the synthetic stream produced by the injector."""

    simulator: XidSimulator

    def poll(self) -> list[XidEvent]:
        return self.simulator.drain()

    @property
    def available(self) -> bool:
        return True


@dataclass
class DcgmXidSource:
    """Real NVIDIA source. M4 -- deliberately not implemented here.

    PLAN.md forbids real DCGM integration before the cloud-GPU week, and
    there is no NVIDIA device on this Mac to develop it against. It raises
    rather than returning `[]`, because a watcher that silently reports
    "no errors" when it cannot see the device is worse than no watcher:
    it would make an unmonitored run look healthy.
    """

    def poll(self) -> list[XidEvent]:
        raise NotImplementedError(
            "DCGM/Xid polling lands in M4 on real NVIDIA hardware. "
            "Use SimulatedXidSource in simulation."
        )

    @property
    def available(self) -> bool:
        return False


@dataclass
class WatcherTier:
    """Escalates fatal driver reports into detections."""

    source: XidSource
    seen: list[XidEvent] = field(default_factory=list)
    fatal_only: bool = True

    def observe(self, *, step: int, **_: object) -> Verdict:
        events = self.source.poll()
        if not events:
            return NO_DETECTION
        self.seen.extend(events)

        actionable = [e for e in events if e.fatal] if self.fatal_only else events
        if not actionable:
            # Non-fatal reports are logged, not acted on: a corrected ECC
            # error means the hardware handled it. Rolling back on those
            # would burn the overhead budget on non-events.
            return NO_DETECTION

        worst = actionable[0]
        return Verdict(
            True,
            step,
            TIER_WATCHER,
            REASON_XID_FATAL,
            {"xid": worst.code, "detail": worst.detail, "n_events": len(actionable)},
        )

    @property
    def silent(self) -> bool:
        return not self.source.available

    def reset(self) -> None:
        self.seen = []
