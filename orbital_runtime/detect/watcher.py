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

import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Protocol

from ..inject.xid import XID_CONTAINED_ECC, XID_DBE, XidEvent, XidSimulator
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


# nvidia-smi's `ecc_errors/volatile` fields, grouped by driver severity.
# Correctable = single-bit; SEC-DED corrected it and the job is fine (log-only).
# Uncorrectable = the error defeated ECC; the affected computation is unsafe.
_ECC_CORRECTABLE_FIELDS = ("sram_correctable", "dram_correctable")
_ECC_UNCORRECTABLE_FIELDS = (
    "sram_uncorrectable_parity",
    "sram_uncorrectable_secded",
    "dram_uncorrectable",
)


@dataclass
class DcgmXidSource:
    """Real NVIDIA ECC/Xid source (M4b -- validated on an NVIDIA L4).

    Reads the device's *volatile* ECC error counters and surfaces any
    INCREASE since the last poll as a driver error report, so the watcher
    tier sees the same shape of events on hardware as it does from the
    simulator: an uncorrectable (double-bit) error becomes a FATAL report
    (the role Xid 48/95 play in the sim), a corrected single-bit error
    becomes non-fatal bookkeeping (Xid 94/63).

    Counters are read from `nvidia-smi -q -x` -- the same NVML data the
    driver exposes, and the counters DCGM surfaces as
    `DCGM_FI_DEV_ECC_{SBE,DBE}_VOL_TOTAL`. We parse nvidia-smi rather than
    link DCGM's Python bindings so the package keeps its two-dependency
    footprint (torch, numpy) and runs against any stock driver; on a host
    with `dcgmi`/`pydcgm` those fields are a drop-in replacement for
    `_read_counters`. dcgmi is not installed on the L4 DLAMI, so the
    nvidia-smi path is the one M4b exercises.

    Honesty invariant (carried from M2): if the source cannot actually
    observe the device it RAISES rather than returning `[]`. A watcher that
    reports "no errors" while it is blind makes an unmonitored run look
    healthy -- worse than no watcher at all.

    Fidelity note: with ECC ON these counts are real hardware ECC events.
    With ECC OFF a single-bit flip is corrected by nothing and reported by
    nothing -- the volatile counters stay 0 -- which is precisely the silent
    regime the application-layer tiers exist to cover. So on the demo's
    default (`ecc_off`) this source is correctly silent, exactly like the
    simulator.
    """

    device_index: int = 0
    _available: bool = field(default=False, init=False, repr=False)
    _baseline: tuple[int, int] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            self._baseline = self._read_counters()
            self._available = True
        except Exception:
            # No device, no driver, or ECC disabled/unreadable: stay blind and
            # loud -- `available` is False and `poll()` refuses.
            self._available = False
            self._baseline = None

    # ------------------------------------------------------------------ #
    # Counter access
    # ------------------------------------------------------------------ #

    def _query_xml(self) -> str:
        return subprocess.run(
            ["nvidia-smi", "-q", "-x", "-i", str(self.device_index)],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        ).stdout

    def _read_counters(self) -> tuple[int, int]:
        """`(corrected, uncorrectable)` volatile ECC totals for this GPU.

        Raises if nvidia-smi is absent, the GPU/ECC section is missing, or the
        counters read "N/A" (ECC disabled) -- every case where we cannot
        truthfully claim "no errors".
        """
        if shutil.which("nvidia-smi") is None:
            raise RuntimeError("nvidia-smi not found")
        root = ET.fromstring(self._query_xml())
        gpu = root.find("gpu")
        if gpu is None:
            raise RuntimeError("nvidia-smi reported no GPU")
        vol = gpu.find("./ecc_errors/volatile")
        if vol is None:
            raise RuntimeError("no volatile ECC section (driver too old, or ECC off)")

        def total(fields: tuple[str, ...]) -> int:
            s = 0
            for f in fields:
                node = vol.find(f)
                if node is None or node.text is None:
                    raise RuntimeError(f"ECC field {f} missing")
                text = node.text.strip()
                if text in ("N/A", "Disabled"):
                    raise RuntimeError("ECC disabled -- counters not meaningful")
                s += int(text)
            return s

        return total(_ECC_CORRECTABLE_FIELDS), total(_ECC_UNCORRECTABLE_FIELDS)

    # ------------------------------------------------------------------ #
    # XidSource protocol
    # ------------------------------------------------------------------ #

    def poll(self) -> list[XidEvent]:
        if not self._available or self._baseline is None:
            raise RuntimeError(
                "DcgmXidSource cannot see an NVIDIA device with readable ECC "
                "counters; use SimulatedXidSource in simulation. Refusing to "
                "report 'no errors' while blind."
            )
        corrected, uncorrectable = self._read_counters()
        base_c, base_u = self._baseline
        self._baseline = (corrected, uncorrectable)

        t = time.monotonic()
        events: list[XidEvent] = []
        # Uncorrectable first: these are the ones that must stop a run.
        for _ in range(max(0, uncorrectable - base_u)):
            events.append(
                XidEvent(
                    t=t,
                    code=XID_DBE,
                    detail="uncorrectable ECC error (volatile)",
                    fatal=True,
                )
            )
        for _ in range(max(0, corrected - base_c)):
            events.append(
                XidEvent(
                    t=t,
                    code=XID_CONTAINED_ECC,
                    detail="corrected single-bit ECC error (volatile)",
                    fatal=False,
                )
            )
        return events

    @property
    def available(self) -> bool:
        return self._available


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
