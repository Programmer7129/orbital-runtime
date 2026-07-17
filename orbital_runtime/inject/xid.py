"""Synthetic ECC / Xid event stream.

Research doc SS1: "Synthetic ECC/Xid event stream (Xid 48/63/64/94/95) for
the watcher tier." PLAN.md: no real DCGM integration until the cloud-GPU
week (M4) -- simulated Xid until then.

What an Xid actually is: an NVIDIA driver error report surfaced in dmesg /
nvidia-smi / DCGM. The watcher tier (M2) consumes this stream; on real
NVIDIA hardware the same consumer reads DCGM counters instead. This module
is therefore the SIMULATION side of an interface, not a pretend detector --
`detect/watcher.py` must not care which side it is talking to.

Fidelity boundary (important for demo honesty)
----------------------------------------------
This stream is CORRELATED WITH, not derived from, the injected flips: we
know a flip happened and emit a plausible driver report for it. Real Xids
are produced by hardware ECC logic we are not simulating at the circuit
level. So the watcher's measured recall against this stream is a measure of
plumbing, not of ECC physics -- M2 must report it as such and not claim it
as a detector-accuracy result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Xid codes we emit, with the meaning the driver assigns them. Research doc
# SS1 names exactly this set (48/63/64/94/95).
XID_DBE = 48  # Double-Bit ECC error (uncorrectable)
XID_ROW_REMAP_PENDING = 63  # ECC page retirement / row-remap recorded
XID_ROW_REMAP_FAILURE = 64  # ECC page retirement / row-remap FAILED
XID_CONTAINED_ECC = 94  # Contained ECC error (app-contained, job may survive)
XID_UNCONTAINED_ECC = 95  # Uncontained ECC error (job must die)

XID_MEANING: dict[int, str] = {
    XID_DBE: "double-bit ECC error (uncorrectable)",
    XID_ROW_REMAP_PENDING: "ECC row-remap recorded",
    XID_ROW_REMAP_FAILURE: "ECC row-remap failed",
    XID_CONTAINED_ECC: "contained ECC error",
    XID_UNCONTAINED_ECC: "uncontained ECC error",
}

# Xids that mean the job cannot be trusted to continue.
FATAL_XIDS = frozenset({XID_DBE, XID_ROW_REMAP_FAILURE, XID_UNCONTAINED_ECC})

# Probability that a given memory upset surfaces as ANY driver report.
#
# !! ENGINEERING ASSUMPTION -- NOT A CITED NUMBER !!
# With ECC disabled (the default demo mode) a real device reports nothing at
# all for a single-bit flip: that is precisely the silent-corruption regime
# the product exists to address. With ECC on, single-bit errors are
# corrected and may be logged. The research doc gives no reporting rate, so
# this is swept, not asserted. Defaults below keep the demo honest: in
# ecc_off mode the stream is SILENT, which is the whole point.
DEFAULT_REPORT_PROB_ECC_ON = 0.5
DEFAULT_REPORT_PROB_ECC_OFF = 0.0


@dataclass(frozen=True)
class XidEvent:
    """A synthetic driver error report."""

    t: float
    code: int
    detail: str
    fatal: bool

    def as_record(self) -> dict:
        return {
            "t_sim": self.t,
            "xid": self.code,
            "detail": self.detail,
            "fatal": self.fatal,
            "meaning": XID_MEANING.get(self.code, "unknown"),
        }


class XidSimulator:
    """Turns injected faults into a plausible driver event stream.

    In `ecc_off` mode (the headline demo) this stream is silent by default:
    an unprotected fp32 training run gets NO warning from the driver, which
    is exactly why detection has to come from the application layer.
    """

    def __init__(
        self,
        *,
        ecc_on: bool = False,
        report_prob: float | None = None,
    ) -> None:
        self.ecc_on = ecc_on
        if report_prob is None:
            report_prob = (
                DEFAULT_REPORT_PROB_ECC_ON if ecc_on else DEFAULT_REPORT_PROB_ECC_OFF
            )
        if not 0.0 <= report_prob <= 1.0:
            raise ValueError(f"report_prob must be in [0,1], got {report_prob}")
        self.report_prob = report_prob
        self.events: list[XidEvent] = []

    @property
    def silent(self) -> bool:
        return self.report_prob == 0.0

    def on_flip(
        self, t: float, rng: np.random.Generator, *, multi_bit: bool = False
    ) -> XidEvent | None:
        """Maybe emit a report for a memory upset at time `t`."""
        if rng.random() >= self.report_prob:
            return None
        # A multi-bit upset defeats SEC-DED and is uncorrectable -> fatal.
        # A single-bit upset under ECC is corrected, and shows up (if at all)
        # as remap bookkeeping rather than an error.
        if multi_bit:
            code = XID_UNCONTAINED_ECC if rng.random() < 0.5 else XID_DBE
        else:
            code = XID_ROW_REMAP_PENDING if rng.random() < 0.7 else XID_CONTAINED_ECC
        return self._emit(t, code, "memory upset")

    def on_sefi(self, t: float, rng: np.random.Generator) -> XidEvent:
        """A SEFI always surfaces: the device fell over."""
        code = XID_ROW_REMAP_FAILURE if rng.random() < 0.3 else XID_UNCONTAINED_ECC
        return self._emit(t, code, "functional interrupt")

    def _emit(self, t: float, code: int, detail: str) -> XidEvent:
        ev = XidEvent(t=t, code=code, detail=detail, fatal=code in FATAL_XIDS)
        self.events.append(ev)
        return ev

    def drain(self) -> list[XidEvent]:
        """Consume pending reports (the watcher tier polls this)."""
        out, self.events = self.events, []
        return out
