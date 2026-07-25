"""What a detector says when it sees something.

Kept in its own module so `train.py`, the tiers, and (M3) the recovery
orchestrator can share the type without importing each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Tier names, in escalating cost order (research doc SS3).
TIER_GUARD = "guard"  # free: isfinite / z-score / loss-spike
TIER_ABFT = "abft"  # sampled checksum verification around GEMMs
TIER_WATCHER = "watcher"  # ECC/Xid event stream (DCGM on real NVIDIA)

# Why a detector fired.
REASON_NONFINITE_LOSS = "nonfinite_loss"
REASON_NONFINITE_GRAD = "nonfinite_grad"
REASON_GRAD_NORM_ZSCORE = "grad_norm_zscore"
REASON_LOSS_SPIKE = "loss_spike"
REASON_ABFT_MISMATCH = "abft_mismatch"
REASON_XID_FATAL = "xid_fatal"
# A single-event functional interrupt (SEFI) or ECC-on double-bit DUE: the
# process fell over. Not found by a tier -- the crash IS the signal -- but it
# routes through the same rollback machinery, so it carries a Verdict.
REASON_SEFI = "sefi_crash"


@dataclass(frozen=True)
class Verdict:
    """A detector's finding at one step."""

    triggered: bool
    step: int
    tier: str = ""
    reason: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def certain(self) -> bool:
        """True when the evidence admits no benign explanation.

        A non-finite loss or a fatal Xid is proof; a gradient-norm z-score
        is an inference that a healthy run can occasionally trip. M3 uses
        this to decide whether to roll back immediately or to escalate --
        rolling back on every z-score blip would spend the overhead budget
        on false alarms.
        """
        return self.reason in (
            REASON_NONFINITE_LOSS,
            REASON_NONFINITE_GRAD,
            REASON_ABFT_MISMATCH,
            REASON_XID_FATAL,
            REASON_SEFI,
        )

    def as_record(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "reason": self.reason,
            "certain": self.certain,
            **self.evidence,
        }


NO_DETECTION = Verdict(triggered=False, step=-1)
