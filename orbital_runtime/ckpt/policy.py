"""When to checkpoint: orbit-aware cadence.

Research doc SS3, the differentiator: "detection intensity and checkpoint
cadence keyed to orbital position (crank ABFT sampling + checkpoint
immediately before SAA entry). Novel; nothing in literature does
position-aware protection scheduling."

The argument in one line: ~90% of upsets arrive during ~10% of the orbit,
and a checkpoint is only worth what it saves you from. A uniform cadence
spends most of its budget insuring against the quiet 90% of the orbit where
almost nothing happens, and then meets the SAA with a checkpoint that is,
on average, half a cadence stale. Checkpointing immediately BEFORE SAA entry
costs one extra save per orbit and guarantees a fresh restore point exactly
where the risk is.

`AbftTier.sample_rate()` is the detection half of the same idea; this is the
recovery half. They are deliberately separate objects keyed to the same
signal (`in_saa`), because they answer different questions: how hard to
look, versus how much work to risk losing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..orbit.track import OrbitTrack

# Baseline cadence outside the SAA, in steps. Loose on purpose: the quiet
# part of the orbit is where the flux is ~1/75th of the SAA's, so frequent
# saves there buy little and cost real wall-clock.
DEFAULT_BASE_INTERVAL = 50

# Cadence inside an SAA transit, in steps. Tighter, because this is where
# the upsets are and where a rollback is most likely to be needed.
DEFAULT_SAA_INTERVAL = 10

# How far before SAA entry to force a save, in steps. Must be >0 so the
# save lands before the flux rises, and small so it is genuinely "fresh".
DEFAULT_PRE_SAA_LEAD = 2


@dataclass
class CheckpointPolicy:
    """Decides, per step, whether now is a good time to checkpoint."""

    track: OrbitTrack
    base_interval: int = DEFAULT_BASE_INTERVAL
    saa_interval: int = DEFAULT_SAA_INTERVAL
    pre_saa_lead: int = DEFAULT_PRE_SAA_LEAD
    adaptive: bool = True

    _last_save_step: int = field(default=-(10**9), init=False)
    _armed_orbits: set[int] = field(default_factory=set, init=False)
    pre_saa_saves: int = field(default=0, init=False)
    interval_saves: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.base_interval < 1 or self.saa_interval < 1:
            raise ValueError("intervals must be >= 1")
        if self.pre_saa_lead < 1:
            raise ValueError("pre_saa_lead must be >= 1")

    def interval(self, *, in_saa: bool) -> int:
        if not self.adaptive:
            return self.base_interval
        return self.saa_interval if in_saa else self.base_interval

    def approaching_saa(self, *, t_sim: float, seconds_per_step: float) -> bool:
        """True when SAA entry is within `pre_saa_lead` steps.

        Looks ahead in SIMULATION time rather than counting steps, so the
        lead is correct regardless of how sim time maps onto steps -- and
        that mapping is not fixed under replay (see injector.py).
        """
        if self.track.in_saa(t_sim):
            return False
        horizon = t_sim + self.pre_saa_lead * seconds_per_step
        return self.track.in_saa(horizon) or any(
            t_sim < lo <= horizon for lo, _ in self.track.saa_windows(t_sim, horizon)
        )

    def should_save(
        self,
        *,
        step: int,
        t_sim: float,
        in_saa: bool,
        seconds_per_step: float,
    ) -> tuple[bool, str]:
        """(save?, why). `why` is recorded in telemetry and the dashboard."""
        # Rule 1: never enter the SAA without a fresh restore point. Once per
        # orbit -- otherwise every step of the approach would re-trigger it.
        if self.adaptive and self.approaching_saa(
            t_sim=t_sim, seconds_per_step=seconds_per_step
        ):
            orbit = self.track.orbit_index(t_sim + self.pre_saa_lead * seconds_per_step)
            if orbit not in self._armed_orbits:
                self._armed_orbits.add(orbit)
                self.pre_saa_saves += 1
                return True, "pre_saa_entry"

        # Rule 2: ordinary cadence, tightened inside the SAA.
        if step - self._last_save_step >= self.interval(in_saa=in_saa):
            self.interval_saves += 1
            return True, "interval"

        return False, ""

    def record_save(self, step: int) -> None:
        self._last_save_step = step

    def reset(self, step: int) -> None:
        """After a rollback the step counter moves backwards; the cadence
        must not think it is overdue and immediately re-save."""
        self._last_save_step = step

    def stats(self) -> dict:
        return {
            "pre_saa_saves": self.pre_saa_saves,
            "interval_saves": self.interval_saves,
        }
