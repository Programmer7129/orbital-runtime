"""When to checkpoint: orbit-aware cadence.

Research doc SS3, the differentiator: "detection intensity and checkpoint
cadence keyed to orbital position (crank ABFT sampling + checkpoint
immediately before SAA entry)." The defensible novelty is position-aware
protection scheduling for general-purpose GPU *training* runtimes;
radiation-aware instrument safing itself is decades-old spacecraft practice.

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

# Target expected upsets between checkpoints when a flux model is available.
# Below 1 means a rollback usually replays less than one upset's worth of work,
# which is the point: bound the replay cost in UPSETS, not in steps.
DEFAULT_TARGET_UPSETS_PER_INTERVAL = 0.5

# Floor on the derived cadence. Saving every step would spend more on
# checkpoint I/O than the step costs to recompute.
DEFAULT_MIN_INTERVAL = 5


@dataclass
class CheckpointPolicy:
    """Decides, per step, whether now is a good time to checkpoint."""

    track: OrbitTrack
    base_interval: int = DEFAULT_BASE_INTERVAL
    saa_interval: int = DEFAULT_SAA_INTERVAL
    pre_saa_lead: int = DEFAULT_PRE_SAA_LEAD
    adaptive: bool = True
    # Optional FluxModel. When supplied, the cadence is derived from the upset
    # rate rather than a fixed step count -- see `interval`.
    flux: object | None = None
    seconds_per_step: float = 0.0
    target_upsets_per_interval: float = DEFAULT_TARGET_UPSETS_PER_INTERVAL
    min_interval: int = DEFAULT_MIN_INTERVAL

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
        """Steps between saves. Keyed to the UPSET RATE when flux is known.

        A fixed step count is the wrong unit. What a checkpoint insures against
        is upsets, and the upset rate scales with resident bits -- so the same
        50-step cadence that is generous on a 10M-parameter model is starvation
        on an 85M-parameter one, which draws ~8x the flux.

        That is not hypothetical. On an L4 at 85.3M parameters the fixed cadence
        produced 3 checkpoints in 33 steps, 3 rollbacks exhausted them, and the
        protected run died "unrecoverable" while detection was working perfectly.
        The failure was recovery economics, not detection.

        So when a flux model is available, the interval is chosen to keep the
        EXPECTED UPSETS PER INTERVAL near `target_upsets_per_interval`. A
        rollback then costs about one upset's worth of replay regardless of
        model size, which is the invariant that actually matters. Without a flux
        model the fixed cadence stands, so nothing changes for callers that do
        not supply one.
        """
        fixed = (
            self.base_interval
            if not self.adaptive
            else (self.saa_interval if in_saa else self.base_interval)
        )
        if self.flux is None:
            return fixed

        rate = (
            self.flux.saa_rate_per_s if in_saa else self.flux.quiescent_rate_per_s
        )
        if rate <= 0 or self.seconds_per_step <= 0:
            return fixed
        upsets_per_step = rate * self.seconds_per_step
        if upsets_per_step <= 0:
            return fixed
        derived = int(self.target_upsets_per_interval / upsets_per_step)
        # Clamp: never rarer than the fixed cadence (that is the ceiling the
        # design already accepted), never so frequent that checkpoint I/O
        # dominates the step it is protecting.
        return max(self.min_interval, min(derived, fixed))

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
