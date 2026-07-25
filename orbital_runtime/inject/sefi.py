"""SEFIs: simulated single-event functional interrupts (hangs / crashes).

Why this channel exists at all
------------------------------
Research doc SS1: "SEFIs: simulated process hang/crash (Jetson RADECS 2024
finding: reboot cross-section EXCEEDS bit-error cross-section -- crashes
matter as much as flips)."

That finding is the reason a bit-flip-only story is incomplete: on real
flight-adjacent hardware the device is at least as likely to fall over
entirely as it is to hand you a quietly wrong number. A runtime that only
catches silent corruption would still lose the job.

Calibration (M4c: now cited and ON by default)
-----------------------------------------------
The beam-data audit (docs/research/beam-calibration-audit.md) closed the gap
the earlier "off by default, uncited" posture was holding open. Suncatcher
(arXiv 2511.19468, 67 MeV protons, UC Davis Crocker cyclotron) measured, on
the SAME device in the SAME beam:

    SEFI cross-section  sigma_SEFI = 2e-11  cm^2/chip   (~1 SEFI per 5 krad)
    SDC  cross-section  sigma_SDC  = 6-9e-9 cm^2/chip

Both are cross-sections against one proton environment, so under any fluence
Phi the EXPECTED-EVENT RATIO is fluence-independent:

    E[SEFI] / E[SDC]  =  sigma_SEFI / sigma_SDC  ~=  2e-11 / 7.5e-9  ~=  2.7e-3

i.e. roughly one functional interrupt per ~375 silent-corruption events. Our
flux model already schedules the SDC-class event stream driven by the SAA
proton flux (`inject/memory.py`), so the SEFI channel is calibrated by
riding that stream at the cross-section ratio -- no new fluence anchor
invented. `SefiInjector.from_flux()` performs exactly this mapping and is
what `RadiationEnvironment` / `run.py` now wire by DEFAULT.

Cross-check of the two Suncatcher numbers via the audit's dose-fluence
conversion 1 rad ~= 7.9e6 p/cm^2 (67 MeV): at 5 krad the fluence is
5000 * 7.9e6 = 3.95e10 p/cm^2, so sigma_SEFI * Phi = 2e-11 * 3.95e10 = 0.79
~= 1 SEFI -- reproducing Suncatcher's stated "~1 SEFI per 5 krad". The two
independently-quoted numbers are mutually consistent, which is why we trust
the ratio.

Modeling assumptions (flagged, not hidden):
  * We identify one modeled memory upset with one SDC-class device event.
    Our upsets ARE the propagating-corruption stream, so this is the natural
    reading, but a real chip aggregates many bit cells per observed SDC.
  * SEFI scales with modeled exposure (upsets/transit), which scales with
    resident bits -- physically a bigger device (more chips) does have more
    SEFI-prone area, so the direction is right even if the constant is a
    proxy. Conditions caveat: Suncatcher is 67 MeV protons on one accelerator
    part; a different device/energy shifts sigma_SEFI/sigma_SDC.

The raw `SefiInjector(track)` constructor still DEFAULTS to p=0.0 (off), so a
test that wants an isolated, SEFI-free memory channel gets one by asking for
it; the calibrated on-by-default behaviour lives in `from_flux`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..orbit.track import OrbitTrack

# SEFI flavours we simulate.
SEFI_CRASH = "crash"  # process dies outright
SEFI_HANG = "hang"  # process stops making progress
SEFI_DUE = "due"  # ECC-on detected-uncorrectable error (functional interrupt)

# Default per-SAA-transit SEFI probability for the RAW constructor. 0.0 = off.
# The calibrated, on-by-default value is computed by `from_flux` below; this
# keeps `SefiInjector(track)` a clean "channel off" for tests that isolate the
# memory-fault stream.
DEFAULT_P_PER_TRANSIT = 0.0

# --- Suncatcher (arXiv 2511.19468) cross-sections, 67 MeV p+, Crocker ---
SUNCATCHER_SEFI_SIGMA_CM2 = 2e-11  # cm^2/chip  (~1 SEFI per 5 krad)
SUNCATCHER_SDC_SIGMA_CM2 = 7.5e-9  # cm^2/chip  (midpoint of the 6-9e-9 range)
# Expected SEFIs per SDC-class event -- a pure, fluence-independent ratio of
# two cross-sections measured in the same beam. ~2.7e-3.
SEFI_PER_SDC_EVENT = SUNCATCHER_SEFI_SIGMA_CM2 / SUNCATCHER_SDC_SIGMA_CM2

# Suncatcher dose-fluence conversion, kept for the docstring cross-check.
DOSE_FLUENCE_P_PER_CM2_PER_RAD = 7.9e6  # p/cm^2 per rad @ 67 MeV

# Split between crash and hang, given that a SEFI occurs.
# Also an assumption; only reachable when the channel fires.
DEFAULT_CRASH_SHARE = 0.5


class SefiCrash(RuntimeError):
    """Raised to simulate a radiation-induced process death.

    Deliberately a hard exception rather than a return code: the
    unprotected run must die the way a real one does, and the protected
    run's recovery path must prove it can catch and resume from it (M3).
    """

    def __init__(self, t_sim: float, orbit: int, flavour: str = SEFI_CRASH) -> None:
        self.t_sim = t_sim
        self.orbit = orbit
        self.flavour = flavour
        super().__init__(
            f"SEFI ({flavour}) at t_sim={t_sim:.1f}s during orbit {orbit} SAA transit"
        )


@dataclass(frozen=True)
class SefiEvent:
    """A scheduled SEFI."""

    t: float
    orbit: int
    flavour: str

    def as_record(self) -> dict:
        return {"t_sim": self.t, "orbit": self.orbit, "flavour": self.flavour}


class SefiInjector:
    """Draws SEFIs as a per-SAA-transit Bernoulli process.

    Per-transit (not per-second) because the RADECS finding is reported as a
    cross-section against the SAA proton environment: the risk is
    concentrated in the transit, which is also what makes the demo's
    "adaptive vigilance" idea (checkpoint before SAA entry) pay off.
    """

    def __init__(
        self,
        track: OrbitTrack,
        *,
        p_per_transit: float = DEFAULT_P_PER_TRANSIT,
        crash_share: float = DEFAULT_CRASH_SHARE,
    ) -> None:
        if not 0.0 <= p_per_transit <= 1.0:
            raise ValueError(f"p_per_transit must be in [0,1], got {p_per_transit}")
        if not 0.0 <= crash_share <= 1.0:
            raise ValueError(f"crash_share must be in [0,1], got {crash_share}")
        self.track = track
        self.p_per_transit = p_per_transit
        self.crash_share = crash_share

    @classmethod
    def from_flux(
        cls,
        flux: "object",
        *,
        crash_share: float = DEFAULT_CRASH_SHARE,
    ) -> "SefiInjector":
        """Calibrate the per-transit SEFI probability from the flux model.

        The conversion chain (all citations in the module docstring):

            mu   = expected SDC-class upsets in one SAA transit   (flux model)
            r    = sigma_SEFI / sigma_SDC = SEFI_PER_SDC_EVENT     (Suncatcher)
            E[SEFI per transit] = r * mu
            p_per_transit       = 1 - exp(-r * mu)                 (>=1 SEFI)

        `1 - exp(-.)` because the injector fires at most one SEFI per transit
        (a Bernoulli), so we convert the Poisson mean `r*mu` to the
        probability of at least one event. At small `r*mu` this is ~= r*mu.
        """
        mu = float(flux.expected_upsets_in_saa_per_orbit())
        mean_sefi_per_transit = SEFI_PER_SDC_EVENT * mu
        p = 1.0 - float(np.exp(-mean_sefi_per_transit))
        p = min(1.0, max(0.0, p))
        return cls(flux.track, p_per_transit=p, crash_share=crash_share)

    @property
    def enabled(self) -> bool:
        return self.p_per_transit > 0.0

    def schedule(self, t0: float, t1: float, rng: np.random.Generator) -> list[SefiEvent]:
        """Draw the SEFI schedule over [t0, t1]. Deterministic given `rng`.

        Drawn up front like the flux schedule, so the whole run's fault
        timeline is fixed before step 0 (PLAN.md design rule 3).
        """
        if not self.enabled:
            return []
        events: list[SefiEvent] = []
        for lo, hi in self.track.saa_windows(t0, t1):
            # Scale by the fraction of the transit actually inside [t0,t1]:
            # a clipped half-transit carries half the risk.
            frac = (hi - lo) / self.track.saa_duration_s
            if rng.random() >= self.p_per_transit * frac:
                continue
            t = float(rng.uniform(lo, hi))
            flavour = SEFI_CRASH if rng.random() < self.crash_share else SEFI_HANG
            events.append(SefiEvent(t, self.track.orbit_index(t), flavour))
        return events
