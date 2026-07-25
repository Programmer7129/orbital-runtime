"""M4c: ECC-on redistribution SDC -> DUE (NSREC'21).

The beam-data audit (docs/research/beam-calibration-audit.md) found that ECC
does not merely SUPPRESS upsets, it REDISTRIBUTES them: with ECC on the
functional-interrupt (DUE) rate exceeds the silent-corruption (SDC) rate by
2.2-2.7x (NSREC'21, arXiv 2108.00554). The flux model already scales the
event rate down to the leaked (multi-bit) share (test_flux.py); here we test
the per-event split the `RadiationEnvironment` performs:

  * a leaked event becomes a DUE (detected-uncorrectable crash, recovered by
    process-restart, same path as a SEFI) with probability ECC_DUE_SHARE, or
  * an SDC (miscorrected silent flip, injected into the tensors) otherwise.

The whole channel is gated on `XidSimulator.ecc_on`, so the headline demo
(ecc_off) is provably untouched -- the DUE branch never draws its RNG there.
"""

from __future__ import annotations

import pytest

from orbital_runtime.inject.injector import RadiationEnvironment
from orbital_runtime.inject.sefi import SEFI_DUE, SefiCrash, SefiInjector
from orbital_runtime.inject.xid import XidSimulator
from orbital_runtime.orbit.flux import (
    ECC_DUE_SHARE,
    ECC_SDC_SHARE,
    MODE_ECC_OFF,
    MODE_ECC_ON,
    FluxModel,
)
from orbital_runtime.orbit.track import OrbitTrack

# A rate high enough to deliver a few hundred events across the run, so the
# DUE/SDC split is measurable. This is a mechanism test, not a calibration.
BUSY_RATE = 2e-3


def _env(workload, *, ecc: str, seed: int, steps: int = 200, orbits: float = 6.0):
    from orbital_runtime.inject.memory import MemoryInjector

    bits = MemoryInjector(workload.model, workload.optimizer).static_resident_bits()
    flux = FluxModel(
        bits_resident=bits,
        track=OrbitTrack(),
        base_rate_upsets_per_bit_day=BUSY_RATE,
        mode=ecc,
    )
    return RadiationEnvironment(
        workload.model,
        workload.optimizer,
        flux=flux,
        seed=seed,
        n_steps=steps,
        orbits=orbits,
        # Isolate the memory/ECC channel from the standalone SEFI channel, so a
        # SEFI crash cannot be mistaken for a DUE.
        sefi=SefiInjector(flux.track, p_per_transit=0.0),
        xid=XidSimulator(ecc_on=(ecc == MODE_ECC_ON)),
    )


def _drive(env, steps: int) -> list[SefiCrash]:
    """Step the environment, catching every crash it raises.

    The cursor advances before an event fires (injector.advance), so a caught
    DUE does not re-fire -- exactly the discipline that keeps replay finite.
    """
    crashes: list[SefiCrash] = []
    for step in range(steps):
        try:
            env.advance(step)
        except SefiCrash as c:
            crashes.append(c)
        env.tick()
    return crashes


# --------------------------------------------------------------------- #
# ecc_off: the headline demo path is untouched
# --------------------------------------------------------------------- #


def test_ecc_off_produces_no_dues_and_a_silent_driver(stepped_workload):
    """The headline demo path: no redistribution, and the driver reports
    nothing (report_prob 0 under ecc_off) -- application-layer detection is all
    there is. Confirms the DUE code cannot leak into the ecc_off story."""
    env = _env(stepped_workload, ecc=MODE_ECC_OFF, seed=1)
    crashes = _drive(env, env.n_steps)
    assert env.stats.dues == 0
    assert not crashes  # no DUE (and SEFI is off) -> nothing raised
    assert env.stats.flips > 0  # SDC events still delivered
    assert env.xid.silent and env.xid.events == []  # the silent-corruption regime


# --------------------------------------------------------------------- #
# ecc_on: SDC -> DUE redistribution
# --------------------------------------------------------------------- #


def test_ecc_on_redistributes_leaked_events_into_dues_and_sdcs(stepped_workload):
    env = _env(stepped_workload, ecc=MODE_ECC_ON, seed=3)
    crashes = _drive(env, env.n_steps)

    dues = env.stats.dues
    sdcs = env.stats.flips
    assert dues > 0, "no DUEs -- redistribution never fired"
    assert sdcs > 0, "no SDCs -- everything crashed"
    # Every DUE raised a SefiCrash flavoured DUE (process-restart recovery).
    assert len(crashes) == dues
    assert all(c.flavour == SEFI_DUE for c in crashes)


def test_ecc_on_due_share_matches_nsrec21(stepped_workload):
    """DUE dominant: DUE:SDC ~ ECC_DUE_SHARE:ECC_SDC_SHARE (~2.3:1). Loose band
    to stay non-flaky on a few-hundred-event sample, but tight enough to fail
    if the split were inverted or dropped."""
    env = _env(stepped_workload, ecc=MODE_ECC_ON, seed=11, steps=400, orbits=12.0)
    _drive(env, env.n_steps)
    dues, sdcs = env.stats.dues, env.stats.flips
    total = dues + sdcs
    assert total > 100, f"too few events ({total}) to measure the split"
    observed_due_share = dues / total
    assert observed_due_share == pytest.approx(ECC_DUE_SHARE, abs=0.12)
    assert dues > sdcs  # DUE dominant, the NSREC'21 headline


def test_ecc_on_dues_are_fatal_but_sdcs_are_silent(stepped_workload):
    """A DUE is a DETECTED uncorrectable error -> fatal Xid. An SDC is by
    definition undetected -> it must NOT surface as a fatal Xid (the SDC path
    forces multi_bit False so `on_flip` cannot emit an uncontained/DBE code)."""
    env = _env(stepped_workload, ecc=MODE_ECC_ON, seed=5)
    _drive(env, env.n_steps)
    fatal = [e for e in env.xid.events if e.fatal]
    # Exactly the DUEs are fatal; no SDC-path report is fatal.
    assert len(fatal) == env.stats.dues
    assert env.stats.dues > 0
