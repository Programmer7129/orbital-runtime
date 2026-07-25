"""End-to-end determinism (PLAN.md design rule 3).

"Seeded runs must reproduce exactly (flip schedule, detection, recovery) --
needed for tests, benchmarks, and a demo that never flakes on stage."

The load-bearing property is stream independence: the protected and
unprotected runs must face a BIT-IDENTICAL bombardment, or the comparison
that produces the headline overhead number is not a controlled experiment.
"""

from __future__ import annotations

import pytest

from orbital_runtime.inject.injector import RadiationEnvironment
from orbital_runtime.inject.memory import MemoryInjector
from orbital_runtime.inject.sefi import SefiInjector
from orbital_runtime.orbit.flux import FluxModel
from orbital_runtime.orbit.track import OrbitTrack
from orbital_runtime.train import TrainConfig, train


def make_env(workload, *, rate: float, seed: int, steps: int, orbits: float = 2.0, **kw):
    bits = MemoryInjector(workload.model, workload.optimizer).static_resident_bits()
    flux = FluxModel(
        bits_resident=bits, track=OrbitTrack(), base_rate_upsets_per_bit_day=rate
    )
    return RadiationEnvironment(
        workload.model,
        workload.optimizer,
        flux=flux,
        seed=seed,
        n_steps=steps,
        orbits=orbits,
        **kw,
    )


def test_identical_seeds_produce_identical_runs(tiny_workload):
    """Same seed, same everything -- to the last bit of the loss."""

    def run():
        w = tiny_workload(seed=11)
        env = make_env(w, rate=2e-4, seed=11, steps=80)
        r = train(w, cfg=TrainConfig(steps=80), env=env)
        return r.losses, r.died, r.death_reason, env.stats.as_dict()

    a = run()
    b = run()
    assert a[0] == b[0]  # exact float equality, not approx
    assert a[1:] == b[1:]


def test_model_init_is_reproducible(tiny_workload):
    a = tiny_workload(seed=5).model
    b = tiny_workload(seed=5).model
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert na == nb
        assert pa.equal(pb)


def test_data_order_is_reproducible_and_step_keyed(tiny_workload):
    """Batches are a pure function of (seed, step).

    This is what makes replay-after-rollback exact in M3: re-running step N
    must see the same data whether it is the first or the second attempt.
    """
    w = tiny_workload(seed=5)
    first = w.loss_for_step(7)
    again = w.loss_for_step(7)
    assert float(first.item()) == float(again.item())

    # And it does not depend on what happened in between.
    w.loss_for_step(0)
    w.loss_for_step(99)
    assert float(w.loss_for_step(7).item()) == float(first.item())


def test_flip_schedule_is_independent_of_protection(tiny_workload):
    """THE controlled-experiment guarantee.

    Turning other subsystems on must not perturb the radiation. If it did,
    the protected run would face different physics from the unprotected one
    and the headline comparison would be meaningless.
    """
    w1 = tiny_workload(seed=21)
    plain = make_env(w1, rate=2e-4, seed=21, steps=80)

    w2 = tiny_workload(seed=21)
    with_extras = make_env(
        w2,
        rate=2e-4,
        seed=21,
        steps=80,
        sefi=SefiInjector(OrbitTrack(), p_per_transit=0.5),
        inject_activations=True,
        activation_share=0.0,  # routing off, but the channel exists
    )

    assert [e.t for e in plain.upsets] == [e.t for e in with_extras.upsets]
    assert [e.in_saa for e in plain.upsets] == [e.in_saa for e in with_extras.upsets]
    with_extras.close()


def test_enabling_sefi_does_not_move_the_flips(tiny_workload):
    """Named streams: the SEFI draw must not consume the memory stream.

    Compared at the SCHEDULE level (env.upsets), not the loss level: a fired
    SEFI legitimately diverges the losses (it crashes the run), but the
    MEMORY-flip schedule -- times, SAA membership -- must be bit-identical
    whether SEFI fires often (p=0.9), never (p=0.0), or at the calibrated
    default. If enabling SEFI moved the flips, the STREAM_SEFI draw would be
    stealing from STREAM_MEMORY.
    """
    w = tiny_workload(seed=31)
    on = make_env(
        w, rate=2e-4, seed=31, steps=60, sefi=SefiInjector(OrbitTrack(), p_per_transit=0.9)
    )
    off = make_env(
        tiny_workload(seed=31), rate=2e-4, seed=31, steps=60,
        sefi=SefiInjector(OrbitTrack(), p_per_transit=0.0),
    )
    default = make_env(tiny_workload(seed=31), rate=2e-4, seed=31, steps=60)  # calibrated

    assert [e.t for e in on.upsets] == [e.t for e in off.upsets]
    assert [e.t for e in on.upsets] == [e.t for e in default.upsets]
    assert [e.in_saa for e in on.upsets] == [e.in_saa for e in off.upsets]


def test_different_seeds_produce_different_runs(tiny_workload):
    """The flip side: seeding must actually vary the physics."""
    w1 = tiny_workload(seed=1)
    r1 = train(w1, cfg=TrainConfig(steps=40), env=make_env(w1, rate=2e-4, seed=1, steps=40))
    w2 = tiny_workload(seed=1)
    r2 = train(w2, cfg=TrainConfig(steps=40), env=make_env(w2, rate=2e-4, seed=2, steps=40))
    assert r1.losses != r2.losses


def test_radiation_free_run_is_identical_to_no_env(tiny_workload):
    """A zero-rate environment must be indistinguishable from no environment.

    Guards against the injector perturbing a run merely by being attached --
    which would silently contaminate every overhead measurement.
    """
    w1 = tiny_workload(seed=8)
    baseline = train(w1, cfg=TrainConfig(steps=50))

    w2 = tiny_workload(seed=8)
    env = make_env(w2, rate=0.0, seed=8, steps=50)
    zero_rate = train(w2, cfg=TrainConfig(steps=50), env=env)

    assert baseline.losses == zero_rate.losses


def test_evaluation_does_not_disturb_training_order(tiny_workload):
    """Eval draws from its own stream, so eval cadence cannot shift training."""
    w1 = tiny_workload(seed=9)
    a = train(w1, cfg=TrainConfig(steps=40, eval_every=0))

    w2 = tiny_workload(seed=9)
    b = train(w2, cfg=TrainConfig(steps=40, eval_every=5))

    assert a.losses == b.losses


def test_evaluate_is_repeatable(tiny_workload):
    w = tiny_workload(seed=9)
    assert w.evaluate(4) == w.evaluate(4)


def test_evaluate_does_not_leave_the_model_in_eval_mode(tiny_workload):
    """A silent train/eval mode leak would change dropout behaviour mid-run."""
    w = tiny_workload(seed=9)
    w.model.train()
    w.evaluate(2)
    assert w.model.training
