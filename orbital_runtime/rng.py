"""Deterministic RNG stream management.

PLAN.md design rule 3 (Determinism): seeded runs must reproduce exactly --
flip schedule, detection, recovery. Two properties are required:

1. **Reproducibility.** Same seed => same everything.
2. **Stream independence.** Adding or removing a consumer must not perturb
   any other consumer's draws. If the injector and the ABFT sampler shared
   one Generator, turning ABFT on would shift the flip schedule and the
   protected/unprotected comparison would no longer be apples-to-apples --
   which is the entire point of the demo.

Both come from `numpy.random.SeedSequence`, which derives statistically
independent child streams from one root seed via a named-key hash. Streams
are addressed by NAME (not by spawn order), so adding a new named stream
leaves existing streams bit-identical.
"""

from __future__ import annotations

import hashlib

import numpy as np

# Stream names used across the runtime. Registered here so the set of
# streams is greppable in one place; `stream()` accepts any name.
STREAM_FLUX = "flux"  # upset arrival times (Poisson engine)
STREAM_MEMORY = "inject.memory"  # which bit of which tensor gets flipped
STREAM_COMPUTE = "inject.compute"  # activation corruption sites
STREAM_SEFI = "inject.sefi"  # hang/crash draws
STREAM_XID = "inject.xid"  # synthetic ECC/Xid event stream
STREAM_ABFT = "detect.abft"  # ABFT checksum sampling schedule
STREAM_WORKLOAD = "workload"  # model init + data order


def _name_to_key(name: str) -> int:
    """Map a stream name to a stable 64-bit spawn key.

    `hash()` is salted per-process (PYTHONHASHSEED), so it cannot be used --
    it would break reproducibility across runs. BLAKE2b is stable forever.
    """
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def stream(seed: int, name: str) -> np.random.Generator:
    """Return the independent Generator for `name` under the root `seed`.

    Deterministic in (seed, name) alone: the same pair always yields the
    same stream, regardless of what other streams exist or when they are
    created.
    """
    root = np.random.SeedSequence(entropy=seed, spawn_key=(_name_to_key(name),))
    return np.random.Generator(np.random.PCG64(root))


def torch_seed(seed: int, name: str) -> int:
    """Derive a stable 63-bit seed for torch's global RNG from a named stream.

    torch's generator is global mutable state (needed for model init and
    dropout), so it can't be a numpy stream -- but the seed fed to it still
    comes from the same deterministic derivation.
    """
    return int(stream(seed, name).integers(0, 2**63 - 1))
