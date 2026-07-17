"""Shared fixtures.

Tests use a deliberately tiny GPT (~50k params vs the demo's 0.81M) so the
suite stays fast. Radiation rates are scaled up to compensate: fewer
resident bits at the same rate means fewer strikes, and the tests are about
mechanism, not calibration (the calibration tests live in test_flux.py and
use the real H100 bit count).
"""

from __future__ import annotations

import pytest
import torch

from demo.workloads.nanogpt import build_nanogpt

TINY = dict(n_layer=1, n_head=2, n_embd=32, block_size=32, batch_size=8)


@pytest.fixture(scope="session")
def corpus_exists() -> bool:
    from demo.workloads.nanogpt.data import CORPUS

    if not CORPUS.exists():
        pytest.skip(f"corpus missing at {CORPUS}; run `make data`")
    return True


@pytest.fixture
def tiny_workload(corpus_exists):
    """A fresh tiny nanoGPT on CPU."""

    def _build(seed: int = 1337, **kw):
        opts = {**TINY, **kw}
        return build_nanogpt(seed=seed, device="cpu", **opts)

    return _build


@pytest.fixture
def stepped_workload(tiny_workload):
    """A tiny workload that has taken one step, so optimizer state exists."""
    w = tiny_workload()
    loss = w.loss_for_step(0)
    loss.backward()
    w.optimizer.step()
    w.optimizer.zero_grad()
    return w


def devices() -> list[str]:
    """CPU plus MPS when present. Never CUDA on this Mac (PLAN.md rule 1)."""
    out = ["cpu"]
    if torch.backends.mps.is_available():
        out.append("mps")
    return out


@pytest.fixture(params=devices())
def device(request) -> str:
    """Every device available here. Guards PLAN.md rule 1 automatically:
    a CPU-only assumption fails the MPS pass of the same test."""
    return request.param
