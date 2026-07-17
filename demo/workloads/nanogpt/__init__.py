"""nanoGPT workload: char-level Shakespeare, CPU/MPS-sized."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from orbital_runtime.rng import STREAM_WORKLOAD, stream, torch_seed

from .data import CharDataset, get_batch, load_corpus
from .model import GPT, GPTConfig

__all__ = ["NanoGPTWorkload", "build_nanogpt", "GPT", "GPTConfig"]


@dataclass
class NanoGPTWorkload:
    """A nanoGPT training workload satisfying `orbital_runtime.workload.Workload`."""

    model: GPT
    optimizer: torch.optim.Optimizer
    device: torch.device
    dataset: CharDataset
    batch_size: int
    block_size: int
    seed: int
    _eval_rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Validation batches come from their own stream so that evaluating
        # more or less often cannot shift the training data order.
        self._eval_rng = stream(self.seed, "workload.eval")

    def batch_rng(self, step: int) -> np.random.Generator:
        """A per-step RNG: data order depends on `step`, not on history.

        This is what makes replay-after-rollback exact. If batches were
        drawn from one long-lived generator, resuming at step N would need
        the generator's internal state at step N -- and any extra draw
        (an eval, a retry) would desynchronise it. Keying on the step index
        makes step N's batch a pure function of (seed, step).
        """
        return stream(self.seed, f"{STREAM_WORKLOAD}.batch.{step}")

    def loss_for_step(self, step: int) -> torch.Tensor:
        x, y = get_batch(
            self.dataset.train,
            batch_size=self.batch_size,
            block_size=self.block_size,
            rng=self.batch_rng(step),
            device=self.device,
        )
        _, loss = self.model(x, y)
        assert loss is not None
        return loss

    @torch.no_grad()
    def evaluate(self, n_batches: int = 8) -> float:
        """Mean validation loss.

        The signal for PLAN.md's failure mode (b): a run that trains to a
        different, worse optimum without ever going NaN.
        """
        was_training = self.model.training
        self.model.eval()
        # Fixed eval batches (seeded per index) so successive evaluations are
        # comparable to each other rather than noisy re-samples.
        losses = []
        for i in range(n_batches):
            x, y = get_batch(
                self.dataset.val,
                batch_size=self.batch_size,
                block_size=self.block_size,
                rng=stream(self.seed, f"workload.eval.{i}"),
                device=self.device,
            )
            _, loss = self.model(x, y)
            assert loss is not None
            losses.append(float(loss.item()))
        if was_training:
            self.model.train()
        return float(np.mean(losses))


def build_nanogpt(
    *,
    seed: int = 1337,
    device: torch.device | str = "cpu",
    batch_size: int = 16,
    block_size: int = 64,
    n_layer: int = 4,
    n_head: int = 4,
    n_embd: int = 128,
    lr: float = 1e-3,
    corpus: Path | str | None = None,
) -> NanoGPTWorkload:
    """Construct a deterministic nanoGPT workload."""
    dataset = load_corpus(corpus) if corpus else load_corpus()
    device = torch.device(device)

    # Seed torch's global RNG from our named stream so model init is
    # reproducible without us owning torch's generator.
    torch.manual_seed(torch_seed(seed, STREAM_WORKLOAD))

    cfg = GPTConfig(
        vocab_size=dataset.vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
    )
    model = GPT(cfg).to(device)
    # AdamW: two state tensors per param (exp_avg, exp_avg_sq), which the
    # injector accounts for in resident bits.
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99))

    return NanoGPTWorkload(
        model=model,
        optimizer=optimizer,
        device=device,
        dataset=dataset,
        batch_size=batch_size,
        block_size=block_size,
        seed=seed,
    )
