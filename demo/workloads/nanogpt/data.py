"""Char-level tinyshakespeare loader.

Deterministic by construction (PLAN.md design rule 3): the vocabulary is
the sorted set of characters in the corpus, and batches are drawn from a
named RNG stream, so the same seed yields the same data order on any
machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

DATA_DIR = Path(__file__).parent / "data"
CORPUS = DATA_DIR / "input.txt"

# Fraction of the corpus held out for validation. 0.1 follows nanoGPT's
# char-level Shakespeare split.
VAL_FRACTION = 0.1


@dataclass
class CharDataset:
    """Encoded corpus + vocabulary."""

    train: np.ndarray
    val: np.ndarray
    stoi: dict[str, int]
    itos: dict[int, str]

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[int(i)] for i in ids)


def load_corpus(path: Path | str = CORPUS) -> CharDataset:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"tinyshakespeare corpus not found at {path}. "
            "Fetch it with: make data"
        )
    text = path.read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}

    ids = np.array([stoi[c] for c in text], dtype=np.uint16)
    split = int(len(ids) * (1 - VAL_FRACTION))
    return CharDataset(train=ids[:split], val=ids[split:], stoi=stoi, itos=itos)


def get_batch(
    data: np.ndarray,
    *,
    batch_size: int,
    block_size: int,
    rng: np.random.Generator,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw one (x, y) batch of next-char prediction targets."""
    if len(data) <= block_size:
        raise ValueError(f"corpus of {len(data)} too short for block_size {block_size}")
    ix = rng.integers(0, len(data) - block_size - 1, size=batch_size)
    x = torch.from_numpy(
        np.stack([data[i : i + block_size].astype(np.int64) for i in ix])
    )
    y = torch.from_numpy(
        np.stack([data[i + 1 : i + 1 + block_size].astype(np.int64) for i in ix])
    )
    return x.to(device), y.to(device)
