"""Data + experiment helpers shared across notebook and CLI scripts.

Corpus source: the prepared `data/shakespeare_lines.csv` — the
Folger Shakespeare dialogue dataset curated for this project and
published at:
    https://www.kaggle.com/datasets/ismailalshofi/shakespeare-for-nlp-91k-tagged-lines

No runtime downloading — the CSV is the single source of truth.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch

DEFAULT_CORPUS_CSV = "data/shakespeare_lines.csv"


def load_corpus(csv_path: str | Path = DEFAULT_CORPUS_CSV, include_speakers: bool = True) -> str:
    """Load the prepared Shakespeare CSV into a single text string.

    With `include_speakers=True` (default), the output is formatted like the
    classic Tiny-Shakespeare corpus — speaker label on its own line, then the
    speaker's lines, with a blank line between speeches:

        HAMLET:
        To be, or not to be, that is the question:
        Whether 'tis nobler in the mind to suffer
        ...

        HORATIO:
        ...

    With `include_speakers=False`, only `line_text` is concatenated.
    """
    df = pd.read_csv(csv_path).fillna({"player": "UNKNOWN", "line_text": ""})
    if not include_speakers:
        return "\n".join(df["line_text"].astype(str))

    # one speech = run of consecutive rows with the same (play, act, scene, player)
    speech_id = (
        (df["player"] != df["player"].shift())
        | (df["play"] != df["play"].shift())
        | (df["act"] != df["act"].shift())
        | (df["scene"] != df["scene"].shift())
    ).cumsum()

    parts: list[str] = []
    for _, group in df.groupby(speech_id, sort=False):
        parts.append(f"{group['player'].iloc[0]}:")
        parts.extend(group["line_text"].astype(str).tolist())
        parts.append("")  # blank line between speeches
    return "\n".join(parts).strip()


def split_corpus(text: str, val_frac: float = 0.1) -> tuple[str, str]:
    n = int(len(text) * (1 - val_frac))
    return text[:n], text[n:]


def make_batches(
    ids: torch.Tensor, batch_size: int, block_size: int, device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch of (input, target) sequences."""
    ix = torch.randint(0, ids.size(0) - block_size - 1, (batch_size,))
    x = torch.stack([ids[i : i + block_size] for i in ix])
    y = torch.stack([ids[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


def perplexity(loss: float) -> float:
    return math.exp(loss)


def bits_per_char(loss: float, chars_per_token: float) -> float:
    """Convert mean cross-entropy (nats per token) to bits per character."""
    return loss / math.log(2) / chars_per_token


def token_length_stats(ids: Iterable[int]) -> dict[str, float]:
    ids = list(ids)
    return {"n_tokens": len(ids)}
