"""Character-level tokenizer.

Vocabulary is every distinct character in the training corpus plus a few
special tokens. There is no OOV: any character seen at training time can be
encoded, and unseen characters fall back to <unk>.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


class CharTokenizer:
    PAD = "<pad>"
    UNK = "<unk>"
    BOS = "<bos>"
    EOS = "<eos>"
    SPECIAL_TOKENS = [PAD, UNK, BOS, EOS]

    def __init__(self, stoi: dict[str, int] | None = None):
        self.stoi: dict[str, int] = stoi or {}
        self.itos: dict[int, str] = {i: s for s, i in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    @property
    def pad_id(self) -> int:
        return self.stoi[self.PAD]

    def train(self, corpus: Iterable[str]) -> "CharTokenizer":
        chars: set[str] = set()
        for line in corpus:
            chars.update(line)
        ordered = self.SPECIAL_TOKENS + sorted(chars)
        self.stoi = {tok: i for i, tok in enumerate(ordered)}
        self.itos = {i: tok for tok, i in self.stoi.items()}
        return self

    def encode(self, text: str, add_special: bool = False) -> list[int]:
        unk = self.stoi[self.UNK]
        ids = [self.stoi.get(c, unk) for c in text]
        if add_special:
            ids = [self.stoi[self.BOS], *ids, self.stoi[self.EOS]]
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        out: list[str] = []
        for i in ids:
            tok = self.itos.get(int(i), self.UNK)
            if tok in self.SPECIAL_TOKENS:
                continue
            out.append(tok)
        return "".join(out)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.stoi, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        stoi = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(stoi=stoi)
