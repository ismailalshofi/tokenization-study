"""Subword (BPE) tokenizer.

Thin wrapper around HuggingFace `tokenizers` so the rest of the codebase
sees the same interface as `CharTokenizer`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer


class SubwordTokenizer:
    PAD = "<pad>"
    UNK = "<unk>"
    BOS = "<bos>"
    EOS = "<eos>"
    SPECIAL_TOKENS = [PAD, UNK, BOS, EOS]

    def __init__(self, tokenizer: Tokenizer | None = None):
        self._tok = tokenizer

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    @property
    def pad_id(self) -> int:
        return self._tok.token_to_id(self.PAD)

    def train(self, corpus: Iterable[str], vocab_size: int = 8000) -> "SubwordTokenizer":
        tok = Tokenizer(BPE(unk_token=self.UNK))
        tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tok.decoder = ByteLevelDecoder()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=self.SPECIAL_TOKENS,
            initial_alphabet=ByteLevel.alphabet(),
            show_progress=False,
        )
        tok.train_from_iterator(corpus, trainer=trainer)
        self._tok = tok
        return self

    def encode(self, text: str, add_special: bool = False) -> list[int]:
        ids = self._tok.encode(text).ids
        if add_special:
            ids = [self._tok.token_to_id(self.BOS), *ids, self._tok.token_to_id(self.EOS)]
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=True)

    def save(self, path: str | Path) -> None:
        self._tok.save(str(path))

    @classmethod
    def load(cls, path: str | Path) -> "SubwordTokenizer":
        return cls(tokenizer=Tokenizer.from_file(str(path)))
