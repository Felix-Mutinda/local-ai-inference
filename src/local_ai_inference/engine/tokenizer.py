"""The tokenizer, rebuilt from the GGUF file's own metadata.

The vocabulary, merge rules, special tokens and pre-tokenizer name all come from
the model file; nothing is downloaded. The BPE merge loop itself is borrowed from
the `tokenizers` library (writing it ourselves is a later post).
"""

from collections.abc import Mapping
from typing import Any

from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers

CONTROL = 3  # GGUF token_type for special tokens such as <|im_start|>


class GGUFTokenizer:
    def __init__(self, metadata: Mapping[str, Any]) -> None:
        kind = metadata.get("tokenizer.ggml.model")
        if kind != "gpt2":
            raise NotImplementedError(f"tokenizer model {kind!r}: only byte-level BPE (gpt2)")

        tokens: list[str] = metadata["tokenizer.ggml.tokens"]
        vocab = {token: i for i, token in enumerate(tokens)}
        merges = [tuple(m.split(" ", 1)) for m in metadata.get("tokenizer.ggml.merges", [])]
        tok = Tokenizer(models.BPE(vocab=vocab, merges=merges, fuse_unk=False))

        byte_level = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
        if metadata.get("tokenizer.ggml.pre") == "smollm":  # digits are split one by one
            tok.pre_tokenizer = pre_tokenizers.Sequence(
                [pre_tokenizers.Digits(individual_digits=True), byte_level]
            )
        else:
            tok.pre_tokenizer = byte_level
        tok.decoder = decoders.ByteLevel()

        types = metadata.get("tokenizer.ggml.token_type", [])
        specials = [t for t, kind in zip(tokens, types, strict=False) if kind == CONTROL]
        tok.add_special_tokens([AddedToken(s, special=True, normalized=False) for s in specials])

        self._tok = tok
        self.vocab_size = len(tokens)
        self.bos_id: int | None = metadata.get("tokenizer.ggml.bos_token_id")
        self.eos_id: int | None = metadata.get("tokenizer.ggml.eos_token_id")
        eot = metadata.get("tokenizer.ggml.eot_token_id")
        self.stop_ids = {i for i in (self.eos_id, eot) if i is not None}
        self.bos_token = tokens[self.bos_id] if self.bos_id is not None else ""
        self.eos_token = tokens[self.eos_id] if self.eos_id is not None else ""

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=True)
