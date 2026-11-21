"""The real engine: GGUF file in, tokens out. Implements the Engine protocol."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from local_ai_inference.engine.llama import Llama
from local_ai_inference.engine.protocol import GenerationParams, ModelInfo
from local_ai_inference.engine.sampler import sample
from local_ai_inference.engine.tokenizer import GGUFTokenizer
from local_ai_inference.parser import GGUFFile


class GGUFEngine:
    def __init__(self, path: str | Path, *, max_context: int = 2048) -> None:
        self.path = Path(path)
        self._file = GGUFFile(self.path)  # reads only the index; weights stay on disk
        meta = self._file.metadata
        self.tokenizer = GGUFTokenizer(meta)
        self.info = ModelInfo(
            model_id=self.path.stem,
            context_length=min(max_context, int(self._file.arch_key("context_length", 2048))),
            chat_template=meta.get("tokenizer.chat_template"),
            eos_token_id=self.tokenizer.eos_id,
            bos_token=self.tokenizer.bos_token,
            eos_token=self.tokenizer.eos_token,
        )
        self.model: Llama | None = None

    def load(self) -> None:
        self.model = Llama(self._file)  # dequantizes every weight: the slow part of startup

    def tokenize(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def detokenize(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    def generate(self, prompt_ids: list[int], params: GenerationParams) -> Iterator[int]:
        if self.model is None:
            raise RuntimeError("call load() first")
        rng = np.random.default_rng(params.seed)
        cache = self.model.new_cache(self.info.context_length)
        logits = self.model.forward(prompt_ids, cache)  # prefill: the whole prompt at once

        for _ in range(params.max_tokens):
            if params.cancel.is_set():
                return
            token = sample(logits, params.temperature, params.top_p, rng)
            if token in self.tokenizer.stop_ids:
                return
            yield token
            if cache.pos >= cache.capacity:
                return
            logits = self.model.forward([token], cache)  # decode: one token at a time
