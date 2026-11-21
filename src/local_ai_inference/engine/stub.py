"""A stub engine: a working stand-in so the whole request path runs before any maths exists.

Its tokenizer is real enough to be honest: one token per UTF-8 byte.
Its "model" replays a fixed reply. Step-03 replaces it with a NumPy forward pass.
"""

import time
from collections.abc import Iterator

from local_ai_inference.engine.protocol import GenerationParams, ModelInfo

DEFAULT_REPLY = (
    "Hello from the stub engine. Every layer above me is real: the API, the server, "
    "admission, streaming and metrics. The real engine arrives in step-03."
)


class StubEngine:
    def __init__(
        self,
        reply: str = DEFAULT_REPLY,
        *,
        token_delay: float = 0.0,
        context_length: int = 2048,
        model_id: str = "stub-echo",
    ) -> None:
        self.info = ModelInfo(model_id=model_id, context_length=context_length)
        self._reply_ids = self.tokenize(reply)
        self._token_delay = token_delay
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def tokenize(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def detokenize(self, ids: list[int]) -> str:
        return bytes(ids).decode("utf-8", errors="replace")

    def generate(self, prompt_ids: list[int], params: GenerationParams) -> Iterator[int]:
        for i, token_id in enumerate(self._reply_ids):
            if i >= params.max_tokens or params.cancel.is_set():
                return
            if self._token_delay:
                time.sleep(self._token_delay)
            yield token_id
