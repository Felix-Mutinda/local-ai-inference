"""The boundary between the server and the engine.

The server only ever hands the engine token IDs and gets token IDs back.
Everything about HTTP, chat messages and stop strings stays on the server side.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from threading import Event
from typing import Protocol


@dataclass(frozen=True)
class ModelInfo:
    """What the server needs to know about the loaded model."""

    model_id: str
    context_length: int
    chat_template: str | None = None  # Jinja template from the model file
    eos_token_id: int | None = None
    bos_token: str = ""  # some chat templates insert these strings
    eos_token: str = ""


@dataclass
class GenerationParams:
    """Per-request knobs. `cancel` lets the server stop generation mid-stream."""

    max_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    seed: int | None = None
    cancel: Event = field(default_factory=Event)


class Engine(Protocol):
    """Tokens in, tokens out. The stub engine and the real engine both implement this."""

    info: ModelInfo

    def load(self) -> None:
        """Read weights and prepare to run. May be slow."""
        ...

    def tokenize(self, text: str) -> list[int]: ...

    def detokenize(self, ids: list[int]) -> str: ...

    def generate(self, prompt_ids: list[int], params: GenerationParams) -> Iterator[int]:
        """Yield new token IDs until max_tokens, end-of-sequence, or params.cancel is set."""
        ...
