"""Prompt preparation: chat messages in, prompt text out.

The server owns this, not the engine. A wrong template still produces text,
just the wrong text, so it belongs where it can be tested on its own.
"""

from collections.abc import Sequence
from typing import Protocol


class Message(Protocol):
    role: str
    content: str


def render_chat(messages: Sequence[Message], template: str | None = None) -> str:
    if template is not None:
        raise NotImplementedError("Jinja chat templates from the model file arrive in step-03")
    parts = [f"<|{m.role}|>\n{m.content}\n" for m in messages]
    parts.append("<|assistant|>\n")
    return "".join(parts)
