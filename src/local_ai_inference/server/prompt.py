"""Prompt preparation: chat messages in, prompt text out.

The server owns this, not the engine. A wrong template still produces text,
just the wrong text, so it belongs where it can be tested on its own.
Real models ship their own Jinja chat template inside the GGUF file.
"""

from collections.abc import Sequence
from functools import lru_cache
from typing import Protocol

from jinja2 import Template
from jinja2.sandbox import ImmutableSandboxedEnvironment


class Message(Protocol):
    role: str
    content: str


def render_chat(
    messages: Sequence[Message],
    template: str | None = None,
    *,
    bos_token: str = "",
    eos_token: str = "",
) -> str:
    if template is None:  # the stub engine has no template: use a readable default
        parts = [f"<|{m.role}|>\n{m.content}\n" for m in messages]
        return "".join(parts) + "<|assistant|>\n"
    return _compile(template).render(
        messages=[{"role": m.role, "content": m.content} for m in messages],
        add_generation_prompt=True,
        bos_token=bos_token,
        eos_token=eos_token,
    )


@lru_cache(maxsize=4)
def _compile(template: str) -> Template:
    # Templates come from model files, which anyone can publish: render them sandboxed.
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)

    def raise_exception(message: str) -> None:
        raise ValueError(f"chat template error: {message}")

    env.globals["raise_exception"] = raise_exception
    return env.from_string(template)
