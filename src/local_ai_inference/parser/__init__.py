"""The boundary between the engine and the model file.

Defined now so the engine's dependency is explicit; the GGUF reader arrives in step-02.
"""

from collections.abc import Mapping
from typing import Any, Protocol


class ModelFile(Protocol):
    @property
    def metadata(self) -> Mapping[str, Any]:
        """Key-value pairs from the file: architecture, sizes, tokenizer, chat template."""
        ...

    def tensor(self, name: str) -> Any:
        """A weight tensor by name, as a NumPy array (step-02)."""
        ...


__all__ = ["ModelFile"]
