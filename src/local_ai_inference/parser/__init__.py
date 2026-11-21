"""The boundary between the engine and the model file.

The engine depends only on the ModelFile protocol; GGUFFile is one implementation.
"""

from collections.abc import Mapping
from typing import Any, Protocol

from local_ai_inference.parser.gguf import GGUFError, GGUFFile, TensorInfo, TensorType, dequantize


class ModelFile(Protocol):
    @property
    def metadata(self) -> Mapping[str, Any]:
        """Key-value pairs from the file: architecture, sizes, tokenizer, chat template."""
        ...

    def tensor(self, name: str) -> Any:
        """A weight tensor by name, as a float32 NumPy array."""
        ...


__all__ = ["GGUFError", "GGUFFile", "ModelFile", "TensorInfo", "TensorType", "dequantize"]
