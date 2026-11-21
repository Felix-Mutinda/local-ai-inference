"""A GGUF reader, written from the file layout up.

A GGUF file has four parts, in order:

    header        magic "GGUF", version, tensor count, metadata count
    metadata      typed key-value pairs: architecture, sizes, tokenizer, chat template
    tensor index  for each tensor: name, shape, element type, offset into the data
    data          the weights, starting at the next `general.alignment` boundary

Nothing is copied when the file opens: the data section is memory-mapped, so the
operating system pages weights in from disk only when a tensor is actually read.
"""

import contextlib
import struct
from collections import Counter
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

GGUF_MAGIC = b"GGUF"
DEFAULT_ALIGNMENT = 32


class GGUFError(ValueError):
    pass


class ValueType(IntEnum):
    """Metadata value types."""

    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


# struct format for each fixed-size scalar type (little-endian)
_SCALAR = {
    ValueType.UINT8: "<B",
    ValueType.INT8: "<b",
    ValueType.UINT16: "<H",
    ValueType.INT16: "<h",
    ValueType.UINT32: "<I",
    ValueType.INT32: "<i",
    ValueType.FLOAT32: "<f",
    ValueType.BOOL: "<?",
    ValueType.UINT64: "<Q",
    ValueType.INT64: "<q",
    ValueType.FLOAT64: "<d",
}


class TensorType(IntEnum):
    """Element types this reader can dequantize. GGUF defines many more."""

    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q8_0 = 8
    BF16 = 30


# (values per block, bytes per block)
_BLOCK = {
    TensorType.F32: (1, 4),
    TensorType.F16: (1, 2),
    TensorType.BF16: (1, 2),
    TensorType.Q8_0: (32, 34),  # fp16 scale + 32 x int8
    TensorType.Q4_0: (32, 18),  # fp16 scale + 16 bytes holding 32 x 4-bit values
}


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: tuple[int, ...]  # NumPy order: slowest-changing dimension first
    type_id: int
    offset: int  # bytes from the start of the data section
    n_bytes: int

    @property
    def type_name(self) -> str:
        try:
            return TensorType(self.type_id).name
        except ValueError:
            return f"type {self.type_id}"

    @property
    def n_elements(self) -> int:
        return int(np.prod(self.shape)) if self.shape else 1


def _tensor_bytes(type_id: int, n_elements: int) -> int:
    try:
        per_block, block_bytes = _BLOCK[TensorType(type_id)]
    except (ValueError, KeyError):
        return -1  # unknown type: size can't be computed, and it can't be read
    if n_elements % per_block:
        raise GGUFError(f"{n_elements} elements is not a whole number of {per_block}-blocks")
    return n_elements // per_block * block_bytes


class _Cursor:
    """Sequential little-endian reads from the start of the file."""

    def __init__(self, f: BinaryIO) -> None:
        self.f = f

    def take(self, n: int) -> bytes:
        b = self.f.read(n)
        if len(b) != n:
            raise GGUFError("file ended early")
        return b

    def scalar(self, vtype: ValueType) -> Any:
        fmt = _SCALAR[vtype]
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def u32(self) -> int:
        return self.scalar(ValueType.UINT32)

    def u64(self) -> int:
        return self.scalar(ValueType.UINT64)

    def string(self) -> str:
        return self.take(self.u64()).decode("utf-8", errors="replace")

    def value(self, vtype: ValueType) -> Any:
        if vtype is ValueType.STRING:
            return self.string()
        if vtype is ValueType.ARRAY:
            item_type = ValueType(self.u32())
            count = self.u64()
            return [self.value(item_type) for _ in range(count)]
        return self.scalar(vtype)


class GGUFFile:
    """Read a GGUF file's metadata and tensors. Implements the ModelFile protocol."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        with self.path.open("rb") as f:
            cur = _Cursor(f)

            # 1. header
            if (magic := cur.take(4)) != GGUF_MAGIC:
                raise GGUFError(f"not a GGUF file (magic {magic!r})")
            self.version = cur.u32()
            if self.version not in (2, 3):
                raise GGUFError(f"unsupported GGUF version {self.version}")
            n_tensors, n_kv = cur.u64(), cur.u64()

            # 2. metadata
            self._metadata: dict[str, Any] = {}
            for _ in range(n_kv):
                key = cur.string()
                self._metadata[key] = cur.value(ValueType(cur.u32()))

            # 3. tensor index
            self.tensors: dict[str, TensorInfo] = {}
            for _ in range(n_tensors):
                name = cur.string()
                dims = [cur.u64() for _ in range(cur.u32())]
                type_id, offset = cur.u32(), cur.u64()
                shape = tuple(reversed(dims))  # GGUF lists the fastest-changing dim first
                n_el = int(np.prod(shape)) if shape else 1
                self.tensors[name] = TensorInfo(
                    name, shape, type_id, offset, _tensor_bytes(type_id, n_el)
                )

            # 4. data starts at the next alignment boundary
            self.alignment = int(self._metadata.get("general.alignment", DEFAULT_ALIGNMENT))
            index_end = f.tell()
            self.data_offset = -(-index_end // self.alignment) * self.alignment

        self._map = np.memmap(self.path, dtype=np.uint8, mode="r")

    # --- ModelFile protocol ----------------------------------------------
    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def tensor(self, name: str) -> np.ndarray:
        """The tensor as float32, dequantized if needed, in NumPy shape order."""
        info = self.info(name)
        return dequantize(self.raw(name), info.type_id).reshape(info.shape)

    # --- extras ---------------------------------------------------------
    def info(self, name: str) -> TensorInfo:
        try:
            return self.tensors[name]
        except KeyError:
            raise KeyError(f"no tensor named {name!r}") from None

    def raw(self, name: str) -> np.ndarray:
        """The tensor's bytes exactly as stored: a view into the mapped file, no copy."""
        info = self.info(name)
        if info.n_bytes < 0:
            raise GGUFError(f"{name}: {info.type_name} is not supported by this reader")
        start = self.data_offset + info.offset
        return self._map[start : start + info.n_bytes]

    @property
    def architecture(self) -> str:
        return str(self._metadata.get("general.architecture", "unknown"))

    def arch_key(self, key: str, default: Any = None) -> Any:
        """Read an architecture-scoped key, e.g. arch_key('block_count') -> llama.block_count."""
        return self._metadata.get(f"{self.architecture}.{key}", default)

    def summary(self) -> dict[str, Any]:
        sizes = [t.n_bytes for t in self.tensors.values() if t.n_bytes > 0]
        return {
            "file": self.path.name,
            "file_bytes": self.path.stat().st_size,
            "gguf_version": self.version,
            "architecture": self.architecture,
            "name": self._metadata.get("general.name"),
            "metadata_keys": len(self._metadata),
            "tensors": len(self.tensors),
            "tensor_types": dict(Counter(t.type_name for t in self.tensors.values())),
            "parameters": sum(t.n_elements for t in self.tensors.values()),
            "weight_bytes": sum(sizes),
            "data_offset": self.data_offset,
            "alignment": self.alignment,
        }

    def close(self) -> None:
        mm = getattr(self._map, "_mmap", None)
        if mm is not None:
            # if arrays still point into the file, the OS unmaps it when they are freed
            with contextlib.suppress(BufferError):
                mm.close()

    def __enter__(self) -> "GGUFFile":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def dequantize(raw: np.ndarray, type_id: int) -> np.ndarray:
    """Turn stored bytes into float32 values (flat)."""
    t = TensorType(type_id)
    if t is TensorType.F32:
        return raw.view("<f4").astype(np.float32)
    if t is TensorType.F16:
        return raw.view("<f2").astype(np.float32)
    if t is TensorType.BF16:
        # bfloat16 is the top 16 bits of a float32
        return (raw.view("<u2").astype(np.uint32) << 16).view(np.float32)

    blocks = raw.reshape(-1, _BLOCK[t][1])
    scale = blocks[:, :2].copy().view("<f2").astype(np.float32)  # (n_blocks, 1)
    if t is TensorType.Q8_0:
        q = blocks[:, 2:].view(np.int8).astype(np.float32)  # (n_blocks, 32)
        return (scale * q).reshape(-1)
    if t is TensorType.Q4_0:
        packed = blocks[:, 2:]  # (n_blocks, 16): low nibbles = values 0-15, high = 16-31
        q = np.concatenate([packed & 0x0F, packed >> 4], axis=1).astype(np.int8) - 8
        return (scale * q.astype(np.float32)).reshape(-1)
    raise GGUFError(f"cannot dequantize {t.name}")
