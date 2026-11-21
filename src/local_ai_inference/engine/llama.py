"""A Llama-family forward pass in NumPy, with a KV cache.

Each block:  x += attention(rms_norm(x))  then  x += feed_forward(rms_norm(x))
Attention uses grouped-query heads (fewer K/V heads than Q heads) and rotary
position embeddings. The cache keeps every past token's K and V, so each new
token costs one pass over the weights instead of re-reading the whole prompt.
"""

import math
from dataclasses import dataclass

import numpy as np

from local_ai_inference.parser import GGUFFile


@dataclass(frozen=True)
class LlamaConfig:
    n_layer: int
    d_model: int
    n_head: int
    n_kv_head: int
    d_ff: int
    rope_base: float
    norm_eps: float
    context_length: int
    vocab_size: int

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    @classmethod
    def from_gguf(cls, f: GGUFFile) -> "LlamaConfig":
        if f.architecture != "llama":
            raise NotImplementedError(f"architecture {f.architecture!r}: only llama")
        n_head = f.arch_key("attention.head_count")
        return cls(
            n_layer=f.arch_key("block_count"),
            d_model=f.arch_key("embedding_length"),
            n_head=n_head,
            n_kv_head=f.arch_key("attention.head_count_kv", n_head),
            d_ff=f.arch_key("feed_forward_length"),
            rope_base=float(f.arch_key("rope.freq_base", 10000.0)),
            norm_eps=float(f.arch_key("attention.layer_norm_rms_epsilon", 1e-5)),
            context_length=f.arch_key("context_length"),
            vocab_size=f.info("token_embd.weight").shape[0],
        )


class KVCache:
    """Keys and values of every token seen so far, per layer and K/V head."""

    def __init__(self, cfg: LlamaConfig, max_tokens: int) -> None:
        shape = (cfg.n_layer, cfg.n_kv_head, max_tokens, cfg.head_dim)
        self.k = np.zeros(shape, dtype=np.float32)
        self.v = np.zeros(shape, dtype=np.float32)
        self.pos = 0  # tokens stored so far

    @property
    def capacity(self) -> int:
        return self.k.shape[2]

    @property
    def nbytes(self) -> int:
        return self.k.nbytes + self.v.nbytes


@dataclass
class Block:
    attn_norm: np.ndarray
    wq: np.ndarray
    wk: np.ndarray
    wv: np.ndarray
    wo: np.ndarray
    ffn_norm: np.ndarray
    w_gate: np.ndarray
    w_up: np.ndarray
    w_down: np.ndarray


class Llama:
    def __init__(self, f: GGUFFile) -> None:
        self.cfg = cfg = LlamaConfig.from_gguf(f)
        # Dequantize every weight to float32 once, at load. Simple and BLAS-friendly,
        # at the cost of ~4x the memory of the Q8_0 file.
        self.embed = f.tensor("token_embd.weight")
        self.blocks = [
            Block(
                attn_norm=f.tensor(f"blk.{i}.attn_norm.weight"),
                wq=f.tensor(f"blk.{i}.attn_q.weight"),
                wk=f.tensor(f"blk.{i}.attn_k.weight"),
                wv=f.tensor(f"blk.{i}.attn_v.weight"),
                wo=f.tensor(f"blk.{i}.attn_output.weight"),
                ffn_norm=f.tensor(f"blk.{i}.ffn_norm.weight"),
                w_gate=f.tensor(f"blk.{i}.ffn_gate.weight"),
                w_up=f.tensor(f"blk.{i}.ffn_up.weight"),
                w_down=f.tensor(f"blk.{i}.ffn_down.weight"),
            )
            for i in range(cfg.n_layer)
        ]
        self.out_norm = f.tensor("output_norm.weight")
        # Tied embeddings: with no separate output matrix, the embedding table is reused.
        self.out = f.tensor("output.weight") if "output.weight" in f.tensors else self.embed
        half = cfg.head_dim // 2
        self.inv_freq = cfg.rope_base ** (-np.arange(half, dtype=np.float64) * 2 / cfg.head_dim)

    @property
    def weight_bytes(self) -> int:
        arrays = [self.embed, self.out_norm, *(a for b in self.blocks for a in vars(b).values())]
        if self.out is not self.embed:
            arrays.append(self.out)
        return sum(a.nbytes for a in arrays)

    def new_cache(self, max_tokens: int) -> KVCache:
        return KVCache(self.cfg, max_tokens)

    def forward(self, tokens: list[int], cache: KVCache) -> np.ndarray:
        """Run new tokens through the model, extend the cache, return the last token's logits."""
        cfg = self.cfg
        n, start = len(tokens), cache.pos
        if start + n > cache.capacity:
            raise ValueError(f"cache full: {start} + {n} > {cache.capacity} tokens")
        positions = np.arange(start, start + n)
        x = self.embed[tokens]  # (n, d_model)

        for i, b in enumerate(self.blocks):
            h = rms_norm(x, b.attn_norm, cfg.norm_eps)
            q = (h @ b.wq.T).reshape(n, cfg.n_head, cfg.head_dim)
            k = (h @ b.wk.T).reshape(n, cfg.n_kv_head, cfg.head_dim)
            v = (h @ b.wv.T).reshape(n, cfg.n_kv_head, cfg.head_dim)
            q, k = self._rope(q, positions), self._rope(k, positions)

            cache.k[i, :, start : start + n] = k.transpose(1, 0, 2)
            cache.v[i, :, start : start + n] = v.transpose(1, 0, 2)
            attn = self._attention(q, cache.k[i, :, : start + n], cache.v[i, :, : start + n], start)
            x = x + attn @ b.wo.T

            h = rms_norm(x, b.ffn_norm, cfg.norm_eps)
            x = x + (silu(h @ b.w_gate.T) * (h @ b.w_up.T)) @ b.w_down.T

        cache.pos += n
        last = rms_norm(x[-1], self.out_norm, cfg.norm_eps)
        return last @ self.out.T  # (vocab_size,)

    def _rope(self, x: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """Rotate adjacent pairs (x0, x1), (x2, x3)... by a position-dependent angle.

        GGUF llama weights are stored so that the pairs are adjacent (llama.cpp's
        "normal" RoPE), not split into two halves as in the original checkpoint.
        """
        angles = np.outer(positions, self.inv_freq)  # (n, head_dim/2)
        cos = np.cos(angles)[:, None, :].astype(np.float32)
        sin = np.sin(angles)[:, None, :].astype(np.float32)
        x0, x1 = x[..., 0::2], x[..., 1::2]
        out = np.empty_like(x)
        out[..., 0::2] = x0 * cos - x1 * sin
        out[..., 1::2] = x0 * sin + x1 * cos
        return out

    def _attention(self, q: np.ndarray, k: np.ndarray, v: np.ndarray, start: int) -> np.ndarray:
        """Grouped-query causal attention. q: (n, H, hd); k, v: (KV, T, hd)."""
        cfg = self.cfg
        n, t = q.shape[0], k.shape[1]
        group = cfg.n_head // cfg.n_kv_head  # query heads sharing one K/V head
        qg = q.reshape(n, cfg.n_kv_head, group, cfg.head_dim)
        # a Python float keeps float32; an np.float64 scalar would silently upcast everything
        scores = np.einsum("nkgd,ktd->kgnt", qg, k) / math.sqrt(cfg.head_dim)
        # token i (at position start+i) may only look at positions <= start+i
        future = np.arange(t)[None, :] > (start + np.arange(n))[:, None]
        scores = np.where(future, -np.inf, scores)
        weights = softmax(scores, axis=-1)
        out = np.einsum("kgnt,ktd->nkgd", weights, v)
        return out.reshape(n, cfg.d_model)


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * weight


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)
