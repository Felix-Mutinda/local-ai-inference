"""Write a tiny random-weight Llama GGUF (with a byte-level tokenizer) for fast tests."""

from pathlib import Path

import gguf
import numpy as np

SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
CHATML = (
    "{% for m in messages %}{{ '<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>\n' }}"
    "{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)
CFG = dict(n_layer=2, d_model=64, n_head=4, n_kv_head=2, d_ff=128, context=64)


def byte_alphabet() -> list[str]:
    """GPT-2's printable stand-in for each of the 256 bytes (byte-level BPE)."""
    keep = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    chars, extra = {}, 0
    for b in range(256):
        if b in keep:
            chars[b] = chr(b)
        else:
            chars[b] = chr(256 + extra)
            extra += 1
    return [chars[b] for b in range(256)]


def write_tiny_llama(path: Path, *, seed: int = 0, quantize: bool = True) -> Path:
    rng = np.random.default_rng(seed)
    c = CFG
    hd = c["d_model"] // c["n_head"]
    tokens = SPECIALS + byte_alphabet() + ["Ġh", "Ġhi"]
    merges = ["Ġ h", "Ġh i"]

    w = gguf.GGUFWriter(path, "llama")
    w.add_name("tiny-llama")
    w.add_block_count(c["n_layer"])
    w.add_context_length(c["context"])
    w.add_embedding_length(c["d_model"])
    w.add_feed_forward_length(c["d_ff"])
    w.add_head_count(c["n_head"])
    w.add_head_count_kv(c["n_kv_head"])
    w.add_rope_freq_base(10000.0)
    w.add_layer_norm_rms_eps(1e-5)
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("default")
    w.add_token_list(tokens)
    w.add_token_types([3] * len(SPECIALS) + [1] * (len(tokens) - len(SPECIALS)))
    w.add_token_merges(merges)
    w.add_bos_token_id(1)
    w.add_eos_token_id(2)
    w.add_chat_template(CHATML)

    def add(name: str, shape: tuple[int, ...], scale: float = 0.2) -> None:
        data = (rng.standard_normal(shape) * scale).astype(np.float32)
        if quantize and len(shape) == 2:
            w.add_tensor(
                name,
                gguf.quants.quantize(data, gguf.GGMLQuantizationType.Q8_0),
                raw_dtype=gguf.GGMLQuantizationType.Q8_0,
            )
        else:
            w.add_tensor(name, data)

    d, kv = c["d_model"], c["n_kv_head"] * hd
    add("token_embd.weight", (len(tokens), d), 1.0)
    for i in range(c["n_layer"]):
        w.add_tensor(f"blk.{i}.attn_norm.weight", np.ones(d, np.float32))
        add(f"blk.{i}.attn_q.weight", (d, d))
        add(f"blk.{i}.attn_k.weight", (kv, d))
        add(f"blk.{i}.attn_v.weight", (kv, d))
        add(f"blk.{i}.attn_output.weight", (d, d))
        w.add_tensor(f"blk.{i}.ffn_norm.weight", np.ones(d, np.float32))
        add(f"blk.{i}.ffn_gate.weight", (c["d_ff"], d))
        add(f"blk.{i}.ffn_up.weight", (c["d_ff"], d))
        add(f"blk.{i}.ffn_down.weight", (d, c["d_ff"]))
    w.add_tensor("output_norm.weight", np.ones(d, np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path
