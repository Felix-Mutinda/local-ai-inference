"""Check this engine against llama.cpp on the same GGUF file: tokens, logits, perplexity.

Needs llama-cpp-python, which is not a project dependency (it compiles C++):
    uv pip install llama-cpp-python
    uv run python scripts/compare_llama_cpp.py models/SmolLM2-135M-Instruct-Q8_0.gguf

Both engines read the same text one token at a time ("teacher forcing"), so their
predictions can be compared position by position, even where greedy outputs would drift.
"""

import sys

import numpy as np
from llama_cpp import Llama as LlamaCpp

from local_ai_inference.engine.gguf_engine import GGUFEngine

# A fixed, varied passage (written for this repo) so the numbers are reproducible.
TEXT = """\
Nairobi wakes early. Before six, matatus are already weaving along Thika Road, their
conductors calling out fares while music spills from open windows. By eight the city's
offices fill, and by noon the queues at the food kiosks stretch onto the pavement.

A small language model on a laptop works differently from one in a data centre. It cannot
lean on racks of accelerators, so every design choice shows up as seconds or megabytes.
The file format decides how quickly the weights can be found. The engine decides how many
bytes must move for each new token. The server decides who waits, and for how long.

Rain changes everything. A short storm in April can turn a twenty-minute drive into two
hours, and the same is true of systems: a cache that was large enough on a quiet day fills
up when every user arrives at once. Engineers who have lived through both learn to measure
before they guess, and to write down what they measured, on which machine, and why.

Three questions are worth asking of any local model. Does it give the same answers as a
trusted reference? How long does the first word take to appear? And how much memory does it
hold while it thinks? Everything else is detail, until it isn't.
"""


def perplexity(logits: np.ndarray, ids: list[int]) -> float:
    z = logits[:-1].astype(np.float64)
    z -= z.max(axis=1, keepdims=True)
    log_probs = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return float(np.exp(-log_probs[np.arange(len(ids) - 1), ids[1:]].mean()))


def main(path: str) -> None:
    ours = GGUFEngine(path)
    ours.load()
    ref = LlamaCpp(model_path=path, n_ctx=1024, logits_all=True, verbose=False)

    ids = ours.tokenize(TEXT)
    same_tokens = ids == ref.tokenize(TEXT.encode(), add_bos=False, special=True)

    ref.reset()
    ref.eval(ids)
    r = np.array(ref.scores[: len(ids)])
    cache = ours.model.new_cache(len(ids))
    o = np.stack([ours.model.forward([t], cache) for t in ids])

    print(f"tokens compared       {len(ids)}")
    print(f"same tokenization     {same_tokens}")
    print(
        f"perplexity            ours {perplexity(o, ids):.3f}   llama.cpp {perplexity(r, ids):.3f}"
    )
    print(f"same top-1 token      {(o.argmax(1) == r.argmax(1)).mean():.1%} of positions")
    print(f"logit correlation     {np.corrcoef(o.ravel(), r.ravel())[0, 1]:.5f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "models/SmolLM2-135M-Instruct-Q8_0.gguf")
