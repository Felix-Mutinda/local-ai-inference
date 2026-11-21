"""The engine: tokenizer, forward pass, KV cache, sampler, and the Engine protocol."""

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from local_ai_inference.app import create_app
from local_ai_inference.engine import Engine, GenerationParams
from local_ai_inference.engine.gguf_engine import GGUFEngine
from local_ai_inference.engine.llama import rms_norm, silu, softmax
from local_ai_inference.engine.sampler import sample
from tests.tiny_llama import write_tiny_llama


@pytest.fixture(scope="module")
def engine(tmp_path_factory) -> GGUFEngine:
    e = GGUFEngine(write_tiny_llama(tmp_path_factory.mktemp("m") / "tiny-llama.gguf"))
    e.load()
    return e


# --- tokenizer --------------------------------------------------------------
def test_tokenizer_round_trips_text(engine: GGUFEngine) -> None:
    text = "hi there, naïve café 42!"
    assert engine.detokenize(engine.tokenize(text)) == text


def test_special_tokens_are_single_ids_and_merges_apply(engine: GGUFEngine) -> None:
    ids = engine.tokenize("<|im_start|>user hi<|im_end|>")
    assert ids[0] == 1 and ids[-1] == 2  # specials are never split into bytes
    assert engine.tokenizer.encode(" hi") == [260]  # the "Ġh" + "i" merge fired


# --- forward pass -----------------------------------------------------------
def reference_logits(model, tokens: list[int]) -> np.ndarray:
    """A deliberately plain re-implementation: no cache, a loop over heads, explicit RoPE."""
    cfg, hd = model.cfg, model.cfg.head_dim
    x = model.embed[tokens].astype(np.float64)
    n = len(tokens)

    def rope(v: np.ndarray) -> np.ndarray:  # v: (n, hd); rotate adjacent pairs
        out = v.copy()
        for p in range(n):
            for i in range(hd // 2):
                a = p * cfg.rope_base ** (-2 * i / hd)
                x0, x1 = v[p, 2 * i], v[p, 2 * i + 1]
                out[p, 2 * i] = x0 * np.cos(a) - x1 * np.sin(a)
                out[p, 2 * i + 1] = x0 * np.sin(a) + x1 * np.cos(a)
        return out

    for b in model.blocks:
        h = rms_norm(x, b.attn_norm, cfg.norm_eps)
        q, k, v = h @ b.wq.T, h @ b.wk.T, h @ b.wv.T
        heads = []
        for head in range(cfg.n_head):
            kv = head // (cfg.n_head // cfg.n_kv_head)
            qh = rope(q[:, head * hd : (head + 1) * hd])
            kh = rope(k[:, kv * hd : (kv + 1) * hd])
            vh = v[:, kv * hd : (kv + 1) * hd]
            s = qh @ kh.T / np.sqrt(hd) + np.triu(np.full((n, n), -np.inf), 1)
            heads.append(softmax(s) @ vh)
        x = x + np.concatenate(heads, axis=1) @ b.wo.T
        h = rms_norm(x, b.ffn_norm, cfg.norm_eps)
        x = x + (silu(h @ b.w_gate.T) * (h @ b.w_up.T)) @ b.w_down.T
    return rms_norm(x[-1], model.out_norm, cfg.norm_eps) @ model.out.T


def test_forward_matches_a_plain_reference(engine: GGUFEngine) -> None:
    tokens = engine.tokenize("<|im_start|>user hi there<|im_end|>")
    ours = engine.model.forward(tokens, engine.model.new_cache(64))
    np.testing.assert_allclose(ours, reference_logits(engine.model, tokens), atol=1e-3)


def test_kv_cache_gives_the_same_logits_as_recomputing(engine: GGUFEngine) -> None:
    tokens = engine.tokenize("hi there, general kenobi")
    m = engine.model
    full = m.forward(tokens, m.new_cache(64))
    cache = m.new_cache(64)
    m.forward(tokens[:3], cache)  # prefill a prefix...
    for t in tokens[3:]:  # ...then decode the rest one token at a time
        step = m.forward([t], cache)
    np.testing.assert_allclose(step, full, atol=1e-4)
    assert cache.pos == len(tokens)


def test_forward_stays_float32(engine: GGUFEngine) -> None:
    logits = engine.model.forward([1, 2, 3], engine.model.new_cache(8))
    assert logits.dtype == np.float32  # a float64 slip makes every matmul copy the weights


# --- sampler ----------------------------------------------------------------
def test_greedy_picks_the_largest_logit() -> None:
    assert sample(np.array([0.1, 3.0, 2.0]), 0.0, 1.0, np.random.default_rng()) == 1


def test_top_p_keeps_only_the_nucleus() -> None:
    logits = np.log(np.array([0.7, 0.2, 0.1]))
    picks = {sample(logits, 1.0, 0.5, np.random.default_rng(s)) for s in range(50)}
    assert picks == {0}  # token 0 alone covers p >= 0.5


def test_same_seed_same_sample() -> None:
    logits = np.random.default_rng(1).standard_normal(100)
    a = [sample(logits, 1.0, 0.9, np.random.default_rng(7)) for _ in range(5)]
    b = [sample(logits, 1.0, 0.9, np.random.default_rng(7)) for _ in range(5)]
    assert a == b


# --- engine protocol --------------------------------------------------------
def test_satisfies_the_engine_protocol(engine: GGUFEngine) -> None:
    e: Engine = engine
    assert e.info.model_id == "tiny-llama"
    assert e.info.chat_template is not None


def test_generation_is_deterministic_when_greedy(engine: GGUFEngine) -> None:
    ids = engine.tokenize("hi")
    run = lambda: list(engine.generate(ids, GenerationParams(max_tokens=6, temperature=0)))  # noqa: E731
    assert run() == run()


def test_generation_respects_max_tokens_and_cancel(engine: GGUFEngine) -> None:
    ids = engine.tokenize("hi")
    assert len(list(engine.generate(ids, GenerationParams(max_tokens=3, temperature=0)))) <= 3
    params = GenerationParams(max_tokens=20, temperature=0)
    out = []
    for token in engine.generate(ids, params):
        out.append(token)
        params.cancel.set()
    assert len(out) == 1


def test_served_end_to_end_through_the_api(tmp_path: Path) -> None:
    e = GGUFEngine(write_tiny_llama(tmp_path / "tiny-llama.gguf"))
    with TestClient(create_app(e)) as client:
        body = client.post(
            "/v1/chat/completions",
            json={
                "model": "tiny-llama",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 5,
                "temperature": 0,
            },
        ).json()
    assert body["usage"]["completion_tokens"] <= 5
    assert body["usage"]["prompt_tokens"] > 0


# --- the real model ---------------------------------------------------------
MODEL = Path("models/SmolLM2-135M-Instruct-Q8_0.gguf")


@pytest.mark.model
def test_real_model_answers_a_factual_question() -> None:
    if not MODEL.exists():
        pytest.skip(f"download it first: see README step-02 ({MODEL})")
    with TestClient(create_app(GGUFEngine(MODEL))) as client:
        body = client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL.stem,
                "messages": [{"role": "user", "content": "What is the capital of France?"}],
                "max_tokens": 16,
                "temperature": 0,
            },
        ).json()
    assert "Paris" in body["choices"][0]["message"]["content"]
