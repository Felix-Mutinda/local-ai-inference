"""The GGUF reader, checked against files written by the official `gguf` package."""

from pathlib import Path

import gguf
import numpy as np
import pytest

from local_ai_inference import main
from local_ai_inference.parser import GGUFError, GGUFFile, ModelFile

Q = gguf.GGMLQuantizationType
RNG = np.random.default_rng(0)
WEIGHTS = {
    "f32": (RNG.standard_normal((2, 3)).astype(np.float32), None),
    "f16": (RNG.standard_normal((4, 8)).astype(np.float16), None),
    "bf16": (RNG.standard_normal((2, 32)).astype(np.float32), Q.BF16),
    "q8_0": (RNG.standard_normal((3, 64)).astype(np.float32), Q.Q8_0),
    "q4_0": (RNG.standard_normal((3, 64)).astype(np.float32), Q.Q4_0),
}


@pytest.fixture(scope="module")
def gguf_path(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("gguf") / "tiny.gguf"
    w = gguf.GGUFWriter(path, "llama")
    w.add_name("tiny-test")
    w.add_block_count(2)
    w.add_context_length(128)
    w.add_float32("llama.rope.freq_base", 10000.0)
    w.add_bool("test.flag", True)
    w.add_int64("test.big", -(2**40))
    w.add_array("tokenizer.ggml.tokens", ["<s>", "</s>", "hello", "ñ"])
    w.add_array("test.ints", [1, 2, 3])
    w.add_chat_template("{% for m in messages %}{{ m.content }}{% endfor %}")
    for name, (data, qtype) in WEIGHTS.items():
        if qtype is None:
            w.add_tensor(name, data)
        else:
            w.add_tensor(name, gguf.quants.quantize(data, qtype), raw_dtype=qtype)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


@pytest.fixture(scope="module")
def f(gguf_path) -> GGUFFile:
    return GGUFFile(gguf_path)


def test_header(f: GGUFFile) -> None:
    assert f.version == 3
    assert f.architecture == "llama"
    assert len(f.tensors) == len(WEIGHTS)


def test_metadata_values_and_types(f: GGUFFile) -> None:
    m = f.metadata
    assert m["general.name"] == "tiny-test"
    assert f.arch_key("block_count") == 2
    assert f.arch_key("context_length") == 128
    assert m["llama.rope.freq_base"] == pytest.approx(10000.0)
    assert m["test.flag"] is True
    assert m["test.big"] == -(2**40)
    assert m["tokenizer.ggml.tokens"] == ["<s>", "</s>", "hello", "ñ"]
    assert m["test.ints"] == [1, 2, 3]
    assert "messages" in m["tokenizer.chat_template"]


def test_data_section_is_aligned(f: GGUFFile) -> None:
    assert f.data_offset % f.alignment == 0
    assert all(t.offset % f.alignment == 0 for t in f.tensors.values())


@pytest.mark.parametrize("name", ["f32", "f16"])
def test_unquantized_tensors_are_exact(f: GGUFFile, name: str) -> None:
    data, _ = WEIGHTS[name]
    out = f.tensor(name)
    assert out.shape == data.shape  # GGUF stores dims reversed; we restore NumPy order
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, data.astype(np.float32))


@pytest.mark.parametrize("name", ["bf16", "q8_0", "q4_0"])
def test_quantized_tensors_match_the_reference_dequantizer(f: GGUFFile, name: str) -> None:
    data, qtype = WEIGHTS[name]
    ours = f.tensor(name)
    reference = gguf.quants.dequantize(f.raw(name).reshape(data.shape[0], -1), qtype)
    assert ours.shape == data.shape
    np.testing.assert_allclose(ours, reference, rtol=0, atol=0)


@pytest.mark.parametrize("name, tolerance", [("bf16", 0.02), ("q8_0", 0.03), ("q4_0", 0.5)])
def test_fewer_bits_means_more_error(f: GGUFFile, name: str, tolerance: float) -> None:
    data, _ = WEIGHTS[name]
    assert np.abs(f.tensor(name) - data).max() < tolerance


def test_raw_is_a_view_into_the_mapped_file(f: GGUFFile) -> None:
    raw = f.raw("q8_0")
    assert np.shares_memory(raw, f._map)  # no copy: the OS pages it in on demand
    assert raw.nbytes == 3 * 64 // 32 * 34


def test_agrees_with_official_reader_on_every_tensor(gguf_path: Path, f: GGUFFile) -> None:
    ref = gguf.GGUFReader(gguf_path)
    assert {t.name for t in ref.tensors} == set(f.tensors)
    for t in ref.tensors:
        ours = f.info(t.name)
        assert ours.type_id == int(t.tensor_type)
        assert ours.n_bytes == int(t.n_bytes)
        assert f.data_offset + ours.offset == int(t.data_offset)


def test_satisfies_the_model_file_protocol(f: GGUFFile) -> None:
    model_file: ModelFile = f
    assert model_file.metadata["general.architecture"] == "llama"


def test_summary(f: GGUFFile) -> None:
    s = f.summary()
    assert s["tensors"] == 5
    assert s["tensor_types"] == {"F32": 1, "F16": 1, "BF16": 1, "Q8_0": 1, "Q4_0": 1}
    assert s["parameters"] == sum(d.size for d, _ in WEIGHTS.values())


def test_rejects_a_file_that_is_not_gguf(tmp_path: Path) -> None:
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"NOPE" + b"\0" * 32)
    with pytest.raises(GGUFError, match="not a GGUF file"):
        GGUFFile(bad)


def test_rejects_a_truncated_file(gguf_path: Path, tmp_path: Path) -> None:
    cut = tmp_path / "cut.gguf"
    cut.write_bytes(gguf_path.read_bytes()[:40])
    with pytest.raises(GGUFError, match="ended early"):
        GGUFFile(cut)


def test_cli_inspect(gguf_path: Path, capsys) -> None:
    main(["inspect", str(gguf_path), "--keys", "--tensors", "2"])
    out = capsys.readouterr().out
    assert '"architecture": "llama"' in out
    assert "tokenizer.ggml.tokens = [4 items]" in out


MODEL = Path("models/SmolLM2-135M-Instruct-Q8_0.gguf")


@pytest.mark.model
def test_real_smollm2_file() -> None:
    if not MODEL.exists():
        pytest.skip(f"download it first: see README step-02 ({MODEL})")
    with GGUFFile(MODEL) as m:
        assert m.architecture == "llama"
        assert m.arch_key("block_count") == 30
        assert m.arch_key("embedding_length") == 576
        assert "tokenizer.chat_template" in m.metadata
        embed = m.tensor("token_embd.weight")
        assert embed.shape == (len(m.metadata["tokenizer.ggml.tokens"]), 576)
        assert np.isfinite(embed).all()
