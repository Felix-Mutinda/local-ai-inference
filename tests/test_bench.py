import json

from local_ai_inference.bench import PROMPTS, append_csv, bench
from local_ai_inference.engine.stub import DEFAULT_REPLY
from tests.conftest import chat


def test_stream_can_report_usage_on_a_final_chunk(client) -> None:
    body = chat(stream=True, stream_options={"include_usage": True})
    with client.stream("POST", "/v1/chat/completions", json=body) as r:
        events = [
            json.loads(ln.removeprefix("data: "))
            for ln in r.iter_lines()
            if ln.startswith("data: ") and ln != "data: [DONE]"
        ]
    last = events[-1]
    assert last["choices"] == []
    assert last["usage"]["completion_tokens"] == len(DEFAULT_REPLY.encode())


def test_bench_measures_every_prompt(client, tmp_path) -> None:
    rows = bench("http://testserver/v1", "stub-echo", runs=2, max_tokens=8, client=client)
    assert [r["prompt"] for r in rows] == list(PROMPTS)
    for r in rows:
        assert r["completion_tokens"] == 8
        assert r["prompt_tokens"] > 0
        assert r["ttft_s"] >= 0
    assert rows[1]["prompt_tokens"] > rows[0]["prompt_tokens"]  # the long prompt is longer

    out = tmp_path / "bench.csv"
    append_csv(out, rows, "stub")
    append_csv(out, rows, "stub-again")
    assert len(out.read_text().splitlines()) == 1 + 2 * len(rows)  # one header, then rows
