import json

from fastapi.testclient import TestClient
from openai import OpenAI

from local_ai_inference.app import create_app
from local_ai_inference.engine import StubEngine
from local_ai_inference.engine.stub import DEFAULT_REPLY
from tests.conftest import MODEL, chat


def test_health_is_ready_after_startup(client) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ready"}


def test_models_lists_the_loaded_model(client) -> None:
    assert client.get("/v1/models").json()["data"][0]["id"] == MODEL


def test_chat_completion_returns_full_reply_and_usage(client) -> None:
    body = client.post("/v1/chat/completions", json=chat()).json()
    assert body["choices"][0]["message"]["content"] == DEFAULT_REPLY
    assert body["choices"][0]["finish_reason"] == "stop"
    usage = body["usage"]
    assert usage["completion_tokens"] == len(DEFAULT_REPLY.encode())
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_max_tokens_truncates_with_length(client) -> None:
    body = client.post("/v1/chat/completions", json=chat(max_tokens=5)).json()
    assert body["choices"][0]["message"]["content"] == DEFAULT_REPLY[:5]
    assert body["choices"][0]["finish_reason"] == "length"


def test_stop_string_ends_generation(client) -> None:
    body = client.post("/v1/chat/completions", json=chat(stop=["stub"])).json()
    assert body["choices"][0]["message"]["content"] == "Hello from the "
    assert body["choices"][0]["finish_reason"] == "stop"


def test_streaming_sends_sse_chunks_then_done(client) -> None:
    with client.stream("POST", "/v1/chat/completions", json=chat(stream=True)) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        lines = [ln for ln in r.iter_lines() if ln.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(ln.removeprefix("data: ")) for ln in lines[:-1]]
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert text == DEFAULT_REPLY
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_unknown_model_is_404_in_openai_error_shape(client) -> None:
    r = client.post("/v1/chat/completions", json={**chat(), "model": "gpt-4o"})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "model_not_found"


def test_prompt_longer_than_context_is_400() -> None:
    with TestClient(create_app(StubEngine(context_length=16))) as c:
        r = c.post("/v1/chat/completions", json=chat("x" * 100))
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "context_length_exceeded"


def test_metrics_report_requests_and_load_time(client) -> None:
    client.post("/v1/chat/completions", json=chat())
    text = client.get("/metrics").text
    assert 'lai_requests_total{finish_reason="stop"} 1.0' in text
    assert "lai_model_load_seconds" in text
    assert "lai_time_to_first_token_seconds_count 1.0" in text


def test_openai_sdk_works_unchanged(client) -> None:
    sdk = OpenAI(base_url="http://testserver/v1", api_key="unused", http_client=client)
    reply = sdk.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": "Hi"}], max_tokens=5
    )
    assert reply.choices[0].message.content == DEFAULT_REPLY[:5]

    streamed = sdk.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": "Hi"}], stream=True
    )
    assert "".join(c.choices[0].delta.content or "" for c in streamed) == DEFAULT_REPLY
