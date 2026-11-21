# local-ai-inference

A custom implementation of the local AI inference stack: a GGUF **parser**, a NumPy **engine**,
a **server** and an OpenAI-compatible **API**, built one layer at a time.

> A custom server, not a production one. It serves one request at a time on purpose, so each
> layer stays small enough to understand.

## Quick start

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                    # create .venv, install dependencies and dev tools
uv run pytest              # fast tests
uv run ruff check .        # lint
uv run ruff format .       # format
uv run local-ai-inference        # start the server on http://127.0.0.1:8000
```

Tests that need a real model file are marked `model` and skipped by default:

```bash
uv run pytest -m model
```

## Steps

Each step is a git tag, so you can check out the repo at any stage.

| Tag | Step | Status |
|---|---|---|
| `step-00` | Tooling: uv, ruff, pytest, CI | done |
| `step-01` | Serving layer: API + server + stub engine | done |
| `step-02` | Parser: GGUF reader | next |
| `step-03` | Engine: forward pass, KV cache, sampler | planned |
| `step-04` | Measure: metrics, memory, cold vs warm, parity | planned |

## Step 01: serving layer

![Step 01: serving layer](docs/diagrams/step-01-serving-layer.svg)

The whole request path works before any maths exists. A stub engine replays a fixed reply
through a real byte-level tokenizer, so the API, the server and the boundaries between them can
be built and tested first. Step 03 swaps in the real engine behind the same `Engine` protocol.

**What each layer owns**

| Layer | Owns | Code |
|---|---|---|
| API | Routes, OpenAI request/response shapes, error format | `api/routes.py`, `api/schemas.py` |
| Server | Lifecycle, admission, prompt prep, output control, streaming, metrics | `server/` |
| Engine | Tokens in, tokens out | `engine/protocol.py`, `engine/stub.py` |
| Parser | Reading the model file (step-02) | `parser/__init__.py` (protocol only) |

**Try it**

```bash
uv run local-ai-inference

# in another terminal
curl -s localhost:8000/health
curl -s localhost:8000/v1/models
curl -sN localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model": "stub-echo", "messages": [{"role": "user", "content": "Hi"}], "stream": true}'
curl -s localhost:8000/metrics | grep '^lai_'
```

Or with the OpenAI SDK, unchanged:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
reply = client.chat.completions.create(
    model="stub-echo", messages=[{"role": "user", "content": "Hi"}]
)
print(reply.choices[0].message.content)
```

**API contract**

| Route | Behaviour |
|---|---|
| `GET /v1/models` | The one loaded model |
| `POST /v1/chat/completions` | Supports `messages`, `max_tokens` / `max_completion_tokens`, `temperature`, `top_p`, `seed`, `stop`, `stream`. Other fields are ignored. |
| `GET /health` | `200 ready`, or `503` with `loading`, `warming` or `failed` |
| `GET /metrics` | Prometheus: time to first token, decode tokens/s, queue depth, context tokens, load time, requests by finish reason |

Errors use the OpenAI shape: `404 model_not_found`, `400 context_length_exceeded`,
`429 server_busy`, `503 model_not_ready`.

## Layout

```
src/local_ai_inference/
  api/        HTTP contract: routes and schemas
  server/     lifecycle, admission, prompt prep, output control, streaming, metrics
  engine/     Engine protocol + stub engine (real engine in step-03)
  parser/     ModelFile protocol (GGUF reader in step-02)
  app.py      wires the layers together
tests/        fast tests; real-model tests are marked `model`
docs/diagrams one diagram per step
models/       downloaded GGUF files (git-ignored)
```

## License

MIT
