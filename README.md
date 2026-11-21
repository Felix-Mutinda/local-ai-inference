# local-ai-inference

A custom implementation of the local AI inference stack: a GGUF **parser**, a NumPy **engine**,
a **server** and an OpenAI-compatible **API**, built one layer at a time.

> A custom server, not a production one. It serves one request at a time on purpose, so each
> layer stays small enough to understand.

## Quick start

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                    # create .venv, install dev tools
uv run pytest              # fast tests
uv run ruff check .        # lint
uv run ruff format .       # format
uv run local-ai-inf        # prints the version
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
| `step-01` | Walking skeleton: API + server + fake engine | next |
| `step-02` | Parser: GGUF reader | planned |
| `step-03` | Engine: forward pass, KV cache, sampler | planned |
| `step-04` | Measure: metrics, memory, cold vs warm, parity | planned |

## Layout

```
src/local_ai_inf/   the package
tests/              fast tests (synthetic fixtures) and model tests
models/             downloaded GGUF files (git-ignored)
```

## License

MIT