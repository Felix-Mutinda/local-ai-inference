"""Benchmark any OpenAI-compatible server the same way: this one, llama-server, Ollama...

For each prompt it streams one greedy completion and records:
    time to first token   the wait before text starts (mostly prompt processing)
    prefill tokens/s      prompt tokens / time to first token
    decode tokens/s       generated tokens after the first / time spent generating them

Every server is measured through the same HTTP path, so the comparison is like for like.
Each request gets a unique tag, so servers with prompt caching can't skip the prompt.
"""

import csv
import json
import statistics
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import httpx

_PARAGRAPH = (
    "A local inference stack has layers. The API is the contract an application calls. "
    "The server decides which request runs and streams the result back. The engine turns "
    "token IDs into the next token. The parser reads the model file, and the hardware "
    "moves the weights from memory to the processor for every token it produces. "
)
PROMPTS = {
    "short": "What is the capital of France?",
    "long": _PARAGRAPH * 6 + "\nSummarise the text above in two sentences.",
}


def run_once(client: httpx.Client, url: str, model: str, prompt: str, max_tokens: int) -> dict:
    # A unique tag per request, so no server can answer from a cached copy of the prompt.
    tagged = f"[{uuid.uuid4().hex[:8]}] {prompt}"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": tagged}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = time.perf_counter()
    first: float | None = None
    text, chunks, usage = "", 0, None
    with client.stream("POST", f"{url}/chat/completions", json=body) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line.removeprefix("data: "))
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                piece = (choice.get("delta") or {}).get("content")
                if piece:
                    first = first or time.perf_counter()
                    text += piece
                    chunks += 1
    ended = time.perf_counter()

    generated = usage["completion_tokens"] if usage else chunks
    prompt_tokens = usage["prompt_tokens"] if usage else None
    ttft = (first or ended) - started
    decode_time = ended - first if first else 0
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": generated,
        "ttft_s": ttft,
        "prefill_tok_s": prompt_tokens / ttft if prompt_tokens and ttft else None,
        "decode_tok_s": (generated - 1) / decode_time if generated > 1 and decode_time else None,
        "text": text,
    }


def bench(
    url: str,
    model: str,
    *,
    runs: int = 3,
    max_tokens: int = 64,
    client: httpx.Client | None = None,
) -> list[dict]:
    url = url.rstrip("/")
    rows = []
    with client or httpx.Client(timeout=600) as client:
        run_once(client, url, model, "Hi", 4)  # warm-up: not measured
        for name, prompt in PROMPTS.items():
            results = [run_once(client, url, model, prompt, max_tokens) for _ in range(runs)]
            row = {"prompt": name, "runs": runs}
            for key in (
                "prompt_tokens",
                "completion_tokens",
                "ttft_s",
                "prefill_tok_s",
                "decode_tok_s",
            ):
                values = [r[key] for r in results if r[key] is not None]
                row[key] = statistics.median(values) if values else None
            row["text"] = results[0]["text"]
            rows.append(row)
    return rows


def print_table(rows: list[dict], label: str) -> None:
    print(f"\n{label}  (median of {rows[0]['runs']} runs, greedy, after one warm-up request)")
    print(
        f"{'prompt':7} {'prompt tok':>10} {'TTFT s':>8} {'prefill tok/s':>14} {'decode tok/s':>13}"
    )
    for r in rows:
        prefill = f"{r['prefill_tok_s']:.1f}" if r["prefill_tok_s"] else "-"
        decode = f"{r['decode_tok_s']:.1f}" if r["decode_tok_s"] else "-"
        print(
            f"{r['prompt']:7} {r['prompt_tokens'] or '-':>10} {r['ttft_s']:>8.2f} "
            f"{prefill:>14} {decode:>13}"
        )
    for r in rows:
        print(f"  {r['prompt']}: {r['text'][:90]!r}")


def append_csv(path: str | Path, rows: list[dict], label: str) -> None:
    path = Path(path)
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(
                [
                    "timestamp",
                    "label",
                    "prompt",
                    "prompt_tokens",
                    "completion_tokens",
                    "ttft_s",
                    "prefill_tok_s",
                    "decode_tok_s",
                    "text",
                ]
            )
        stamp = datetime.now(UTC).isoformat(timespec="seconds")
        for r in rows:
            w.writerow(
                [
                    stamp,
                    label,
                    r["prompt"],
                    r["prompt_tokens"],
                    r["completion_tokens"],
                    r["ttft_s"],
                    r["prefill_tok_s"],
                    r["decode_tok_s"],
                    r["text"],
                ]
            )
