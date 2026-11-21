"""local_ai_inference: a custom local AI inference stack.

Layers, top to bottom:
    api     -> OpenAI-compatible HTTP contract (step-01)
    server  -> lifecycle, admission, prompt prep, output control, streaming, metrics (step-01)
    engine  -> tokens in, tokens out: a NumPy Llama with a KV cache (step-03), or a stub
    parser  -> reads a GGUF file (step-02)
"""

import argparse
import json

__version__ = "0.4.0"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="local-ai-inference", description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="version", version=f"local-ai-inference {__version__}")
    sub = parser.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="start the OpenAI-compatible server (default)")
    serve.add_argument("--host", default="127.0.0.1", help="bind address (default: localhost)")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--model", help="GGUF file to serve (default: the stub engine)")
    serve.add_argument("--ctx", type=int, default=2048, help="max context tokens (KV cache size)")
    serve.add_argument(
        "--token-delay",
        type=float,
        default=0.02,
        help="seconds between stub tokens, so streaming is visible",
    )

    inspect = sub.add_parser("inspect", help="summarise a GGUF file")
    inspect.add_argument("path")
    inspect.add_argument("--keys", action="store_true", help="also list every metadata key")
    inspect.add_argument(
        "--tensors", type=int, default=0, metavar="N", help="list the first N tensors"
    )

    bench = sub.add_parser("bench", help="benchmark any OpenAI-compatible server")
    bench.add_argument("--url", default="http://127.0.0.1:8000/v1", help="server base URL")
    bench.add_argument("--model", required=True, help="model ID the server expects")
    bench.add_argument("--label", help="name for this run (default: the URL)")
    bench.add_argument("--runs", type=int, default=3)
    bench.add_argument("--max-tokens", type=int, default=64)
    bench.add_argument("--csv", help="append results to this CSV file")

    args = parser.parse_args(argv)
    if args.command == "bench":
        from local_ai_inference.bench import append_csv, bench, print_table

        rows = bench(args.url, args.model, runs=args.runs, max_tokens=args.max_tokens)
        label = args.label or args.url
        print_table(rows, label)
        if args.csv:
            append_csv(args.csv, rows, label)
    elif args.command == "inspect":
        _inspect(args.path, show_keys=args.keys, n_tensors=args.tensors)
    else:
        if args.command is None:  # `local-ai-inference` alone means serve, with defaults
            args = serve.parse_args([])
        _serve(args.host, args.port, args.model, args.ctx, args.token_delay)


def _serve(host: str, port: int, model: str | None, ctx: int, token_delay: float) -> None:
    import uvicorn

    from local_ai_inference.app import create_app
    from local_ai_inference.engine import StubEngine

    if model:
        from local_ai_inference.engine.gguf_engine import GGUFEngine

        engine = GGUFEngine(model, max_context=ctx)
    else:
        engine = StubEngine(token_delay=token_delay)
    uvicorn.run(create_app(engine), host=host, port=port)


def _inspect(path: str, *, show_keys: bool, n_tensors: int) -> None:
    from local_ai_inference.parser import GGUFFile

    with GGUFFile(path) as f:
        print(json.dumps(f.summary(), indent=2))
        if show_keys:
            print("\nmetadata:")
            for key, value in f.metadata.items():
                if isinstance(value, list):
                    value = f"[{len(value)} items]"
                elif isinstance(value, str) and len(value) > 80:
                    value = value[:77] + "..."
                print(f"  {key} = {value}")
        if n_tensors:
            print("\ntensors:")
            for info in list(f.tensors.values())[:n_tensors]:
                print(f"  {info.name:32} {info.type_name:6} {info.shape}")
