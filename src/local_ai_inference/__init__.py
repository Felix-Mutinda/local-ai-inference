"""local_ai_inference: a custom local AI inference stack.

Layers, top to bottom:
    api     -> OpenAI-compatible HTTP contract (step-01)
    server  -> lifecycle, admission, prompt prep, output control, streaming, metrics (step-01)
    engine  -> tokens in, tokens out; a stub engine until step-03
    parser  -> reads a GGUF file (step-02)
"""

import argparse

__version__ = "0.1.0"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="local-ai-inference", description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="version", version=f"local-ai-inference {__version__}")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: localhost)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--token-delay",
        type=float,
        default=0.02,
        help="seconds between stub tokens, so streaming is visible",
    )
    args = parser.parse_args(argv)

    import uvicorn

    from local_ai_inference.app import create_app
    from local_ai_inference.engine import StubEngine

    uvicorn.run(
        create_app(StubEngine(token_delay=args.token_delay)), host=args.host, port=args.port
    )
