"""local_ai_inference: a custom local AI inference stack.

Layers, each added in its own step:
    parser  -> reads a GGUF file (step-02)
    engine  -> runs the model (step-03)
    server  -> lifecycle, admission, streaming, metrics (step-01)
    api     -> OpenAI-compatible HTTP contract (step-01)
"""

__version__ = "0.0.0"


def main() -> None:
    print(f"local-ai-inference {__version__}")
