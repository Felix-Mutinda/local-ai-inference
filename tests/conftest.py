import pytest
from fastapi.testclient import TestClient

from local_ai_inference.app import create_app
from local_ai_inference.engine import StubEngine

MODEL = "stub-echo"


@pytest.fixture
def engine() -> StubEngine:
    return StubEngine()


@pytest.fixture
def client(engine: StubEngine):
    with TestClient(create_app(engine)) as c:  # `with` runs startup: load + warm
        yield c


def chat(content: str = "Hi", **extra) -> dict:
    return {"model": MODEL, "messages": [{"role": "user", "content": content}], **extra}
