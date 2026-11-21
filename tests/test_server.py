import asyncio

import pytest

from local_ai_inference.engine import GenerationParams, StubEngine
from local_ai_inference.server import Done, InferenceServer, ModelState, NotReady, Overloaded


def test_lifecycle_moves_to_ready_and_records_load_time() -> None:
    server = InferenceServer(StubEngine())
    assert server.state is ModelState.LOADING
    server.load()
    assert server.state is ModelState.READY
    assert server.metrics.model_load._value.get() >= 0


def test_admission_refuses_before_ready() -> None:
    with pytest.raises(NotReady):
        InferenceServer(StubEngine()).admit()


def test_admission_refuses_when_queue_is_full() -> None:
    server = InferenceServer(StubEngine(), max_queue=0)
    server.load()
    with pytest.raises(Overloaded):
        server.admit()


def test_requests_run_one_at_a_time() -> None:
    server = InferenceServer(StubEngine(reply="abc", token_delay=0.01))
    server.load()
    order: list[str] = []

    async def run(name: str) -> None:
        ids = server.prepare([_Msg("user", "hi")])
        async for event in server.stream(ids, GenerationParams()):
            if not isinstance(event, Done):
                order.append(name)

    async def both() -> None:
        await asyncio.gather(run("A"), run("B"))

    asyncio.run(both())
    assert order in (["A"] * 3 + ["B"] * 3, ["B"] * 3 + ["A"] * 3)


def test_client_disconnect_cancels_generation() -> None:
    server = InferenceServer(StubEngine())
    server.load()

    async def gone() -> bool:
        return True

    async def run() -> list:
        ids = server.prepare([_Msg("user", "hi")])
        return [e async for e in server.stream(ids, GenerationParams(), is_disconnected=gone)]

    events = asyncio.run(run())
    assert events[-1] == Done("cancelled", events[-1].prompt_tokens, 1)


class _Msg:
    def __init__(self, role: str, content: str) -> None:
        self.role, self.content = role, content
