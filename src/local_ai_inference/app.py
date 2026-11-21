"""Wire the layers together: API routes on top of a server that owns an engine."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.concurrency import run_in_threadpool

from local_ai_inference import __version__
from local_ai_inference.api import router
from local_ai_inference.engine import Engine, StubEngine
from local_ai_inference.server import InferenceServer


def create_app(engine: Engine | None = None, *, max_queue: int = 8) -> FastAPI:
    server = InferenceServer(engine or StubEngine(), max_queue=max_queue)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await run_in_threadpool(server.load)  # loading can be slow; keep the event loop free
        yield

    app = FastAPI(title="local-ai-inference", version=__version__, lifespan=lifespan)
    app.state.server = server
    app.include_router(router)
    return app
