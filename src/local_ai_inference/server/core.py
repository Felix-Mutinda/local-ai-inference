"""The server: everything between the HTTP contract and the engine's maths.

Lifecycle (load, warm, ready), admission (one slot, a bounded queue),
prompt preparation, output control, streaming with cancellation, and metrics.
"""

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import StrEnum

from starlette.concurrency import iterate_in_threadpool

from local_ai_inference.engine.protocol import Engine, GenerationParams
from local_ai_inference.server.metrics import Metrics
from local_ai_inference.server.output import StopMatcher
from local_ai_inference.server.prompt import Message, render_chat


class ModelState(StrEnum):
    LOADING = "loading"
    WARMING = "warming"
    READY = "ready"
    FAILED = "failed"


class NotReady(Exception):
    pass


class Overloaded(Exception):
    pass


class PromptTooLong(Exception):
    pass


@dataclass
class Delta:
    text: str


@dataclass
class Done:
    finish_reason: str  # "stop", "length" or "cancelled"
    prompt_tokens: int
    completion_tokens: int


class InferenceServer:
    def __init__(self, engine: Engine, *, max_queue: int = 8) -> None:
        self.engine = engine
        self.state = ModelState.LOADING
        self.metrics = Metrics()
        self.max_queue = max_queue
        self._slot = asyncio.Lock()  # one request at a time, on purpose
        self._waiting = 0

    # --- lifecycle -------------------------------------------------------
    def load(self) -> None:
        started = time.perf_counter()
        try:
            self.state = ModelState.LOADING
            self.engine.load()
            self.state = ModelState.WARMING
            warm = GenerationParams(max_tokens=1)
            list(self.engine.generate(self.engine.tokenize("Hi"), warm))
            self.state = ModelState.READY
        except Exception:
            self.state = ModelState.FAILED
            raise
        finally:
            self.metrics.model_load.set(time.perf_counter() - started)

    # --- admission -------------------------------------------------------
    def admit(self) -> None:
        if self.state is not ModelState.READY:
            self.metrics.rejected.labels("not_ready").inc()
            raise NotReady(f"model is {self.state}")
        if self._waiting >= self.max_queue:
            self.metrics.rejected.labels("queue_full").inc()
            raise Overloaded(f"{self._waiting} requests already waiting")

    @asynccontextmanager
    async def _hold_slot(self) -> AsyncIterator[None]:
        self._waiting += 1
        self.metrics.queue_depth.set(self._waiting)
        try:
            await self._slot.acquire()
        finally:
            self._waiting -= 1
            self.metrics.queue_depth.set(self._waiting)
        try:
            yield
        finally:
            self._slot.release()

    # --- prompt preparation ---------------------------------------------
    def prepare(self, messages: Sequence[Message]) -> list[int]:
        prompt = render_chat(messages, self.engine.info.chat_template)
        ids = self.engine.tokenize(prompt)
        if len(ids) >= self.engine.info.context_length:
            self.metrics.rejected.labels("prompt_too_long").inc()
            raise PromptTooLong(
                f"prompt is {len(ids)} tokens; context is {self.engine.info.context_length}"
            )
        return ids

    # --- generation, streaming, output control ---------------------------
    async def stream(
        self,
        prompt_ids: list[int],
        params: GenerationParams,
        stops: Sequence[str] = (),
        is_disconnected: Callable[[], Awaitable[bool]] | None = None,
    ) -> AsyncIterator[Delta | Done]:
        room = self.engine.info.context_length - len(prompt_ids)
        params.max_tokens = min(params.max_tokens, room)

        async with self._hold_slot():
            matcher = StopMatcher(stops)
            started = time.perf_counter()
            first_at: float | None = None
            out_ids: list[int] = []
            emitted = ""
            finish = "length"
            tokens = self.engine.generate(prompt_ids, params)
            try:
                async for token_id in iterate_in_threadpool(tokens):
                    if first_at is None:
                        first_at = time.perf_counter()
                        self.metrics.ttft.observe(first_at - started)
                    out_ids.append(token_id)
                    self.metrics.context_tokens.set(len(prompt_ids) + len(out_ids))

                    text = self.engine.detokenize(out_ids)
                    if text.endswith("�"):  # partial multi-byte character: wait
                        continue
                    safe, hit = matcher.push(text[len(emitted) :])
                    emitted = text
                    if safe:
                        yield Delta(safe)
                    if hit:
                        finish = "stop"
                        break
                    if is_disconnected is not None and await is_disconnected():
                        finish = "cancelled"
                        break
                else:
                    if rest := matcher.flush():
                        yield Delta(rest)
                    if len(out_ids) < params.max_tokens:
                        finish = "stop"  # the engine ended on its own (end-of-sequence)
            except (GeneratorExit, asyncio.CancelledError):
                finish = "cancelled"  # the client went away while we were streaming
                raise
            finally:
                params.cancel.set()
                tokens.close()
                ended = time.perf_counter()
                if first_at is not None and len(out_ids) > 1 and ended > first_at:
                    self.metrics.decode_tps.set((len(out_ids) - 1) / (ended - first_at))
                self.metrics.generated_tokens.inc(len(out_ids))
                self.metrics.requests.labels(finish).inc()

            yield Done(finish, len(prompt_ids), len(out_ids))
