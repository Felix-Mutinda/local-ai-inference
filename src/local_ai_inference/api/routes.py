"""HTTP routes: translate between the API contract and the server.

No generation logic lives here. Each route validates, calls the server,
and shapes the result into the OpenAI format.
"""

import time
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from local_ai_inference.api.schemas import (
    AssistantMessage,
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionRequest,
    Choice,
    ChunkChoice,
    ChunkDelta,
    ModelCard,
    ModelList,
    Usage,
)
from local_ai_inference.engine.protocol import GenerationParams
from local_ai_inference.server import (
    Delta,
    Done,
    InferenceServer,
    ModelState,
    NotReady,
    Overloaded,
    PromptTooLong,
)

router = APIRouter()


def _server(request: Request) -> InferenceServer:
    return request.app.state.server


def _error(status: int, message: str, code: str) -> JSONResponse:
    body = {"error": {"message": message, "type": "invalid_request_error", "code": code}}
    return JSONResponse(status_code=status, content=body)


@router.get("/health")
def health(request: Request) -> JSONResponse:
    state = _server(request).state
    return JSONResponse(
        status_code=200 if state is ModelState.READY else 503, content={"status": state}
    )


@router.get("/metrics")
def metrics(request: Request) -> Response:
    m = _server(request).metrics
    return Response(content=m.render(), media_type=m.content_type)


@router.get("/v1/models")
def list_models(request: Request) -> ModelList:
    return ModelList(data=[ModelCard(id=_server(request).engine.info.model_id)])


@router.post("/v1/chat/completions", response_model=None)
async def chat_completions(
    body: ChatCompletionRequest, request: Request
) -> ChatCompletion | StreamingResponse | JSONResponse:
    server = _server(request)
    model_id = server.engine.info.model_id
    if body.model != model_id:
        return _error(
            404, f"model '{body.model}' is not loaded; use '{model_id}'", "model_not_found"
        )
    try:
        server.admit()
        prompt_ids = server.prepare(body.messages)
    except NotReady as exc:
        return _error(503, str(exc), "model_not_ready")
    except Overloaded as exc:
        return _error(429, str(exc), "server_busy")
    except PromptTooLong as exc:
        return _error(400, str(exc), "context_length_exceeded")

    params = GenerationParams(
        max_tokens=body.token_budget(),
        temperature=body.temperature,
        top_p=body.top_p,
        seed=body.seed,
    )
    events = server.stream(prompt_ids, params, body.stop_list(), request.is_disconnected)
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    if body.stream:
        usage = bool(body.stream_options and body.stream_options.include_usage)
        return StreamingResponse(
            _sse(events, completion_id, created, model_id, usage), media_type="text/event-stream"
        )

    text: list[str] = []
    done: Done | None = None
    async for event in events:
        if isinstance(event, Delta):
            text.append(event.text)
        else:
            done = event
    assert done is not None
    return ChatCompletion(
        id=completion_id,
        created=created,
        model=model_id,
        choices=[
            Choice(
                message=AssistantMessage(content="".join(text)), finish_reason=done.finish_reason
            )
        ],
        usage=Usage(
            prompt_tokens=done.prompt_tokens,
            completion_tokens=done.completion_tokens,
            total_tokens=done.prompt_tokens + done.completion_tokens,
        ),
    )


async def _sse(
    events: AsyncIterator[Delta | Done],
    completion_id: str,
    created: int,
    model_id: str,
    include_usage: bool = False,
) -> AsyncIterator[str]:
    def chunk(delta: ChunkDelta, finish: str | None = None) -> str:
        c = ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=model_id,
            choices=[ChunkChoice(delta=delta, finish_reason=finish)],
        )
        return f"data: {c.model_dump_json(exclude_none=True)}\n\n"

    yield chunk(ChunkDelta(role="assistant", content=""))
    async for event in events:
        if isinstance(event, Delta):
            yield chunk(ChunkDelta(content=event.text))
        else:
            yield chunk(ChunkDelta(), event.finish_reason)
            if include_usage:  # OpenAI's convention: one extra chunk, no choices, with usage
                last = ChatCompletionChunk(
                    id=completion_id,
                    created=created,
                    model=model_id,
                    choices=[],
                    usage=Usage(
                        prompt_tokens=event.prompt_tokens,
                        completion_tokens=event.completion_tokens,
                        total_tokens=event.prompt_tokens + event.completion_tokens,
                    ),
                )
                yield f"data: {last.model_dump_json(exclude_none=True)}\n\n"
    yield "data: [DONE]\n\n"
