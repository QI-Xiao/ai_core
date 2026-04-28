import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ai_core.schemas.chat import ChatRequest, ChatResponse
from ai_core.services import chat_service
from ai_core.tools.base import BaseTool


def make_router(
    *,
    tools: list[BaseTool],
    system_prompt: str | None,
    model_catalog: list[dict],
) -> APIRouter:
    router = APIRouter()

    @router.post("/ai/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        return chat_service.chat(
            request,
            tools=tools,
            system_prompt=system_prompt,
            model_catalog=model_catalog,
        )

    @router.post("/ai/chat/stream")
    def chat_stream(request: ChatRequest) -> StreamingResponse:
        session_id, model, events = chat_service.stream_chat(
            request,
            tools=tools,
            system_prompt=system_prompt,
            model_catalog=model_catalog,
        )

        def event_stream():
            yield f"data: {json.dumps({'type': 'meta', 'session_id': session_id, 'model': model})}\n\n"
            try:
                for event in events:
                    yield f"data: {json.dumps(event)}\n\n"
            except Exception as exc:
                msg = _friendly_error(exc)
                yield f"data: {json.dumps({'type': 'token', 'token': msg})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return router


def _friendly_error(exc: Exception) -> str:
    s = str(exc)
    if "529" in s or "overloaded" in s.lower():
        return "The AI provider is temporarily overloaded. Please try again in a moment."
    if "401" in s or "authentication" in s.lower():
        return "API authentication failed."
    if "timeout" in s.lower():
        return "The request timed out. Please try again."
    return f"An error occurred: {s}"
