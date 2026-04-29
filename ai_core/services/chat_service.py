"""Orchestrates session history, tool loop, and provider call. Provider-agnostic.

Project-specific behaviour (system prompt, tools, model catalog) is supplied by
the caller — typically the create_app() factory binds a project's config and
hands it to the API routes.
"""
import uuid
from typing import Any, Iterator

from ai_core.providers.base import BaseProvider
from ai_core.providers.router import get_provider
from ai_core.schemas.chat import ChatRequest, ChatResponse, Message
from ai_core.schemas.tool import ToolResult
from ai_core.services import session as session_store
from ai_core.tools.base import BaseTool

_MAX_TOOL_ITERATIONS = 5


def chat(
    request: ChatRequest,
    *,
    tools: list[BaseTool],
    system_prompt: str | None,
    model_catalog: list[dict] | None = None,
) -> ChatResponse:
    session_id = request.session_id or str(uuid.uuid4())
    provider = get_provider(request.model, catalog=model_catalog)
    tool_map = {t.name: t for t in tools}

    history = session_store.get_history(session_id)
    api_messages = _to_api_messages(history + request.messages)

    response = _tool_loop(provider, api_messages, system_prompt, tools, tool_map)

    assistant_msg = Message(role="assistant", content=response.content)
    session_store.append_messages(session_id, request.messages + [assistant_msg])

    return ChatResponse(message=assistant_msg, session_id=session_id, model=response.model)


def stream_chat(
    request: ChatRequest,
    *,
    tools: list[BaseTool],
    system_prompt: str | None,
    model_catalog: list[dict] | None = None,
) -> tuple[str, str, Iterator[dict[str, Any]]]:
    """Returns (session_id, model_name, event_iterator).

    Events are typed dicts:
      {"type": "tool_call",  "name": "list_stations"}
      {"type": "attachment", "kind": ..., ...}
      {"type": "token",      "token": "Here are..."}
    """
    session_id = request.session_id or str(uuid.uuid4())
    provider = get_provider(request.model, catalog=model_catalog)
    tool_map = {t.name: t for t in tools}

    history = session_store.get_history(session_id)
    api_messages = _to_api_messages(history + request.messages)

    def _generate() -> Iterator[dict[str, Any]]:
        collected: list[str] = []

        if provider.supports_tools() and tools:
            messages = api_messages
            for _ in range(_MAX_TOOL_ITERATIONS):
                resp = provider.generate_response(
                    messages, system=system_prompt, tools=tools
                )
                if not resp.tool_calls:
                    break
                for tc in resp.tool_calls:
                    yield {"type": "tool_call", "name": tc.name}
                results = _execute_tool_calls(resp, tool_map)
                for r in results:
                    for att in r.attachments:
                        yield {
                            "type": "attachment",
                            "kind": att.kind,
                            "url":  att.url,
                            "data": att.data,
                        }
                messages = provider.build_tool_result_messages(messages, resp, results)
            for token in provider.stream_response(messages, system=system_prompt):
                collected.append(token)
                yield {"type": "token", "token": token}
        else:
            for token in provider.stream_response(api_messages, system=system_prompt):
                collected.append(token)
                yield {"type": "token", "token": token}

        assistant_msg = Message(role="assistant", content="".join(collected))
        session_store.append_messages(session_id, request.messages + [assistant_msg])

    return session_id, provider.model_name(), _generate()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_api_messages(messages: list[Message]) -> list[dict[str, Any]]:
    return [{"role": m.role, "content": m.content} for m in messages]


def _tool_loop(
    provider: BaseProvider,
    api_messages: list[dict[str, Any]],
    system_prompt: str | None,
    tools: list[BaseTool],
    tool_map: dict[str, BaseTool],
):
    use_tools = tools if (provider.supports_tools() and tools) else None
    response = None
    for _ in range(_MAX_TOOL_ITERATIONS):
        response = provider.generate_response(
            api_messages, system=system_prompt, tools=use_tools
        )
        if not response.tool_calls:
            return response
        results = _execute_tool_calls(response, tool_map)
        api_messages = provider.build_tool_result_messages(api_messages, response, results)
    return response


def _execute_tool_calls(response, tool_map: dict[str, BaseTool]) -> list[ToolResult]:
    results = []
    for tc in response.tool_calls:
        tool = tool_map.get(tc.name)
        if tool is None:
            results.append(ToolResult(
                tool_call_id=tc.id, name=tc.name,
                content=f"Unknown tool: {tc.name!r}", is_error=True,
            ))
            continue
        try:
            content, attachments = tool.run_with_attachments(tc.input)
            results.append(ToolResult(
                tool_call_id=tc.id, name=tc.name,
                content=content, attachments=attachments,
            ))
        except Exception as exc:
            results.append(ToolResult(
                tool_call_id=tc.id, name=tc.name, content=str(exc), is_error=True,
            ))
    return results
