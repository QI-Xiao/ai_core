"""Tests for the typed Attachment protocol that flows through the chat pipeline.

Covers the four contracts that matter:
  1. `Attachment` model accepts `kind` plus optional `url` / `data`.
  2. `BaseTool.run_with_attachments` default returns `(content, [])`.
  3. `_execute_tool_calls` calls the new method and packs attachments into ToolResult.
  4. `stream_chat` emits one `{type: "attachment", ...}` SSE event per attachment.
"""
from typing import Any, Iterator

import pytest

from ai_core import Attachment, BaseTool
from ai_core.providers.base import BaseProvider
from ai_core.schemas.chat import ChatRequest, Message
from ai_core.schemas.provider import ProviderResponse
from ai_core.schemas.tool import ToolCall, ToolResult
from ai_core.services.chat_service import _execute_tool_calls, stream_chat


# ── Attachment model ─────────────────────────────────────────────────────────

def test_attachment_with_url_only():
    a = Attachment(kind="image", url="http://example.com/img.png")
    assert a.kind == "image"
    assert a.url == "http://example.com/img.png"
    assert a.data is None


def test_attachment_with_data_only():
    a = Attachment(kind="timeseries", data={"series": [1, 2, 3]})
    assert a.kind == "timeseries"
    assert a.url is None
    assert a.data == {"series": [1, 2, 3]}


def test_attachment_with_both():
    a = Attachment(kind="future_kind", url="/foo", data={"x": 1})
    assert a.url == "/foo"
    assert a.data == {"x": 1}


def test_attachment_kind_is_required():
    with pytest.raises(Exception):
        Attachment(url="/foo")  # type: ignore[call-arg]


# ── BaseTool default ─────────────────────────────────────────────────────────

class _NoSideEffectTool(BaseTool):
    name = "no_side_effect"
    description = "Returns plain text only."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    def run(self, input: dict[str, Any]) -> str:
        return "hello"


def test_run_with_attachments_default_returns_empty_list():
    tool = _NoSideEffectTool()
    content, attachments = tool.run_with_attachments({})
    assert content == "hello"
    assert attachments == []


# ── _execute_tool_calls dispatch ─────────────────────────────────────────────

class _PlotTool(BaseTool):
    name = "make_plot"
    description = "Returns a synthetic plot URL."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    def run(self, input: dict[str, Any]) -> str:
        return "rendered"

    def run_with_attachments(self, input):
        return "rendered", [Attachment(kind="image", url="http://example.com/p.png")]


class _DummyResponse:
    def __init__(self, tool_calls):
        self.tool_calls = tool_calls


def test_execute_tool_calls_packs_attachments_into_tool_result():
    tool = _PlotTool()
    response = _DummyResponse([ToolCall(id="c1", name="make_plot", input={})])
    results = _execute_tool_calls(response, {tool.name: tool})

    assert len(results) == 1
    r = results[0]
    assert isinstance(r, ToolResult)
    assert r.content == "rendered"
    assert len(r.attachments) == 1
    assert r.attachments[0].kind == "image"
    assert r.attachments[0].url == "http://example.com/p.png"


def test_execute_tool_calls_handles_unknown_tool():
    response = _DummyResponse([ToolCall(id="c1", name="ghost", input={})])
    results = _execute_tool_calls(response, {})
    assert results[0].is_error
    assert "Unknown tool" in results[0].content
    assert results[0].attachments == []


def test_execute_tool_calls_captures_exceptions_as_error_results():
    class _BoomTool(_NoSideEffectTool):
        name = "boom"
        def run_with_attachments(self, input):
            raise RuntimeError("kaboom")

    tool = _BoomTool()
    response = _DummyResponse([ToolCall(id="c1", name="boom", input={})])
    results = _execute_tool_calls(response, {tool.name: tool})
    assert results[0].is_error
    assert "kaboom" in results[0].content
    assert results[0].attachments == []


# ── stream_chat SSE emission ─────────────────────────────────────────────────

class _StubProvider(BaseProvider):
    """Two-turn provider: turn 1 calls make_plot; turn 2 returns a final message."""

    def __init__(self) -> None:
        self._calls = 0

    def generate_response(self, messages, system=None, tools=None) -> ProviderResponse:
        self._calls += 1
        if self._calls == 1:
            return ProviderResponse(
                content="",
                model="stub",
                tool_calls=[ToolCall(id="c1", name="make_plot", input={})],
            )
        return ProviderResponse(content="done", model="stub", tool_calls=[])

    def stream_response(self, messages, system=None) -> Iterator[str]:
        yield "done"

    def supports_tools(self) -> bool:
        return True

    def supports_streaming(self) -> bool:
        return True

    def model_name(self) -> str:
        return "stub"

    def build_tool_result_messages(self, messages, response, results):
        new = list(messages)
        for r in results:
            new.append({"role": "tool", "content": r.content, "tool_call_id": r.tool_call_id})
        return new


def test_stream_chat_emits_attachment_event_for_typed_attachment(monkeypatch):
    monkeypatch.setattr(
        "ai_core.services.chat_service.get_provider",
        lambda model, catalog=None: _StubProvider(),
    )
    request = ChatRequest(
        messages=[Message(role="user", content="make me a plot")],
        model="stub",
    )
    _, _, events = stream_chat(
        request, tools=[_PlotTool()], system_prompt=None, model_catalog=None,
    )
    events_list = list(events)

    types = [e["type"] for e in events_list]
    assert "tool_call" in types
    assert "attachment" in types

    attachment = next(e for e in events_list if e["type"] == "attachment")
    assert attachment["kind"] == "image"
    assert attachment["url"] == "http://example.com/p.png"
    assert attachment["data"] is None


def test_stream_chat_emits_no_attachment_for_pure_text_tool(monkeypatch):
    """A tool that doesn't override run_with_attachments shouldn't produce any
    attachment events. Guards against regressions in the default."""
    class _ChatStubProvider(_StubProvider):
        def generate_response(self, messages, system=None, tools=None) -> ProviderResponse:
            self._calls += 1
            if self._calls == 1:
                return ProviderResponse(
                    content="",
                    model="stub",
                    tool_calls=[ToolCall(id="c1", name="no_side_effect", input={})],
                )
            return ProviderResponse(content="done", model="stub", tool_calls=[])

    monkeypatch.setattr(
        "ai_core.services.chat_service.get_provider",
        lambda model, catalog=None: _ChatStubProvider(),
    )
    request = ChatRequest(
        messages=[Message(role="user", content="say hi")],
        model="stub",
    )
    _, _, events = stream_chat(
        request, tools=[_NoSideEffectTool()], system_prompt=None, model_catalog=None,
    )
    events_list = list(events)
    assert not any(e["type"] == "attachment" for e in events_list)
