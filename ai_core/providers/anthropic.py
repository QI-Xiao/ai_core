from typing import TYPE_CHECKING, Any, Iterator

import anthropic

from ai_core.config import ANTHROPIC_API_KEY, ANTHROPIC_DEFAULT_MODEL
from ai_core.providers.base import BaseProvider
from ai_core.schemas.provider import ProviderResponse
from ai_core.schemas.tool import ToolCall, ToolResult

if TYPE_CHECKING:
    from ai_core.tools.base import BaseTool


class AnthropicProvider(BaseProvider):
    def __init__(self, model: str | None = None) -> None:
        self._client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self._model = model or ANTHROPIC_DEFAULT_MODEL

    def model_name(self) -> str:
        return self._model

    def supports_tools(self) -> bool:
        return True

    def supports_streaming(self) -> bool:
        return True

    def generate_response(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: "list[BaseTool] | None" = None,
    ) -> ProviderResponse:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": 4096,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = [t.to_anthropic_spec() for t in tools]

        response = self._client.messages.create(**kwargs)

        text_content = ""
        tool_calls: list[ToolCall] = []
        raw_content: list[dict[str, Any]] = []

        for block in response.content:
            if block.type == "text":
                text_content = block.text
                raw_content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, input=block.input))
                raw_content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })

        return ProviderResponse(
            content=text_content,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            stop_reason=response.stop_reason,
            tool_calls=tool_calls,
            raw_content=raw_content,
        )

    def build_tool_result_messages(
        self,
        messages: list[dict[str, Any]],
        response: ProviderResponse,
        results: list[ToolResult],
    ) -> list[dict[str, Any]]:
        return [
            *messages,
            {"role": "assistant", "content": response.raw_content},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": r.tool_call_id,
                        "content": r.content,
                        **({"is_error": True} if r.is_error else {}),
                    }
                    for r in results
                ],
            },
        ]

    def stream_response(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
    ) -> Iterator[str]:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": 4096,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system

        with self._client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                yield text
