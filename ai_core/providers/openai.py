import json
from typing import TYPE_CHECKING, Any, Iterator

from ai_core.config import OPENAI_API_KEY, OPENAI_DEFAULT_MODEL
from ai_core.providers.base import BaseProvider
from ai_core.schemas.provider import ProviderResponse
from ai_core.schemas.tool import ToolCall, ToolResult

if TYPE_CHECKING:
    from ai_core.tools.base import BaseTool


class OpenAIProvider(BaseProvider):
    def __init__(self, model: str | None = None) -> None:
        import openai as _openai
        self._client = _openai.OpenAI(api_key=OPENAI_API_KEY)
        self._model = model or OPENAI_DEFAULT_MODEL

    def model_name(self) -> str:
        return self._model

    def supports_tools(self) -> bool:
        return True

    def supports_streaming(self) -> bool:
        return True

    def _with_system(self, messages: list[dict[str, Any]], system: str | None) -> list[dict]:
        if system:
            return [{"role": "system", "content": system}, *messages]
        return messages

    def generate_response(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: "list[BaseTool] | None" = None,
    ) -> ProviderResponse:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": self._with_system(messages, system),
        }
        if tools:
            kwargs["tools"] = [t.to_openai_spec() for t in tools]
            kwargs["tool_choice"] = "auto"

        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        msg = choice.message

        tool_calls: list[ToolCall] = []
        raw_tool_calls: list[dict[str, Any]] = []
        for tc in msg.tool_calls or []:
            args = json.loads(tc.function.arguments)
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, input=args))
            raw_tool_calls.append({
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            })

        assistant_message: dict[str, Any] = {"role": "assistant", "content": msg.content}
        if raw_tool_calls:
            assistant_message["tool_calls"] = raw_tool_calls

        return ProviderResponse(
            content=msg.content or "",
            model=response.model,
            input_tokens=response.usage.prompt_tokens if response.usage else None,
            output_tokens=response.usage.completion_tokens if response.usage else None,
            stop_reason=choice.finish_reason,
            tool_calls=tool_calls,
            raw_content=[assistant_message],
        )

    def build_tool_result_messages(
        self,
        messages: list[dict[str, Any]],
        response: ProviderResponse,
        results: list[ToolResult],
    ) -> list[dict[str, Any]]:
        assistant_msg = response.raw_content[0]
        tool_msgs = [
            {"role": "tool", "tool_call_id": r.tool_call_id, "content": r.content}
            for r in results
        ]
        return [*messages, assistant_msg, *tool_msgs]

    def stream_response(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
    ) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=self._with_system(messages, system),
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
