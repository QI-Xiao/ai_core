"""vLLM provider — uses the OpenAI-compatible API exposed by vLLM."""
from typing import TYPE_CHECKING, Any, Iterator

from ai_core.config import AI_DEFAULT_MODEL, VLLM_BASE_URL, VLLM_DEFAULT_MODEL
from ai_core.providers.base import BaseProvider
from ai_core.schemas.provider import ProviderResponse

if TYPE_CHECKING:
    from ai_core.tools.base import BaseTool


class VLLMProvider(BaseProvider):
    def __init__(self) -> None:
        import openai as _openai
        self._client = _openai.OpenAI(
            api_key="vllm",  # vLLM ignores the key by default
            base_url=f"{VLLM_BASE_URL}/v1",
        )
        self._model = AI_DEFAULT_MODEL or VLLM_DEFAULT_MODEL

    def model_name(self) -> str:
        return self._model

    def supports_tools(self) -> bool:
        return False

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
        response = self._client.chat.completions.create(
            model=self._model,
            messages=self._with_system(messages, system),
        )
        choice = response.choices[0]
        return ProviderResponse(
            content=choice.message.content or "",
            model=self._model,
            stop_reason=choice.finish_reason,
        )

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
