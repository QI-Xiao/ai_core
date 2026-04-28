from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Iterator

from ai_core.schemas.provider import ProviderResponse
from ai_core.schemas.tool import ToolResult

if TYPE_CHECKING:
    from ai_core.tools.base import BaseTool


class BaseProvider(ABC):
    """All providers implement this interface. Upper layers never touch SDK types.

    messages are plain dicts: {"role": "user"|"assistant", "content": str}
    The tool loop may add provider-specific continuation dicts — only the
    originating provider's build_tool_result_messages should produce those.
    """

    @abstractmethod
    def generate_response(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: "list[BaseTool] | None" = None,
    ) -> ProviderResponse: ...

    @abstractmethod
    def stream_response(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
    ) -> Iterator[str]: ...

    @abstractmethod
    def supports_tools(self) -> bool: ...

    @abstractmethod
    def supports_streaming(self) -> bool: ...

    @abstractmethod
    def model_name(self) -> str: ...

    def build_tool_result_messages(
        self,
        messages: list[dict[str, Any]],
        response: ProviderResponse,
        results: list[ToolResult],
    ) -> list[dict[str, Any]]:
        """Append the assistant tool-use turn + tool results and return the new list.

        Only providers where supports_tools() is True need to implement this.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support tools")
