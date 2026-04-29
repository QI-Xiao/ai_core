from abc import ABC, abstractmethod
from typing import Any

from ai_core.schemas.attachment import Attachment


class BaseTool(ABC):
    name: str
    description: str

    @property
    @abstractmethod
    def input_schema(self) -> dict[str, Any]: ...

    @abstractmethod
    def run(self, input: dict[str, Any]) -> str: ...

    def run_with_attachments(
        self, input: dict[str, Any]
    ) -> tuple[str, list[Attachment]]:
        """Run the tool and return (content, attachments).

        Override in tools that produce side-effect output (rendered images,
        structured datasets, etc.). The default delegates to `run` and
        returns no attachments.

        Each `Attachment` is forwarded to the SSE stream as a typed event;
        `ai_core` never inspects `attachment.kind`, so consumers are free to
        introduce new kinds without library changes.
        """
        return self.run(input), []

    def to_anthropic_spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def to_openai_spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }
