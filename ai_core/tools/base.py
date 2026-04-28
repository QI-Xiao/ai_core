from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    name: str
    description: str

    @property
    @abstractmethod
    def input_schema(self) -> dict[str, Any]: ...

    @abstractmethod
    def run(self, input: dict[str, Any]) -> str: ...

    def run_with_metadata(self, input: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Run the tool and return (content_str, metadata_dict).

        Override in tools that produce side effects like files.
        Default: calls run() with empty metadata.
        """
        return self.run(input), {}

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
