from typing import Any

from ai_core.retrieval.pipeline import RAGIndex
from ai_core.tools.base import BaseTool


class SearchDocsTool(BaseTool):
    """Generic RAG-backed documentation search tool.

    Each project constructs this with its own RAGIndex and (optionally) a
    domain-specific description so the model knows what's in the corpus.
    """

    name = "search_docs"

    def __init__(
        self,
        rag: RAGIndex,
        description: str | None = None,
    ) -> None:
        self._rag = rag
        self.description = description or (
            "Search the project documentation for explanations, methodology, "
            "data formats, API details, or operational notes. "
            "Use this when the user asks 'how does X work', 'what is Y', or needs "
            "background context rather than live data."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The question or topic to search for in the documentation.",
                }
            },
            "required": ["query"],
        }

    def run(self, input: dict[str, Any]) -> str:
        result = self._rag.search(input["query"])
        return result if result else "No relevant documentation found for that query."
