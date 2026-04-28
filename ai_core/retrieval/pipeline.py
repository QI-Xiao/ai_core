"""RAG pipeline: lazily indexes a knowledge directory and exposes a search method."""
from pathlib import Path

from ai_core.retrieval.chunker import chunk_knowledge_base
from ai_core.retrieval.store import TFIDFStore


class RAGIndex:
    """Lazily-built TF-IDF index over a directory of markdown files.

    Construct once with the project's knowledge_dir; pass into SearchDocsTool.
    Indexing happens on the first call to search().
    """

    def __init__(self, knowledge_dir: str | Path) -> None:
        self._knowledge_dir = Path(knowledge_dir)
        self._store = TFIDFStore()

    def _ensure_indexed(self) -> None:
        if self._store.is_indexed:
            return
        if not self._knowledge_dir.exists():
            return
        chunks = chunk_knowledge_base(self._knowledge_dir)
        self._store.index(chunks)

    def search(self, query: str, n: int = 4) -> str:
        """Return formatted context from the most relevant doc chunks.

        Returns an empty string if nothing relevant is found.
        """
        self._ensure_indexed()
        results = self._store.search(query, n=n)
        if not results:
            return ""

        parts = [
            f"[{r['source']} — {r['section']}]\n{r['text']}"
            for r in results
        ]
        return "\n\n---\n\n".join(parts)
