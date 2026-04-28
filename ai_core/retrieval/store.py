"""TF-IDF in-memory retrieval store. Zero extra dependencies."""
import math
import re
from collections import Counter


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class TFIDFStore:
    """Lightweight TF-IDF store for small document sets (<1000 chunks)."""

    def __init__(self) -> None:
        self._docs: list[dict] = []
        self._idf: dict[str, float] = {}
        self._vectors: list[dict[str, float]] = []

    def index(self, docs: list[dict]) -> None:
        self._docs = docs
        tokenized = [_tokenize(d["text"]) for d in docs]
        N = len(docs)

        df: dict[str, int] = {}
        for tokens in tokenized:
            for term in set(tokens):
                df[term] = df.get(term, 0) + 1
        self._idf = {t: math.log((N + 1) / (f + 1)) for t, f in df.items()}

        self._vectors = []
        for tokens in tokenized:
            tf = Counter(tokens)
            total = len(tokens) or 1
            vec = {t: (c / total) * self._idf.get(t, 0.0) for t, c in tf.items()}
            self._vectors.append(vec)

    def search(self, query: str, n: int = 4) -> list[dict]:
        if not self._docs:
            return []

        q_tokens = _tokenize(query)
        q_tf = Counter(q_tokens)
        q_total = len(q_tokens) or 1
        q_vec = {t: (c / q_total) * self._idf.get(t, 0.0) for t, c in q_tf.items()}

        q_norm = math.sqrt(sum(v ** 2 for v in q_vec.values())) or 1.0

        scores: list[tuple[float, int]] = []
        for i, doc_vec in enumerate(self._vectors):
            dot = sum(q_vec.get(t, 0.0) * v for t, v in doc_vec.items())
            d_norm = math.sqrt(sum(v ** 2 for v in doc_vec.values())) or 1.0
            scores.append((dot / (q_norm * d_norm), i))

        scores.sort(reverse=True)
        return [
            self._docs[i]
            for score, i in scores[:n]
            if score > 0
        ]

    @property
    def is_indexed(self) -> bool:
        return bool(self._docs)
