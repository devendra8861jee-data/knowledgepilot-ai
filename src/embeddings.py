"""Embedding backends.

* ``sentence-transformers`` (default): real semantic embeddings, runs locally and
  for free. The model is downloaded from Hugging Face on first use.
* ``hashing``: a tiny, deterministic, *lexical* embedder with no downloads. It is
  meant for offline tests and quick demos. It matches shared words, not meaning.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Raised when embeddings cannot be produced (user-facing message)."""


class Embedder(Protocol):
    """Interface every embedding backend implements."""

    name: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,:][0-9]+)*")
_STOPWORDS = frozenset(
    "a an the of to in on for and or is are was were be been by with at as it its this that "
    "do does did how what which who when where why can i you we they from per".split()
)


class HashingEmbedder:
    """Feature-hashing bag-of-words embedder (L2-normalised)."""

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _tokens(self, text: str) -> list[str]:
        return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]

    def _vector(self, text: str) -> list[float]:
        vec = np.zeros(self.dim, dtype=np.float64)
        for token in self._tokens(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            sign = 1.0 if (value >> 63) & 1 else -1.0
            vec[value % self.dim] += sign
        norm = math.sqrt(float(np.dot(vec, vec)))
        if norm > 0:
            vec /= norm
        return vec.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class SentenceTransformerEmbedder:
    """Wrapper around ``sentence_transformers.SentenceTransformer`` (lazy loaded)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingError(
                    "The 'sentence-transformers' package is not installed. Run "
                    "'pip install -r requirements.txt' or switch the embedding backend to "
                    "'hashing' for an offline test."
                ) from exc
            try:
                logger.info("Loading embedding model %s", self.model_name)
                self._model = SentenceTransformer(self.model_name)
            except Exception as exc:
                raise EmbeddingError(
                    f"Could not load embedding model '{self.model_name}'. The first run needs an "
                    "internet connection to download it from Hugging Face. "
                    f"Details: {type(exc).__name__}"
                ) from exc
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        vectors = model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False, batch_size=32
        )
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def get_embedder(backend: str, model_name: str = "") -> Embedder:
    """Create an embedder from a backend name."""
    if backend == "hashing":
        return HashingEmbedder()
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder(model_name or "sentence-transformers/all-MiniLM-L6-v2")
    raise ValueError(f"Unknown embedding backend '{backend}'.")
