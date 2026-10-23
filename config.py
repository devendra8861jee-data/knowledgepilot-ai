"""Central configuration for KnowledgePilot AI.

All settings come from environment variables (optionally loaded from a local
``.env`` file) with safe defaults. API keys are never hardcoded and are hidden
from ``repr`` output so they cannot leak into logs by accident.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

FALLBACK_MESSAGE = "I could not find sufficient evidence in the uploaded documents."

SUPPORTED_EMBEDDING_BACKENDS = ("sentence-transformers", "hashing")
SUPPORTED_LLM_PROVIDERS = ("none", "openai_compatible")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logging.getLogger(__name__).warning("Invalid integer for %s; using %s", name, default)
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logging.getLogger(__name__).warning("Invalid number for %s; using %s", name, default)
        return default


@dataclass(frozen=True)
class Settings:
    """Runtime settings. Immutable; use :meth:`with_overrides` to change values."""

    # Chunking / retrieval
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 4
    min_relevance: float = 0.10

    # Embeddings
    embedding_backend: str = "sentence-transformers"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # LLM (OpenAI-compatible: OpenAI, Groq, OpenRouter, Gemini compat endpoint, Ollama, ...)
    llm_provider: str = "none"
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = field(default="", repr=False)
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 2
    llm_temperature: float = 0.0

    # Optional pricing (USD per 1M tokens) used only for cost *estimates*
    price_input_per_m: float = 0.0
    price_output_per_m: float = 0.0

    # Storage
    data_dir: Path = BASE_DIR / "data"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def has_api_key(self) -> bool:
        return bool(self.llm_api_key.strip())

    def with_overrides(self, **kwargs: object) -> "Settings":
        """Return a copy with selected fields replaced."""
        return replace(self, **kwargs)  # type: ignore[arg-type]


def load_settings() -> Settings:
    """Build :class:`Settings` from environment variables."""
    return Settings(
        chunk_size=_env_int("CHUNK_SIZE", 800),
        chunk_overlap=_env_int("CHUNK_OVERLAP", 120),
        top_k=_env_int("TOP_K", 4),
        min_relevance=_env_float("MIN_RELEVANCE", 0.10),
        embedding_backend=os.getenv("EMBEDDING_BACKEND", "sentence-transformers").strip(),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ).strip(),
        llm_provider=os.getenv("LLM_PROVIDER", "none").strip().lower(),
        llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini").strip(),
        llm_base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").strip(),
        llm_api_key=os.getenv("LLM_API_KEY", "").strip(),
        llm_timeout_s=_env_float("LLM_TIMEOUT_S", 60.0),
        llm_max_retries=_env_int("LLM_MAX_RETRIES", 2),
        llm_temperature=_env_float("LLM_TEMPERATURE", 0.0),
        price_input_per_m=_env_float("PRICE_INPUT_PER_M", 0.0),
        price_output_per_m=_env_float("PRICE_OUTPUT_PER_M", 0.0),
        data_dir=Path(os.getenv("DATA_DIR", str(BASE_DIR / "data"))),
    )


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging once (safe to call repeatedly)."""
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
