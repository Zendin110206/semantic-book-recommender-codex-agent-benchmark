from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class BuildConfig:
    raw_books_path: Path = PROJECT_ROOT / "data" / "books.csv"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    min_description_chars: int = 120
    chunk_size: int = 700
    chunk_overlap: int = 120
    max_features: int = 35000
    svd_components: int = 256
    semantic_provider: str = os.getenv("SEMANTIC_PROVIDER", "tfidf").lower()
    metadata_provider: str = os.getenv("METADATA_PROVIDER", "keyword").lower()
    openai_embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    zero_shot_model: str = os.getenv("ZERO_SHOT_MODEL", "facebook/bart-large-mnli")
    emotion_model: str = os.getenv(
        "EMOTION_MODEL",
        "j-hartmann/emotion-english-distilroberta-base",
    )

    @property
    def processed_dir(self) -> Path:
        return self.artifacts_dir / "processed"

    @property
    def reports_dir(self) -> Path:
        return self.artifacts_dir / "reports"

    @property
    def models_dir(self) -> Path:
        return self.artifacts_dir / "models"

    @property
    def processed_books_path(self) -> Path:
        return self.processed_dir / "books_enriched.csv"

    @property
    def chunk_data_path(self) -> Path:
        return self.processed_dir / "book_chunks.csv"

    @property
    def tfidf_index_path(self) -> Path:
        return self.models_dir / "tfidf_semantic_index.pkl"

    @property
    def index_metadata_path(self) -> Path:
        return self.models_dir / "semantic_index_metadata.json"

    @property
    def chroma_dir(self) -> Path:
        return self.artifacts_dir / "chroma"

    @property
    def data_profile_path(self) -> Path:
        return self.reports_dir / "data_profile.json"

    @property
    def sample_recommendations_path(self) -> Path:
        return self.reports_dir / "sample_recommendations.csv"

    def ensure_directories(self) -> None:
        for path in (
            self.artifacts_dir,
            self.processed_dir,
            self.reports_dir,
            self.models_dir,
            self.chroma_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
