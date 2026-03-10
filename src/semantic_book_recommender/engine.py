from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd
import sklearn

from .config import BuildConfig
from .retrieval import build_semantic_index, load_semantic_index


DISPLAY_COLUMNS = [
    "full_title",
    "authors",
    "categories",
    "average_rating",
    "ratings_count",
    "num_pages",
    "published_year",
    "topic_display",
    "mood_display",
    "emotion_display",
    "audience_label",
    "reading_length",
    "reason",
    "semantic_score",
    "final_score",
]


@dataclass
class BookRecommender:
    books: pd.DataFrame
    index: Any
    config: BuildConfig

    @classmethod
    def from_artifacts(cls, config: BuildConfig | None = None) -> "BookRecommender":
        config = config or BuildConfig()
        if not config.processed_books_path.exists():
            raise FileNotFoundError(
                f"Missing processed catalog: {config.processed_books_path}. Run the build pipeline first."
            )

        _ensure_semantic_artifacts_current(config)

        books = pd.read_csv(config.processed_books_path)
        persist_path = config.chroma_dir if config.semantic_provider == "openai" else config.tfidf_index_path
        index = load_semantic_index(
            provider=config.semantic_provider,
            persist_path=persist_path,
            openai_model_name=config.openai_embedding_model,
        )
        books["book_id"] = books["book_id"].astype(str)
        return cls(books=books, index=index, config=config)

    def recommend_from_query(
        self,
        query: str,
        n_results: int = 10,
        topic: str | None = None,
        mood: str | None = None,
        min_rating: float = 0.0,
        max_pages: int | None = None,
        exclude_book_id: str | None = None,
        source_title: str | None = None,
    ) -> pd.DataFrame:
        if not query.strip():
            return self.curated_picks(
                n_results=n_results,
                topic=topic,
                mood=mood,
                min_rating=min_rating,
                max_pages=max_pages,
            )

        semantic_scores = dict(self.index.search(query=query, top_n_chunks=max(n_results * 12, 60)))
        candidates = self.books[self.books["book_id"].isin(semantic_scores)].copy()

        if exclude_book_id:
            candidates = candidates[candidates["book_id"] != str(exclude_book_id)]

        candidates = self._apply_filters(
            candidates=candidates,
            topic=topic,
            mood=mood,
            min_rating=min_rating,
            max_pages=max_pages,
        )

        if candidates.empty:
            return self.curated_picks(
                n_results=n_results,
                topic=topic,
                mood=mood,
                min_rating=min_rating,
                max_pages=max_pages,
            )

        candidates["semantic_score"] = candidates["book_id"].map(semantic_scores).fillna(0.0)
        candidates["topic_match_score"] = 1.0 if not topic else (candidates["topic_label"] == topic).astype(float)
        candidates["mood_match_score"] = 1.0 if not mood else (candidates["mood_label"] == mood).astype(float)
        candidates["final_score"] = (
            0.72 * candidates["semantic_score"]
            + 0.18 * candidates["quality_score"]
            + 0.05 * candidates["topic_match_score"]
            + 0.05 * candidates["mood_match_score"]
        )
        candidates["reason"] = candidates.apply(
            lambda row: self._build_reason(
                row=row,
                selected_topic=topic,
                selected_mood=mood,
                source_title=source_title,
            ),
            axis=1,
        )

        candidates = candidates.sort_values(["final_score", "ratings_count"], ascending=[False, False])
        return self._format_results(candidates.head(n_results))

    def recommend_similar_books(
        self,
        title: str,
        n_results: int = 10,
        topic: str | None = None,
        mood: str | None = None,
        min_rating: float = 0.0,
        max_pages: int | None = None,
    ) -> pd.DataFrame:
        anchor = self.find_best_title_match(title)
        if anchor is None:
            raise ValueError(f"No title match found for: {title}")

        return self.recommend_from_query(
            query=str(anchor["search_document"]),
            n_results=n_results,
            topic=topic,
            mood=mood,
            min_rating=min_rating,
            max_pages=max_pages,
            exclude_book_id=str(anchor["book_id"]),
            source_title=str(anchor["full_title"]),
        )

    def curated_picks(
        self,
        n_results: int = 10,
        topic: str | None = None,
        mood: str | None = None,
        min_rating: float = 0.0,
        max_pages: int | None = None,
    ) -> pd.DataFrame:
        candidates = self._apply_filters(
            candidates=self.books.copy(),
            topic=topic,
            mood=mood,
            min_rating=min_rating,
            max_pages=max_pages,
        )
        if candidates.empty:
            return pd.DataFrame(columns=DISPLAY_COLUMNS)

        candidates["semantic_score"] = 0.0
        candidates["final_score"] = candidates["quality_score"]
        candidates["reason"] = candidates.apply(
            lambda row: self._build_reason(row=row, selected_topic=topic, selected_mood=mood, source_title=None),
            axis=1,
        )
        candidates = candidates.sort_values(["quality_score", "ratings_count"], ascending=[False, False])
        return self._format_results(candidates.head(n_results))

    def find_best_title_match(self, title: str) -> pd.Series | None:
        if not title.strip():
            return None

        lowered = title.strip().lower()
        matches = self.books[self.books["full_title"].str.lower().str.contains(lowered, regex=False)].copy()
        if matches.empty:
            matches = self.books[self.books["title"].str.lower().str.contains(lowered, regex=False)].copy()
        if matches.empty:
            return None

        matches = matches.sort_values(["quality_score", "ratings_count"], ascending=[False, False])
        return matches.iloc[0]

    def available_titles(self, limit: int = 500) -> list[str]:
        titles = (
            self.books.sort_values(["ratings_count", "quality_score"], ascending=[False, False])["full_title"]
            .drop_duplicates()
            .head(limit)
            .tolist()
        )
        return titles

    def dataset_summary(self) -> dict[str, Any]:
        return {
            "books_after_cleaning": int(len(self.books)),
            "median_rating": round(float(self.books["average_rating"].median()), 2),
            "median_pages": round(float(self.books["num_pages"].median()), 0),
            "top_topics": self.books["topic_display"].value_counts().head(8).to_dict(),
            "top_moods": self.books["mood_display"].value_counts().head(8).to_dict(),
        }

    def _apply_filters(
        self,
        candidates: pd.DataFrame,
        topic: str | None,
        mood: str | None,
        min_rating: float,
        max_pages: int | None,
    ) -> pd.DataFrame:
        filtered = candidates[candidates["average_rating"] >= min_rating].copy()
        if topic:
            filtered = filtered[filtered["topic_label"] == topic]
        if mood:
            filtered = filtered[filtered["mood_label"] == mood]
        if max_pages is not None:
            filtered = filtered[filtered["num_pages"] <= max_pages]
        return filtered

    def _build_reason(
        self,
        row: pd.Series,
        selected_topic: str | None,
        selected_mood: str | None,
        source_title: str | None,
    ) -> str:
        parts: list[str] = []

        if source_title:
            parts.append(f"Semantic match to {source_title}")
        else:
            parts.append(f"Strong fit for {row['topic_display']}")

        if selected_topic and row["topic_label"] == selected_topic:
            parts.append("topic filter matched")
        else:
            parts.append(row["topic_display"])

        if selected_mood and row["mood_label"] == selected_mood:
            parts.append("mood filter matched")
        else:
            parts.append(row["mood_display"])

        if float(row["average_rating"]) >= 4.2:
            parts.append("high average rating")
        elif float(row["ratings_count"]) >= 1000:
            parts.append("well-rated by many readers")

        return " | ".join(parts[:4])

    def _format_results(self, results: pd.DataFrame) -> pd.DataFrame:
        frame = results.copy()
        frame["semantic_score"] = frame["semantic_score"].round(4)
        frame["final_score"] = frame["final_score"].round(4)
        frame["average_rating"] = frame["average_rating"].round(2)
        frame["published_year"] = frame["published_year"].astype(int)
        frame["ratings_count"] = frame["ratings_count"].astype(int)
        frame["num_pages"] = frame["num_pages"].astype(int)
        return frame[DISPLAY_COLUMNS + ["thumbnail", "description_preview", "book_id"]]


def _ensure_semantic_artifacts_current(config: BuildConfig) -> None:
    if config.semantic_provider == "openai":
        return

    if _tfidf_artifact_is_current(config):
        return

    if not config.chunk_data_path.exists():
        raise FileNotFoundError(
            f"Missing chunk artifact: {config.chunk_data_path}. Run the build pipeline first."
        )

    chunks = pd.read_csv(config.chunk_data_path)
    build_semantic_index(
        chunks=chunks,
        provider=config.semantic_provider,
        max_features=config.max_features,
        svd_components=config.svd_components,
        persist_path=config.tfidf_index_path,
        openai_model_name=config.openai_embedding_model,
    )

    with config.index_metadata_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "provider": config.semantic_provider,
                "metadata_provider": config.metadata_provider,
                "sklearn_version": sklearn.__version__,
                "openai_embedding_model": config.openai_embedding_model,
            },
            file,
            indent=2,
        )


def _tfidf_artifact_is_current(config: BuildConfig) -> bool:
    if not config.tfidf_index_path.exists() or not config.index_metadata_path.exists():
        return False

    try:
        metadata = json.loads(config.index_metadata_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False

    return (
        metadata.get("provider") == config.semantic_provider
        and metadata.get("sklearn_version") == sklearn.__version__
    )
