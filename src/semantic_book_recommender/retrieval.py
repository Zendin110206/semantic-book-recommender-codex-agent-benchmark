from __future__ import annotations

import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


@dataclass
class TfidfSemanticIndex:
    vectorizer: TfidfVectorizer
    svd: TruncatedSVD
    chunk_embeddings: np.ndarray
    chunk_book_ids: np.ndarray
    chunk_ids: np.ndarray

    @classmethod
    def build(
        cls,
        chunks: pd.DataFrame,
        max_features: int = 35000,
        svd_components: int = 256,
    ) -> "TfidfSemanticIndex":
        vectorizer = TfidfVectorizer(
            stop_words="english",
            strip_accents="unicode",
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=max_features,
        )
        tfidf_matrix = vectorizer.fit_transform(chunks["text"])

        max_components = min(
            svd_components,
            tfidf_matrix.shape[0] - 1,
            tfidf_matrix.shape[1] - 1,
        )
        max_components = max(max_components, 2)

        svd = TruncatedSVD(n_components=max_components, random_state=42)
        dense_embeddings = svd.fit_transform(tfidf_matrix)
        dense_embeddings = normalize(dense_embeddings)

        return cls(
            vectorizer=vectorizer,
            svd=svd,
            chunk_embeddings=dense_embeddings.astype(np.float32),
            chunk_book_ids=chunks["book_id"].astype(str).to_numpy(),
            chunk_ids=chunks["chunk_id"].astype(str).to_numpy(),
        )

    def search(self, query: str, top_n_chunks: int = 80) -> list[tuple[str, float]]:
        query_embedding = self._encode_query(query)
        scores = self.chunk_embeddings @ query_embedding

        limit = min(top_n_chunks, len(scores))
        top_indices = np.argpartition(scores, -limit)[-limit:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        book_scores: dict[str, dict[str, float]] = {}
        for index in top_indices:
            book_id = str(self.chunk_book_ids[index])
            score = float(scores[index])
            stats = book_scores.setdefault(book_id, {"max": score, "sum": 0.0, "count": 0.0})
            stats["max"] = max(stats["max"], score)
            stats["sum"] += score
            stats["count"] += 1.0

        aggregated = []
        for book_id, stats in book_scores.items():
            mean_score = stats["sum"] / max(stats["count"], 1.0)
            aggregated_score = 0.70 * stats["max"] + 0.30 * mean_score
            aggregated.append((book_id, round(float(aggregated_score), 6)))

        aggregated.sort(key=lambda item: item[1], reverse=True)
        return aggregated

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path: str | Path) -> "TfidfSemanticIndex":
        with Path(path).open("rb") as file:
            loaded = pickle.load(file)
        if not isinstance(loaded, cls):
            raise TypeError("The loaded artifact is not a TfidfSemanticIndex")
        return loaded

    def _encode_query(self, query: str) -> np.ndarray:
        query_tfidf = self.vectorizer.transform([query])
        dense_query = self.svd.transform(query_tfidf)
        dense_query = normalize(dense_query)
        return dense_query[0].astype(np.float32)


class OpenAIChromaIndex:
    def __init__(self, persist_directory: str | Path, model_name: str) -> None:
        self.persist_directory = Path(persist_directory)
        self.model_name = model_name
        self._vectorstore = self._load_vectorstore()

    @classmethod
    def build(
        cls,
        chunks: pd.DataFrame,
        persist_directory: str | Path,
        model_name: str,
    ) -> "OpenAIChromaIndex":
        Document, Chroma, OpenAIEmbeddings = _require_openai_stack()

        persist_directory = Path(persist_directory)
        if persist_directory.exists():
            shutil.rmtree(persist_directory)
        persist_directory.mkdir(parents=True, exist_ok=True)

        documents = []
        for row in chunks.itertuples(index=False):
            documents.append(
                Document(
                    page_content=row.text,
                    metadata={
                        "book_id": row.book_id,
                        "chunk_id": row.chunk_id,
                        "full_title": row.full_title,
                        "authors": row.authors,
                        "categories": row.categories,
                    },
                )
            )

        embeddings = OpenAIEmbeddings(model=model_name)
        Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=str(persist_directory),
        )
        return cls(persist_directory=persist_directory, model_name=model_name)

    def search(self, query: str, top_n_chunks: int = 80) -> list[tuple[str, float]]:
        results = self._vectorstore.similarity_search_with_relevance_scores(query, k=top_n_chunks)
        book_scores: dict[str, dict[str, float]] = {}

        for document, score in results:
            book_id = str(document.metadata["book_id"])
            stats = book_scores.setdefault(book_id, {"max": score, "sum": 0.0, "count": 0.0})
            stats["max"] = max(float(stats["max"]), float(score))
            stats["sum"] += float(score)
            stats["count"] += 1.0

        aggregated = []
        for book_id, stats in book_scores.items():
            mean_score = stats["sum"] / max(stats["count"], 1.0)
            aggregated_score = 0.70 * stats["max"] + 0.30 * mean_score
            aggregated.append((book_id, round(float(aggregated_score), 6)))

        aggregated.sort(key=lambda item: item[1], reverse=True)
        return aggregated

    def _load_vectorstore(self) -> Any:
        _, Chroma, OpenAIEmbeddings = _require_openai_stack()
        embeddings = OpenAIEmbeddings(model=self.model_name)
        return Chroma(
            persist_directory=str(self.persist_directory),
            embedding_function=embeddings,
        )


def build_semantic_index(
    chunks: pd.DataFrame,
    provider: str,
    max_features: int,
    svd_components: int,
    persist_path: Path,
    openai_model_name: str,
) -> Any:
    provider = provider.lower()
    if provider == "openai":
        return OpenAIChromaIndex.build(
            chunks=chunks,
            persist_directory=persist_path,
            model_name=openai_model_name,
        )

    index = TfidfSemanticIndex.build(
        chunks=chunks,
        max_features=max_features,
        svd_components=svd_components,
    )
    index.save(persist_path)
    return index


def load_semantic_index(provider: str, persist_path: Path, openai_model_name: str) -> Any:
    provider = provider.lower()
    if provider == "openai":
        return OpenAIChromaIndex(persist_directory=persist_path, model_name=openai_model_name)
    return TfidfSemanticIndex.load(persist_path)


def _require_openai_stack() -> tuple[Any, Any, Any]:
    try:
        from langchain_chroma import Chroma
        from langchain_core.documents import Document
        from langchain_openai import OpenAIEmbeddings
    except ImportError as exc:
        raise ImportError(
            "OpenAI mode requires langchain-core, langchain-openai, and langchain-chroma."
        ) from exc
    return Document, Chroma, OpenAIEmbeddings
