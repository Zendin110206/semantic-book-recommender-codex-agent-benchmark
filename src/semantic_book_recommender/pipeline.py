from __future__ import annotations

import argparse
import json
from dataclasses import replace

import pandas as pd
import sklearn
from dotenv import load_dotenv

from .config import BuildConfig
from .data import build_book_chunks, build_data_profile, clean_books, load_raw_books
from .engine import BookRecommender
from .retrieval import build_semantic_index
from .taxonomy import enrich_with_metadata


SAMPLE_QUERIES = [
    "a reflective literary novel about family, faith, and grief",
    "a fast-paced murder mystery with clever investigators",
    "an inspiring business book about leadership and personal growth",
]


def build_project_artifacts(config: BuildConfig | None = None) -> dict[str, object]:
    load_dotenv()
    config = config or BuildConfig()
    config.ensure_directories()

    raw_books = load_raw_books(config.raw_books_path)
    cleaned_books = clean_books(raw_books, min_description_chars=config.min_description_chars)
    enriched_books = enrich_with_metadata(
        cleaned_books,
        provider=config.metadata_provider,
        zero_shot_model=config.zero_shot_model,
        emotion_model=config.emotion_model,
    )
    chunks = build_book_chunks(
        enriched_books,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )

    persist_path = config.chroma_dir if config.semantic_provider == "openai" else config.tfidf_index_path
    index = build_semantic_index(
        chunks=chunks,
        provider=config.semantic_provider,
        max_features=config.max_features,
        svd_components=config.svd_components,
        persist_path=persist_path,
        openai_model_name=config.openai_embedding_model,
    )

    enriched_books.to_csv(config.processed_books_path, index=False)
    chunks.to_csv(config.chunk_data_path, index=False)

    profile = build_data_profile(raw_books=raw_books, books=enriched_books, chunks=chunks)
    profile["semantic_provider"] = config.semantic_provider
    profile["metadata_provider"] = config.metadata_provider

    with config.data_profile_path.open("w", encoding="utf-8") as file:
        json.dump(profile, file, indent=2)

    engine = BookRecommender(books=enriched_books.copy(), index=index, config=config)
    sample_frame = _build_sample_recommendations(engine)
    sample_frame.to_csv(config.sample_recommendations_path, index=False)

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

    return profile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build semantic book recommender artifacts.")
    parser.add_argument(
        "--semantic-provider",
        choices=["tfidf", "openai"],
        default=None,
        help="Semantic retrieval backend. Default comes from config/.env.",
    )
    parser.add_argument(
        "--metadata-provider",
        choices=["keyword", "transformers"],
        default=None,
        help="Metadata enrichment backend. Default comes from config/.env.",
    )
    parser.add_argument("--min-description-chars", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = BuildConfig()
    overrides = {}

    if args.semantic_provider:
        overrides["semantic_provider"] = args.semantic_provider
    if args.metadata_provider:
        overrides["metadata_provider"] = args.metadata_provider
    if args.min_description_chars is not None:
        overrides["min_description_chars"] = args.min_description_chars
    if args.chunk_size is not None:
        overrides["chunk_size"] = args.chunk_size
    if args.chunk_overlap is not None:
        overrides["chunk_overlap"] = args.chunk_overlap

    if overrides:
        config = replace(config, **overrides)

    profile = build_project_artifacts(config)
    print(json.dumps(profile, indent=2))
    return 0


def _build_sample_recommendations(engine: BookRecommender) -> pd.DataFrame:
    frames = []
    for query in SAMPLE_QUERIES:
        recommendations = engine.recommend_from_query(query=query, n_results=5, min_rating=3.7).copy()
        recommendations.insert(0, "query", query)
        frames.append(recommendations)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
