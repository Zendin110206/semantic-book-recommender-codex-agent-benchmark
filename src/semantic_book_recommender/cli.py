from __future__ import annotations

import argparse
from dataclasses import replace

from dotenv import load_dotenv

from .config import BuildConfig
from .engine import BookRecommender
from .pipeline import main as build_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interact with the semantic book recommender.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser_cmd = subparsers.add_parser("build", help="Build processed data and semantic artifacts.")
    build_parser_cmd.add_argument("--semantic-provider", choices=["tfidf", "openai"], default=None)
    build_parser_cmd.add_argument("--metadata-provider", choices=["keyword", "transformers"], default=None)
    build_parser_cmd.add_argument("--min-description-chars", type=int, default=None)
    build_parser_cmd.add_argument("--chunk-size", type=int, default=None)
    build_parser_cmd.add_argument("--chunk-overlap", type=int, default=None)

    query_parser = subparsers.add_parser("query", help="Query the recommender with natural language.")
    _add_shared_recommendation_args(query_parser)
    query_parser.add_argument("--query", required=True)

    similar_parser = subparsers.add_parser("similar", help="Find similar books from a known title.")
    _add_shared_recommendation_args(similar_parser)
    similar_parser.add_argument("--title", required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "build":
        build_args = []
        if args.semantic_provider:
            build_args.extend(["--semantic-provider", args.semantic_provider])
        if args.metadata_provider:
            build_args.extend(["--metadata-provider", args.metadata_provider])
        if args.min_description_chars is not None:
            build_args.extend(["--min-description-chars", str(args.min_description_chars)])
        if args.chunk_size is not None:
            build_args.extend(["--chunk-size", str(args.chunk_size)])
        if args.chunk_overlap is not None:
            build_args.extend(["--chunk-overlap", str(args.chunk_overlap)])
        return build_main(build_args)

    config = BuildConfig()
    if getattr(args, "semantic_provider", None):
        config = replace(config, semantic_provider=args.semantic_provider)

    engine = BookRecommender.from_artifacts(config)

    if args.command == "query":
        results = engine.recommend_from_query(
            query=args.query,
            n_results=args.n_results,
            topic=args.topic,
            mood=args.mood,
            min_rating=args.min_rating,
            max_pages=args.max_pages,
        )
    else:
        results = engine.recommend_similar_books(
            title=args.title,
            n_results=args.n_results,
            topic=args.topic,
            mood=args.mood,
            min_rating=args.min_rating,
            max_pages=args.max_pages,
        )

    if results.empty:
        print("No results matched the current filters.")
        return 0

    printable = results.drop(columns=["thumbnail", "description_preview", "book_id"], errors="ignore")
    print(printable.to_string(index=False))
    return 0


def _add_shared_recommendation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--topic", default=None)
    parser.add_argument("--mood", default=None)
    parser.add_argument("--min-rating", type=float, default=0.0)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--n-results", type=int, default=5)
