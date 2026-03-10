from __future__ import annotations

import math
import os
import re
from typing import Any, Union

import numpy as np
import pandas as pd


EXPECTED_COLUMNS = [
    "isbn13",
    "isbn10",
    "title",
    "subtitle",
    "authors",
    "categories",
    "thumbnail",
    "description",
    "published_year",
    "average_rating",
    "num_pages",
    "ratings_count",
]


PathLike = Union[str, os.PathLike[str]]


def normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_raw_books(path: PathLike) -> pd.DataFrame:
    books = pd.read_csv(path)
    missing = [column for column in EXPECTED_COLUMNS if column not in books.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return books


def clean_books(raw_books: pd.DataFrame, min_description_chars: int = 120) -> pd.DataFrame:
    books = raw_books.copy()

    for column in ["title", "subtitle", "authors", "categories", "thumbnail", "description"]:
        books[column] = books[column].map(normalize_text)

    books["authors"] = books["authors"].replace("", "Unknown Author")
    books["categories"] = books["categories"].replace("", "Unknown")
    books["thumbnail"] = books["thumbnail"].str.replace(r"^http://", "https://", regex=True)
    books["full_title"] = np.where(
        books["subtitle"].eq(""),
        books["title"],
        books["title"] + ": " + books["subtitle"],
    )
    books["description_length"] = books["description"].str.len()

    numeric_columns = ["published_year", "average_rating", "num_pages", "ratings_count"]
    for column in numeric_columns:
        books[column] = pd.to_numeric(books[column], errors="coerce")

    books["average_rating"] = books["average_rating"].fillna(books["average_rating"].median())
    books["num_pages"] = books["num_pages"].replace(0, np.nan)
    books["num_pages"] = books["num_pages"].fillna(books["num_pages"].median())
    books["ratings_count"] = books["ratings_count"].fillna(0)
    books["published_year"] = books["published_year"].fillna(books["published_year"].median())

    books = books[books["description_length"] >= min_description_chars].copy()
    books = books.sort_values(
        ["ratings_count", "average_rating", "description_length"],
        ascending=[False, False, False],
    )
    books = books.drop_duplicates(subset=["title", "authors", "description"], keep="first")

    books["book_id"] = books["isbn13"].astype(str)
    books["description_preview"] = books["description"].map(_preview_text)
    books["search_document"] = (
        books["full_title"]
        + ". Author: "
        + books["authors"]
        + ". Category: "
        + books["categories"]
        + ". "
        + books["description"]
    )

    max_ratings = max(float(books["ratings_count"].max()), 1.0)
    year_min = float(books["published_year"].min())
    year_max = float(books["published_year"].max())
    year_span = max(year_max - year_min, 1.0)

    books["rating_score"] = books["average_rating"] / 5.0
    books["popularity_score"] = np.log1p(books["ratings_count"]) / math.log1p(max_ratings)
    books["recency_score"] = (books["published_year"] - year_min) / year_span
    books["quality_score"] = (
        0.55 * books["rating_score"]
        + 0.35 * books["popularity_score"]
        + 0.10 * (books["description_length"] / books["description_length"].max())
    )

    return books.reset_index(drop=True)


def build_book_chunks(
    books: pd.DataFrame,
    chunk_size: int = 700,
    chunk_overlap: int = 120,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    for row in books.itertuples(index=False):
        prefix = f"{row.full_title}. Author: {row.authors}. Category: {row.categories}. "
        description_chunks = split_text(row.description, chunk_size=chunk_size, overlap=chunk_overlap)

        if not description_chunks:
            description_chunks = [row.description]

        for chunk_index, description_chunk in enumerate(description_chunks):
            records.append(
                {
                    "chunk_id": f"{row.book_id}-{chunk_index}",
                    "book_id": row.book_id,
                    "chunk_index": chunk_index,
                    "text": prefix + description_chunk,
                    "full_title": row.full_title,
                    "authors": row.authors,
                    "categories": row.categories,
                }
            )

    return pd.DataFrame.from_records(records)


def build_data_profile(
    raw_books: pd.DataFrame,
    books: pd.DataFrame,
    chunks: pd.DataFrame,
) -> dict[str, Any]:
    top_categories = books["categories"].value_counts().head(10).to_dict()
    top_topics = books["topic_display"].value_counts().head(10).to_dict()
    top_moods = books["mood_display"].value_counts().head(10).to_dict()

    return {
        "raw_rows": int(len(raw_books)),
        "clean_rows": int(len(books)),
        "chunk_rows": int(len(chunks)),
        "dropped_rows": int(len(raw_books) - len(books)),
        "median_description_length": float(books["description_length"].median()),
        "median_rating": float(books["average_rating"].median()),
        "median_pages": float(books["num_pages"].median()),
        "top_categories": top_categories,
        "top_topics": top_topics,
        "top_moods": top_moods,
    }


def split_text(text: str, chunk_size: int = 700, overlap: int = 120) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks: list[str] = []
    start = 0

    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        if end < len(normalized):
            window = normalized[start:end]
            boundary = window.rfind(" ")
            if boundary > int(chunk_size * 0.6):
                end = start + boundary
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(normalized):
            break
        start = max(end - overlap, start + 1)

    return chunks


def _preview_text(text: str, max_chars: int = 320) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."
