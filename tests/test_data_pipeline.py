from pathlib import Path
import sys
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semantic_book_recommender.data import build_book_chunks, clean_books
from semantic_book_recommender.taxonomy import enrich_with_metadata


class DataPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.raw_books = pd.DataFrame(
            [
                {
                    "isbn13": "1",
                    "isbn10": "1",
                    "title": "Dragon Crown",
                    "subtitle": "",
                    "authors": "A. Writer",
                    "categories": "Fantasy",
                    "thumbnail": "",
                    "description": "A young wizard must protect the kingdom by mastering ancient magic and facing a dragon.",
                    "published_year": 2008,
                    "average_rating": 4.3,
                    "num_pages": 410,
                    "ratings_count": 1200,
                },
                {
                    "isbn13": "2",
                    "isbn10": "2",
                    "title": "Silent Clue",
                    "subtitle": "",
                    "authors": "B. Author",
                    "categories": "Detective and mystery stories",
                    "thumbnail": "",
                    "description": "A detective follows a murder investigation through a tense maze of clues and conspiracies.",
                    "published_year": 2014,
                    "average_rating": 4.0,
                    "num_pages": 320,
                    "ratings_count": 900,
                },
                {
                    "isbn13": "3",
                    "isbn10": "3",
                    "title": "Short Entry",
                    "subtitle": "",
                    "authors": "C. Author",
                    "categories": "Fiction",
                    "thumbnail": "",
                    "description": "Too short.",
                    "published_year": 2016,
                    "average_rating": 3.8,
                    "num_pages": 180,
                    "ratings_count": 15,
                },
            ]
        )

    def test_clean_books_removes_short_descriptions(self) -> None:
        cleaned = clean_books(self.raw_books, min_description_chars=50)
        self.assertEqual(len(cleaned), 2)
        self.assertIn("search_document", cleaned.columns)
        self.assertTrue(cleaned["description_length"].ge(50).all())

    def test_metadata_enrichment_adds_labels(self) -> None:
        cleaned = clean_books(self.raw_books, min_description_chars=50)
        enriched = enrich_with_metadata(cleaned, provider="keyword")
        self.assertIn("topic_label", enriched.columns)
        self.assertIn("mood_label", enriched.columns)
        self.assertIn("emotion_label", enriched.columns)

    def test_chunk_builder_generates_chunk_ids(self) -> None:
        cleaned = clean_books(self.raw_books, min_description_chars=50)
        enriched = enrich_with_metadata(cleaned, provider="keyword")
        chunks = build_book_chunks(enriched, chunk_size=80, chunk_overlap=20)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertIn("chunk_id", chunks.columns)


if __name__ == "__main__":
    unittest.main()
