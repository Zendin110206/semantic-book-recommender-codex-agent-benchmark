from pathlib import Path
import sys
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semantic_book_recommender.config import BuildConfig
from semantic_book_recommender.data import build_book_chunks, clean_books
from semantic_book_recommender.engine import BookRecommender
from semantic_book_recommender.retrieval import TfidfSemanticIndex
from semantic_book_recommender.taxonomy import enrich_with_metadata


class RecommenderTests(unittest.TestCase):
    def setUp(self) -> None:
        raw_books = pd.DataFrame(
            [
                {
                    "isbn13": "1",
                    "isbn10": "1",
                    "title": "Dragon Crown",
                    "subtitle": "",
                    "authors": "A. Writer",
                    "categories": "Fantasy",
                    "thumbnail": "",
                    "description": "A young wizard must protect the kingdom by mastering ancient magic and facing a dragon in war.",
                    "published_year": 2008,
                    "average_rating": 4.4,
                    "num_pages": 430,
                    "ratings_count": 1800,
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
                    "average_rating": 4.1,
                    "num_pages": 320,
                    "ratings_count": 900,
                },
                {
                    "isbn13": "3",
                    "isbn10": "3",
                    "title": "Quiet Faith",
                    "subtitle": "",
                    "authors": "C. Author",
                    "categories": "Fiction",
                    "thumbnail": "",
                    "description": "A reflective novel about faith, grief, memory, and family in a small town.",
                    "published_year": 2005,
                    "average_rating": 4.3,
                    "num_pages": 260,
                    "ratings_count": 1200,
                },
            ]
        )
        cleaned = clean_books(raw_books, min_description_chars=50)
        enriched = enrich_with_metadata(cleaned, provider="keyword")
        chunks = build_book_chunks(enriched, chunk_size=90, chunk_overlap=20)
        index = TfidfSemanticIndex.build(chunks, max_features=5000, svd_components=16)
        self.engine = BookRecommender(books=enriched, index=index, config=BuildConfig())

    def test_query_prefers_matching_theme(self) -> None:
        results = self.engine.recommend_from_query(
            query="wizard magic kingdom dragon adventure",
            n_results=2,
        )
        self.assertEqual(results.iloc[0]["full_title"], "Dragon Crown")

    def test_similar_books_excludes_anchor_title(self) -> None:
        results = self.engine.recommend_similar_books(title="Dragon Crown", n_results=2)
        self.assertNotIn("Dragon Crown", results["full_title"].tolist())

    def test_topic_filter_limits_results(self) -> None:
        results = self.engine.recommend_from_query(
            query="leadership and business growth",
            topic="business-career",
            n_results=2,
        )
        self.assertTrue(results.empty or (results["topic_display"] == "Business & Career").all())


if __name__ == "__main__":
    unittest.main()
