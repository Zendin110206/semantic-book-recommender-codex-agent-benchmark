from __future__ import annotations

import warnings
from typing import Any

import pandas as pd

from .data import normalize_text


TOPIC_KEYWORDS = {
    "literary-fiction": [
        "fiction",
        "novel",
        "literary",
        "family",
        "society",
        "relationship",
        "coming of age",
    ],
    "mystery-thriller": [
        "mystery",
        "detective",
        "thriller",
        "crime",
        "murder",
        "suspense",
        "investigation",
        "spy",
    ],
    "fantasy-sci-fi": [
        "fantasy",
        "science fiction",
        "dragon",
        "magic",
        "wizard",
        "space",
        "alien",
        "future",
        "kingdom",
        "myth",
    ],
    "romance-relationships": [
        "romance",
        "love",
        "marriage",
        "relationship",
        "heart",
        "wedding",
        "romantic",
    ],
    "history-politics": [
        "history",
        "war",
        "empire",
        "political",
        "government",
        "civilization",
        "president",
        "historical",
    ],
    "biography-memoir": [
        "biography",
        "autobiography",
        "memoir",
        "life",
        "personal story",
        "journal",
        "letters",
    ],
    "business-career": [
        "business",
        "economics",
        "leadership",
        "career",
        "startup",
        "management",
        "finance",
        "productivity",
    ],
    "self-help-wellness": [
        "self-help",
        "mindfulness",
        "habit",
        "wellness",
        "health",
        "motivation",
        "healing",
        "growth",
    ],
    "science-technology": [
        "science",
        "computer",
        "technology",
        "physics",
        "mathematics",
        "engineering",
        "research",
        "data",
    ],
    "philosophy-religion": [
        "philosophy",
        "religion",
        "spiritual",
        "faith",
        "ethics",
        "theology",
        "belief",
        "meditation",
    ],
    "children-young-adult": [
        "juvenile",
        "young adult",
        "children",
        "school",
        "teen",
        "friendship",
        "adventure",
        "illustrated",
    ],
    "arts-culture": [
        "art",
        "music",
        "performing arts",
        "culture",
        "film",
        "design",
        "criticism",
        "essay",
    ],
    "poetry-drama-comics": [
        "poetry",
        "drama",
        "play",
        "graphic novel",
        "comics",
        "humor",
        "verse",
    ],
    "general-nonfiction": [
        "travel",
        "education",
        "reference",
        "language",
        "social science",
        "guide",
        "manual",
    ],
}


TOPIC_DISPLAY_NAMES = {
    "literary-fiction": "Literary Fiction",
    "mystery-thriller": "Mystery & Thriller",
    "fantasy-sci-fi": "Fantasy & Sci-Fi",
    "romance-relationships": "Romance & Relationships",
    "history-politics": "History & Politics",
    "biography-memoir": "Biography & Memoir",
    "business-career": "Business & Career",
    "self-help-wellness": "Self-Help & Wellness",
    "science-technology": "Science & Technology",
    "philosophy-religion": "Philosophy & Religion",
    "children-young-adult": "Children & Young Adult",
    "arts-culture": "Arts & Culture",
    "poetry-drama-comics": "Poetry, Drama & Comics",
    "general-nonfiction": "General Nonfiction",
}


MOOD_KEYWORDS = {
    "uplifting": ["hope", "uplifting", "joy", "celebration", "kindness", "healing", "friendship"],
    "reflective": ["memory", "loss", "grief", "identity", "faith", "family", "meditative", "quiet"],
    "dark": ["dark", "tragedy", "violence", "revenge", "despair", "haunting", "bleak"],
    "tense": ["suspense", "danger", "murder", "escape", "conspiracy", "threat", "war"],
    "adventurous": ["journey", "quest", "exploration", "discovery", "expedition", "adventure"],
    "romantic": ["love", "passion", "heart", "romance", "marriage", "affair"],
    "inspiring": ["leadership", "achievement", "growth", "success", "dream", "purpose"],
    "playful": ["humor", "funny", "witty", "comic", "lighthearted", "mischief"],
}


MOOD_DISPLAY_NAMES = {
    "uplifting": "Uplifting",
    "reflective": "Reflective",
    "dark": "Dark",
    "tense": "Tense",
    "adventurous": "Adventurous",
    "romantic": "Romantic",
    "inspiring": "Inspiring",
    "playful": "Playful",
}


EMOTION_DISPLAY_NAMES = {
    "joy": "Joy",
    "sadness": "Sadness",
    "fear": "Fear",
    "anger": "Anger",
    "surprise": "Surprise",
    "love": "Love",
    "neutral": "Neutral",
}


TOPIC_CANDIDATE_TEXT = {
    "literary-fiction": "literary fiction",
    "mystery-thriller": "mystery or thriller",
    "fantasy-sci-fi": "fantasy or science fiction",
    "romance-relationships": "romance and relationships",
    "history-politics": "history and politics",
    "biography-memoir": "biography or memoir",
    "business-career": "business and career",
    "self-help-wellness": "self-help and wellness",
    "science-technology": "science and technology",
    "philosophy-religion": "philosophy and religion",
    "children-young-adult": "children or young adult",
    "arts-culture": "arts and culture",
    "poetry-drama-comics": "poetry drama or comics",
    "general-nonfiction": "general nonfiction",
}


MOOD_CANDIDATE_TEXT = {
    "uplifting": "uplifting and hopeful",
    "reflective": "reflective and introspective",
    "dark": "dark and tragic",
    "tense": "tense and suspenseful",
    "adventurous": "adventurous and expansive",
    "romantic": "romantic and emotional",
    "inspiring": "inspiring and motivational",
    "playful": "playful and humorous",
}


def enrich_with_metadata(
    books: pd.DataFrame,
    provider: str = "keyword",
    zero_shot_model: str = "facebook/bart-large-mnli",
    emotion_model: str = "j-hartmann/emotion-english-distilroberta-base",
) -> pd.DataFrame:
    provider = provider.lower()

    if provider == "transformers":
        try:
            return _enrich_with_transformers(
                books=books,
                zero_shot_model=zero_shot_model,
                emotion_model=emotion_model,
            )
        except Exception as exc:
            warnings.warn(
                f"Transformer metadata enrichment failed and keyword mode will be used instead: {exc}",
                stacklevel=2,
            )

    return _enrich_with_keywords(books)


def label_to_display(label: str, mapping: dict[str, str]) -> str:
    return mapping.get(label, label.replace("-", " ").title())


def _enrich_with_keywords(books: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    for row in books.itertuples(index=False):
        category_text = normalize_text(row.categories).lower()
        body_text = normalize_text(f"{row.full_title}. {row.description}").lower()

        topic_scores = _score_label_map(category_text, body_text, TOPIC_KEYWORDS)
        topic_label, topic_confidence = _select_label(topic_scores, fallback="general-nonfiction")

        mood_scores = _score_label_map("", body_text, MOOD_KEYWORDS)
        mood_label, mood_confidence = _select_label(mood_scores, fallback="reflective")

        emotion_label = _mood_to_emotion(mood_label)
        audience_label = _infer_audience(row.categories, row.description)
        reading_length = _infer_reading_length(row.num_pages)
        publication_era = _infer_publication_era(row.published_year)

        records.append(
            {
                "topic_label": topic_label,
                "topic_display": label_to_display(topic_label, TOPIC_DISPLAY_NAMES),
                "topic_confidence": topic_confidence,
                "mood_label": mood_label,
                "mood_display": label_to_display(mood_label, MOOD_DISPLAY_NAMES),
                "mood_confidence": mood_confidence,
                "emotion_label": emotion_label,
                "emotion_display": label_to_display(emotion_label, EMOTION_DISPLAY_NAMES),
                "audience_label": audience_label,
                "reading_length": reading_length,
                "publication_era": publication_era,
            }
        )

    return pd.concat([books.reset_index(drop=True), pd.DataFrame.from_records(records)], axis=1)


def _enrich_with_transformers(
    books: pd.DataFrame,
    zero_shot_model: str,
    emotion_model: str,
) -> pd.DataFrame:
    from transformers import pipeline

    zero_shot = pipeline("zero-shot-classification", model=zero_shot_model)
    emotion_classifier = pipeline(
        "text-classification",
        model=emotion_model,
        top_k=None,
    )

    topic_candidates = list(TOPIC_CANDIDATE_TEXT.values())
    topic_lookup = {value: key for key, value in TOPIC_CANDIDATE_TEXT.items()}
    mood_candidates = list(MOOD_CANDIDATE_TEXT.values())
    mood_lookup = {value: key for key, value in MOOD_CANDIDATE_TEXT.items()}

    records: list[dict[str, Any]] = []

    for row in books.itertuples(index=False):
        snippet = normalize_text(row.description)[:1200]

        topic_result = zero_shot(snippet, topic_candidates, multi_label=False)
        topic_candidate = topic_result["labels"][0]
        topic_label = topic_lookup[topic_candidate]
        topic_confidence = float(topic_result["scores"][0])

        mood_result = zero_shot(snippet, mood_candidates, multi_label=False)
        mood_candidate = mood_result["labels"][0]
        mood_label = mood_lookup[mood_candidate]
        mood_confidence = float(mood_result["scores"][0])

        emotion_result = emotion_classifier(snippet[:512])[0]
        emotion_sorted = sorted(
            emotion_result,
            key=lambda item: float(item["score"]),
            reverse=True,
        )
        emotion_label = str(emotion_sorted[0]["label"]).lower()
        if emotion_label not in EMOTION_DISPLAY_NAMES:
            emotion_label = _mood_to_emotion(mood_label)

        records.append(
            {
                "topic_label": topic_label,
                "topic_display": label_to_display(topic_label, TOPIC_DISPLAY_NAMES),
                "topic_confidence": topic_confidence,
                "mood_label": mood_label,
                "mood_display": label_to_display(mood_label, MOOD_DISPLAY_NAMES),
                "mood_confidence": mood_confidence,
                "emotion_label": emotion_label,
                "emotion_display": label_to_display(emotion_label, EMOTION_DISPLAY_NAMES),
                "audience_label": _infer_audience(row.categories, row.description),
                "reading_length": _infer_reading_length(row.num_pages),
                "publication_era": _infer_publication_era(row.published_year),
            }
        )

    return pd.concat([books.reset_index(drop=True), pd.DataFrame.from_records(records)], axis=1)


def _score_label_map(category_text: str, body_text: str, label_map: dict[str, list[str]]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for label, keywords in label_map.items():
        score = 0.0
        for keyword in keywords:
            token = keyword.lower()
            if category_text and token in category_text:
                score += 2.2
            if token in body_text:
                score += 1.0 + 0.1 * token.count(" ")
        scores[label] = score
    return scores


def _select_label(scores: dict[str, float], fallback: str) -> tuple[str, float]:
    if not scores:
        return fallback, 0.0

    label = max(scores, key=scores.get)
    total = sum(scores.values())
    top_score = scores[label]

    if top_score <= 0:
        return fallback, 0.0

    confidence = top_score / total if total > 0 else 0.0
    return label, round(float(confidence), 4)


def _mood_to_emotion(mood_label: str) -> str:
    mapping = {
        "uplifting": "joy",
        "reflective": "sadness",
        "dark": "anger",
        "tense": "fear",
        "adventurous": "surprise",
        "romantic": "love",
        "inspiring": "joy",
        "playful": "joy",
    }
    return mapping.get(mood_label, "neutral")


def _infer_audience(categories: str, description: str) -> str:
    text = normalize_text(f"{categories} {description}").lower()
    if any(token in text for token in ["juvenile", "children", "young adult", "teen"]):
        return "Children & YA"
    if any(token in text for token in ["family", "illustrated", "friendship", "school"]):
        return "All Ages"
    return "Adult"


def _infer_reading_length(num_pages: float) -> str:
    if num_pages <= 220:
        return "Short"
    if num_pages <= 420:
        return "Medium"
    return "Long"


def _infer_publication_era(year: float) -> str:
    if year < 1980:
        return "Classic"
    if year < 2000:
        return "Modern"
    return "Contemporary"
