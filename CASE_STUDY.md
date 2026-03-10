# Case Study: Semantic Book Recommender

## Problem

Readers rarely search for books by exact title alone. They usually describe intent:

- "I want a reflective novel about grief and faith."
- "Give me a tense detective story."
- "Recommend an inspiring business book."

Traditional catalog filters are weak for this workflow. This project solves that gap with semantic retrieval and metadata-aware reranking.

## Dataset

- Source: 7K books dataset by Dylan Castillo on Kaggle
- Raw rows: 6,810
- Clean rows after preprocessing: 5,805
- Retrieval chunks created: 7,419

## Solution Design

1. Clean and deduplicate catalog metadata
2. Build retrieval text from title, author, category, and description
3. Split descriptions into chunks for better semantic recall
4. Enrich books with topic, mood, emotion, audience, and reading-length tags
5. Retrieve candidates with semantic similarity
6. Re-rank with quality signals such as rating and popularity
7. Deliver results through CLI and Gradio app surfaces

## Technical Decisions

- Default retrieval backend: TF-IDF + latent semantic projection for a strong offline baseline
- Upgrade path: OpenAI embeddings with LangChain + Chroma for higher-quality semantic search
- Metadata provider: keyword heuristics by default, optional transformer enrichment for richer labeling
- Product feature set: topic filter, mood filter, rating threshold, and page-count cap

## Why this is portfolio-grade

- Reproducible build pipeline instead of notebook-only work
- Clear separation between data prep, enrichment, retrieval, ranking, and presentation layers
- Local tests for the core pipeline
- App-ready interface for a hiring manager or recruiter demo
- Sensible fallback strategy when API access is unavailable

## Resume bullets

- Built an end-to-end semantic book recommender that converts unstructured descriptions into a searchable retrieval index across 5.8K cleaned titles.
- Designed a modular Python pipeline for preprocessing, metadata enrichment, semantic retrieval, and reranking with reproducible artifact generation.
- Implemented dual retrieval paths: an offline latent semantic baseline and an optional OpenAI + LangChain + Chroma upgrade path.
- Shipped a Gradio-based demo interface with natural-language search and mood/topic filters for non-technical stakeholders.
