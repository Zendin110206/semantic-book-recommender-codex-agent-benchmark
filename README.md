# Semantic Book Recommender

Semantic recommendation system for books built on the `7K Books` dataset by Dylan Castillo. The application lets users search by meaning, tone, and reading intent instead of relying only on exact titles or rigid categories.

This repository version is a **benchmark build implemented end to end with OpenAI Codex agent** under direct human direction. The project owner provided the dataset, project goal, review feedback, and final approval. The codebase, app structure, UI rebuild, pipeline layout, and repository documentation in this version were produced by Codex agent.

## What the project does

- Cleans and restructures the raw Kaggle book catalog
- Builds retrieval-ready text from title, author, category, and description
- Splits descriptions into semantic chunks for better search recall
- Adds topic, mood, emotion, audience, and reading-length metadata
- Ranks recommendations with semantic similarity plus book quality signals
- Exposes the system through a Gradio interface and CLI commands

## AI stack in this version

The current default build already uses AI-style semantic matching:

- `Semantic retrieval`: TF-IDF + latent semantic analysis
- `Metadata enrichment`: local topic and mood labeling rules
- `Optional upgrade path`: OpenAI embeddings + LangChain + Chroma + transformer classification

So yes, this project is already semantic and AI-driven even when the OpenAI path is not enabled.

## Repository structure

```text
.
|-- app.py
|-- data/
|-- notebooks/
|-- scripts/
|-- src/semantic_book_recommender/
|-- tests/
|-- artifacts/
|-- README.md
|-- CASE_STUDY.md
|-- pyproject.toml
```

## Main components

- `src/semantic_book_recommender/data.py`: loading, cleaning, chunking, and profiling
- `src/semantic_book_recommender/taxonomy.py`: topic, mood, emotion, audience, and reading-length labeling
- `src/semantic_book_recommender/retrieval.py`: semantic index backends
- `src/semantic_book_recommender/engine.py`: recommendation logic and reranking
- `src/semantic_book_recommender/app.py`: Gradio interface
- `scripts/build_pipeline.py`: build processed artifacts
- `scripts/recommend_cli.py`: command-line recommendations

## Local setup

Recommended environment:

```powershell
conda create -n semantic_book_recommender python=3.10
conda activate semantic_book_recommender
pip install -r requirements.txt
```

Minimal local baseline only:

```powershell
pip install -e .
```

## Build the artifacts

```powershell
python scripts/build_pipeline.py --semantic-provider tfidf --metadata-provider keyword
```

Build outputs include:

- `artifacts/processed/books_enriched.csv`
- `artifacts/processed/book_chunks.csv`
- `artifacts/models/tfidf_semantic_index.pkl`
- `artifacts/models/semantic_index_metadata.json`
- `artifacts/reports/data_profile.json`
- `artifacts/reports/sample_recommendations.csv`

## Run the application

Start the Gradio app:

```powershell
python app.py
```

Start the CLI:

```powershell
python scripts/recommend_cli.py query --query "a reflective literary novel about family, faith, and grief"
python scripts/recommend_cli.py similar --title "Gilead"
```

## Optional OpenAI upgrade

This repository also supports a stronger cloud-backed path.

1. Add `OPENAI_API_KEY` to `.env`
2. Install the optional dependencies from `requirements.txt`
3. Rebuild with:

```powershell
python scripts/build_pipeline.py --semantic-provider openai --metadata-provider transformers
```

## Notes

- The default metadata enrichment path is local and reproducible.
- The TF-IDF semantic index is automatically rebuilt if the installed `scikit-learn` version changes.
- Some category and topic labels remain heuristic because the source dataset is noisy.

## Attribution note

If this repository is used as a reference point for later projects, the most accurate description is:

`This benchmark version was implemented by OpenAI Codex agent under human direction and review.`
