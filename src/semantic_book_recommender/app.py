from __future__ import annotations

from html import escape

import pandas as pd
from dotenv import load_dotenv

from .config import BuildConfig
from .engine import BookRecommender
from .taxonomy import MOOD_DISPLAY_NAMES, TOPIC_DISPLAY_NAMES


APP_CSS = """
:root {
  --paper: #f6f0e4;
  --paper-strong: #fffaf1;
  --ink: #1f2a28;
  --muted: #5e6b68;
  --forest: #173a34;
  --forest-soft: #285248;
  --gold: #b98746;
  --line: rgba(23, 58, 52, 0.12);
  --shadow: 0 24px 60px rgba(23, 58, 52, 0.10);
}

body,
.gradio-container {
  background:
    radial-gradient(circle at top left, rgba(185, 135, 70, 0.14), transparent 22%),
    linear-gradient(180deg, #f3ead9 0%, #f7f2ea 38%, #eee3d1 100%);
  color: var(--ink);
  font-family: "Palatino Linotype", "Book Antiqua", Georgia, serif;
}

.gradio-container {
  --body-background-fill: transparent;
  --block-background-fill: rgba(255, 250, 241, 0.82);
  --block-border-color: var(--line);
  --input-background-fill: rgba(255, 250, 241, 0.94);
  --input-border-color: rgba(23, 58, 52, 0.16);
  --button-primary-background-fill: var(--forest);
  --button-primary-background-fill-hover: var(--forest-soft);
  --button-primary-text-color: #fff8ef;
  --color-accent: var(--gold);
  --color-accent-soft: rgba(185, 135, 70, 0.18);
}

.gradio-container .prose,
.gradio-container p,
.gradio-container span,
.gradio-container label,
.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container h4,
.gradio-container td,
.gradio-container th {
  color: var(--ink) !important;
}

.gradio-container input,
.gradio-container textarea,
.gradio-container button,
.gradio-container select,
.gradio-container [data-testid="dropdown"] *,
.gradio-container [role="listbox"] *,
.gradio-container [role="option"] *,
.gradio-container svg {
  color: var(--ink) !important;
  fill: currentColor !important;
  stroke: currentColor !important;
  -webkit-text-fill-color: var(--ink) !important;
  opacity: 1 !important;
}

.gradio-container input::placeholder,
.gradio-container textarea::placeholder {
  color: rgba(31, 42, 40, 0.34) !important;
  opacity: 1 !important;
}

.gradio-container [data-testid="dropdown"] > div,
.gradio-container [data-testid="dropdown"] button,
.gradio-container [data-testid="dropdown"] input,
.gradio-container [data-testid="dropdown"] .wrap,
.gradio-container [data-testid="dropdown"] .single-select,
.gradio-container [data-testid="dropdown"] span,
.gradio-container [data-testid="dropdown"] label {
  background: rgba(255, 250, 241, 0.95) !important;
  color: var(--ink) !important;
  -webkit-text-fill-color: var(--ink) !important;
  opacity: 1 !important;
}

.gradio-container [role="listbox"] {
  background: rgba(255, 250, 241, 0.98) !important;
  border: 1px solid var(--line) !important;
}

.gradio-container [role="option"] {
  background: transparent !important;
  color: var(--ink) !important;
}

.gradio-container [role="option"][aria-selected="true"] {
  background: rgba(185, 135, 70, 0.12) !important;
}

.gradio-container [data-testid="number-input"] *,
.gradio-container .wrap *,
.gradio-container .form * {
  color: var(--ink) !important;
  -webkit-text-fill-color: var(--ink) !important;
}

.gradio-container .tabs {
  border: none;
  background: transparent;
}

.gradio-container .tab-nav {
  gap: 10px;
  border-bottom: 1px solid var(--line);
  padding-bottom: 6px;
}

.gradio-container .tab-nav button {
  border-radius: 999px;
  border: 1px solid var(--line);
  background: rgba(255, 250, 241, 0.72);
  color: var(--muted);
  padding: 10px 16px;
  font-weight: 700;
}

.gradio-container .tab-nav button.selected {
  background: var(--forest);
  color: #fff8ef !important;
  border-color: transparent;
}

.hero-shell {
  overflow: hidden;
  border: 1px solid rgba(23, 58, 52, 0.10);
  border-radius: 30px;
  padding: 28px;
  margin-bottom: 20px;
  background: linear-gradient(135deg, rgba(255, 250, 241, 0.96) 0%, rgba(243, 232, 211, 0.92) 100%);
  box-shadow: var(--shadow);
}

.hero-kicker {
  display: inline-flex;
  padding: 7px 12px;
  border-radius: 999px;
  background: rgba(23, 58, 52, 0.08);
  color: var(--forest);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-size: 12px;
  font-weight: 700;
}

.hero-title {
  margin: 14px 0 10px;
  font-size: clamp(34px, 5vw, 56px);
  line-height: 1.02;
  letter-spacing: -0.03em;
  color: var(--forest);
}

.hero-copy {
  max-width: 760px;
  margin: 0;
  font-size: 18px;
  line-height: 1.68;
  color: var(--muted);
}

.hero-grid,
.explain-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin-top: 24px;
}

.hero-card,
.explain-card,
.panel {
  border: 1px solid rgba(23, 58, 52, 0.10);
  border-radius: 24px;
  background: rgba(255, 250, 241, 0.86);
  box-shadow: 0 14px 34px rgba(23, 58, 52, 0.08);
}

.hero-card,
.explain-card {
  padding: 16px 18px;
}

.hero-label {
  display: block;
  margin-bottom: 8px;
  color: var(--muted);
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.10em;
}

.hero-value {
  display: block;
  color: var(--forest);
  font-size: 28px;
  font-weight: 700;
}

.stack-card {
  grid-column: span 2;
}

.stack-card h3,
.explain-card h3,
.section-title {
  margin: 0 0 10px;
  color: var(--forest);
}

.stack-copy,
.explain-card p,
.section-copy {
  margin: 0;
  color: var(--muted);
  line-height: 1.68;
}

.stack-pills {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 12px;
}

.stack-pill {
  padding: 8px 12px;
  border-radius: 999px;
  background: rgba(185, 135, 70, 0.13);
  color: var(--forest);
  font-size: 13px;
  font-weight: 700;
}

.panel {
  padding: 20px;
}

.section-kicker {
  color: var(--gold);
  text-transform: uppercase;
  letter-spacing: 0.10em;
  font-size: 12px;
  font-weight: 700;
  margin-bottom: 8px;
}

#search-button button,
#similar-button button {
  min-height: 54px;
  border-radius: 18px;
  border: none;
  font-size: 16px;
  font-weight: 700;
  box-shadow: 0 16px 28px rgba(23, 58, 52, 0.16);
}

.empty-state {
  border: 1px dashed rgba(23, 58, 52, 0.18);
  border-radius: 24px;
  padding: 28px;
  background: rgba(255, 250, 241, 0.70);
  text-align: center;
}

.empty-state h3 {
  margin: 0 0 8px;
  color: var(--forest);
  font-size: 24px;
}

.empty-state p {
  margin: 0;
  color: var(--muted);
  line-height: 1.6;
}

.results-shell {
  display: grid;
  gap: 18px;
}

.result-card {
  display: grid;
  grid-template-columns: 112px minmax(0, 1fr);
  gap: 18px;
  padding: 18px;
  border-radius: 24px;
  border: 1px solid rgba(23, 58, 52, 0.10);
  background: rgba(255, 250, 241, 0.96);
  box-shadow: 0 16px 34px rgba(23, 58, 52, 0.08);
}

.result-cover,
.result-cover img {
  width: 112px;
  height: 164px;
  border-radius: 18px;
}

.result-cover img {
  display: block;
  object-fit: cover;
  box-shadow: 0 14px 24px rgba(23, 58, 52, 0.16);
}

.result-cover-placeholder {
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
  padding: 14px;
  background: linear-gradient(180deg, rgba(23, 58, 52, 0.88) 0%, rgba(40, 82, 72, 0.96) 100%);
  color: #fff8ef;
  font-size: 13px;
  line-height: 1.5;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.result-top {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 10px;
}

.result-rank {
  color: var(--gold);
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.09em;
  font-weight: 700;
}

.score-chip {
  padding: 9px 12px;
  border-radius: 999px;
  background: rgba(185, 135, 70, 0.13);
  color: var(--forest);
  font-size: 13px;
  font-weight: 700;
  white-space: nowrap;
}

.result-title {
  margin: 0 0 6px;
  color: var(--forest);
  font-size: clamp(22px, 2vw, 30px);
  line-height: 1.12;
}

.result-author,
.preview-copy {
  color: var(--muted);
}

.result-author {
  margin: 0 0 12px;
  font-size: 16px;
}

.badge-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 14px;
}

.meta-badge {
  padding: 7px 12px;
  border-radius: 999px;
  background: rgba(23, 58, 52, 0.07);
  color: var(--forest);
  font-size: 13px;
  font-weight: 700;
}

.fact-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
  margin-bottom: 14px;
}

.fact-card {
  border-radius: 16px;
  border: 1px solid rgba(23, 58, 52, 0.08);
  background: rgba(244, 236, 221, 0.88);
  padding: 10px 12px;
}

.fact-label {
  display: block;
  color: var(--muted);
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-bottom: 6px;
}

.fact-value {
  display: block;
  color: var(--forest);
  font-size: 15px;
  font-weight: 700;
}

.reason-note {
  margin: 0 0 10px;
  color: var(--forest);
  font-size: 14px;
  line-height: 1.6;
}

.reason-note strong {
  color: var(--gold);
}

.preview-copy {
  margin: 0;
  font-size: 15px;
  line-height: 1.76;
}

.table-shell {
  overflow: auto;
  border: 1px solid rgba(23, 58, 52, 0.10);
  border-radius: 24px;
  background: rgba(255, 250, 241, 0.96);
  box-shadow: 0 16px 34px rgba(23, 58, 52, 0.07);
}

.table-shell table {
  width: 100%;
  min-width: 820px;
  border-collapse: collapse;
}

.table-shell thead th {
  background: var(--forest);
  color: #fff8ef !important;
  text-align: left;
  padding: 14px 16px;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.table-shell tbody tr:nth-child(even) {
  background: rgba(243, 234, 217, 0.62);
}

.table-shell td {
  padding: 14px 16px;
  border-top: 1px solid rgba(23, 58, 52, 0.08);
  color: var(--ink) !important;
  line-height: 1.55;
  vertical-align: top;
}

.table-title {
  color: var(--forest);
  font-weight: 700;
}

.table-muted {
  color: var(--muted);
}

@media (max-width: 1100px) {
  .hero-grid,
  .explain-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .stack-card {
    grid-column: span 2;
  }
}

@media (max-width: 768px) {
  .hero-grid,
  .explain-grid,
  .fact-grid {
    grid-template-columns: 1fr;
  }

  .stack-card {
    grid-column: span 1;
  }

  .result-card {
    grid-template-columns: 1fr;
  }

  .result-cover,
  .result-cover img {
    width: 100%;
    max-width: 180px;
    height: 240px;
  }
}
"""


def launch(config: BuildConfig | None = None) -> None:
    try:
        import gradio as gr
    except ImportError as exc:
        raise ImportError("Gradio is not installed. Install the app dependencies first.") from exc

    major_version = int(str(gr.__version__).split(".", maxsplit=1)[0])
    app = build_demo(config=config, gr=gr, include_css_in_blocks=major_version < 6)

    launch_kwargs = {}
    if major_version >= 6:
        launch_kwargs["css"] = APP_CSS

    app.launch(**launch_kwargs)


def build_demo(
    config: BuildConfig | None = None,
    gr=None,
    include_css_in_blocks: bool | None = None,
):
    if gr is None:
        try:
            import gradio as gr
        except ImportError as exc:
            raise ImportError("Gradio is not installed. Install the app dependencies first.") from exc

    load_dotenv()
    config = config or BuildConfig()
    engine = BookRecommender.from_artifacts(config)

    if include_css_in_blocks is None:
        include_css_in_blocks = int(str(gr.__version__).split(".", maxsplit=1)[0]) < 6

    topic_choices = [("All Topics", None)] + [
        (display_name, slug) for slug, display_name in TOPIC_DISPLAY_NAMES.items()
    ]
    mood_choices = [("All Moods", None)] + [
        (display_name, slug) for slug, display_name in MOOD_DISPLAY_NAMES.items()
    ]
    title_choices = engine.available_titles(limit=800)
    summary = engine.dataset_summary()
    query_examples = [
        "a reflective literary novel about family, faith, and grief",
        "a tense detective story with a clever investigator",
        "an inspiring business book about leadership and growth",
        "a dark philosophical novel about memory and identity",
    ]

    blocks_kwargs = {"title": "Semantic Book Recommender"}
    if include_css_in_blocks:
        blocks_kwargs["css"] = APP_CSS

    with gr.Blocks(**blocks_kwargs) as demo:
        gr.HTML(_build_hero(summary, config))

        with gr.Tabs():
            with gr.Tab("Discover"):
                with gr.Row():
                    with gr.Column(scale=4, min_width=320, elem_classes=["panel"]):
                        gr.HTML(
                            """
                            <div class="section-kicker">Discovery Desk</div>
                            <h2 class="section-title">Describe your next read</h2>
                            <p class="section-copy">
                              Search the catalog with plain language. Mention mood, topic, pace, theme,
                              or the kind of emotional atmosphere you want.
                            </p>
                            """
                        )
                        query = gr.Textbox(
                            label="What kind of book are you looking for?",
                            placeholder="Example: a dark, elegant novel about grief, memory, and identity",
                            lines=4,
                            elem_id="query-box",
                        )
                        gr.Examples(examples=query_examples, inputs=query)
                        with gr.Row():
                            topic = gr.Dropdown(topic_choices, value=None, label="Topic")
                            mood = gr.Dropdown(mood_choices, value=None, label="Mood")
                        with gr.Row():
                            min_rating = gr.Slider(0.0, 5.0, value=3.9, step=0.1, label="Minimum Rating")
                            max_pages = gr.Slider(120, 1200, value=650, step=10, label="Maximum Pages")
                        n_results = gr.Slider(3, 12, value=6, step=1, label="Number of Results")
                        search_button = gr.Button("Recommend Books", variant="primary", elem_id="search-button")

                    with gr.Column(scale=6, min_width=380, elem_classes=["panel"]):
                        gr.HTML(
                            """
                            <div class="section-kicker">Recommendation Shelf</div>
                            <h2 class="section-title">Curated matches</h2>
                            <p class="section-copy">
                              Results combine semantic similarity with quality signals such as rating,
                              popularity, and filter alignment.
                            </p>
                            """
                        )
                        cards_html = gr.HTML(_empty_state_html())
                        with gr.Accordion("Open structured table", open=False):
                            results_table = gr.HTML(_empty_table_html())

                search_button.click(
                    fn=lambda q, t, m, r, p, n: _search_books(engine, q, t, m, r, p, int(n)),
                    inputs=[query, topic, mood, min_rating, max_pages, n_results],
                    outputs=[cards_html, results_table],
                )
                demo.load(
                    fn=lambda: _search_books(engine, "", None, None, 3.9, 650, 6),
                    outputs=[cards_html, results_table],
                )

            with gr.Tab("Similar"):
                with gr.Row():
                    with gr.Column(scale=4, min_width=320, elem_classes=["panel"]):
                        gr.HTML(
                            """
                            <div class="section-kicker">Anchor Search</div>
                            <h2 class="section-title">Start from a title you already like</h2>
                            <p class="section-copy">
                              Pick a known book, then let the recommender search for nearby themes,
                              tone, and descriptive language across the catalog.
                            </p>
                            """
                        )
                        title = gr.Dropdown(title_choices, label="Choose a book", elem_id="similar-title")
                        with gr.Row():
                            similar_topic = gr.Dropdown(topic_choices, value=None, label="Topic")
                            similar_mood = gr.Dropdown(mood_choices, value=None, label="Mood")
                        with gr.Row():
                            similar_rating = gr.Slider(0.0, 5.0, value=3.9, step=0.1, label="Minimum Rating")
                            similar_pages = gr.Slider(120, 1200, value=650, step=10, label="Maximum Pages")
                        similar_n = gr.Slider(3, 12, value=6, step=1, label="Number of Results")
                        similar_button = gr.Button("Find Similar Titles", variant="primary", elem_id="similar-button")

                    with gr.Column(scale=6, min_width=380, elem_classes=["panel"]):
                        gr.HTML(
                            """
                            <div class="section-kicker">Related Shelf</div>
                            <h2 class="section-title">Books with a similar profile</h2>
                            <p class="section-copy">
                              The selected book description becomes the search anchor, then the anchor title
                              itself is removed from the final ranking.
                            </p>
                            """
                        )
                        similar_cards = gr.HTML(_empty_state_html("Select a title to generate a matching shelf."))
                        with gr.Accordion("Open structured table", open=False):
                            similar_table = gr.HTML(_empty_table_html("The table view will appear after you choose an anchor title."))

                similar_button.click(
                    fn=lambda selected_title, t, m, r, p, n: _find_similar_books(
                        engine,
                        selected_title,
                        t,
                        m,
                        r,
                        p,
                        int(n),
                    ),
                    inputs=[title, similar_topic, similar_mood, similar_rating, similar_pages, similar_n],
                    outputs=[similar_cards, similar_table],
                )

            with gr.Tab("How It Works"):
                gr.HTML(_build_explain_panel(summary, config))
                with gr.Row():
                    with gr.Column(elem_classes=["panel"]):
                        gr.HTML(_summary_table_html("Top Topics After Cleaning", summary["top_topics"]))
                    with gr.Column(elem_classes=["panel"]):
                        gr.HTML(_summary_table_html("Top Moods After Enrichment", summary["top_moods"]))

    return demo


def _search_books(
    engine: BookRecommender,
    query: str,
    topic: str | None,
    mood: str | None,
    min_rating: float,
    max_pages: int,
    n_results: int,
):
    results = engine.recommend_from_query(
        query=query,
        n_results=n_results,
        topic=topic or None,
        mood=mood or None,
        min_rating=min_rating,
        max_pages=max_pages,
    )
    return _results_to_html(results), _results_to_table_html(results)


def _find_similar_books(
    engine: BookRecommender,
    title: str,
    topic: str | None,
    mood: str | None,
    min_rating: float,
    max_pages: int,
    n_results: int,
):
    if not title:
        empty = engine.curated_picks(n_results=n_results, topic=topic or None, mood=mood or None)
        return _results_to_html(empty), _results_to_table_html(empty)

    results = engine.recommend_similar_books(
        title=title,
        n_results=n_results,
        topic=topic or None,
        mood=mood or None,
        min_rating=min_rating,
        max_pages=max_pages,
    )
    return _results_to_html(results), _results_to_table_html(results)


def _results_to_html(results: pd.DataFrame) -> str:
    if results.empty:
        return _empty_state_html("No books matched the current filters. Try widening the mood, rating, or page limits.")

    cards: list[str] = []
    for rank, row in enumerate(results.head(6).itertuples(index=False), start=1):
        cards.append(
            f"""
            <article class="result-card">
              <div class="result-cover">{_cover_markup(getattr(row, 'thumbnail', ''), str(row.full_title))}</div>
              <div>
                <div class="result-top">
                  <div>
                    <div class="result-rank">Shelf Match {rank}</div>
                    <h3 class="result-title">{escape(str(row.full_title))}</h3>
                    <p class="result-author">{escape(str(row.authors))}</p>
                  </div>
                  <div class="score-chip">Score {float(row.final_score):.2f}</div>
                </div>
                <div class="badge-row">
                  <span class="meta-badge">{escape(str(row.topic_display))}</span>
                  <span class="meta-badge">{escape(str(row.mood_display))}</span>
                  <span class="meta-badge">{escape(str(row.emotion_display))}</span>
                  <span class="meta-badge">{escape(str(row.reading_length))}</span>
                </div>
                <div class="fact-grid">
                  <div class="fact-card"><span class="fact-label">Rating</span><span class="fact-value">{float(row.average_rating):.2f}</span></div>
                  <div class="fact-card"><span class="fact-label">Readers</span><span class="fact-value">{int(row.ratings_count):,}</span></div>
                  <div class="fact-card"><span class="fact-label">Pages</span><span class="fact-value">{int(row.num_pages)}</span></div>
                  <div class="fact-card"><span class="fact-label">Published</span><span class="fact-value">{int(row.published_year)}</span></div>
                </div>
                <p class="reason-note"><strong>Why it matched:</strong> {escape(str(row.reason))}</p>
                <p class="preview-copy">{escape(str(row.description_preview))}</p>
              </div>
            </article>
            """
        )

    return f'<section class="results-shell">{"".join(cards)}</section>'


def _cover_markup(thumbnail: str, title: str) -> str:
    safe_title = escape(title)
    safe_thumbnail = escape(thumbnail)
    if safe_thumbnail:
        return f"<img src='{safe_thumbnail}' alt='Cover for {safe_title}'>"
    return f"<div class='result-cover-placeholder'>{escape(_title_initials(title))}<br>Library Edition</div>"


def _empty_state_html(message: str | None = None) -> str:
    body = message or "Start with a mood, topic, or story idea. The recommender will build a curated shelf from the cleaned catalog."
    return f"<section class='empty-state'><h3>Your recommendation shelf will appear here</h3><p>{escape(body)}</p></section>"


def _results_to_table_html(results: pd.DataFrame) -> str:
    if results.empty:
        return _empty_table_html()

    rows: list[str] = []
    for row in results.itertuples(index=False):
        rows.append(
            f"""
            <tr>
              <td><span class="table-title">{escape(str(row.full_title))}</span></td>
              <td><span class="table-muted">{escape(str(row.authors))}</span></td>
              <td>{escape(str(row.topic_display))}</td>
              <td>{escape(str(row.mood_display))}</td>
              <td>{float(row.average_rating):.2f}</td>
              <td>{int(row.num_pages)}</td>
              <td>{int(row.published_year)}</td>
              <td>{float(row.final_score):.2f}</td>
            </tr>
            """
        )

    return (
        "<div class='table-shell'><table><thead><tr>"
        "<th>Title</th><th>Author</th><th>Topic</th><th>Mood</th>"
        "<th>Rating</th><th>Pages</th><th>Year</th><th>Score</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def _empty_table_html(message: str | None = None) -> str:
    body = message or "A clean table summary of the current recommendation shelf will appear here."
    return f"<section class='empty-state'><h3>Table view</h3><p>{escape(body)}</p></section>"

def _build_hero(summary: dict[str, object], config: BuildConfig) -> str:
    semantic_label, semantic_copy = _semantic_mode_description(config.semantic_provider)
    metadata_label, metadata_copy = _metadata_mode_description(config.metadata_provider)
    return f"""
    <section class='hero-shell'>
      <div class='hero-kicker'>Semantic Library Engine</div>
      <h1 class='hero-title'>Semantic Book Recommender</h1>
      <p class='hero-copy'>
        Discover books by describing meaning, tone, and emotional texture. The interface now uses a calm
        editorial library style so the results stay readable and premium.
      </p>
      <div class='hero-grid'>
        <div class='hero-card'>
          <span class='hero-label'>Books After Cleaning</span>
          <span class='hero-value'>{int(summary['books_after_cleaning']):,}</span>
        </div>
        <div class='hero-card'>
          <span class='hero-label'>Median Rating</span>
          <span class='hero-value'>{float(summary['median_rating']):.2f}</span>
        </div>
        <div class='hero-card'>
          <span class='hero-label'>Median Pages</span>
          <span class='hero-value'>{int(summary['median_pages'])}</span>
        </div>
        <div class='hero-card stack-card'>
          <h3>AI Status</h3>
          <p class='stack-copy'>{escape(semantic_copy)} {escape(metadata_copy)}</p>
          <div class='stack-pills'>
            <span class='stack-pill'>{escape(semantic_label)}</span>
            <span class='stack-pill'>{escape(metadata_label)}</span>
          </div>
        </div>
      </div>
    </section>
    """


def _build_explain_panel(summary: dict[str, object], config: BuildConfig) -> str:
    semantic_label, semantic_copy = _semantic_mode_description(config.semantic_provider)
    metadata_label, metadata_copy = _metadata_mode_description(config.metadata_provider)
    return f"""
    <section class='explain-grid'>
      <div class='explain-card'>
        <h3>Semantic Retrieval</h3>
        <p><strong>{escape(semantic_label)}</strong><br>{escape(semantic_copy)}</p>
      </div>
      <div class='explain-card'>
        <h3>Metadata Enrichment</h3>
        <p><strong>{escape(metadata_label)}</strong><br>{escape(metadata_copy)}</p>
      </div>
      <div class='explain-card'>
        <h3>Is this already AI?</h3>
        <p>Yes. The app already uses AI-style semantic text matching. In the default mode it understands similarity
        through vectorized text and latent semantic structure, not only exact keyword matches.</p>
      </div>
      <div class='explain-card'>
        <h3>What is still optional?</h3>
        <p>The stronger cloud path is optional: OpenAI embeddings with LangChain and Chroma, plus transformer-based
        topic and emotion labeling. The local baseline already works end to end.</p>
      </div>
    </section>
    """


def _summary_table_html(title: str, values: dict[str, int]) -> str:
    rows = []
    for key, value in values.items():
        rows.append(
            f"<tr><td><span class='table-title'>{escape(str(key))}</span></td><td>{int(value):,}</td></tr>"
        )
    return (
        f"<div class='section-kicker'>Dataset Summary</div><h2 class='section-title'>{escape(title)}</h2>"
        f"<div class='table-shell'><table><thead><tr><th>Label</th><th>Books</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def _semantic_mode_description(provider: str) -> tuple[str, str]:
    if provider == "openai":
        return (
            "OpenAI Embeddings + Chroma",
            "The current build uses embedding vectors from OpenAI with a LangChain-Chroma vector database.",
        )
    return (
        "Local Semantic Retrieval",
        "The current build uses TF-IDF plus latent semantic analysis to match meaning across descriptions without calling OpenAI.",
    )


def _metadata_mode_description(provider: str) -> tuple[str, str]:
    if provider == "transformers":
        return (
            "Transformer Classification",
            "Topic, mood, and emotion metadata are generated with transformer models for richer AI labeling.",
        )
    return (
        "Keyword Enrichment Baseline",
        "Topic, mood, and emotion labels are generated locally with curated rules so the project stays reproducible offline.",
    )


def _title_initials(title: str) -> str:
    words = [word for word in title.split() if word]
    initials = "".join(word[0] for word in words[:3]).upper()
    return initials or "Book"


def _dict_to_rows(values: dict[str, int], key_name: str, value_name: str) -> list[dict[str, int | str]]:
    return [{key_name: key, value_name: value} for key, value in values.items()]
