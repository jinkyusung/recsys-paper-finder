import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import logging
from rank_bm25 import BM25Okapi
import altair as alt

# --- Constants ---
DB_FILE        = 'paper_database.parquet'
RECSYS_KEYWORDS_RAW      = ['recommend', 'collaborative filtering', 'cf', 'matrix factorization', 'personalization', 'personalized']
RECSYS_MASTER_REGEX      = re.compile('|'.join(RECSYS_KEYWORDS_RAW), re.IGNORECASE)
RECOMMEND_ONLY_REGEX     = re.compile(r'recommend', re.IGNORECASE)

APP_CSS = """
<style>
    /* Layout */
    .main > div { max-width: 960px; margin: 0 auto; }

    /* Paper entry — Google Scholar style, no box */
    .gs-paper {
        display: flex;
        flex-direction: row;
        padding: 12px 0 10px 0;
        border-top: 1px solid #e8e8e8;
    }
    .gs-paper:first-of-type {
        border-top: none;
        padding-top: 2px;
    }
    /* Left: index number column */
    .gs-index-col {
        min-width: 36px;
        max-width: 36px;
        padding-top: 2px;
        padding-right: 12px;
        text-align: right;
        color: #aaa;
        font-size: 0.80rem;
        font-weight: 400;
        line-height: 1.6;
        flex-shrink: 0;
    }
    /* Right: content column */
    .gs-content-col { flex: 1; min-width: 0; }
    .gs-title {
        font-size: 1.00rem;
        font-weight: 500;
        line-height: 1.4;
        margin-bottom: 3px;
    }
    .gs-title a { color: #1558d6; text-decoration: none; }
    .gs-title a:hover { text-decoration: underline; }

    /* External link icon */
    .gs-ext-link {
        font-size: 0.85rem;
        color: #1558d6;
        margin-left: 5px;
        text-decoration: none;
        vertical-align: middle;
    }
    .gs-ext-link:hover { text-decoration: underline; }

    /* Author · Venue · Scores line */
    .gs-meta {
        font-size: 0.84rem;
        color: #555;
        margin-bottom: 4px;
        line-height: 1.5;
    }
    .gs-authors       { color: #2d6a2d; font-weight: 500; }
    .gs-venue         { color: #555; }
    .gs-dot           { color: #999; margin: 0 4px; }
    .gs-scores-inline { color: #888; font-size: 0.79rem; }

    /* Keyword pills */
    .keyword-pill {
        display: inline-block;
        background-color: #f1f3f4;
        color: #3c4043;
        padding: 2px 9px;
        margin: 2px 5px 4px 0;
        border-radius: 10px;
        font-size: 0.76rem;
        font-weight: 500;
        border: 1px solid #dadce0;
    }

    /* Bottom row: keywords + abstract toggle */
    .gs-bottom-row {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 4px;
        margin: 4px 0 2px;
    }

    /* Abstract toggle (<details>) */
    details.gs-toggle { display: inline-block; vertical-align: middle; margin: 2px 0 4px 0; }
    details.gs-toggle > summary {
        display: inline-block;
        cursor: pointer;
        color: #1558d6;
        font-size: 0.78rem;
        font-weight: 500;
        padding: 2px 9px;
        border-radius: 10px;
        border: 1px solid #c5d2f6;
        background: #f0f4ff;
        list-style: none;
        user-select: none;
    }
    details.gs-toggle > summary::-webkit-details-marker { display: none; }
    details.gs-toggle > summary::after       { content: " ▾"; font-size: 0.70rem; }
    details.gs-toggle[open] > summary::after { content: " ▴"; font-size: 0.70rem; }
    details.gs-toggle > summary:hover { background: #dce6ff; }
    .gs-abstract-body {
        margin-top: 8px;
        font-size: 0.90rem;
        line-height: 1.65;
        color: #3c4043;
        padding: 8px 4px 4px 4px;
    }

    /* Keyword highlight (Search results) */
    mark {
        background-color: #FFF3CD;
        color: #856404;
        font-weight: 600;
        padding: 0 2px;
        border-radius: 2px;
    }
    /* RecSys keyword highlight (Blue) */
    .recsys-mark {
        background-color: #D1E8FF;
        color: #004085;
        font-weight: 600;
        padding: 0 2px;
        border-radius: 2px;
    }
</style>
"""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data
def load_search_database():
    default = (None, 2000, 2025, None, None)

    if not os.path.exists(DB_FILE):
        st.error(f"Could not find '{DB_FILE}'.")
        st.info("Run 'python update.py --force' to build the database.")
        return default

    try:
        df = pd.read_parquet(DB_FILE)

        for col in ('Title', 'Author', 'Abstract', 'Keywords'):
            df[col] = df[col].fillna('')

        df['Year_Num'] = pd.to_numeric(df['Year'], errors='coerce').fillna(0).astype(int)

        valid_years = df[df['Year_Num'] > 1900]['Year_Num']
        min_year = int(valid_years.min()) if not valid_years.empty else 2000
        max_year = int(df['Year_Num'].max()) if not df.empty else 2025

        # DB summary table
        summary_df = pd.DataFrame(columns=['Conference', 'Year', 'Paper Count'])
        if 'Conference Name (Book Title)' in df.columns:
            try:
                summary_df = (
                    df.groupby(['Conference Name (Book Title)', 'Year_Num'])
                    .size()
                    .reset_index(name='Paper Count')
                    .rename(columns={'Conference Name (Book Title)': 'Conference', 'Year_Num': 'Year'})
                )
                summary_df = summary_df[summary_df['Year'] > 1900].sort_values(['Conference', 'Year'])
            except Exception as e:
                print(f"Warning: summary table failed. {e}")

        # Search corpus: Title + Abstract + Keywords (no references)
        df['search_corpus_lower'] = (
            df['Title'].str.lower() + ' ' +
            df['Abstract'].str.lower() + ' ' +
            df['Keywords'].str.lower()
        )

        # RecSys classification via keyword match count
        df['recsys_match_count'] = df['search_corpus_lower'].apply(
            lambda x: len(RECSYS_MASTER_REGEX.findall(x)) if isinstance(x, str) else 0
        )
        df['is_recsys'] = df['recsys_match_count'] > 0

        # BM25 index
        bm25 = BM25Okapi([doc.split() for doc in df['search_corpus_lower']])

        st.success(f"Loaded {len(df)} papers.")
        return df, min_year, max_year, summary_df, bm25

    except Exception as e:
        st.error(f"Failed to load database: {e}")
        st.info("Try rebuilding with 'python update.py --force'.")
        return default


# ---------------------------------------------------------------------------
# Search helpers
# ---------------------------------------------------------------------------

def bm25_search(bm25, df, query_str, top_k=None):
    """Rank subset df by BM25 score for query_str."""
    tokens = query_str.lower().split()
    if not tokens:
        return df
    scores = bm25.get_scores(tokens)[df.index]
    result = df.copy()
    result['BM25_Score'] = scores
    result = result.sort_values('BM25_Score', ascending=False)
    return result.head(top_k) if top_k else result


def filter_by_keywords(df, query, mode='AND'):
    """Hard keyword filter on search_corpus_lower."""
    terms = [t for t in query.lower().split() if t]
    if not terms:
        return df
    if mode == 'AND':
        mask = pd.Series(True, index=df.index)
        for t in terms:
            mask &= df['search_corpus_lower'].str.contains(t, na=False)
    else:  # OR
        mask = pd.Series(False, index=df.index)
        for t in terms:
            mask |= df['search_corpus_lower'].str.contains(t, na=False)
    return df[mask]


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def generate_keyword_pills(keywords_str):
    if not keywords_str or pd.isna(keywords_str):
        return ''
    spans = []
    for k in re.split(r'[;,]', keywords_str):
        key = re.sub(r'[^a-zA-Z0-9\s]', '', k).strip()
        if key:
            spans.append(f'<span class="keyword-pill">{key.title()}</span>')
    return ' '.join(spans)


def _first_author(author_str):
    """Return 'First Author et al.' from a bibtex author string."""
    if not author_str or pd.isna(author_str):
        return ''
    authors = re.split(r'\s+and\s+', author_str.strip(), flags=re.IGNORECASE)
    first = authors[0].strip()
    if ',' in first:
        parts = first.split(',', 1)
        first = f"{parts[1].strip()} {parts[0].strip()}"
    return f"{first} et al." if len(authors) > 1 else first


def _render_abstract(text, highlight_query):
    """Return abstract text with search terms highlighted in yellow and RecSys terms in blue."""
    if not text:
        return '<em>No abstract provided.</em>'
    
    # 1. Highlight RecSys terms (Blue)
    try:
        # Use RECSYS_KEYWORDS_RAW defined above
        recsys_pattern = re.compile('|'.join(rf'\b{re.escape(t)}\b' if len(t) < 4 else re.escape(t) 
                                            for t in RECSYS_KEYWORDS_RAW), re.IGNORECASE)
        text = recsys_pattern.sub(r'<span class="recsys-mark">\g<0></span>', text)
    except Exception:
        pass

    # 2. Highlight Search terms (Yellow)
    if not highlight_query:
        return text
    try:
        terms = [t for t in highlight_query.lower().split() if t]
        if not terms:
            return text
        search_pattern = re.compile('|'.join(re.escape(t) for t in terms), re.IGNORECASE)
        # To avoid highlighting terms inside already created span tags, we'd need a more complex regex.
        # But for simple display, nested or consecutive tags are often handled okay by browsers.
        text = search_pattern.sub(r'<mark>\g<0></mark>', text)
    except Exception:
        pass
    return text


def display_paper(row, highlight_query_str, index):
    """Return a single paper card HTML in Google Scholar style."""
    title        = row.get('Title', 'No Title')
    url          = row.get('url', '')
    conf         = row.get('Conference Name (Book Title)', 'N/A')
    year         = row.get('Year', 'N/A')
    author_str   = row.get('Author', '')
    keywords_str = row.get('Keywords', '')

    # Title + external link
    has_url  = pd.notna(url) and url
    ext_link = (f'<a class="gs-ext-link" href="{url}" target="_blank" title="Open paper">↗</a>'
                if has_url else '')
    title_html = f'<div class="gs-title">{title}{ext_link}</div>'

    # Meta line: author · venue · scores
    meta_parts = []
    author_display = _first_author(author_str)
    if author_display:
        meta_parts.append(f'<span class="gs-authors">{author_display}</span>')
    meta_parts.append(f'<span class="gs-venue">{conf} ({year})</span>')

    score_parts = []
    if 'BM25_Score' in row and pd.notna(row['BM25_Score']):
        score_parts.append(f"BM25&nbsp;<strong>{row['BM25_Score']:.2f}</strong>")
    if 'recsys_match_count' in row and pd.notna(row['recsys_match_count']):
        score_parts.append(f"RS Match&nbsp;<strong>{int(row['recsys_match_count'])}</strong>")
    meta_parts.append(f'<span class="gs-scores-inline">{" · ".join(score_parts)}</span>')

    meta_html = '<div class="gs-meta">' + '<span class="gs-dot">·</span>'.join(meta_parts) + '</div>'

    # Bottom row: keyword pills + abstract toggle
    pills_html        = generate_keyword_pills(keywords_str)
    rendered_abstract = _render_abstract(row.get('Abstract', ''), highlight_query_str)
    abstract_toggle   = (
        f'<details class="gs-toggle">'
        f'<summary>Abstract</summary>'
        f'<div class="gs-abstract-body">{rendered_abstract}</div>'
        f'</details>'
    )
    bottom_row = f'<div class="gs-bottom-row">{pills_html}{abstract_toggle}</div>'

    return (
        f'<div class="gs-paper">'
        f'  <div class="gs-index-col">{index}</div>'
        f'  <div class="gs-content-col">{title_html}{meta_html}{bottom_row}</div>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(layout="wide", page_title="RecSys Paper Finder", page_icon="🔎")
    st.title("RecSys Paper Finder")
    st.markdown(APP_CSS, unsafe_allow_html=True)

    papers_df, min_year, max_year, summary_df, bm25 = load_search_database()

    if papers_df is None:
        st.warning("Failed to load essential data. The app cannot start.")
        return

    # Database summary chart
    if summary_df is not None and not summary_df.empty:
        with st.expander("View Database Summary", expanded=False):
            st.markdown("#### Paper Counts by Conference and Year")
            base  = alt.Chart(summary_df).encode(
                x=alt.X('Year', axis=alt.Axis(format='d')),
                y='Paper Count',
                color='Conference',
                tooltip=['Conference', 'Year', 'Paper Count']
            )
            st.altair_chart(base.mark_line() + base.mark_point(size=100, filled=True),
                            width='stretch')

    # Search mode
    search_type = st.radio(
        "Search Mode:",
        ('bm25', 'exact'),
        format_func=lambda x: {'bm25': '🔍 BM25 (Ranked by Relevance)', 'exact': '🔎 Exact (Keyword Filter Only)'}[x],
        horizontal=True
    )

    bm25_query = st.text_input(
        "BM25 Search Query:",
        placeholder="e.g., graph neural network collaborative filtering",
        disabled=(search_type == 'exact'),
        help="Ranks results by BM25 relevance across Title, Abstract, and Keywords."
    )

    col_must, col_any = st.columns(2)
    with col_must:
        must_include_query = st.text_input(
            "Must-include keywords (AND)",
            placeholder="e.g., graph cf",
            help="Results MUST contain ALL of these terms in Title / Abstract / Keywords."
        )
    with col_any:
        any_include_query = st.text_input(
            "Include-at-least-one (OR)",
            placeholder="e.g., privacy fairness",
            help="Results MUST contain AT LEAST ONE of these terms in Title / Abstract / Keywords."
        )

    col_year, col_topk = st.columns(2)
    with col_year:
        if min_year < max_year:
            selected_year_range = st.slider("Filter by Year Range:", min_value=min_year, max_value=max_year, value=(min_year, max_year))
        else:
            st.info(f"Data for year {min_year} only.")
            selected_year_range = (min_year, max_year)
    with col_topk:
        top_k = st.number_input("Max Results to Display", min_value=1, max_value=5000, value=50, step=10)

    if st.button("Search"):
        highlight_query_str = ""

        # Step 1: Year filter
        filtered_df = papers_df
        if min_year < max_year:
            s, e = selected_year_range
            filtered_df = filtered_df[(filtered_df['Year_Num'] >= s) & (filtered_df['Year_Num'] <= e)]

        # Step 2: Keyword hard filters
        if must_include_query:
            filtered_df = filter_by_keywords(filtered_df, must_include_query, mode='AND')
        if any_include_query:
            filtered_df = filter_by_keywords(filtered_df, any_include_query, mode='OR')

        # Step 3: Rank by mode
        if search_type == 'bm25':
            if not bm25_query:
                st.warning("Please enter a BM25 query to search.")
                st.stop()
            results_df = bm25_search(bm25, filtered_df, bm25_query, top_k=top_k) if not filtered_df.empty else filtered_df
            highlight_query_str = bm25_query
        else:
            if not must_include_query and not any_include_query:
                st.warning("Exact mode requires at least one keyword filter (AND or OR).")
                st.stop()
            results_df = filtered_df.head(top_k)
            highlight_query_str = (must_include_query + ' ' + any_include_query).strip()

        # Step 4: Display results
        if results_df is None or results_df.empty:
            st.info("No papers found matching all criteria.")
        else:
            recsys_df = results_df[results_df['is_recsys']]
            other_df  = results_df[~results_df['is_recsys']]

            st.subheader(f"Search Results ({len(results_df)} total)")
            tab1, tab2 = st.tabs([
                f"Recommender System Papers ({len(recsys_df)})",
                f"Other Papers ({len(other_df)})"
            ])
            with tab1:
                if recsys_df.empty:
                    st.info("No matching recommender system papers found.")
                else:
                    html = ''.join(
                        display_paper(row, highlight_query_str, i)
                        for i, (_, row) in enumerate(recsys_df.iterrows(), 1)
                    )
                    st.markdown(html, unsafe_allow_html=True)
            with tab2:
                if other_df.empty:
                    st.info("No matching other papers found.")
                else:
                    html = ''.join(
                        display_paper(row, highlight_query_str, i)
                        for i, (_, row) in enumerate(other_df.iterrows(), 1)
                    )
                    st.markdown(html, unsafe_allow_html=True)


if __name__ == "__main__":
    main()