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
RECSYS_KEYWORDS_RAW      = ['recommend', 'collaborative filtering', 'cf', 'matrix factorization']
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
        font-size: 1.05rem; /* Increased size to match title weight */
        color: #1a73e8; /* Premium Google blue */
        margin-left: 6px;
        text-decoration: none;
        display: inline-block;
        vertical-align: middle;
        line-height: 1;
        margin-top: -2px; /* Fine-tuned alignment */
    }
    .gs-ext-link:hover { color: #1558d6; text-decoration: underline; }

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
        vertical-align: middle; /* Fix: Match toggle alignment */
        background-color: #f1f3f4;
        color: #3c4043;
        padding: 2px 9px;
        margin: 2px 0; /* Simpler margin */
        border-radius: 10px;
        font-size: 0.76rem;
        font-weight: 500;
        border: 1px solid #dadce0;
    }

    /* Abstract & Actions Layout */
    .gs-paper-toggle { border: none; }
    .gs-paper-toggle > summary {
        display: block;
        outline: none;
        list-style: none;
        user-select: none;
        cursor: pointer;
    }
    .gs-paper-toggle > summary::-webkit-details-marker { display: none; }

    /* 2-line Preview snippet */
    .gs-snippet-preview {
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
        text-overflow: ellipsis;
        color: #5f6368;
        font-size: 0.88rem;
        line-height: 1.55;
        margin-top: 4px;
        margin-bottom: 4px;
    }
    .gs-paper-toggle[open] .gs-snippet-preview { display: none; }

    /* Fixed Actions Row (more/less + keywords) */
    .gs-actions-row {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 6px;
        margin-top: 2px;
    }
    .gs-toggle-btn {
        color: #1558d6;
        font-size: 0.82rem;
        font-weight: 500;
        min-width: 50px; /* Keep consistent width */
    }
    .gs-toggle-btn:hover { text-decoration: underline; }

    .gs-paper-toggle:not([open]) .gs-toggle-btn::after { content: "more ▾"; }
    .gs-paper-toggle[open] .gs-toggle-btn::after { content: "less ▴"; font-weight: 600; }

    /* Full abstract box */
    .gs-abstract-full-box {
        margin-top: 10px;
        font-size: 0.88rem;
        line-height: 1.6;
        color: #5f6368;
        padding: 14px 18px;
        background: #fdfdfd;
        border-radius: 8px;
        border: 1px solid #eef2ff;
        border-left: 4px solid #1558d6;
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

        # RecSys classification via granular keyword match count
        def get_hits(text):
            if not text: return 0
            return len(RECSYS_MASTER_REGEX.findall(text.lower()))

        title_hits    = df['Title'].apply(get_hits)
        keyword_hits  = df['Keywords'].apply(get_hits)
        abstract_hits = df['Abstract'].apply(get_hits)

        # Classification Logic:
        # 1. Strict RS: Title match > 0 OR Keywords match > 0 OR Abstract match >= 2
        # 2. Ambiguous: (NOT Strict) AND Abstract match == 1
        df['recsys_match_count'] = title_hits + keyword_hits + abstract_hits
        
        df['recsys_class'] = 'other'
        df.loc[abstract_hits == 1, 'recsys_class'] = 'ambiguous'
        df.loc[(title_hits > 0) | (keyword_hits > 0) | (abstract_hits >= 2), 'recsys_class'] = 'recsys'

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

def clean_latex(text):
    """Remove LaTeX/BibTeX escaping backslashes outside of math formulas ($...$)."""
    if not text or not isinstance(text, str):
        return text
    
    # 1. Protect math formulas ($...$) to preserve LaTeX commands inside them
    math_blocks = []
    def save_math(m):
        math_blocks.append(m.group(0))
        return f"__MATH_LATEX_{len(math_blocks)-1}__"
    
    # Find $...$ blocks
    text = re.sub(r'\$.*?\$', save_math, text)
    
    # 2. Clean outside math blocks
    # Remove escaping backslashes for special characters: \&, \%, \_, \{, \}
    text = re.sub(r'\\([&%_{}])', r'\1', text)
    
    # Replace LaTeX accent commands: \'{o}, \"{a}, \v{s} -> o, a, s
    text = re.sub(r"\\(?:['\"^`~vh])\{([a-zA-Z])\}", r"\1", text)
    text = re.sub(r"\\(?:['\"^`~vh])([a-zA-Z])", r"\1", text)
    
    # Remove remaining backslashes that are not part of a word (junk)
    text = re.sub(r'\\(?![a-zA-Z])', '', text)
    
    # Remove BibTeX grouping braces: {BERT} -> BERT
    text = text.replace('{', '').replace('}', '')

    # 3. Restore math blocks
    for i, block in enumerate(math_blocks):
        text = text.replace(f"__MATH_LATEX_{i}__", block)
        
    return text


def generate_keyword_pills(keywords_str):
    if not keywords_str or pd.isna(keywords_str):
        return ''
    spans = []
    # Common suffixes that indicate a hyphen might be a broken word (e.g., learn-ing)
    suffix_pattern = re.compile(r'(\w+)-(ing|ed|ion|er|ation|ive|ment|all?y|able|ness|s|es)\b', re.IGNORECASE)
    
    for k in re.split(r'[;,]', keywords_str):
        # 1. LaTeX cleanup
        k = clean_latex(k)
        # 2. Whitelist: allow alphanumeric, space, and hyphen
        key = re.sub(r'[^a-zA-Z0-9\s\-]', '', k).strip()
        # 3. Fix broken words (e.g., learn-ing -> learning)
        key = suffix_pattern.sub(r'\1\2', key)
        # 4. Final trim
        key = key.strip('-').strip()
        
        if key:
            # key.title() will capitalize both parts of hyphenated words: Graph-Based
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
    
    # Clean LaTeX junk first
    text = clean_latex(text)
    
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
    title        = clean_latex(row.get('Title', 'No Title'))
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

    # Combined Action Row logic
    pills_html = generate_keyword_pills(keywords_str)
    
    # Simple assembly using the new CSS structure
    abstract_section = f'''
    <details class="gs-paper-toggle">
        <summary>
            <div class="gs-snippet-preview">{plain_abs}</div>
            <div class="gs-actions-row">
                <span class="gs-toggle-btn"></span>
                {pills_html}
            </div>
        </summary>
        <div class="gs-full-content">
            <div class="gs-actions-row">
                <span class="gs-toggle-btn"></span>
                {pills_html}
            </div>
            <div class="gs-abstract-full-box">{highlighted_abs}</div>
        </div>
    </details>
    '''

    return (
        f'<div class="gs-paper">'
        f'  <div class="gs-index-col">{index}</div>'
        f'  <div class="gs-content-col">{title_html}{meta_html}{abstract_section}</div>'
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
            recsys_df    = results_df[results_df['recsys_class'] == 'recsys']
            ambiguous_df = results_df[results_df['recsys_class'] == 'ambiguous']
            other_df     = results_df[results_df['recsys_class'] == 'other']

            st.subheader(f"Search Results ({len(results_df)} total)")
            tab1, tab2, tab3 = st.tabs([
                f"RS Papers ({len(recsys_df)})",
                f"Potential RS? ({len(ambiguous_df)})",
                f"Other Papers ({len(other_df)})"
            ])
            with tab1:
                if recsys_df.empty:
                    st.info("No matching recommender system papers found.")
                else:
                    html = ''.join(display_paper(row, highlight_query_str, i)
                                   for i, (_, row) in enumerate(recsys_df.iterrows(), 1))
                    st.markdown(html, unsafe_allow_html=True)
            with tab2:
                if ambiguous_df.empty:
                    st.info("No ambiguous/potential RS papers found.")
                else:
                    html = ''.join(display_paper(row, highlight_query_str, i)
                                   for i, (_, row) in enumerate(ambiguous_df.iterrows(), 1))
                    st.markdown(html, unsafe_allow_html=True)
            with tab3:
                if other_df.empty:
                    st.info("No matching other papers found.")
                else:
                    html = ''.join(display_paper(row, highlight_query_str, i)
                                   for i, (_, row) in enumerate(other_df.iterrows(), 1))
                    st.markdown(html, unsafe_allow_html=True)


if __name__ == "__main__":
    main()