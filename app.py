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
RECSYS_MASTER_REGEX      = re.compile('|'.join(rf'\b{re.escape(t)}\b' if len(t) < 4 else re.escape(t) for t in RECSYS_KEYWORDS_RAW), re.IGNORECASE)

APP_CSS = """
<style>
    /* Layout - Optimized for central column (Google-style focus) */
    [data-testid="stAppViewBlockContainer"] {
        max-width: 720px !important;
        padding-top: 3rem !important;
        margin: auto;
    }

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
        padding-top: 4px; /* Increased to align with larger title text */
        padding-right: 12px;
        text-align: right;
        color: #aaa;
        font-size: 0.80rem;
        font-weight: 400;
        line-height: 1.4; /* Match title line-height for alignment */
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

    /* Score Badges */
    .score-badge {
        display: inline-block;
        padding: 0px 6px;
        border-radius: 4px;
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        margin-right: 6px; /* Changed from left to right for row-start placement */
        vertical-align: middle;
        line-height: 1.6;
        letter-spacing: 0.03em;
    }
    .badge-bm25 {
        background-color: #FFF3CD; /* Yellow for search match alignment */
        color: #856404;
        border: 1px solid #ffeeba;
    }
    .badge-rs {
        background-color: #e8f0fe;
        color: #1967d2;
        border: 1px solid #c5d2f6;
    }

    /* Keyword pills - Unified with score badges */
    .keyword-pill {
        display: inline-block;
        vertical-align: middle;
        background-color: #f1f3f4;
        color: #5f6368;
        padding: 0px 6px;
        margin: 2px 0;
        border-radius: 4px;
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        border: 1px solid #dadce0;
        line-height: 1.6;
        letter-spacing: 0.02em;
    }
    /* Unified Abstract Zone */
    .gs-abstract-zone {
        margin-top: 4px;
        background: #f9f9f9;
        border-radius: 6px;
        border: 1px solid #eee; /* Uniform subtle border */
        box-shadow: none;
    }
    
    .gs-paper-toggle > summary {
        display: block;
        outline: none;
        list-style: none;
        user-select: none;
        cursor: pointer;
    }
    .gs-paper-toggle > summary::-webkit-details-marker { display: none; }

    /* Content inside the zone */
    .gs-abstract-content {
        padding: 6px 10px; /* Reduced padding */
        font-size: 0.88rem;
        line-height: 1.55;
        color: #5f6368;
    }

    /* 2-line Preview (only visible when closed) */
    .gs-snippet-preview {
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .gs-paper-toggle[open] .gs-snippet-preview { display: none; }

    /* Full Text (only visible when open) */
    .gs-full-content { 
        display: none; 
    }
    .gs-paper-toggle[open] .gs-full-content { 
        display: block;
    }

    /* Fixed Actions Row stays outside */
    .gs-actions-row {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 6px;
        margin-top: 8px;
        margin-bottom: 4px;
    }
    
    /* Toggle button next to Venue */
    .gs-toggle-btn {
        color: #1558d6;
        font-size: 0.82rem;
        font-weight: 500;
        cursor: pointer;
        display: inline-block;
        vertical-align: middle;
    }
    .gs-toggle-btn:hover { text-decoration: underline; }

    .gs-paper-toggle:not([open]) .gs-toggle-btn::after { content: "more ▾"; }
    .gs-paper-toggle[open] .gs-toggle-btn::after { content: "less ▴"; font-weight: 600; }

    /* Keyword highlight (Search results) */
    mark {
        background-color: #FFF3CD;
        color: #856404;
        font-weight: 600;
        padding: 0 2px;
        border-radius: 2px;
    }
    .recsys-mark {
        color: #1967d2;
        background-color: #e8f0fe;
        font-weight: 600;
        padding: 0 2px;
        border-radius: 2px;
    }
    /* Responsive Flex Grid */
    [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
        gap: 8px !important;
    }
    [data-testid="stHorizontalBlock"] > div {
        min-width: 130px !important; /* Forces one per row if width < 130px */
        flex-basis: 130px !important;
    }
    /* Specific narrow columns like 'Limit' */
    div[data-testid="column"]:has(input[type="number"]) {
        min-width: 80px !important;
        flex-basis: 80px !important;
    }

    .section-label {
        font-size: 1.1rem; /* Significantly larger */
        font-weight: 700;
        color: #202124; /* Darker for better contrast */
        margin-top: 32px; /* More breathing room */
        margin-bottom: 12px;
        letter-spacing: -0.01em;
    }
    .gs-filter-section {
        background-color: #fcfcfc;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #f0f0f0;
        margin-top: 10px;
    }
    
    /* Bigger labels for Streamlit widgets */
    .stSlider label, .stSelectbox label, .stMultiSelect label, .stTextInput label, .stNumberInput label {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        color: #3c4043 !important;
        margin-bottom: 8px !important;
    }
    .stat-item {
        font-size: 0.85rem;
        color: #5f6368;
        padding: 4px 0;
        border-bottom: 1px solid #f1f3f4;
    }
    .stat-value {
        font-weight: 700;
        color: #1a73e8;
        float: right;
    }

    /* Extremely tight checkbox */
    [data-testid="stCheckbox"] {
        margin-bottom: -22px !important;
        padding-top: 0px !important;
        padding-bottom: 0px !important;
    }
    [data-testid="stCheckbox"] label p {
        font-size: 0.78rem !important;
        margin: 0 !important;
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

        # Search corpus: Title + Abstract + Keywords
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

        # BM25 index - using space-splitting for the corpus words
        bm25 = BM25Okapi([doc.split() for doc in df['search_corpus_lower']])

        return df, min_year, max_year, summary_df, bm25

    except Exception as e:
        st.error(f"Failed to load database: {e}")
        st.info("Try rebuilding with 'python update.py --force'.")
        return default


# ---------------------------------------------------------------------------
# Search helpers
# ---------------------------------------------------------------------------

def bm25_search(bm25, df, query_str, top_k=None):
    """Rank subset df by BM25 score for query_str, only returning matches."""
    # Split by comma instead of spaces. 
    # To keep BM25 working with the word-level index, we flatten the comma-separated parts into words.
    tokens = [w for t in query_str.lower().split(',') if t.strip() for w in t.split()]
    if not tokens:
        return df
    
    # Get scores and filter out non-matches (0.0)
    all_scores = bm25.get_scores(tokens)
    scores = all_scores[df.index]
    
    result = df.copy()
    result['BM25_Score'] = scores
    # Filter for score > 0 to remove irrelevant fallback results
    result = result[result['BM25_Score'] > 0]
    result = result.sort_values('BM25_Score', ascending=False)
    
    return result.head(top_k) if top_k else result


def filter_by_keywords(df, query, mode='AND'):
    """Hard keyword filter on search_corpus_lower. Splits query by comma."""
    terms = [t.strip() for t in query.lower().split(',') if t.strip()]
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
    
    keys = []
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
            keys.append(key.title())
    
    # Sort alphabetically
    keys.sort()
    
    for key in keys:
        spans.append(f'<span class="keyword-pill">{key}</span>')
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
        terms = [t.strip() for t in highlight_query.lower().split(',') if t.strip()]
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
    
    # [NEW] Meta includes Venue(Year) + more/less toggle
    venue_html = f'<span class="gs-venue">{conf} ({year})</span>'
    toggle_html = '<span class="gs-toggle-btn"></span>'
    meta_parts.append(f'{venue_html} <span class="gs-dot">·</span> {toggle_html}')

    badge_html = ""
    if 'BM25_Score' in row and pd.notna(row['BM25_Score']):
        badge_html += f'<span class="score-badge badge-bm25">BM25 {row["BM25_Score"]:.1f}</span>'
    if 'recsys_match_count' in row and pd.notna(row['recsys_match_count']):
        badge_html += f'<span class="score-badge badge-rs">RS Match {int(row["recsys_match_count"])}</span>'

    # Meta line now ONLY bibliographic info + toggle
    meta_line = '<div class="gs-meta">' + '<span class="gs-dot">·</span>'.join(meta_parts) + '</div>'

    # Combined Action Row logic: Badges (Left) + Keywords (Right)
    pills_html = generate_keyword_pills(keywords_str)
    
    # Abstract versions
    raw_abstract    = row.get('Abstract', '')
    plain_abs       = clean_latex(raw_abstract)
    highlighted_abs = _render_abstract(raw_abstract, highlight_query_str)

    # Professional HTML Assembly
    html = (
        f'<div class="gs-paper">'
        f'  <div class="gs-index-col">{index}</div>'
        f'  <div class="gs-content-col">'
        f'    <div class="gs-title">{title}{ext_link}</div>'
        f'    <details class="gs-paper-toggle">'
        f'        <summary>'
        f'            {meta_line}'
        f'            <div class="gs-abstract-zone">'
        f'                <div class="gs-abstract-content">'
        f'                    <div class="gs-snippet-preview">{plain_abs}</div>'
        f'                    <div class="gs-full-content">{highlighted_abs}</div>'
        f'                </div>'
        f'            </div>'
        f'        </summary>'
        f'    </details>'
        f'    <div class="gs-actions-row">'
        f'        {badge_html}'
        f'        {pills_html}'
        f'    </div>'
        f'  </div>'
        f'</div>'
    )
    return html.strip()


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(layout="centered", page_title="RecSys Paper Finder")
    st.title("RecSys Paper Finder")
    st.markdown(APP_CSS, unsafe_allow_html=True)

    papers_df, min_year, max_year, summary_df, bm25 = load_search_database()

    if papers_df is None:
        st.warning("Failed to load essential data. The app cannot start.")
        return

    # --- 1. Intelligent Dashboard Row ---
    if summary_df is not None and not summary_df.empty:
        # Detailed Stats in Toggle
        with st.expander("Show Detailed Database Analytics", expanded=False):
            base = alt.Chart(summary_df).encode(
                x=alt.X('Year', axis=alt.Axis(format='d')),
                y='Paper Count',
                color='Conference',
                tooltip=['Conference', 'Year', 'Paper Count']
            )
            st.altair_chart(base.mark_line() + base.mark_point(size=100, filled=True),
                            width='stretch')

    # --- 2. Advanced Search Stage ---
    st.markdown('<div class="section-label">Define Search Strategy</div>', unsafe_allow_html=True)
    
    search_type = st.radio(
        "Select your primary search intent:",
        ('bm25', 'exact', 'author'),
        format_func=lambda x: {
            'bm25': 'Semantic Discovery (BM25)', 
            'exact': 'Specific Keywords (Exact)',
            'author': 'Researcher Search (Author)'
        }[x],
        horizontal=True
    )

    # Only show primary query what is needed
    if search_type == 'bm25':
        bm25_query = st.text_input(
            label="What are you looking for?",
            placeholder="e.g., Graph Neural Networks, Diffusion",
            help="Results are ranked by semantic relevance. Use commas to separate multiple terms."
        )
        author_query = ""
    elif search_type == 'author':
        author_query = st.text_input(
            label="Enter Author Name",
            placeholder="e.g., He, Xiangnan",
            help="Find specific researchers in the RecSys community."
        )
        bm25_query = ""
    else: # exact mode
        bm25_query = ""
        author_query = ""

    # Integrated Keyword Constraints (AND/OR) inside the Define block
    must_include_query = st.text_input("Must Contain (AND)", placeholder="Optional terms (e.g., bert, attention)", help="Papers MUST contain all these comma-separated terms.")
    any_include_query = st.text_input("Include Any (OR)", placeholder="Optional terms (e.g., graph, multi-modal)", help="Papers with ANY of these comma-separated terms.")

    # --- 3. Refinement Stage ---
    st.markdown('<div class="section-label">Refine Results</div>', unsafe_allow_html=True)
    
    all_confs = sorted(papers_df['Conference Name (Book Title)'].unique().tolist())
    selected_confs = st.multiselect(
        "Target Venues", 
        options=all_confs,
        default=all_confs,
        help="Select specific conferences to include in your search."
    )
    
    # Bottom refinement set
    top_k = st.number_input("Max Results", min_value=1, max_value=5000, value=100, step=10)
    selected_year_range = st.slider("Publication Year Range", min_value=min_year, max_value=max_year, value=(min_year, max_year))

    st.markdown('<div style="margin-top: 24px;"></div>', unsafe_allow_html=True)
    if st.button("Find Papers", use_container_width=True, type="primary"):
        highlight_query_str = ""

        # Step 1: Year & Conference filters
        filtered_df = papers_df
        
        # Conference filter
        if selected_confs:
            filtered_df = filtered_df[filtered_df['Conference Name (Book Title)'].isin(selected_confs)]
        else:
            st.warning("Please select at least one conference.")
            st.stop()
            
        # Year filter
        if min_year < max_year:
            s, e = selected_year_range
            filtered_df = filtered_df[(filtered_df['Year_Num'] >= s) & (filtered_df['Year_Num'] <= e)]

        # Step 2: Keyword hard filters
        if must_include_query:
            filtered_df = filter_by_keywords(filtered_df, must_include_query, mode='AND')
        if any_include_query:
            filtered_df = filter_by_keywords(filtered_df, any_include_query, mode='OR')
        
        # Step 2.1: Author filter
        if author_query:
            terms = [t.strip().lower() for t in author_query.split(',') if t.strip()]
        # Step 3: Handle Search Modes
        if search_type == 'bm25':
            if not bm25_query:
                st.warning("Please enter a BM25 query.")
                st.stop()
            results_df = bm25_search(bm25, filtered_df, bm25_query, top_k=top_k) if not filtered_df.empty else filtered_df
            highlight_query_str = bm25_query
            
        elif search_type == 'author':
            if not author_query:
                st.warning("Please enter an author name.")
                st.stop()
            terms = [t.strip().lower() for t in author_query.split(',') if t.strip()]
            mask = pd.Series(True, index=filtered_df.index)
            for t in terms:
                mask &= filtered_df['Author'].str.lower().str.contains(t, na=False)
            results_df = filtered_df[mask].head(top_k)
            highlight_query_str = author_query
            
        else: # exact mode
            if not must_include_query and not any_include_query:
                st.warning("Exact mode requires at least one keyword filter.")
                st.stop()
            results_df = filtered_df.head(top_k)
            highlight_query_str = (must_include_query + ',' + any_include_query).strip(',')

        # Step 4: Display results
        if results_df is None or results_df.empty:
            st.info("No papers found matching all criteria.")
        else:
            recsys_df    = results_df[results_df['recsys_class'] == 'recsys']
            ambiguous_df = results_df[results_df['recsys_class'] == 'ambiguous']
            other_df     = results_df[results_df['recsys_class'] == 'other']

            st.subheader(f"Search Results ({len(results_df)} total)")
            tab1, tab2, tab3 = st.tabs([
                f"RecSys Papers ({len(recsys_df)})",
                f"Potential RecSys ({len(ambiguous_df)})",
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