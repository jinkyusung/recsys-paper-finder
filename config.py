import re

# --- Constants ---
DB_FILE = 'paper_database.parquet'
RECSYS_KEYWORDS_RAW = ['recommend', 'collaborative filtering', 'cf', 'matrix factorization']
RECSYS_MASTER_REGEX = re.compile('|'.join(rf'\b{re.escape(t)}\b' if len(t) < 4 else re.escape(t) for t in RECSYS_KEYWORDS_RAW), re.IGNORECASE)

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

    .app-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #202124;
        margin-bottom: 12px;
        letter-spacing: -0.01em;
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
