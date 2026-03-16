import re
import pandas as pd
from config import RECSYS_KEYWORDS_RAW

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
