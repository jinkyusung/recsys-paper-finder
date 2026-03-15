import streamlit as st
import pandas as pd
import numpy as np
import os
import re
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
import altair as alt

# --- 상수 정의 ---
DB_FILE = 'paper_database.parquet'
EMBEDDING_FILE = 'paper_embeddings.npy'
MODEL_NAME = 'all-MiniLM-L6-v2'

RECSYS_CONCEPT_QUERY = "A paper about recommendation systems, collaborative filtering, personalization, or recommender models."
RECSYS_CONCEPT_THRESHOLD = 0.1
RECSYS_KEYWORDS_RAW = [
    'recommend', 'collaborative filtering', 'cf', 
    'matrix factorization', 'personalization', 'personalized'
]
RECSYS_MASTER_REGEX = re.compile('|'.join(RECSYS_KEYWORDS_RAW), re.IGNORECASE)
RECOMMEND_ONLY_REGEX = re.compile(r'recommend', re.IGNORECASE)

# [수정됨] KEYWORD_CSS 상수 (파일 상단)

KEYWORD_CSS = """
<style>
    /* ── Layout ── */
    .main > div { max-width: 960px; margin: 0 auto; }

    /* ── Paper entry (Google Scholar style, no box) ── */
    .gs-paper {
        padding: 14px 0 6px 0;
    }
    .gs-separator {
        border: none;
        border-top: 1px solid #e8e8e8;
        margin: 8px 0 0 0;
    }

    /* Title: blue, slightly larger, inline */
    .gs-title {
        font-size: 1.08rem;
        font-weight: 500;
        line-height: 1.4;
        margin-bottom: 3px;
    }
    .gs-title a {
        color: #1558d6;
        text-decoration: none;
    }
    .gs-title a:hover { text-decoration: underline; }
    /* External link icon at end of title */
    .gs-ext-link {
        font-size: 0.85rem;
        color: #1558d6;
        margin-left: 5px;
        text-decoration: none;
        vertical-align: middle;
    }
    .gs-ext-link:hover { text-decoration: underline; }

    /* Author · Venue line */
    .gs-meta {
        font-size: 0.84rem;
        color: #555;
        margin-bottom: 4px;
        line-height: 1.5;
    }
    .gs-authors { color: #2d6a2d; font-weight: 500; }
    .gs-venue   { color: #555; }
    .gs-dot     { color: #999; margin: 0 4px; }
    .gs-scores-inline { color: #888; font-size: 0.79rem; }

    /* Scores */
    .gs-scores {
        font-size: 0.80rem;
        color: #888;
        margin-bottom: 5px;
    }

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

    /* Abstract expander – minimal Google Scholar style */
    div[data-testid="stExpander"] {
        border: none !important;
        box-shadow: none !important;
        background: transparent !important;
    }
    div[data-testid="stExpander"] summary {
        padding: 2px 0 !important;
    }
    div[data-testid="stExpander"] summary p {
        font-size: 0.82rem !important;
        font-weight: 400 !important;
        color: #1558d6 !important;
    }
    div[data-testid="stExpander"] summary:hover p {
        text-decoration: underline;
    }

    /* Abstract text */
    .abstract-text {
        font-size: 0.90rem;
        line-height: 1.65;
        color: #3c4043;
        margin-top: 6px;
    }

    /* Highlight */
    mark {
        background-color: #FFF3CD;
        color: #856404;
        font-weight: 600;
        padding: 0 2px;
        border-radius: 2px;
    }
</style>
"""

@st.cache_resource
def load_model(model_name):
    try:
        model = SentenceTransformer(model_name)
        return model
    except Exception as e:
        st.error(f"Error: Failed to load embedding model ({model_name}). {e}")
        return None

@st.cache_data
def load_search_database(_model):
    default_return = (None, None, 2000, 2025, None, None)
    
    if not os.path.exists(DB_FILE) or not os.path.exists(EMBEDDING_FILE):
        st.error(f"Error: Could not find '{DB_FILE}' or '{EMBEDDING_FILE}'.")
        st.info("Please run 'python update.py --force' first to build the database.")
        return default_return
    
    try:
        df = pd.read_parquet(DB_FILE)
        embeddings = np.load(EMBEDDING_FILE)
        
        df['Title'] = df['Title'].fillna('')
        df['Author'] = df['Author'].fillna('')
        df['Abstract'] = df['Abstract'].fillna('')
        df['Keywords'] = df['Keywords'].fillna('')

        df['Year_Num'] = pd.to_numeric(df['Year'], errors='coerce').fillna(0).astype(int)
        
        valid_years = df[df['Year_Num'] > 1900]['Year_Num']
        min_year = int(valid_years.min()) if not valid_years.empty else 2000
        max_year = int(df['Year_Num'].max()) if not df.empty else 2025

        # 요약 테이블 생성 (이전과 동일)
        summary_df = None
        if 'Conference Name (Book Title)' in df.columns and 'Year_Num' in df.columns:
            try:
                summary_df = df.groupby(['Conference Name (Book Title)', 'Year_Num']).size().reset_index(name='Paper Count')
                summary_df.rename(columns={
                    'Conference Name (Book Title)': 'Conference',
                    'Year_Num': 'Year'
                }, inplace=True)
                summary_df = summary_df[summary_df['Year'] > 1900]
                summary_df = summary_df.sort_values(by=['Conference', 'Year'], ascending=[True, True])
            except Exception as e:
                print(f"Warning: Could not generate summary table. {e}")
                summary_df = pd.DataFrame(columns=['Conference', 'Year', 'Paper Count'])
        else:
                summary_df = pd.DataFrame(columns=['Conference', 'Year', 'Paper Count'])

        df['search_corpus_lower'] = (
            df['Title'].str.lower() + " " +
            df['Abstract'].str.lower() + " " +
            df['Keywords'].str.lower()
        )

        concept_embedding = _model.encode(RECSYS_CONCEPT_QUERY, normalize_embeddings=True)
        scores = util.dot_score(concept_embedding, embeddings)[0].numpy()
        df['recsys_score'] = scores
        is_recsys_semantic = (df['recsys_score'] > RECSYS_CONCEPT_THRESHOLD)
        
        df['has_recommend_keyword'] = df['search_corpus_lower'].str.contains(RECOMMEND_ONLY_REGEX, na=False, regex=True)
        is_recsys_keyword = df['search_corpus_lower'].str.contains(RECSYS_MASTER_REGEX, na=False, regex=True)
        
        df['is_recsys'] = is_recsys_semantic | is_recsys_keyword

        # BM25 인덱스 빌드 (토큰화: 공백 기준 split)
        tokenized_corpus = [doc.split() for doc in df['search_corpus_lower']]
        bm25 = BM25Okapi(tokenized_corpus)

        st.success(f"Successfully loaded {len(df)} papers and their embeddings.")
        return df, embeddings, min_year, max_year, summary_df, bm25
        
    except Exception as e:
        st.error(f"Error: Failed to load database files. {e}")
        st.info("Please try rebuilding with 'python update.py --force'.")
        return default_return


def bm25_search(bm25, df, query_str, top_k=None):
    """BM25 점수로 df 내 문서를 랭킹하여 반환. df는 원본 인덱스 유지 subset이어도 동작."""
    tokens = query_str.lower().split()
    if not tokens:
        return df
    # df의 각 행에 대해 BM25 점수를 계산 (원본 인덱스 기준)
    scores = bm25.get_scores(tokens)  # 전체 corpus 크기와 동일한 배열
    subset_scores = scores[df.index]  # df에 해당하는 행만 추출
    result_df = df.copy()
    result_df['BM25_Score'] = subset_scores
    result_df = result_df.sort_values(by='BM25_Score', ascending=False)
    if top_k is not None:
        result_df = result_df.head(top_k)
    return result_df

def filter_by_keywords(df, query, mode='AND'):
    terms = query.lower().split()
    
    if not terms:
        return df

    if mode == 'AND':
        mask = pd.Series([True] * len(df), index=df.index)
        for term in terms:
            if term:
                mask = mask & df['search_corpus_lower'].str.contains(term, na=False)
        return df[mask]
        
    elif mode == 'OR':
        mask = pd.Series([False] * len(df), index=df.index)
        for term in terms:
            if term:
                mask = mask | df['search_corpus_lower'].str.contains(term, na=False)
        return df[mask]

    return df 

# [제거됨] search_semantic 함수는 더 이상 사용되지 않음

def generate_keyword_pills(keywords_str):
    if not keywords_str or pd.isna(keywords_str):
        return ""
    html_spans = []
    keys = re.split(r'[;,]', keywords_str)
    for k in keys:
        # 알파벳/숫자/공백 외의 문자 제거
        key = re.sub(r'[^a-zA-Z0-9\s]', '', k)
        key = key.strip()
        if key:
            # 단어별 첫 글자 대문자화 (title casing)
            key = key.title()
            html_spans.append(f'<span class="keyword-pill">{key}</span>')
    return ' '.join(html_spans)

def _first_author(author_str):
    """bibtex 저자 문자열에서 첫 저자만 추출, 나머지는 et al. 처리."""
    if not author_str or pd.isna(author_str):
        return ""
    # bibtex 'and' 구분자로 분리
    authors = re.split(r'\s+and\s+', author_str.strip(), flags=re.IGNORECASE)
    first = authors[0].strip()
    # 'Lastname, Firstname' 형식 → 'Firstname Lastname' 변환
    if ',' in first:
        parts = first.split(',', 1)
        first = f"{parts[1].strip()} {parts[0].strip()}"
    return f"{first} et al." if len(authors) > 1 else first


def _render_abstract(abstract_text, highlight_query_str):
    """Abstract 텍스트를 하이라이트 처리하여 반환."""
    if not abstract_text:
        return "<em>No abstract provided.</em>"
    if not highlight_query_str:
        return abstract_text
    try:
        terms = [t for t in highlight_query_str.lower().split() if t]
        if not terms:
            return abstract_text
        pattern = re.compile('|'.join(re.escape(t) for t in terms), re.IGNORECASE)
        return pattern.sub(r'<mark>\g<0></mark>', abstract_text)
    except Exception:
        return abstract_text


def display_paper(row, highlight_query_str, index):
    """Google Scholar 스타일 논문 카드 (박스 없음)."""
    title      = row.get('Title', 'No Title')
    url        = row.get('url', '')
    conf       = row.get('Conference Name (Book Title)', 'N/A')
    year       = row.get('Year', 'N/A')
    author_str = row.get('Author', '')
    keywords_str = row.get('Keywords', '')

    # ── 제목 + 끝에 바로가기 ↗ ──
    has_url = pd.notna(url) and url
    ext_link = f'<a class="gs-ext-link" href="{url}" target="_blank" title="Open paper">↗</a>' if has_url else ''
    title_html = f'<div class="gs-title">{index}. {title}{ext_link}</div>'

    # ── 저자 (첫 저자 et al.) · 학회 (연도) · 점수 (같은 줄) ──
    author_display = _first_author(author_str)
    meta_parts = []
    if author_display:
        meta_parts.append(f'<span class="gs-authors">{author_display}</span>')
    meta_parts.append(f'<span class="gs-venue">{conf} ({year})</span>')

    # 점수를 venue 바로 옆에 표시
    score_parts = []
    if 'BM25_Score' in row and pd.notna(row['BM25_Score']):
        score_parts.append(f"BM25&nbsp;<strong>{row['BM25_Score']:.2f}</strong>")
    if 'recsys_score' in row and pd.notna(row['recsys_score']):
        score_parts.append(f"RecSys&nbsp;<strong>{row['recsys_score']:.3f}</strong>")
    has_recommend = row.get('has_recommend_keyword', False)
    score_parts.append(f"rec&nbsp;<strong>{'✓' if has_recommend else '✗'}</strong>")
    if score_parts:
        meta_parts.append(f'<span class="gs-scores-inline">{" · ".join(score_parts)}</span>')

    meta_html = '<div class="gs-meta">' + '<span class="gs-dot">·</span>'.join(meta_parts) + '</div>'

    # ── 키워드 필 ──
    pills_html = generate_keyword_pills(keywords_str)
    pills_block = f'<div style="margin: 4px 0 2px;">{pills_html}</div>' if pills_html else ''

    # ── 완성 HTML 출력 (제목 + 메타+점수 + 키워드) ──
    st.markdown(
        f'<div class="gs-paper">{title_html}{meta_html}{pills_block}</div>',
        unsafe_allow_html=True
    )

    # ── Abstract 토글 (키워드 다음, 구분선 전) ──
    with st.expander("Abstract"):
        abstract_text = row.get('Abstract', '')
        rendered = _render_abstract(abstract_text, highlight_query_str)
        st.markdown(f'<div class="abstract-text">{rendered}</div>', unsafe_allow_html=True)

    # ── 구분선 (Abstract 토글 이후) ──
    st.markdown('<hr class="gs-separator">', unsafe_allow_html=True)


def main():
    st.set_page_config(layout="wide", page_title="RecSys Paper Finder", page_icon="🔎")
    st.title("RecSys Paper Finder")
    st.markdown(KEYWORD_CSS, unsafe_allow_html=True)

    model = load_model(MODEL_NAME) 
    
    if model:
        papers_df, paper_embeddings, min_year, max_year, summary_df, bm25 = load_search_database(model)
    else:
        papers_df, paper_embeddings, min_year, max_year, summary_df, bm25 = (None, None, 2000, 2025, None, None)

    if papers_df is None or paper_embeddings is None:
        st.warning("Failed to load essential data. The app cannot start.")
        return
    
    if summary_df is not None and not summary_df.empty:
        # [수정] expanded=True를 expanded=False로 변경하여 기본적으로 닫힘
        with st.expander("View Database Summary", expanded=False):
            st.markdown("#### Paper Counts by Conference and Year")
            
            base = alt.Chart(summary_df).encode(
                x=alt.X('Year', axis=alt.Axis(format='d')),
                y=alt.Y('Paper Count'),
                color='Conference',
                tooltip=['Conference', 'Year', 'Paper Count']
            )
            line = base.mark_line()
            points = base.mark_point(size=100, filled=True)
            
            chart = (line + points) 
            
            st.altair_chart(chart, use_container_width=True)
        
    # --- 검색 모드 선택 ---
    search_type = st.radio(
        "Search Mode:",
        ('bm25', 'exact'),
        format_func=lambda x: {
            'bm25':  '🔍 BM25 (Ranked by Relevance)',
            'exact': '🔎 Exact  (Keyword Filter Only)'
        }[x],
        index=0,
        horizontal=True
    )

    # BM25 쿼리 입력 (Exact 모드에서는 비활성화)
    bm25_query = st.text_input(
        "BM25 Search Query:",
        placeholder="e.g., graph neural network collaborative filtering",
        disabled=(search_type == 'exact'),
        help="Ranks results by BM25 relevance score across Title, Abstract, and Keywords."
    )

    # AND / OR 키워드 필터 (두 모드 공통)
    col_must, col_any = st.columns(2)
    with col_must:
        must_include_query = st.text_input(
            "Must-include keywords (AND)",
            placeholder="e.g., graph cf",
            help="Space-separated. Results MUST contain ALL of these terms in Title / Abstract / Keywords."
        )
    with col_any:
        any_include_query = st.text_input(
            "Include-at-least-one (OR)",
            placeholder="e.g., privacy fairness",
            help="Space-separated. Results MUST contain AT LEAST ONE of these terms in Title / Abstract / Keywords."
        )

    col2, col3 = st.columns(2)
    with col2:
        if min_year < max_year:
            selected_year_range = st.slider(
                "Filter by Year Range:",
                min_value=min_year,
                max_value=max_year,
                value=(min_year, max_year)
            )
        else:
            st.info(f"Data for year {min_year} only.")
            selected_year_range = (min_year, max_year)
    with col3:
        top_k = st.number_input(
            "Max Results to Display",
            min_value=1,
            max_value=5000,
            value=50,
            step=10,
        )

    if st.button("Search"):

        results_df = None
        highlight_query_str = ""

        # --- 1. 연도 필터 ---
        filtered_df = papers_df
        if min_year < max_year:
            start_year, end_year = selected_year_range
            filtered_df = filtered_df[
                (filtered_df['Year_Num'] >= start_year) &
                (filtered_df['Year_Num'] <= end_year)
            ]

        # --- 2. AND / OR 키워드 필터
        #     search_corpus_lower = Title + Abstract + Keywords (References 제외)
        if must_include_query:
            filtered_df = filter_by_keywords(filtered_df, must_include_query, mode='AND')
        if any_include_query:
            filtered_df = filter_by_keywords(filtered_df, any_include_query, mode='OR')

        # --- 3. 모드별 랭킹 ---
        if search_type == 'bm25':
            if not bm25_query:
                st.warning("Please enter a BM25 query to search.")
                st.stop()
            if not filtered_df.empty:
                results_df = bm25_search(bm25, filtered_df, bm25_query, top_k=top_k)
            else:
                results_df = filtered_df
            highlight_query_str = bm25_query

        else:  # exact — AND/OR 필터 결과를 그대로 사용 (추가 랭킹 없음)
            if not must_include_query and not any_include_query:
                st.warning("Exact mode requires at least one keyword filter (AND or OR).")
                st.stop()
            results_df = filtered_df.head(top_k)
            highlight_query_str = (must_include_query + " " + any_include_query).strip()

        # --- 4. 결과 표시 ---
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
                    for i, (_, row) in enumerate(recsys_df.iterrows(), 1):
                        display_paper(row, highlight_query_str, i)

            with tab2:
                if other_df.empty:
                    st.info("No matching other papers found.")
                else:
                    for i, (_, row) in enumerate(other_df.iterrows(), 1):
                        display_paper(row, highlight_query_str, i)


if __name__ == "__main__":
    main()