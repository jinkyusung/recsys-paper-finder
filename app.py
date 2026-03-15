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
    /* Content area max-width: half of 1920px (16:9 full screen) */
    .main > div {
        max-width: 960px;
        margin: 0 auto;
    }
    /* Paper Title */
    .paper-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #1a1a1a;
        display: block;
        margin-bottom: 4px;
        line-height: 1.35;
    }
    /* Authors */
    .paper-authors {
        font-size: 0.88rem;
        color: #2E7D32;
        margin-bottom: 3px;
        line-height: 1.4;
    }
    /* Meta info like conference and year */
    .paper-meta {
        font-size: 0.85rem;
        color: #555555;
        margin-bottom: 5px;
    }
    /* Scores */
    .paper-scores {
        font-size: 0.82rem;
        color: #777777;
        margin-bottom: 8px;
    }
    /* Keyword Pills */
    .keyword-pill {
        display: inline-block; 
        background-color: #F0F2F6; 
        color: #31333F;
        padding: 3px 9px; 
        margin: 2px 5px 5px 0; 
        border-radius: 12px;
        font-size: 0.78rem; 
        font-weight: 500;
        border: 1px solid #E0E0E0;
    }
    /* Expander adjustment */
    div[data-testid="stExpander"] {
        border-color: #E0E0E0;
        border-radius: 8px;
    }
    div[data-testid="stExpander"] summary p {
        font-size: 0.95rem !important;
        font-weight: 600;
        color: #444444;
    }
    /* Abstract Text */
    .abstract-text {
        font-size: 0.92rem;
        line-height: 1.6;
        color: #333333;
        text-align: justify;
    }
    /* Highlighting */
    mark {
        background-color: #FFF3CD;
        color: #856404;
        font-weight: bold;
        padding: 0 3px;
        border-radius: 3px;
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

# [수정됨] 'Similarity' 열 이름을 사용하도록 유지
# [수정됨] display_paper 함수 - UI 개선
def display_paper(row, highlight_query_str, index):
    with st.container(border=True):
        title = row.get('Title', 'No Title')
        url = row.get('url', '')
        conf = row.get('Conference Name (Book Title)', 'N/A')
        year = row.get('Year', 'N/A')
        author_str = row.get('Author', '')
        keywords_str = row.get('Keywords', '')

        # 제목은 일반 텍스트 (링크 없음, 크기 다운)
        title_html = f'<span class="paper-title">{index}. {title}</span>'

        author_html = f'<div class="paper-authors">{author_str}</div>' if pd.notna(author_str) and author_str else ''

        meta_html = f'<div class="paper-meta"><strong>{conf}</strong> ({year})</div>'

        score_parts = []
        if 'Similarity' in row and pd.notna(row['Similarity']):
            score_parts.append(f"Semantic Relevance: <strong>{row['Similarity']:.4f}</strong>")
        if 'BM25_Score' in row and pd.notna(row['BM25_Score']):
            score_parts.append(f"BM25 Score: <strong>{row['BM25_Score']:.3f}</strong>")
        if 'recsys_score' in row and pd.notna(row['recsys_score']):
            score_parts.append(f"RecSys Relevance: <strong>{row['recsys_score']:.3f}</strong>")
        has_recommend = row.get('has_recommend_keyword', False)
        score_parts.append(f"Has 'recommend': <strong>{'Yes' if has_recommend else 'No'}</strong>")

        scores_html = f'<div class="paper-scores">{" | ".join(score_parts)}</div>' if score_parts else ''

        pills_html = generate_keyword_pills(keywords_str)
        if pills_html:
            pills_html = f'<div style="margin-bottom: 8px;">{pills_html}</div>'

        header_html = f"""
        <div style="margin-bottom: 4px;">
            {title_html}
            {author_html}
            {meta_html}
            {scores_html}
            {pills_html}
        </div>
        """
        st.markdown(header_html, unsafe_allow_html=True)

        # URL 링크는 별도 버튼으로 분리
        btn_col, exp_col = st.columns([1, 4])
        with btn_col:
            if pd.notna(url) and url:
                st.link_button("🔗 Open Paper", url)
        with exp_col:
            pass  # 여백 확보

        with st.expander("Abstract & Details"):
            abstract_text = row.get('Abstract', 'No abstract provided')
            if highlight_query_str and abstract_text:
                try:
                    terms_to_highlight = highlight_query_str.lower().split()
                    terms_regex = '|'.join(re.escape(term) for term in terms_to_highlight if term)
                    
                    if terms_regex:
                        pattern = re.compile(terms_regex, re.IGNORECASE)
                        # highlight using standard mark tag
                        highlighted_text = pattern.sub(r'<mark>\g<0></mark>', abstract_text)
                        st.markdown(f'<div class="abstract-text">{highlighted_text}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="abstract-text">{abstract_text}</div>', unsafe_allow_html=True)
                except Exception:
                    st.markdown(f'<div class="abstract-text">{abstract_text}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="abstract-text">{abstract_text}</div>', unsafe_allow_html=True)


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
        
    search_type = st.radio(
        "Search Mode:",
        ('semantic', 'exact'), 
        format_func=lambda x: {'semantic': 'Semantic (Primary Query + Filters)', 'exact': 'Exact (Filters Only)'}[x],
        index=0,
        horizontal=True 
    )

    query = st.text_input(
        "Enter Semantic query (ignored in Exact mode):", 
        placeholder="e.g., Graph Neural Network",
        disabled=(search_type == 'exact')
    )
    
    col_must, col_any = st.columns(2)
    with col_must:
        must_include_query = st.text_input(
            "Must-include keywords (AND)",
            placeholder="e.g., graph cf",
            help="Space-separated. Results MUST contain ALL of these terms."
        )
    with col_any:
        any_include_query = st.text_input(
            "Include-at-least-one (OR)",
            placeholder="e.g., privacy fairness",
            help="Space-separated. Results MUST contain AT LEAST ONE of these terms."
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
        
        # --- 공통 필터링 단계 (연도, AND, OR) ---
        # 1. 연도 필터 적용
        filtered_df = papers_df 
        if min_year < max_year:
            start_year, end_year = selected_year_range
            filtered_df = filtered_df[
                (filtered_df['Year_Num'] >= start_year) &
                (filtered_df['Year_Num'] <= end_year)
            ]
        
        # 2. Must-Include (AND) 필터 적용
        if must_include_query:
            filtered_df = filter_by_keywords(filtered_df, must_include_query, mode='AND')

        # 3. Include-at-Least-One (OR) 필터 적용
        if any_include_query:
            filtered_df = filter_by_keywords(filtered_df, any_include_query, mode='OR')

        # --- 모드별 최종 결과 생성 ---
        if search_type == 'semantic':
            if not query:
                st.warning("Please enter a Semantic query to start.")
                st.stop()
            
            # 4. Semantic: 필터링된 결과 내에서 유사도 계산 및 상위 K개 선택
            if not filtered_df.empty:
                filtered_indices = filtered_df.index
                filtered_embeddings = paper_embeddings[filtered_indices]
                
                query_embedding = model.encode(query, normalize_embeddings=True)
                cos_scores = util.dot_score(query_embedding, filtered_embeddings)[0].numpy()
                
                filtered_df['Similarity'] = cos_scores
                
                results_df = filtered_df.sort_values(by='Similarity', ascending=False).head(top_k)
            else:
                results_df = filtered_df 

            highlight_query_str = query 
            
        else: # search_type == 'exact'
            # 4. Exact: AND/OR 필터 후 BM25로 랭킹
            bm25_query = (must_include_query + " " + any_include_query).strip()
            if bm25_query and bm25 is not None and not filtered_df.empty:
                results_df = bm25_search(bm25, filtered_df, bm25_query, top_k=top_k)
            else:
                results_df = filtered_df.head(top_k)

            highlight_query_str = bm25_query

        # --- 결과 표시 ---
            
        if results_df is None or results_df.empty:
             st.info("No papers found matching all criteria.")
        else:
            recsys_df = results_df[results_df['is_recsys']]
            other_df = results_df[~results_df['is_recsys']]
            
            st.subheader(f"Search Results ({len(results_df)} total)") 

            tab1, tab2 = st.tabs([
                f"Recommender System Papers ({len(recsys_df)})", 
                f"Other Papers ({len(other_df)})"
            ])

            # --- Recommender System Papers 탭 ---
            with tab1:
                if recsys_df.empty:
                    st.info("No matching recommender system papers found.")
                else:
                    if search_type == 'semantic':
                         recsys_df = recsys_df.sort_values(by='recsys_score', ascending=False)
                    
                    for i, (_, row) in enumerate(recsys_df.iterrows(), 1):
                        display_paper(row, highlight_query_str, i)

            # --- Other Papers 탭 ---
            with tab2:
                if other_df.empty:
                    st.info("No matching other papers found.")
                else:
                    for i, (_, row) in enumerate(other_df.iterrows(), 1):
                        display_paper(row, highlight_query_str, i)


if __name__ == "__main__":
    main()