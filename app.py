import streamlit as st
import pandas as pd
import numpy as np
import os
import re
from sentence_transformers import SentenceTransformer, util
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
    .keyword-pill {
        display: inline-block; background-color: #e8e8e8; color: #31333F;
        padding: 4px 10px; margin: 2px 4px 2px 0; border-radius: 16px;
        font-size: 0.85em; font-weight: 600; border: 0px solid #e3d5ca;
    }
    
    /* [추가됨] Expander 레이블(summary) 내부의 p 태그 텍스트 크기 조절 */
    div[data-testid="stExpander"] summary p {
        font-size: 0.9rem !important; /* 기본값(1rem)보다 10% 작게 설정 */
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
    default_return = (None, None, 2000, 2025, None) 
    
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

        # [수정] st.success를 사용하여 완료 메시지 표시
        st.success(f"Successfully loaded {len(df)} papers and their embeddings.")
        return df, embeddings, min_year, max_year, summary_df 
        
    except Exception as e:
        st.error(f"Error: Failed to load database files. {e}")
        st.info("Please try rebuilding with 'python update.py --force'.")
        return default_return

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
        key = k.strip()
        if key:
            html_spans.append(f'<span class="keyword-pill">{key}</span>')
    return ' '.join(html_spans)

# [수정됨] 'Similarity' 열 이름을 사용하도록 유지
# [수정됨] display_paper 함수
def display_paper(row, highlight_query_str, index):
    with st.container(border=True):
        
        # --- 1. 논문명 ---
        # 제목을 더 눈에 띄게 하기 위해 '######' (h6)로 변경합니다.
        title = row.get('Title', 'No Title')
        st.markdown(f"##### {index}. {title}")

        # --- 2. 학회명 (년도), URL ---
        # 학회, 연도, URL을 한 줄에 결합하여 표시합니다.
        url = row.get('url', '')
        conf = row.get('Conference Name (Book Title)', 'N/A')
        year = row.get('Year', 'N/A')
        
        url_display = f"[{url}]({url})" if pd.notna(url) and url else "No URL Provided"
        st.markdown(f"_{conf} ({year})_ &emsp; _{url_display}_")

        # --- 3. 저자 ---
        # 저자 정보는 caption을 사용하여 부가 정보임을 나타냅니다.
        author_str = row.get('Author', '')
        if pd.notna(author_str) and author_str:
            st.caption(f"{author_str}") 

        # --- 4. 판단 기준 점수 ---
        # 모든 점수 관련 정보를 한 줄의 caption으로 묶어 표시합니다.
        score_parts = []
        
        # Semantic 검색 시 'Similarity' 점수
        if 'Similarity' in row and pd.notna(row['Similarity']):
            score_parts.append(f"Semantic Relevance: {row['Similarity']:.4f}")
        
        # RecSys 관련성 점수
        if 'recsys_score' in row and pd.notna(row['recsys_score']):
            score_parts.append(f"RecSys Relevance: {row['recsys_score']:.3f}")
        
        # 'recommend' 키워드 포함 여부
        has_recommend = row.get('has_recommend_keyword', False)
        score_parts.append(f"Contains 'recommend': {'Yes' if has_recommend else 'No'}")

        if score_parts:
            st.caption("  |  ".join(score_parts))

        # (선택 사항) 키워드 표시는 메타데이터와 Abstract 사이에 두는 것이
        # 논문 파악에 유용하므로 유지하는 것을 권장합니다.
        keywords_str = row.get('Keywords', '')
        pills_html = generate_keyword_pills(keywords_str)
        if pills_html:
            st.markdown(pills_html, unsafe_allow_html=True)

        # --- 5. Abstract ---
        # Abstract는 expander 내부에 두어 UI를 깔끔하게 유지합니다.
        with st.expander("View Abstract"):
            abstract_text = row.get('Abstract', 'No abstract provided')
            if highlight_query_str and abstract_text:
                try:
                    terms_to_highlight = highlight_query_str.lower().split()
                    terms_regex = '|'.join(re.escape(term) for term in terms_to_highlight if term)
                    
                    if terms_regex:
                        pattern = re.compile(terms_regex, re.IGNORECASE)
                        highlighted_text = pattern.sub(r'<mark>\g<0></mark>', abstract_text)
                        st.markdown(highlighted_text, unsafe_allow_html=True)
                    else:
                        st.write(abstract_text)
                except Exception:
                    st.write(abstract_text) 
            else:
                st.write(abstract_text)
        

def main():
    st.set_page_config(layout="centered", page_title="RecSys Paper Finder", page_icon="🔎")
    st.title("RecSys Paper Finder")
    st.markdown(KEYWORD_CSS, unsafe_allow_html=True)

    model = load_model(MODEL_NAME) 
    
    if model:
        # load_search_database 함수가 st.text() 및 st.success()를 직접 호출
        papers_df, paper_embeddings, min_year, max_year, summary_df = load_search_database(model)
    else:
        papers_df, paper_embeddings, min_year, max_year, summary_df = (None, None, 2000, 2025, None)

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
            # 4. Exact: 필터링된 결과 전체를 사용하고, 상위 K개만 표시
            results_df = filtered_df.head(top_k) 
            
            highlight_query_str = must_include_query + " " + any_include_query 

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