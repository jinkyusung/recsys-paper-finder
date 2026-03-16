import os
import pandas as pd
import streamlit as st
from rank_bm25 import BM25Okapi
from config import DB_FILE, RECSYS_MASTER_REGEX

def get_db_mtime():
    if os.path.exists(DB_FILE):
        import datetime
        mtime = os.path.getmtime(DB_FILE)
        return datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
    return "Never"

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
