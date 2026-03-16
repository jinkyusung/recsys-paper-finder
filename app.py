import streamlit as st
import pandas as pd
import altair as alt

from update import sync_csv_files, sync_database
from config import APP_CSS
from data_loader import get_db_mtime, load_search_database
from search_engine import bm25_search, filter_by_keywords
from ui_components import display_paper

# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(layout="centered", page_title="RecSys Paper Finder")
    st.markdown('<div class="app-title">RecSys Paper Finder</div>', unsafe_allow_html=True)
    st.markdown(APP_CSS, unsafe_allow_html=True)

    # --- 0. Database Maintenance (Formerly in Sidebar) ---
    with st.expander("Database Sync & Maintenance", expanded=False):
        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown(f"""
                <div style="font-size: 0.85rem; color: #5f6368; margin-bottom: 16px;">
                Update your paper database by parsing current BibTeX files in the `bibtex/` folder.<br>
                <b>Last updated:</b> {get_db_mtime()}
                </div>
            """, unsafe_allow_html=True)
        with c2:
            if st.button("Sync Now", use_container_width=True, type="secondary"):
                with st.spinner("Syncing Database..."):
                    sync_csv_files(force_rebuild=False)
                    sync_database(force_rebuild=False)
                    st.cache_data.clear()
                st.success("Database synced successfully!")
                st.rerun()

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

            st.markdown(f'<div class="section-label">Search Results ({len(results_df)} total)</div>', unsafe_allow_html=True)
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
