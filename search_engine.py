import pandas as pd

def bm25_search(bm25, df, query_str, top_k=None):
    """Rank subset df by BM25 score for query_str, only returning matches."""
    # Split by comma instead of spaces.
    # To keep BM25 working with the word-level index, we flatten the comma-separated parts into words.
    tokens = [w for t in query_str.lower().split(',') if t.strip() for w in t.split()]
    if not tokens:
        # Default: Sort by year descending if no query
        result = df.sort_values('Year_Num', ascending=False)
        return result.head(top_k) if top_k else result

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
