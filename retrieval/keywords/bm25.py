from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from nltk import download

from retrieval.util import get_chunked_codebase

download('punkt_tab')

def get_top_n_chunks(chunks: list[str], scores: list[float], n: int) -> list[str]:
    top_chunks = []
    for i in range(n):
        max_score = max(scores)
        if max_score != 0:
            index = scores.index(max_score)
            scores.pop(index)
            top_chunk = chunks.pop(index)
            top_chunks.append(top_chunk)

    return top_chunks

def get_scores_of_chunks(chunks: list[str], query: str) -> list[float]:
    tokenized_docs = [word_tokenize(doc) for doc in chunks]
    bm25 = BM25Okapi(tokenized_docs)

    query_tokens = word_tokenize(query)

    return bm25.get_scores(query_tokens).tolist()