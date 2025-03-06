from retrieval.keywords.bm25 import get_scores_of_chunks, get_top_n_chunks
from retrieval.util import get_chunked_codebase


def keyword_search(codebase_id: str, query: str, n: int = 3) -> list[str]:
    chunked_codebase = get_chunked_codebase(codebase_id)

    chunks = [chunk['text'] for file in chunked_codebase for chunk in file]
    scores = get_scores_of_chunks(chunks, query)

    return get_top_n_chunks(chunks, scores, n)