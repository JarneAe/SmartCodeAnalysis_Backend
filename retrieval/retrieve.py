from retrieval.embeddings import create_qdrant_collection_of_codebase
from retrieval.embeddings.qdrant import query_qdrant_collection
from retrieval.keywords import keyword_search
from retrieval.reranking import relevance_rank


def retrieve(codebase_id: str, query: str, n: int):
    if n > 5: n = 5 # Limit amount of returned docs

    # Embeddings
    qdrant_collection = create_qdrant_collection_of_codebase(codebase_id)
    embeddings_docs = query_qdrant_collection(qdrant_collection, query)

    # BM25 keyword search
    keyword_docs = keyword_search(codebase_id, query)

    # Reranking
    all_docs = keyword_docs + embeddings_docs

    return relevance_rank(query, all_docs, n)
