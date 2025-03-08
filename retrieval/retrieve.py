from retrieval.embeddings import create_qdrant_collection_of_codebase
from retrieval.embeddings.qdrant import query_qdrant_collection
from retrieval.keywords import keyword_search
from retrieval.reranking import relevance_rank
from pydantic import BaseModel

class RetrievalRequest(BaseModel):
    codebase_id: str
    query: str
    n: int

def retrieve(request: RetrievalRequest):
    if request.n > 5: request.n = 5 # Limit amount of returned docs

    # Embeddings
    qdrant_collection = create_qdrant_collection_of_codebase(request.codebase_id)
    embeddings_docs = query_qdrant_collection(qdrant_collection, request.query)

    # BM25 keyword search
    keyword_docs = keyword_search(request.codebase_id, request.query)

    # Reranking
    all_docs = keyword_docs + embeddings_docs

    return relevance_rank(request.query, all_docs, request.n)
