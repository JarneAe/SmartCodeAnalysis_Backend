from retrieval.embeddings.qdrant import create_qdrant_collection
from retrieval.embeddings.embeddings import embed_chunked_codebase
from retrieval.util import get_chunked_codebase


def create_qdrant_collection_of_codebase(codebase_id: str) -> str:
    """
    Automates the process of retrieving a codebase and making a qdrant collection of it.

    Args:
        codebase_id: The GUID of the codebase

    Returns:
        str: name of the qdrant collection
    """

    chunked_codebase = get_chunked_codebase(codebase_id)
    embedded_codebase = embed_chunked_codebase(chunked_codebase)
    return create_qdrant_collection(codebase_id, embedded_codebase)