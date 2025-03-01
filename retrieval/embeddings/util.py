from retrieval.embeddings.chunking import flatten_codebase, chunk_file
from retrieval.embeddings.qdrant import create_qdrant_collection
from retrieval.embeddings.embeddings import embed_chunk, embed_chunked_codebase
from retrieval.embeddings.pg_comm import get_codebase


def create_qdrant_collection_of_codebase(codebase_id: str) -> str:
    """
    Automates the process of retrieving a codebase and making a qdrant collection of it.

    Args:
        codebase_id: The GUID of the codebase

    Returns:
        str: name of the qdrant collection
    """

    codebase = get_codebase(codebase_id)
    flattened = flatten_codebase(codebase)
    chunked_codebase = []

    for key in flattened:
        chunked_codebase.append(chunk_file(key, flattened[key]))

    embedded_codebase = embed_chunked_codebase(chunked_codebase)
    return create_qdrant_collection(codebase_id, embedded_codebase)