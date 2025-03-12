from retrieval.chunking import flatten_codebase, chunk_file
from retrieval.pg_comm import get_codebase


def get_chunked_codebase(codebase_id: str) -> list[list[dict[str, int | str]]]:
    codebase = get_codebase(codebase_id)
    flattened = flatten_codebase(codebase)
    chunked_codebase = []

    for key in flattened:
        chunked_codebase.append(chunk_file(key, flattened[key]))

    return chunked_codebase