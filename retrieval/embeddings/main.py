from retrieval.embeddings.chunking import flatten_codebase, chunk_file
from retrieval.embeddings.create_embeddings import embed_chunks
from retrieval.embeddings.pg_comm import get_codebase

codebase = get_codebase("8b22027e-d4b6-47f4-b23c-0a750944b03d")
flattened = flatten_codebase(codebase)
chunked_codebase = []

for key in flattened:
    chunked_codebase.append(chunk_file(key, flattened[key]))



embedded_codebase = []

for x in chunked_codebase:
    embedded_codebase.append(embed_chunks(x))

print(embedded_codebase)
