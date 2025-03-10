from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import torch

model_name = "microsoft/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def embed_chunk(chunk: dict[str, int | str]) -> Tensor:
    chunked_text = chunk["text"]
    inputs = tokenizer(chunked_text, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        embeddings: Tensor = model(**inputs).last_hidden_state.mean(dim=1)
        return embeddings


def embed_chunked_codebase(chunked_codebase: list[list[dict[str, int | str]]]) -> list[dict[str, int | str | Tensor]]:
    embedded_codebase = []
    for chunked_file in chunked_codebase:
        for chunk in chunked_file:
            embedding = embed_chunk(chunk)
            embedded_codebase.append(
                {
                    "file_name": chunk["file_name"],
                    "chunk_index": chunk["chunk_index"],
                    "content": chunk["text"],
                    "embedding": embedding
                }
            )

    return embedded_codebase

def embed_query(query: str) -> Tensor:
    inputs = tokenizer(query, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        embeddings: Tensor = model(**inputs).last_hidden_state.mean(dim=1)
        return embeddings