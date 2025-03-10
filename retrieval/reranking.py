from transformers import AutoModel, AutoTokenizer

model_name = "microsoft/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


def relevance_rank(query: str, docs: list[str], n: int = 0) -> list[str]:
    """
    Ranks the provided docs on relevance based on the query provided and returns the top n results.
    Args:
        query: The query the user provided
        docs: The code snippets the retrievers returned
        n: The amount of snippets to return, 0 to return all

    Returns:
        list[str]: The top n snippets.

    """
    inputs = tokenizer([f"{query} {doc}" for doc in docs], return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs).pooler_output  # Get relevance scores

    # Rank documents
    ranked_docs = [doc for _, doc in sorted(zip(outputs[:, 0].tolist(), docs), reverse=True)]

    if n == 0: return ranked_docs
    else: return ranked_docs[0:n]