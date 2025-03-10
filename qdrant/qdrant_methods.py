import os
from typing import List

import ollama
from qdrant_client import QdrantClient
import nltk
from nltk.tokenize import sent_tokenize
from qdrant_client.http.models import models, CountResult

from models.ContextRequest import ContextFile

# Download necessary NLTK resources for sentence tokenization
nltk.download('punkt')

# Constants for configuration
COLLECTION_NAME = "TestCollection"
SAVE_DIR = "../markdown_files"
OLLAMA_URI = os.getenv("OLLAMA_URI", "http://localhost:11434")
QDRANT_URI = os.getenv("QDRANT_URI", "http://localhost:6333")
MAX_CHARS = 500

# Initialize clients for Ollama and Qdrant
oclient = ollama.Client(OLLAMA_URI)
qclient = QdrantClient(url=QDRANT_URI)


def chunk_markdown_by_sentences(markdown_text: str, max_chars: int = MAX_CHARS) -> List[str]:
    """
    Splits the input markdown text into smaller chunks based on sentences, ensuring each chunk
    does not exceed the specified character limit.

    Args:
        markdown_text (str): The markdown content to chunk.
        max_chars (int): The maximum number of characters per chunk.

    Returns:
        List[str]: A list of sentence chunks.
    """
    sentences = sent_tokenize(markdown_text, language='dutch')
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chars:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def get_embeddings(text: str) -> List[float]:
    """
    Generates embeddings for the provided text using Ollama's embeddings model.

    Args:
        text (str): The text for which to generate embeddings.

    Returns:
        List[float]: The generated embeddings.
    """
    response = oclient.embeddings(model="nomic-embed-text", prompt=text)
    return response["embedding"]


def upsert_embeddings(texts: List[str], file_name: str, collection_name: str):
    """
    Upserts (inserts or updates) embeddings into a Qdrant collection.

    Args:
        texts (List[str]): The list of text chunks to upsert.
        file_name (str): The name of the file to associate with the texts.
        collection_name (str): The name of the Qdrant collection.
    """
    embeddings_list = [get_embeddings(text) for text in texts]

    # Create collection if it doesn't exist
    if not qclient.collection_exists(collection_name):
        qclient.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=len(embeddings_list[0]), distance=models.Distance.COSINE
            ),
        )

    # Get last ID to continue from
    last_id = qclient.get_collection(collection_name).points_count or 0

    # Prepare points for upsert
    points = [
        models.PointStruct(
            id=i + 1 + last_id,
            vector=embeddings,
            payload={
                "text": text,
                "file_name": file_name,
                "chunk_index": i + 1,
                "char_count": len(text),
            }
        )
        for i, (text, embeddings) in enumerate(zip(texts, embeddings_list))
    ]

    qclient.upsert(collection_name=collection_name, points=points)


def search_similar_text_qdrant(query_text: str, collection_name: str, top_k: int = 5) -> List[dict]:
    """
    Searches for the top K most similar text chunks from a Qdrant collection based on the input query text.

    Args:
        query_text (str): The query text to search for similar chunks.
        collection_name (str): The name of the Qdrant collection.
        top_k (int): The number of results to return.

    Returns:
        List[dict]: A list of dictionaries containing the file name, text, chunk index, and similarity score for the top K results.
    """
    query_embedding = get_embeddings(query_text)

    search_results = qclient.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=top_k,
    )

    results = []
    for result in search_results:
        payload = result.payload
        similarity_score = result.score
        results.append({
            "file_name": payload.get("file_name"),
            "text": payload.get("text"),
            "chunk_index": payload.get("chunk_index"),
            "similarity_score": similarity_score,
        })

    return results


def instantiate_qdrant_and_fill_collection() -> str:
    """
    Instantiates a Qdrant collection and fills it with embeddings from a markdown file.

    Returns:
        str: A success message.
    """
    markdown_file = "../markdown_files/improved_case.md"

    with open(markdown_file, "r", encoding="utf-8") as file:
        markdown_text = file.read()

    chunks = chunk_markdown_by_sentences(markdown_text, max_chars=300)
    upsert_embeddings(chunks, file_name=os.path.basename(markdown_file), collection_name=COLLECTION_NAME)

    return "Qdrant collection filled successfully."


def add_collection(collection_name: str, context_files: List[ContextFile]) -> str:
    """
    Adds multiple context files to an existing Qdrant collection.

    Args:
        collection_name (str): The name of the Qdrant collection.
        context_files (List[ContextFile]): A list of context files to be added.

    Returns:
        str: A success message.
    """
    for file in context_files:
        context_chunks = chunk_markdown_by_sentences(file.content, max_chars=300)
        upsert_embeddings(context_chunks, file_name=file.name, collection_name=collection_name)

    return f"Collection {collection_name} added successfully."


def get_collection_details(collection_name: str) -> CountResult:
    """
    Retrieves details about a specific Qdrant collection.

    Args:
        collection_name (str): The name of the collection to retrieve details for.

    Returns:
        CountResult: The count result for the collection.
    """
    return qclient.count(
        collection_name=collection_name,
        count_filter=models.Filter(
            must=[
                models.FieldCondition(key="color", match=models.MatchValue(value="red")),
            ]
        ),
        exact=True,
    )
