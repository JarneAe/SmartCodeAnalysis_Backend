import os
import ollama
from qdrant_client import QdrantClient
from qdrant_client.http.models import models
from qdrant_client.models import Distance, VectorParams
from PDFConvertor import PDFConvertor
import nltk
from nltk.tokenize import sent_tokenize

# Download necessary NLTK data
nltk.download('punkt')

# Constants
COLLECTION_NAME = "TestCollection"
SAVE_DIR = "markdown_files"

# Initialize clients
qclient = QdrantClient(url="http://localhost:6333")
oclient = ollama.Client("localhost")


def chunk_markdown_by_sentences(markdown_text, max_chars=300):
    """
    Split the text into smaller chunks by sentences with a max character limit.
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


def get_embeddings(text):
    response = oclient.embeddings(model="nomic-embed-text", prompt=text)
    return response["embedding"]


def upsert_embeddings(texts, file_name):
    embeddings_list = [get_embeddings(text) for text in texts]

    if not qclient.collection_exists(COLLECTION_NAME):
        qclient.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=len(embeddings_list[0]), distance=models.Distance.COSINE
            ),
        )

    points = [
        models.PointStruct(
            id=i + 1,
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

    qclient.upsert(collection_name=COLLECTION_NAME, points=points)


def search_similar_text(query_text, top_k=5):
    """
    Embed the input text and return the top_k most similar results from Qdrant.
    """
    query_embedding = get_embeddings(query_text)

    search_results = qclient.search(
        collection_name=COLLECTION_NAME,
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


pdf_file = "files/improved_case.pdf"
converter = PDFConvertor(pdf_file)
markdown_text = converter.convert()

chunks = chunk_markdown_by_sentences(markdown_text, max_chars=300)
upsert_embeddings(chunks, file_name=os.path.basename(pdf_file))

print(f"Processed {len(chunks)} fine-grained chunks from {pdf_file} and upserted successfully.")
print(f"Markdown saved in directory: {SAVE_DIR}")

# Search query
query = """
public Long getWarehouseIdByLicensePlate(String licensePlate) {
    try {
        Pair<Long, String> data = appointmentRetrievalService.getClientIdAndMineralByLicensePlate(licensePlate);
        if (data == null) {
            throw new NoSuchElementException("No data found for license plate: " + licensePlate);
        }

        Long sellerId = data.getFirst();
        String mineralName = data.getSecond();

        log.info("Fetching Warehouse id for seller with id {} and mineral {}", sellerId, mineralName);

        String url = String.format(warehouseApiUrl, sellerId, mineralName);

        ResponseEntity<Long> response = restTemplate.getForEntity(url, Long.class);

        if (response.getStatusCode().is2xxSuccessful()) {
            Long warehouseId = response.getBody();
            log.info("Warehouse id retrieved: {}", warehouseId);
            return warehouseId;
        } else {
            log.error("Failed to retrieve warehouse id. Status code: {}", response.getStatusCode());
            throw new CouldNotRetrieveWarehouseException("Failed to retrieve warehouse id. Status code: " + response.getStatusCode());
        }
    } catch (NoSuchElementException e) {
        log.error("Error retrieving warehouse id for license plate: {}", licensePlate, e);
        throw new CouldNotRetrieveWarehouseException("Error retrieving warehouse id for license plate: " + licensePlate, e);
    }
}
"""

results = search_similar_text(query)

# Print search results
for i, result in enumerate(results, 1):
    print(f"Result {i}:")
    print(f"File: {result['file_name']}")
    print(f"Chunk Index: {result['chunk_index']}")
    print(f"Similarity Score: {result['similarity_score']:.4f}")
    print(f"Text: {result['text']}\n")
