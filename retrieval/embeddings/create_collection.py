from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
import ollama

qdrantClient = QdrantClient(host="localhost", port=6333)
ollamaClient = ollama.Client("localhost")

qdrantClient.create_collection(
    collection_name="code_1"
)
