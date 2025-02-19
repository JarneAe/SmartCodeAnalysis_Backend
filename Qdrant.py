import ollama
from qdrant_client import QdrantClient
from qdrant_client.http.models import models
from qdrant_client.models import Distance, VectorParams

COLLECTION_NAME = "TestCollection"
qclient = QdrantClient(url="http://localhost:6333")
oclient = ollama.Client("localhost")


def get_embeddings(text):
    response = oclient.embeddings(model="nomic-embed-text", prompt=text)
    return response["embedding"]


def upsert_embeddings(texts):
    embeddings_list = [get_embeddings(text) for text in texts]

    if not qclient.collection_exists(COLLECTION_NAME):
        qclient.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=len(embeddings_list[0]), distance=models.Distance.COSINE
            ),
        )

    points = [
        models.PointStruct(id=i + 1, vector=embeddings, payload={"text": text, "id": i + 1})
        for i, (text, embeddings) in enumerate(zip(texts, embeddings_list))
    ]

    qclient.upsert(collection_name=COLLECTION_NAME, points=points)


texts = ["Goodbye, world!", "Goodbye, universe!"]
upsert_embeddings(texts)
print("Embeddings upserted successfully.")