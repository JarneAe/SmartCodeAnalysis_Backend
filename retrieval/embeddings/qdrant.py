from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import VectorParams, Distance
import ollama
from qdrant_client.models import PointStruct
from torch import Tensor

from retrieval.embeddings.embeddings import embed_query

qdrantClient = QdrantClient(host="localhost", port=6333)
ollamaClient = ollama.Client("localhost")


def create_qdrant_collection(codebase_id: str, embedded_codebase: list[dict[str, int | str | Tensor]]) -> str:
    collection_name = "codebase_" + codebase_id
    qdrantClient.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=len(embedded_codebase[0]["embedding"][0]),
                                    distance=Distance.COSINE)
    )

    points = []

    for index, item in enumerate(embedded_codebase):
        embedding = item["embedding"].squeeze(0).numpy()
        metadata = {key: value for key, value in item.items() if key != "embedding"}
        point = PointStruct(id=index, vector=embedding.tolist(), payload=metadata)
        points.append(point)

    qdrantClient.upsert(
        collection_name=collection_name,
        points=points
    )

    return collection_name


def query_qdrant_collection(collection_name: str, query: str, n: int = 3) -> [str]:
    embedded_query = embed_query(query).squeeze(0).tolist()
    search_results = qdrantClient.query_points(
        collection_name=collection_name,
        query=embedded_query,
        limit=n
    )

    return [point.payload['content'] for point in search_results.points]

def codebase_has_collection(codebase_id: str) -> bool:
    try:
        qdrantClient.get_collection("codebase_"+codebase_id)
        return True
    except UnexpectedResponse as e:
        if "404" in str(e):
            return False
        else:
            raise