from fastapi import FastAPI, HTTPException, Query

from ExplainAgent import explain_business, CodeRequest
from Models.ContextRequest import ContextRequest
from Qdrant import instantiate_qdrant_and_fill_collection, search_similar_text_qdrant, add_collection
from typing import Dict, Any, List
from fastapi.responses import RedirectResponse
from retrieval import RetrievalRequest, retrieve
from chatbot_methods import ask_question

app = FastAPI(
    title="Smart Code Analysis",
    description="API to interact with the RAG Business Explanation Agent for code snippets.",
    version="1.0.0",
    contact={
        "name": "Gang of Three",
        "email": "jarne.aerts@student.kdg.be",
    },
    docs_url="/docs",
    redoc_url="/redoc",
)

app = FastAPI()


@app.post(
    "/analyze-code",
    summary="Analyze a code snippet and explain it using business context",
    tags=["Code Analysis"],
    response_model=Dict[str, Any],
)
async def analyze_code(request: CodeRequest):
    """
    Analyzes a given code snippet and provides a business explanation.

    Parameters:
    - **request**: Contains the code snippet and user role.

    Returns:
    - JSON response containing the explanation.
    """
    try:
        return await explain_business(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing the code: {str(e)}")


@app.post(
    "/chat",
    summary="Analyze a code snippet and explain it using business context",
    tags=["Chat"],
    response_model=Dict[str, Any],
)
async def analyze_code(question: str):
    """
    Analyzes a given code snippet and provides a business explanation.

    Parameters:
    - **request**: Contains the code snippet and user role.

    Returns:
    - JSON response containing the explanation.
    """
    try:
        return await ask_question(question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error Responding {str(e)}")


@app.post("/qdrant/instantiate", tags=["Qdrant"], response_model=Dict[str, Any])
def instantiate_qdrant():
    """
    Instantiates Qdrant and fills the collection with embeddings.

    Returns:
    - JSON response indicating success or failure.
    """
    try:
        return {"message": instantiate_qdrant_and_fill_collection()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error initializing Qdrant: {str(e)}")


@app.post("/qdrant/add_collection", tags=["Qdrant"], response_model=Dict[str, Any])
def add_qdrant_collection(request: ContextRequest):
    try:
        return {"message": add_collection(request.collection_name, request.context_files)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding collection to Qdrant: {str(e)}")


@app.get("/qdrant/search_similar", tags=["Qdrant"], response_model=List[Dict[str, Any]])
def search_similar_text(
    query_text: str = Query(..., min_length=3, description="The text to search for similar embeddings in Qdrant"),
    collection_name: str = Query("TestCollection", description="The name of the collection to search in"),
):
    """
    Searches for similar text embeddings in Qdrant.

    Parameters:
    - query_text (str): The text query for similarity search.

    Returns:
    - JSON response containing similar text results.
    """
    try:
        return search_similar_text_qdrant(query_text, collection_name=collection_name)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid search query: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error performing search: {str(e)}")


@app.get("/qdrant/search_similar", tags=["Qdrant"], response_model=List[Dict[str, Any]])
def search_similar_text(
    query_text: str = Query(..., min_length=3, description="The text to search for similar embeddings in Qdrant")
):
    """
    Searches for similar text embeddings in Qdrant.

    Parameters:
    - query_text (str): The text query for similarity search.

    Returns:
    - JSON response containing similar text results.
    """
    try:
        return search_similar_text_qdrant(query_text)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid search query: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error performing search: {str(e)}")

@app.get("/", include_in_schema=False)
def docs_redirect():
    return RedirectResponse(url='/docs')

@app.post("/retrieve-code")
async def retrieve_code(request: RetrievalRequest):
    try:
        docs = retrieve(request)
        return docs
    except ValueError:
        raise HTTPException(status_code=404, detail="Codebase not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}")
