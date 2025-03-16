from fastapi import FastAPI, HTTPException, Query
from agents.ExplainAgent import explain_business
from agents.TestGenerationAgent import generate_tests_agent
from models.ChatRequest import ChatRequest
from models.CodeRequest import CodeRequest
from models.CodeTestGenerationRequest import CodeTestGenerationRequest
from models.ContextRequest import ContextRequest
from qdrant.qdrant_methods import instantiate_qdrant_and_fill_collection, search_similar_text_qdrant, add_collection
from typing import Dict, Any, List
from fastapi.responses import RedirectResponse
from chatbot.chatbot_methods import ask_question

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


@app.post("/generate_tests", tags=["Test Generation"], response_model=Dict[str, Any])
async def generate_tests(request: CodeTestGenerationRequest):
    """
    Generates test cases for a given code snippet.

    Parameters:
    - **request**: Contains the code snippet, user role, and test framework.

    Returns:
    - JSON response containing the generated test cases.
    """
    try:
        return await generate_tests_agent(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating tests: {str(e)}")



@app.post(
    "/chat",
    summary="Respond to a chat message with a business explanation",
    tags=["Chat"],
    response_model=Dict[str, Any],
)
async def analyze_code(request: ChatRequest):
    """
    Analyzes a given code snippet and provides a business explanation.

    Parameters:
    - **request**: Contains the code snippet and user role.

    Returns:
    - JSON response containing the explanation.
    """
    try:
        return await ask_question(request)
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