from fastapi import FastAPI, HTTPException, Query

from FormatCodeAgent import format_code
from ExplainAgent import explain_business, CodeRequest
from Qdrant import instantiate_qdrant_and_fill_collection, search_similar_text_qdrant
from typing import Dict, Any, List
from fastapi.responses import RedirectResponse

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
    response_model=Dict[str, Any],  # Adjust this to match actual return structure
)
async def analyze_code(request: CodeRequest):
    """
    Analyzes a given code snippet and provides a business explanation.

    Parameters:
    - request (CodeRequest): Contains the code snippet to be analyzed.

    Returns:
    - JSON response containing the explanation.
    """
    try:
        return await explain_business(request)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid input: {ve}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing the code: {str(e)}")


@app.get("/full-chain/{code_snippet}", tags=["Code Analysis"], response_model=Dict[str, Any])
async def full_chain(
        query_text: str = Query(..., min_length=3, description="The text to search for similar embeddings in Qdrant")
):
    """
    Analyzes a given code snippet and provides a business explanation.

    Parameters:
    - code_snippet (str): The code snippet to be analyzed.

    Returns:
    - JSON response containing the explanation.
    """
    try:
        formatted_code = await format_code(query_text)
        rag_results = search_similar_text_qdrant(formatted_code)

        # Extract RAG text into a single formatted string
        rag_context = "\n\n".join([f"- {item['text']}" for item in rag_results])

        explanation = await explain_business(CodeRequest(code_snippet=query_text), formatted_code, rag_context)

        return {
            "formatted_code": formatted_code,
            "RAG": rag_results,
            "explanation": explanation["explanation"]
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid input: {ve}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing the code: {str(e)}")

@app.post("/qdrant/instantiate", tags=["Qdrant"], response_model=Dict[str, Any])
async def instantiate_qdrant():
    """
    Instantiates Qdrant and fills the collection with embeddings.

    Returns:
    - JSON response indicating success or failure.
    """
    try:
        return instantiate_qdrant_and_fill_collection()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error initializing Qdrant: {str(e)}")


@app.get("/qdrant/search_similar", tags=["Qdrant"], response_model=List[Dict[str, Any]])
async def search_similar_text(
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
        return search_similar_text_qdrant(query_text)  # This already returns a list
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid search query: {ve}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error performing search: {str(e)}")

@app.get("/", include_in_schema=False)
async def docs_redirect():
    return RedirectResponse(url='/docs')
