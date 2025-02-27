from fastapi import FastAPI, HTTPException
from PydanticAgent import explain_business, CodeRequest
from Qdrant import instantiate_qdrant_and_fill_collection
from database_methods.qdrant_methods import get_collection_details

app = FastAPI(
    title="Smart Code Analysis",
    description=(
        "API to interact with the RAG Business Explanation Agent for code snippets. "
    ),
    version="1.0.0",
    contact={
        "name": "Gang of Three",
        "email": "jarne.aerts@student.kdg.be",
    },
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.post("/analyze-code",
          summary="Analyze a code snippet and explain it using business context",
          tags=["Code Analysis"])
async def analyze_code(request: CodeRequest):
    try:
        return await explain_business(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing the code: {e}")



@app.get("/qdrant/info",tags=["Qdrant"])
async def get_qdrant_info(colection_name: str):
    return get_collection_details(colection_name)


@app.post("/qdrant/instantiate",tags=["Qdrant"])
async def instantiate_qdrant():
    return instantiate_qdrant_and_fill_collection()