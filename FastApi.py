from fastapi import FastAPI, HTTPException
from PydanticAgent import explain_business, CodeRequest
from Qdrant import instantiate_qdrant_and_fill_collection, search_similar_text_qdrant

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



@app.post("/qdrant/instantiate",tags=["Qdrant"])
async def instantiate_qdrant():
    return instantiate_qdrant_and_fill_collection()

@app.get("/qdrant/search_similar",tags=["Qdrant"])
async def search_similar_text(query_text: str):
    return search_similar_text_qdrant(query_text)