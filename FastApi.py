from fastapi import FastAPI, HTTPException
from PydanticAgent import explain_business, CodeRequest
from retrieval import RetrievalRequest, retrieve

app = FastAPI()

@app.post("/analyze-code")
async def analyze_code(request: CodeRequest):
    try:
        return await explain_business(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing the code: {e}")

@app.post("/retrieve-code")
async def retrieve_code(request: RetrievalRequest):
    try:
        docs = retrieve(request)
        return docs
    except ValueError:
        raise HTTPException(status_code=404, detail="Codebase not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}")
