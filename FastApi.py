from fastapi import FastAPI, HTTPException
from PydanticAgent import explain_business, CodeRequest
app = FastAPI()

@app.post("/analyze-code")
async def analyze_code(request: CodeRequest):
    try:
        return await explain_business(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing the code: {e}")
