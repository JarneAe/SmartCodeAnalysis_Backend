from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PydanticAgent import BusinessExplanationAgent

app = FastAPI()

business_explanation_agent = BusinessExplanationAgent()

class CodeRequest(BaseModel):
    code_snippet: str
    deps: str = "TruckQueueService.java"

@app.post("/analyze-code")
async def analyze_code(request: CodeRequest):
    try:
        explanation = await business_explanation_agent.analyze(request.code_snippet, request.deps)
        return {"business_explanation": explanation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))