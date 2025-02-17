from fastapi import FastAPI
from PydanticAgent import explain_business, CodeRequest
app = FastAPI()

@app.post("/analyze-code")
async def analyze_code(request: CodeRequest):
    return await explain_business(request)
