import asyncio
import os

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
import logfire
from PDFConvertor import PDFConvertor
from Models.ResponseTemplate import ResponseTemplate

OLLAMA_URI = os.getenv("OLLAMA_URI", "http://localhost:11434/v1")

logfire.configure()
logfire.instrument_httpx(capture_all=True)

pdf_convertor = PDFConvertor(file_path="files/SA 24-25 - Mineral Flow-1.pdf")
business_context = pdf_convertor.convert()

ollama_model = OpenAIModel(
    model_name='qwen2.5:7b',
    base_url='http://localhost:11434/v1',
    api_key='ollama',
)

business_explanation_agent = Agent(
    ollama_model,
    deps_type=str,
    result_type=ResponseTemplate,
    retries=10
)

@business_explanation_agent.system_prompt
def add_business_context() -> str:
    return (
        "You are a business analyst specializing in translating technical implementations into real-world business value. "
        "Your task is to explain what the provided code accomplishes in terms of business operations, without using technical jargon.\n\n"
        "### Primary Task:\n"
        "Analyze the provided code and explain its purpose in the context of real-world business operations. Focus on:\n"
        "- What business process the code supports\n"
        "- How it improves operations\n"
        "- Who benefits and how\n"
        "- Why this process matters to the business\n\n"
        "### Business Context:\n"
        f"{business_context}\n\n"
        "### Response Structure:\n"
        "1. **Practical use case:** Describe the specific business scenario where this code is used\n"
        "2. **Business impact:** Explain how this affects operational efficiency, costs, or revenue\n"
        "3. **User impact:** Describe how this affects the daily work of employees or customers\n"
        "4. **Why is this important:** Explain the strategic value to the business\n"
        "5. **Day in the life scenario:** Provide a concrete example of how this is used in daily operations\n\n"
        "Put the main focus on the business context that was given and ensure that the explanation is clear and easy to understand for non-technical stakeholders.\n\n"
        "### Critical Constraints:\n"
        "Focus exclusively on:\n"
        "- Real-world business processes\n"
        "- Operational workflows\n"
        "- User experiences\n"
        "- Organizational outcomes\n\n"
        "Absolute Prohibitions:\n"
        "× Technical terms (APIs, repositories, methods)\n"
        "× Code structure discussions\n"
        "× System architecture details\n"
        "× Implementation quality assessments\n\n"
        "### Success Criteria:\n"
        "A perfect response:\n"
        "- Clearly connects the code to a specific business process\n"
        "- Explains the value in terms of business outcomes\n"
        "- Uses relatable examples from physical operations\n"
        "- Could be understood by a non-technical manager\n"
        "- Focuses on 'what' the code enables, not 'how' it works\n\n"
        "Ensure that each explanation ties directly to the given code snippet, linking it to the business process it supports, the operational improvements it provides, and its impact on stakeholders."
    )

message_history = []

class CodeRequest(BaseModel):
    code_snippet: str

async def explain_business(request: CodeRequest):
    # Define a synchronous wrapper that creates its own event loop.
    def run_agent():
        # This ensures the thread gets a new event loop.
        return asyncio.run(
            business_explanation_agent.run(
                request.code_snippet,
                deps="TruckQueueService.java",
                message_history=message_history
            )
        )

    # Offload the blocking work to a worker thread.
    result = await asyncio.to_thread(run_agent)
    message_history.extend(result.new_messages())
    return {"explanation": result.data}
