import asyncio
import os

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
import logfire

from FormatCodeAgent import format_code
from Models.CodeRequest import CodeRequest
from Models.ExplainAgentDependencies import ExplainAgentDependencies
from Qdrant import search_similar_text_qdrant
from PDFConvertor import PDFConvertor
from Models.ResponseTemplate import ResponseTemplate

OLLAMA_URI = os.getenv("OLLAMA_URI", "http://localhost:11434")

logfire.configure()
logfire.instrument_httpx(capture_all=True)


ollama_model = OpenAIModel(
    model_name='qwen2.5:7b',
    base_url=OLLAMA_URI + "/v1",
    api_key='ollama',
)

business_explanation_agent = Agent(
    ollama_model,
    deps_type=ExplainAgentDependencies,
    result_type=ResponseTemplate,
    retries=10
)


@business_explanation_agent.system_prompt
def add_business_context(run_context) -> str:
    deps = run_context.deps

    if deps.complexity >= 0.75:
        complexity_message = "This explanation will be detailed, covering advanced concepts and deeper analysis. Expect a thorough breakdown of the technical aspects."
    elif deps.complexity >= 0.50:
        complexity_message = "This explanation will strike a balance between depth and simplicity, focusing on the key points with some technical details where necessary."
    elif deps.complexity >= 0.25:
        complexity_message = "This explanation will be relatively simple, providing essential information with minimal technical detail. Focus will be on the high-level benefits."
    else:
        complexity_message = "This explanation will be very simple, with a focus on the most basic concepts, ensuring clarity for non-technical stakeholders."

    return f"""
You are a business analyst translating technical implementations into business value. Your task is to explain how the provided code creates value in the context of a {deps.user_role}'s responsibilities and workflows.

### Key Requirements:
1. **Role Context**: Frame the explanation through the lens of a {deps.user_role}'s daily operations and priorities.
2. **Relevance**: Highlight aspects most impactful to their specific business goals and challenges.
3. **Practicality**: Use examples from common scenarios they encounter.
4. **Value Focus**: Show how this enables better outcomes in their area of responsibility.

### Complexity of Answer
Code complexity of answer = {deps.complexity}
{complexity_message}

### Prohibited Elements:
× Technical jargon (APIs, frameworks, etc.)
× Code implementation details
× Developer-specific terminology
× Architecture discussions

### Structural Guidance:
1. **Process Impact**: How does this code improve workflows the {deps.user_role} manages?
2. **Outcome Alignment**: What business goals does this help the {deps.user_role} achieve?
3. **Stakeholder Value**: Which other roles benefit from this functionality and how?
4. **Strategic Fit**: How does this contribute to the organization's core objectives?

### Business Context:
{deps.business_context}

### Success Criteria:
- Direct ties to the {deps.user_role}'s key performance indicators (KPIs)
- Clear examples of application in their daily work environment
- Explanation of how this creates measurable business impact
- Language a non-technical manager would naturally understand
- No technical implementation details whatsoever

### Example Framing (DO NOT USE PHRASES LIKE "YOU" OR "YOUR"):
"For a {deps.user_role}, this functionality streamlines... by automating..., allowing more focus on..."
"""



async def explain_business(request: CodeRequest):
    def run_agent():
        formatted_code = asyncio.run(
            format_code(request.code_snippet)
        )

        business_context = "\n\n".join([f"- {item['text']}" for item in
                                        search_similar_text_qdrant(formatted_code, request.collection_name)])

        dependencies = ExplainAgentDependencies(
            business_context=business_context,
            user_role=request.user_role,
            complexity=request.complexity
        )

        print(f"user_role: {request.user_role}")
        print(f"business_context: {business_context}")

        result = asyncio.run(
            business_explanation_agent.run(
                formatted_code,
                deps=dependencies
            )
        )
        return result

    result = await asyncio.to_thread(run_agent)
    logfire.info(f"Result: {result.data}")
    return {"explanation": result.data}