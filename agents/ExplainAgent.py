import os
import logfire
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from models.CodeRequest import CodeRequest
from models.ExplainAgentDependencies import ExplainAgentDependencies
from models.ResponseTemplate import ResponseTemplate
from database.Qdrant import search_similar_text_qdrant
from agents.FormatCodeAgent import format_code

OLLAMA_URI = os.getenv("OLLAMA_URI", "http://localhost:11434")
logfire.configure()
logfire.instrument_httpx(capture_all=True)


ollama_model = OpenAIModel(
    model_name='qwen2.5:7b',
    base_url=OLLAMA_URI + "/v1",
    api_key=os.getenv("OLLAMA_API_KEY", "default_api_key"),
)

business_explanation_agent = Agent(
    ollama_model,
    deps_type=ExplainAgentDependencies,
    result_type=ResponseTemplate,
    retries=10
)


def get_complexity_message(complexity: float) -> str:
    """Return a complexity message based on the provided complexity score."""
    if complexity >= 0.75:
        return "This explanation will be detailed, covering advanced concepts and deeper analysis."
    elif complexity >= 0.50:
        return "This explanation will strike a balance between depth and simplicity."
    elif complexity >= 0.25:
        return "This explanation will be relatively simple, providing essential information with minimal technical detail."
    else:
        return "This explanation will be very simple, focusing on the most basic concepts."


@business_explanation_agent.system_prompt
def add_business_context(run_context) -> str:
    """Construct the business context message based on complexity and role."""
    complexity_message = get_complexity_message(run_context.deps.complexity)

    return f"""
You are a business analyst translating technical implementations into business value. Your task is to explain how the provided code creates value in the context of a {run_context.deps.user_role}'s responsibilities and workflows.

### Key Requirements:
1. **Role Context**: Frame the explanation through the lens of a {run_context.deps.user_role}'s daily operations and priorities.
2. **Relevance**: Highlight aspects most impactful to their specific business goals and challenges.
3. **Practicality**: Use examples from common scenarios they encounter.
4. **Value Focus**: Show how this enables better outcomes in their area of responsibility.

### Complexity of Answer
Code complexity of answer = {run_context.deps.complexity} take this number as a percent into account. 
{complexity_message}

### Prohibited Elements:
× Technical jargon (APIs, frameworks, etc.)
× Code implementation details
× Developer-specific terminology
× Architecture discussions

### Structural Guidance:
1. **Process Impact**: How does this code improve workflows the {run_context.deps.user_role} manages?
2. **Outcome Alignment**: What business goals does this help the {run_context.deps.user_role} achieve?
3. **Stakeholder Value**: Which other roles benefit from this functionality and how?
4. **Strategic Fit**: How does this contribute to the organization's core objectives?

### Business Context:
{run_context.deps.business_context}

### Success Criteria:
- Direct ties to the {run_context.deps.user_role}'s key performance indicators (KPIs)
- Clear examples of application in their daily work environment
- Explanation of how this creates measurable business impact
- Language a non-technical manager would naturally understand
- No technical implementation details whatsoever
"""


async def get_business_context(formatted_code: str, collection_name: str) -> str:
    """Retrieve the business context by searching for similar code."""
    similar_texts = search_similar_text_qdrant(formatted_code, collection_name)
    return "\n\n".join([f"- {item['text']}" for item in similar_texts])


async def explain_business(request: CodeRequest):
    """Explain the business value of a code snippet for a given user role."""

    try:
        formatted_code = await format_code(request.code_snippet)

        business_context = await get_business_context(formatted_code, request.collection_name)

        dependencies = ExplainAgentDependencies(
            business_context=business_context,
            user_role=request.user_role,
            complexity=request.complexity
        )

        result = await business_explanation_agent.run(
            formatted_code,
            deps=dependencies
        )

        logfire.info(f"Explanation result: {result.data}")
        return {"explanation": result.data}

    except Exception as e:
        logfire.error(f"Error during business explanation: {str(e)}")
        return {"error": "An error occurred while processing the request."}

