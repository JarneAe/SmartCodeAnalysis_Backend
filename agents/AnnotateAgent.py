import os
from typing import List

import logfire
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

from models.AnnotateResponse import Annotation
from models.CodeRequest import CodeRequest
from models.ExplainAgentDependencies import ExplainAgentDependencies
from agents.ExplainAgent import get_complexity_message, get_business_context

OLLAMA_URI = os.getenv("OLLAMA_URI", "http://localhost:11434")

ollama_model = OpenAIModel(
    model_name='qwen2.5:7b',
    base_url=OLLAMA_URI + "/v1",
    api_key='ollama',
)

business_annotate_agent = Agent(
    ollama_model,
    deps_type=ExplainAgentDependencies,
    result_type=List[Annotation],
    retries=10
)


@business_annotate_agent.system_prompt
def add_business_context(run_context) -> str:
    """Construct the business context message based on complexity and role."""
    complexity_message = get_complexity_message(run_context.deps.complexity)

    return f"""
    You are a business analyst translating technical implementations into business value. Your task is to annotate the provided code that creates value in the context of a {run_context.deps.user_role}'s responsibilities and workflows.
    
    ### Key Requirements:
    1. **Role Context**: Frame the explanation through the lens of a {run_context.deps.user_role}'s daily operations and priorities.
    2. **Annotation**: Provide annotations above blocks of code (methods, classes,...) to explain their purpose and value.
    3. **Relevance**: Highlight aspects most impactful to their specific business goals and challenges.
    4. **Practicality**: Use examples from common scenarios they encounter.
    5. **Value Focus**: Show how this enables better outcomes in their area of responsibility.
    
    ### Complexity of Answer
    Code complexity of answer = {run_context.deps.complexity} take this number as a percent into account. 
    {complexity_message}
    
    ### Prohibited Elements:
    × Technical jargon (APIs, frameworks, etc.)
    × Code implementation details
    × Developer-specific terminology
    × Architecture discussions
    
    ### Structural Guidance:
    1. **Annotations**: Provide a list of annotations for the code blocks in the provided code snippet.
        1.1. **Annotation Line**: The first line of the annotated block.
        1.2. **Explanation**: The explanation of the annotated block.
    
    ### Business Context:
    {run_context.deps.business_context}
    
    ### Success Criteria:
    - Direct ties to the {run_context.deps.user_role}'s responsibilities and workflows.
    - Clear and concise annotations that explain the value of the code.
    - No technical jargon or code implementation details whatsoever.
    """


async def annotate_code(request: CodeRequest):
    """Annotates codeblocks in the provided code snippet."""
    try:
        business_context = await get_business_context(request.code_snippet, request.collection_name)
        dependencies = ExplainAgentDependencies(
            business_context=business_context,
            user_role=request.user_role,
            complexity=request.complexity
        )

        result = await business_annotate_agent.run(
            request.code_snippet,
            deps=dependencies
        )

        logfire.info(f"Annotated result: {result.data}")
        return result.data
    except Exception as e:
        logfire.error(f"Error annotating the code: {str(e)}")
        raise {"error": f"Error annotating the code: {str(e)}"}