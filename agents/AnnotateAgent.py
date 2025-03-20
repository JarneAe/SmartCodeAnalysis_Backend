import os
import ast
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
    2. **Annotation**: Provide annotations for relevant blocks of code (methods, classes, or significant logic).
    3. **Relevance**: Highlight aspects most impactful to their specific business goals and challenges.
    4. **Practicality**: Use examples from common scenarios they encounter.
    5. **Value Focus**: Show how this enables better outcomes in their area of responsibility.

    ### Complexity of Answer
    Code complexity of answer = {run_context.deps.complexity} take this number as a percent into account. 
    {complexity_message}
    
    ### Link to Business Context:
    Make sure to provide annotations that are relevant to the business context. Give business examples and not pure code stuff.

    ### Annotation Structure:
    Provide a **JSON array** where each object follows this structure:
    ```json
    [
      {{
        "start_line": <integer: first line of the annotated block>,
        "end_line": <integer: last line of the annotated block>,
        "explanation": "<string: clear business-friendly explanation>"
      }}
    ]
    ```

    ### Example Annotation Output:
    ```json
    [
      {{
        "start_line": 5,
        "end_line": 10,
        "explanation": "This function processes user input to validate and clean the data before saving."
      }},
      {{
        "start_line": 15,
        "end_line": 18,
        "explanation": "This loop iterates through a dataset to calculate total revenue based on user transactions."
      }}
    ]
    ```

    ### Business Context:
    {run_context.deps.business_context}

    ### Success Criteria:
    - Annotations include valid `start_line` and `end_line` values.
    - Clear and concise annotations explaining the business value of the code.
    - No technical jargon or code implementation details.
    - Ensure at least one annotation per 25% of the code length. (or atleast 5 annotations)
    """

def add_line_numbers(code: str) -> str:
    """Add line numbers to the provided code snippet."""
    lines = code.split("\n")
    return "\n".join([f"{i + 1}. {line}" for i, line in enumerate(lines)])

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
            add_line_numbers(request.code_snippet),
            deps=dependencies
        )

        logfire.info(f"Raw LLM Output: {result.data}")

        if not result.data:
            logfire.error("Received empty annotation response from the model. Requesting regeneration.")
            return {"error": "No annotations generated. Please retry."}

        annotations = []
        total_lines = len(request.code_snippet.split("\n"))
        min_annotations = max(1, total_lines // 25)

        for annotation in result.data:
            try:
                annotations.append(
                    Annotation(
                        start_line=int(annotation.start_line),
                        end_line=int(annotation.end_line),
                        explanation=str(annotation.explanation)
                    )
                )
            except (AttributeError, ValueError) as e:
                logfire.error(f"Invalid annotation format: {annotation}, Error: {e}")

        if len(annotations) < min_annotations:
            logfire.warning("Insufficient annotations generated. Requesting regeneration.")
            return {"error": "Insufficient annotations. Please retry."}

        logfire.info(f"Final Annotated Result: {annotations}")
        return annotations

    except Exception as e:
        logfire.error(f"Error annotating the code: {str(e)}")
        raise {"error": f"Error annotating the code: {str(e)}"}