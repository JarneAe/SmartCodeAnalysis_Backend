import os
import logfire
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from models.CodeTestGenerationRequest import CodeTestGenerationRequest
from models.ResponseTemplate import ResponseTemplate
from models.TestAgentDependencies import TestAgentDependencies
from qdrant.qdrant_methods import search_similar_text_qdrant
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
    deps_type=TestAgentDependencies,
)


@business_explanation_agent.system_prompt
def add_business_context(run_context) -> str:
    """Construct the business context message based on complexity and role."""
    return f"""
    You are a highly skilled testing engineer specializing in {run_context.deps.test_framework}. 
    Your primary objective is to design a robust and comprehensive suite of test cases for the given code.

    ### Responsibilities:
    - Identify core functionalities and edge cases to ensure full test coverage.
    - Apply best practices in {run_context.deps.test_framework}, leveraging appropriate assertions and mocking where necessary.
    - Consider potential failure points, security vulnerabilities, and performance bottlenecks.
    - Ensure tests align with industry standards and maintainability principles.

    ### Business Context:
    The following business context should guide your test case generation, ensuring that your tests align with real-world usage and critical business functions:
    {run_context.deps.business_context}

    Generate well-structured test cases that include clear setup, execution, and validation steps.
    Provide rationale for each test, explaining its importance in ensuring code reliability and business alignment.
    """

async def get_business_context(formatted_code: str, collection_name: str) -> str:
    """Retrieve the business context by searching for similar code."""
    try:
        similar_texts = search_similar_text_qdrant(formatted_code, collection_name)
        return "\n\n".join([f"- {item['text']}" for item in similar_texts])
    except Exception as e:
        logfire.error(f"Error retrieving business context: {str(e)}")
        return ""


import re

async def generate_tests_agent(request: CodeTestGenerationRequest):
    """Generate test cases for a given code snippet."""
    try:
        formatted_code = await format_code(request.code_file)
        business_context = await get_business_context(formatted_code, request.collection_name)

        dependencies = TestAgentDependencies(
            test_framework=request.testing_framework,
            business_context=business_context,
        )

        logfire.info(f"Dependencies: {dependencies}")  # Debug logging

        result = await business_explanation_agent.run(
            formatted_code,
            deps=dependencies
        )

        logfire.info(f"Raw result: {result.data}")

        # Extract the code block using regex
        match = re.search(r"```(?:\w+)?\n(.*?)```", result.data, re.DOTALL)
        extracted_code = match.group(1) if match else result.data  # Return extracted code or fallback to raw response

        return {"code": extracted_code.strip()}

    except FileNotFoundError:
        return {"error": "The provided file path does not exist."}
    except RecursionError:
        logfire.error("Recursion error detected! Check function calls.")
        return {"error": "Recursion depth exceeded. Please check function calls."}
    except Exception as e:
        logfire.error(f"Error generating tests: {str(e)}")
        return {"error": "An error occurred while processing the request."}