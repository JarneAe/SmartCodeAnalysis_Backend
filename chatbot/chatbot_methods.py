import asyncio

import logfire
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

from models.ChatDependencies import ChatDependencies
from database.Qdrant import search_similar_text_qdrant

OLLAMA_URI = "http://localhost:11434"

logfire.configure()
logfire.instrument_httpx(capture_all=True)


ollama_model = OpenAIModel(
    model_name='qwen2.5:7b',
    base_url=OLLAMA_URI + "/v1",
    api_key='ollama',
)

question_agent = Agent(
    ollama_model,
    deps_type=ChatDependencies)

@question_agent.system_prompt()
def system_prompt(run_context) -> str:
    deps = run_context.deps

    return f"""
    You are a business expert. Use the following context to provide clear explanations:
    - Relevant information: {deps.business_context}
    """



async def ask_question(question):

    def run_agent():

        business_context = "\n\n".join([f"- {item['text']}" for item in search_similar_text_qdrant(question)])

        dependencies = ChatDependencies(
            business_context=business_context,
        )

        result = asyncio.run(
            question_agent.run(
                question,
                deps=dependencies,
            )
        )

        return result


    result = await asyncio.to_thread(run_agent)

    return {"explanation": result.data}
