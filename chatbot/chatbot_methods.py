import os
import logfire
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from models.ChatDependencies import ChatDependencies
from models.ChatRequest import ChatRequest
from qdrant.qdrant_methods import search_similar_text_qdrant
from retrieval import retrieve, RetrievalRequest

OLLAMA_URI = os.getenv("OLLAMA_URI", "http://localhost:11434")
logfire.configure()
logfire.instrument_httpx(capture_all=True)


ollama_model = OpenAIModel(
    model_name='qwen2.5:7b',
    base_url=OLLAMA_URI + "/v1",
    api_key=os.getenv("OLLAMA_API_KEY", "default_api_key"),
)

question_agent = Agent(
    ollama_model,
    deps_type=ChatDependencies)

@question_agent.system_prompt()
def system_prompt(run_context) -> str:
    deps = run_context.deps

    return f"""
    You are an expert in business strategy, operations, and decision-making.  
    Your goal is to provide **clear, concise, and actionable explanations** using the relevant context below:

    **Relevant Business Context:**
    {deps.business_context if deps.business_context else "No specific context available. Provide a general but insightful response."}

    **Relevant Code Snippets:**
    {deps.code_snippets if deps.code_snippets else "No specific code snippets available." }

    ### **Instructions:**
    - Keep responses **brief yet comprehensive** (ideally under 200 words).
    - Use **concrete examples** and **specific details** where possible.
    - **Avoid vague statements**—ensure explanations are direct and useful.
    - Do not mention the code snippets. If they do not provide enough context, ignore them. 
    - If additional information is needed, state what’s missing.
    """

async def get_business_context(chat_message: str, collection_name: str) -> str:
    business_context = "\n\n".join([f"- {item['text']}" for item in search_similar_text_qdrant(query_text=chat_message, collection_name=collection_name)])
    return business_context


async def ask_question(request: ChatRequest):

    try:
        business_context = await get_business_context(request.chat_message, request.collection_name)

        codebase_id = request.collection_name[-36:] # very scuffed
        code_snippets = retrieve(RetrievalRequest(codebase_id=codebase_id, query=request.chat_message, n=3))

        dependencies = ChatDependencies(
            business_context=business_context,
            code_snippets=code_snippets
        )

        result = await question_agent.run(
            request.chat_message,
            deps=dependencies,
        )

        return {"explanation": result.data}

    except Exception as e:
        logfire.error(f"Error during business explanation: {str(e)}")
        return {"error": "An error occurred while processing the request."}
