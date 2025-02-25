import os
import logfire
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
import asyncio

OLLAMA_URI = os.getenv("OLLAMA_URI", "http://localhost:11434/v1")

logfire.configure()

ollama_model = OpenAIModel(
    model_name='qwen2.5:7b',
    base_url=OLLAMA_URI,
    api_key='ollama',
)


format_code_agent = Agent(
    model=ollama_model,
    deps_type=str,
    retries=5,
    system_prompt=
    "You are a programmer tasked with formatting code to improve readability.\n"
    "Format the provided code snippet according to best practices and conventions.\n"
    "Add comments where necessary to explain the code.\n"
    "Only return the formatted code and comments, do not add external information.\n"
)


async def format_code(unformatted_code: str):
    def run_agent():
        try:
            return asyncio.run(format_code_agent.run(unformatted_code))
        except Exception as e:
            logfire.error(f"Error formatting code: {e}")
            return unformatted_code

    result = await asyncio.to_thread(run_agent)
    logfire.info(f'Result: {result.data}')
    return result.data


#asyncio.run(format_code("public Long getWarehouseIdByLicensePlate(String licensePlate) { try { Pair<Long, String> data = appointmentRetrievalService.getClientIdAndMineralByLicensePlate(licensePlate);\n if (data == null) {\n throw new NoSuchElementException(\"No data found for license plate: \" + licensePlate);\n }\n\n Long sellerId = data.getFirst();\n String mineralName = data.getSecond();\n log.info(\"Fetching Warehouse id for seller with id {} and mineral {}\", sellerId, mineralName);\n\n String url = String.format(warehouseApiUrl, sellerId, mineralName);\n\n ResponseEntity<Long> response = restTemplate.getForEntity(url, Long.class);\n\n if (response.getStatusCode().is2xxSuccessful()) {\n Long warehouseId = response.getBody();\n log.info(\"Warehouse id retrieved: {}\", warehouseId);\n return warehouseId;\n } else {\n log.error(\"Failed to retrieve warehouse id. Status code: {}\", response.getStatusCode());\n throw new CouldNotRetrieveWarehouseException(\"Failed to retrieve warehouse id. Status code: \" + response.getStatusCode());\n }\n } catch (NoSuchElementException e) {\n log.error(\"Error retrieving warehouse id for license plate: {}\", licensePlate, e);\n throw new CouldNotRetrieveWarehouseException(\"Error retrieving warehouse id for license plate: \" + licensePlate, e);\n }\n}"))