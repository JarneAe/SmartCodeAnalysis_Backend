from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
import logfire
from PDFConvertor import PDFConvertor
from ResponseTemplate import ResponseTemplate

logfire.configure()
logfire.instrument_httpx(capture_all=True)

pdf_convertor = PDFConvertor(file_path="files/SA 24-25 - Mineral Flow-1.pdf")

business_context = pdf_convertor.convert()

ollama_model = OpenAIModel(
    model_name='qwen2.5:14b',
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
    
        "Put the main focus on the business context that was given and ensure that the explanation is clear and easy to understand for non-technical stakeholders."
        
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

code_snippet = """
    This is the code that should be explained:
    
    public Long getWarehouseIdByLicensePlate(String licensePlate) {
        try {
            Pair<Long, String> data = appointmentRetrievalService.getClientIdAndMineralByLicensePlate(licensePlate);
            if (data == null) {
                throw new NoSuchElementException("No data found for license plate: " + licensePlate);
            }

            Long sellerId = data.getFirst();
            String mineralName = data.getSecond();

            log.info("Fetching Warehouse id for seller with id {} and mineral {}", sellerId, mineralName);

            String url = String.format(warehouseApiUrl, sellerId, mineralName);

            ResponseEntity<Long> response = restTemplate.getForEntity(url, Long.class);

            if (response.getStatusCode().is2xxSuccessful()) {
                Long warehouseId = response.getBody();
                log.info("Warehouse id retrieved: {}", warehouseId);
                return warehouseId;
            } else {
                log.error("Failed to retrieve warehouse id. Status code: {}", response.getStatusCode());
                throw new CouldNotRetrieveWarehouseException("Failed to retrieve warehouse id. Status code: " + response.getStatusCode());
            }
        } catch (NoSuchElementException e) {
            log.error("Error retrieving warehouse id for license plate: {}", licensePlate, e);
            throw new CouldNotRetrieveWarehouseException("Error retrieving warehouse id for license plate: " + licensePlate, e);
        }
    }

}"""

result = business_explanation_agent.run_sync(
    code_snippet,
    deps="TruckQueueService.java",
    message_history=message_history
)

message_history.extend(result.new_messages())

print(result.data)
