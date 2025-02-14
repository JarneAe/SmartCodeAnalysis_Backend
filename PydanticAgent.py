import os
import pdfplumber
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
import logfire


class Response(BaseModel):
    practical_use_case: str
    business_impact: str
    user_impact: str
    why_code_important: str
    day_in_the_life_usecase: str



def extract_text_from_pdfs(folder_path = "files"):
    """Extracts text from all PDF files in a folder."""
    all_text = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if file_name.lower().endswith(".pdf") and os.path.isfile(file_path):
            with pdfplumber.open(file_path) as pdf:
                text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
                all_text.append(text)
    return "\n".join(all_text)


business_context = extract_text_from_pdfs()

logfire.configure()
logfire.instrument_httpx(capture_all=True)

ollama_model = OpenAIModel(
    model_name='qwen2.5:7b',
    base_url='http://localhost:11434/v1',
    api_key='ollama',
)

business_explanation_agent = Agent(
    ollama_model,
    deps_type=str,
    result_type=Response,
    retries=5
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

    @Transactional
    public DockOperationDto createDockOperation(DockArrivalDto dockArrivalDto) {
        PurchaseOrder order = purchaseOrderRepository.findByPoNumberIgnoreCase(dockArrivalDto.getPoNumber())
                .orElseThrow(() -> new IllegalArgumentException("Purchase order not found"));
        Vessel vessel = vesselRepository.findByVesselNumber(dockArrivalDto.getVesselNumber())
                .orElseThrow(() -> new IllegalArgumentException("Vessel not found"));
        DockOperation dockOperation = new DockOperation();
        dockOperation.setPurchaseOrder(order);
        dockOperation.setArrivalTime(LocalDateTime.now());
        dockOperation.setCurrent(true);
        dockOperation.setVessel(vessel);
        dockOperation.setShipOperations(new ArrayList<>());
        inspectionOperationCreationService.planInspectionOperation(dockOperation);
        loadingOperationCreationService.planLoadingOperation(dockOperation, dockArrivalDto.getClientId(), dockArrivalDto.getPoNumber());
        dockOperationRepository.save(dockOperation);
        vessel.getDockOperations().add(dockOperation);
        vesselRepository.save(vessel);
        return dockOperationMapper.convertToDto(dockOperation);
    }
}"""

result = business_explanation_agent.run_sync(
    code_snippet,
    deps="TruckQueueService.java",
    message_history=message_history
)

message_history.extend(result.new_messages())

print(result.data)
