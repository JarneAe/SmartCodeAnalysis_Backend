import os
import pdfplumber
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_ollama import ChatOllama

#get text from pdf

folder_path = "files"

def extract_text_from_pdfs(folder_path):
    """Extracts text from all PDF files in a folder."""
    all_text = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if file_name.lower().endswith(".pdf") and os.path.isfile(file_path):
            with pdfplumber.open(file_path) as pdf:
                text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
                all_text.append(text)
    return "\n".join(all_text)

business_context = extract_text_from_pdfs(folder_path)

model = ChatOllama(
    model="llama3.2",
    temperature=0.2,
    system="You are skilled at explaining technical implementations in simple business terms that anyone can understand."
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are an expert in translating technical concepts into easy-to-understand business explanations. Your job is to analyze the provided code and explain how it supports business goals and day-to-day operations.

You will be given:
1. **Business Context** → Background information extracted from documents.
2. **Code Snippet** → A technical implementation to analyze.

### Your Task:
- **Business Purpose:** Explain why this code is important for business operations.
- **Real-World Impact:** Describe how this code helps in everyday business activities.
- **Clear & Simple Explanation:** Avoid technical jargon and keep it easy to understand.
- Explain everything in such a manner that a non-technical person would understand the importance of the code. If you do use any technical terms explain them very briefly.



**DO NOT**:
- Use technical language.
- Suggest improvements or alternative implementations.
- Repeat the business context without linking it to the code.

Focus on a clear and practical business explanation."""),

    ("human", """## Business Context:
{business_context}

## Code Analysis Task:
Analyze the following {language} code in file **{file_name}**:

```
{code}
```

### Provide:
1. **Why this matters for the business.**
2. **How it helps in daily operations.**
3. **A simple explanation that connects the code to real-world business activities.**

Keep it clear and easy to understand for non-technical stakeholders.""")
])

analysis_chain = (
    prompt_template
    | model
    | StrOutputParser()
)

analysis_result = analysis_chain.invoke({
    "business_context": business_context,
    "language": "Java",
    "file_name": "DockOperationsController.java",
    "code": """
 package be.kdg.sa.jarne_milan.Water.controllers;


import be.kdg.sa.jarne_milan.Water.controllers.dto.DockArrivalDto;
import be.kdg.sa.jarne_milan.Water.controllers.dto.DockOperationDto;
import be.kdg.sa.jarne_milan.Water.controllers.dto.DockOperationStatusResponseDto;
import be.kdg.sa.jarne_milan.Water.services.DockOperationCreationService;
import be.kdg.sa.jarne_milan.Water.services.DockOperationService;
import lombok.extern.log4j.Log4j2;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;

import java.util.Optional;

@RestController
@RequestMapping("/api/dock-operations")
@Log4j2
public class DockOperationsController {

    private final DockOperationService dockOperationService;
    private final DockOperationCreationService dockOperationCreationService;

    public DockOperationsController(DockOperationService dockOperationService, DockOperationCreationService dockOperationCreationService) {
        this.dockOperationService = dockOperationService;
        this.dockOperationCreationService = dockOperationCreationService;
    }

    @GetMapping("/get-by-po/{purchaseOrderId}")
    @PreAuthorize("hasAuthority('captain')")
    public ResponseEntity<Optional<DockOperationDto>> getDockOperationByPurchaseOrderId(@PathVariable Long purchaseOrderId) {
        log.info("Getting dock operation by purchase order id");
        return ResponseEntity.ok(Optional.ofNullable(dockOperationService.getDockOperationById(purchaseOrderId)));
    }

    @GetMapping("/get-by-po/{purchaseOrderId}/status")
    @PreAuthorize("hasAuthority('captain')")
    public ResponseEntity<DockOperationStatusResponseDto> getDockOperationByPurchaseOrderIdStatus(@PathVariable Long purchaseOrderId) {
        log.info("Getting dock operation status by purchase order id {}", purchaseOrderId);
        DockOperationStatusResponseDto statusResponse = dockOperationService.getDockOperationByIdStatus(purchaseOrderId);
        return ResponseEntity.ok(statusResponse);
    }

    @PostMapping
    @PreAuthorize("hasAuthority('captain')")
    public ResponseEntity<DockOperationDto> planOperations(@RequestBody DockArrivalDto dockArrivalDto) {
        log.info("Creating dock operation and planning operations");
        return ResponseEntity.ok(dockOperationCreationService.createDockOperation(dockArrivalDto));
    }
}
""",
})

print("Business-Technical Analysis:\n")
print(analysis_result)
