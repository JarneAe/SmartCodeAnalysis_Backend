from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_ollama import ChatOllama

model = ChatOllama(
    model="llama3.2",
    temperature=0.2,
    system="You are skilled at explaining technical implementations in business terms without suggesting code improvements."
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are an analyst specializing in {business_context}. Explain how the given code connects to:
1. Business rules
2. Operational processes

Focus only on explanation. Do not suggest improvements or alternative implementations.

Use clear business language while keeping technical accuracy."""),

### Few shot example for the llm to use.

    ("human", """Analyze this Python code for Retail Order Processing:
def validate_order(order):
    return order.quantity > 0 and order.customer_active"""),
    ("ai", """1. Business Rules: 
- Ensures all orders contain at least 1 item (no zero-quantity orders)
- Verifies customer accounts are active before accepting orders

2. Operational Processes:
- Automates quality checks during order entry
- Prevents invalid orders from entering fulfillment pipelines"""),





    ("human", """Analyze this {language} code for {business_context}:
{code}

Explain:
- How it works in business terms
- Its impact on operations

Do not suggest changes to the code—only analyze what is present.""")
])

analysis_chain = (
    prompt_template
    | model
    | StrOutputParser()
)

analysis_result = analysis_chain.invoke({
    "business_context": "Retail Order Processing",
    "language": "Python",
    "code": """def apply_discount(price, discount):
    return max(price - (price * discount), 0)"""
})

print("Business-Technical Analysis:\n")
print(analysis_result)
