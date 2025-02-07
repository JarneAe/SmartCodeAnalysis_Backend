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
3. Efficiency improvements
4. Key risks or limitations

Focus only on explanation. Do not suggest improvements or alternative implementations.

Use clear business language while keeping technical accuracy."""),

    ("human", """Analyze this {language} code for {business_context}:
{code}

Explain:
- How it works in business terms
- Its impact on operations
- Any potential risks or inefficiencies

Do not suggest changes to the code—only analyze what is present. Don't show possible risks or improvements / limitations just purely explain the code in a readable manner.""")
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
