from pydantic import BaseModel, condecimal

class TestAgentDependencies(BaseModel):
    test_framework: str
    business_context: str