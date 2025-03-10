from pydantic import BaseModel, condecimal

class ExplainAgentDependencies(BaseModel):
    business_context: str
    user_role: str
    complexity: condecimal(ge=0, le=1)