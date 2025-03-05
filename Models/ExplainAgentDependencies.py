from pydantic import BaseModel


class ExplainAgentDependencies(BaseModel):
    business_context: str
    user_role: str