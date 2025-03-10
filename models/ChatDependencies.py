from pydantic import BaseModel


class ChatDependencies(BaseModel):
    business_context: str
