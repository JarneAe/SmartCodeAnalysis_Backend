from pydantic import BaseModel


class ChatDependencies(BaseModel):
    business_context: str
    code_snippets: list[str]
