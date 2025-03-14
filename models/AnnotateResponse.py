from pydantic import BaseModel, Field


class Annotation(BaseModel):
    startLine: int = Field(default=0)
    endLine: int = Field(default=0)
    explanation: str = Field(default="")