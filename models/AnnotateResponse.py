from pydantic import BaseModel, Field


class Annotation(BaseModel):
    annotationLine: int = Field(default=0)
    explanation: str = Field(default="")