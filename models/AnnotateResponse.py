from pydantic import BaseModel, Field

class Annotation(BaseModel):
    start_line: int = Field(default=0, description="The line number where the annotated block starts")
    end_line: int = Field(default=0, description="The line number where the annotated block ends")
    explanation: str = Field(default="", description="The explanation of the annotated block")
