from typing import List, Dict

from pydantic import BaseModel, Field


class ContextFile(BaseModel):
    name: str
    content: str


class ContextRequest(BaseModel):
    collection_name: str = Field(default="context_collection")
    context_files: List[ContextFile] = Field()