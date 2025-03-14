from pydantic import BaseModel, Field, condecimal


class ChatRequest(BaseModel):
    chat_message: str = Field(
        default="""
        How does this work?
        """
    )
    collection_name: str = Field(default="TestCollection")
