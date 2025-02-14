from pydantic import BaseModel


class ResponseTemplate(BaseModel):
    practical_use_case: str
    business_impact: str
    user_impact: str
    why_code_important: str
    day_in_the_life_usecase: str