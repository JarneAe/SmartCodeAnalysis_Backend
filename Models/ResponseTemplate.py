from pydantic import BaseModel


class ResponseTemplate(BaseModel):
    business_scenario: str
    impact_on_operations: str
    user_experience: str
    strategic_value: str
    day_in_the_life_usecase: str