from pydantic import BaseModel

class ResponseTemplate(BaseModel):
    role_specific_use_case: str = "How this supports the role's daily responsibilities"
    process_impact: str = "Improvements to workflows managed by this role"
    kpi_alignment: str = "Connection to the role's key performance indicators"

    stakeholder_value: str = "Benefits to other teams/roles"
    strategic_relevance: str = "Alignment with organizational goals"

    workflow_example: str = "Concrete example of usage in daily operations"
    outcome_summary: str = "Measurable business impact (efficiency, cost, revenue)"