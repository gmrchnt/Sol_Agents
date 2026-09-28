from typing import Literal
from pydantic import BaseModel,ConfigDict,Field

class RequirementResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    requirement_id: str = Field(pattern = r"^REQ-\d{3}$")
    status: Literal["pass","fail","uncertain"]
    evidence:str = Field(min_length=1)


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^F-\d{3}$")
    severity: Literal["critical", "high", "medium", "low"]
    category: Literal["bug", "requirement", "security", "performance", "quality", "test"]
    requirement_id: str | None = None
    file: str | None = None
    line: int | None = Field(default=None, ge=1)
    claim: str = Field(min_length=1)
    evidence: str = Field(min_length=1)
    reproduction: str | None = None
    suggested_fix: str | None = None

class Review(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["approved", "changes_required"]
    requirements: list[RequirementResult]
    findings: list[Finding] = Field(default_factory=list)