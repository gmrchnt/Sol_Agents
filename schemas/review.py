from typing import Literal
from pydantic import BaseModel,ConfigDict,Field

# F- findings come from the Reviewer, H- from the human's own feedback and
# S- from Reviewer suggestions the human adopted.
FINDING_ID = r"^[FHS]-\d{3}$"


class RequirementResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    requirement_id: str = Field(pattern = r"^REQ-\d{3}$")
    status: Literal["pass","fail","uncertain"]
    evidence:str = Field(min_length=1)


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=FINDING_ID)
    severity: Literal["critical", "high", "medium", "low"]
    category: Literal["bug", "requirement", "security", "performance", "quality", "test"]
    requirement_id: str | None = None
    file: str | None = None
    line: int | None = Field(default=None, ge=1)
    claim: str = Field(min_length=1)
    evidence: str = Field(min_length=1)
    reproduction: str | None = None
    suggested_fix: str | None = None


class Suggestion(BaseModel):
    """Optional and never blocking: an improvement to existing code, or a new addition."""
    model_config = ConfigDict(extra="forbid")

    # Only a label: Sol renumbers suggestions S-001, S-002, ... after the review,
    # so a model that writes "SUG-1" does not cost the run.
    id: str = ""
    kind: Literal["improvement", "addition"]
    file: str | None = None
    line: int | None = Field(default=None, ge=1)
    description: str = Field(min_length=1)
    rationale: str = Field(min_length=1)


class Review(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["approved", "changes_required"]
    requirements: list[RequirementResult]
    findings: list[Finding] = Field(default_factory=list)
    suggestions: list[Suggestion] = Field(default_factory=list)
