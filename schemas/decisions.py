from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

FindingId = Annotated[str, Field(pattern=r"^F-\d{3}$")]


class HumanDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    finding_id: FindingId
    action: Literal["approve", "reject", "investigate", "instruct"]
    note: str | None = None


class FindingDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    finding_id: FindingId
    decision: Literal["accept", "reject", "partially_accept"]
    reasoning: str = Field(min_length=1)
    evidence: str | None = None
    action: str | None = None


class DeveloperResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decisions: list[FindingDecision] = Field(default_factory=list)
