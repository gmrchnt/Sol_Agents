from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from schemas.decisions import DeveloperResponse, HumanDecision, HumanFeedback
from schemas.project import ProjectSummary
from schemas.requirements import Specification
from schemas.review import Finding, Review
from schemas.validation import ValidationReport

Status = Literal[
    "requirements",
    "implement",
    "test",
    "review",
    "human_review",
    "developer_response",
    "fix",
    "validate",
    "final_review",
    "complete",
    "failed",
]


class Checkpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    commit: str
    label: str


class ProjectState(BaseModel):
    """Everything about a run. Only the orchestrator changes it."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    user_prompt: str
    specification: Specification
    project_path: str
    original_branch: str
    run_branch: str
    base_commit: str

    status: Status = "requirements"
    iteration: int = 0
    developer_summary: str | None = None
    validation: ValidationReport | None = None
    review: Review | None = None
    human_decisions: list[HumanDecision] = Field(default_factory=list)
    developer_response: DeveloperResponse | None = None
    dismissed_findings: list[Finding] = Field(default_factory=list)
    checkpoints: list[Checkpoint] = Field(default_factory=list)
    outcome: str | None = None

    # What the project already was when the run started (survey or .sol document).
    project_summary: ProjectSummary | None = None
    # Feedback the human gave at final review, and requests that amended the spec.
    feedback: list[HumanFeedback] = Field(default_factory=list)
    amendments: list[str] = Field(default_factory=list)
