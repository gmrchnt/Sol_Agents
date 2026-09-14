from typing import Literal
from pydantic import BaseModel, Field, ConfigDict

class Requirement(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )
    id: str = Field(pattern = r"^REQ-\d{3}$")
    description: str
    acceptance_criteria: list[str]
    source: Literal[
        "explicit",
    ]
    source_text: str

class Specification(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )
    original_prompt: str
    summary: str
    requirements: list[Requirement]
    assumptions: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)