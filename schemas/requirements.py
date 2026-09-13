from typing import Literal
from pydantic import BaseModel, Field

class Requirements(BaseModel):
    id: str = Field(pattern = r"^REQ-\d{3}$")
    description: str
    acceptance_criteria: list[str]
    source: Literal[
        "explicit",
        "derived",
    ]

class Specification(BaseModel):
    original_prompt: str
    summary: str
    requirements: list[Requirements]
    assumptions: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)