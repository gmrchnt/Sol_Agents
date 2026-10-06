from typing import Annotated, Literal
from pydantic import BaseModel, Field, ConfigDict

#Schema of a single requirement in the specification.
class Requirement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern = r"^REQ-\d{3}$")
    description: str = Field(min_length=1)
    acceptance_criteria: list[Annotated[str, Field(min_length=1)]] = Field(min_length=1)
    source: Literal["explicit",]
    source_text: str


#Schema for the entire specification generated from the user's request.
class Specification(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: str = Field(min_length=1)
    requirements: list[Requirement] = Field(min_length=1)
    assumptions: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)