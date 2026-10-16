from pydantic import BaseModel, ConfigDict, Field


class ModuleInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    responsibility: str = Field(min_length=1)


class ProjectSummary(BaseModel):
    """What the project is, as the Reviewer understands it from reading it."""

    model_config = ConfigDict(extra="forbid")

    purpose: str = Field(min_length=1)
    entry_points: list[str] = Field(default_factory=list)
    modules: list[ModuleInfo] = Field(default_factory=list)
    how_to_run: str = ""
    tests: str = ""
    dependencies: list[str] = Field(default_factory=list)
    known_issues: list[str] = Field(default_factory=list)


class RunRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    request: str
    requirements: list[str] = Field(default_factory=list)
    outcome: str


class ProjectDocument(BaseModel):
    """.sol/project.json: carried from one run to the next."""

    model_config = ConfigDict(extra="forbid")

    commit: str
    summary: ProjectSummary
    history: list[RunRecord] = Field(default_factory=list)
