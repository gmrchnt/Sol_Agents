from pathlib import PurePosixPath

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Service(BaseModel):
    """A container the component's tests can reach by name - a database, a cache."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[a-z][a-z0-9-]{0,30}$")
    image: str = Field(min_length=1)
    env: dict[str, str] = Field(default_factory=dict)
    # Run inside the service until it succeeds, e.g. "pg_isready -U postgres".
    ready_command: str | None = None


class Component(BaseModel):
    """One buildable part of the project: its folder, sandbox image and commands.

    Every command runs with `sh -c` in the component's folder.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,40}$")
    language: str = "other"
    path: str = "."
    # The image commands run in; a project-relative Dockerfile is built instead.
    image: str | None = None
    dockerfile: str | None = None
    # Install dependencies. These run with network; everything else runs without.
    setup: list[str] = Field(default_factory=list)
    # Files (globs, relative to the component) whose change means setup must run again,
    # and the folder setup creates, which must exist for setup to count as done.
    setup_inputs: list[str] = Field(default_factory=list)
    setup_creates: str | None = None
    build: str | None = None
    test: str | None = None
    lint: str | None = None
    static_analysis: str | None = None
    # How a person starts the program, for "try it yourself".
    run: str | None = None
    ports: list[int] = Field(default_factory=list)
    # Run checks under a virtual X display (needs xvfb-run in the image).
    display: bool = False
    test_network: bool = False
    env: dict[str, str] = Field(default_factory=dict)
    services: list[Service] = Field(default_factory=list)

    @field_validator("path", "dockerfile")
    @classmethod
    def _inside_project(cls, value: str | None) -> str | None:
        if value is None:
            return None
        path = PurePosixPath(value.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts or ":" in (path.parts or ("",))[0]:
            raise ValueError(f"{value!r} must be a path inside the project")
        if path.parts and path.parts[0].lower() in {".git", ".sol"}:
            raise ValueError(f"{value!r} is inside a folder Sol protects")
        return path.as_posix()

    @field_validator("ports")
    @classmethod
    def _valid_ports(cls, value: list[int]) -> list[int]:
        bad = [port for port in value if not 1 <= port <= 65535]
        if bad:
            raise ValueError(f"not valid port numbers: {bad}")
        return value


class Toolchain(BaseModel):
    model_config = ConfigDict(extra="forbid")

    components: list[Component] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_names(self) -> "Toolchain":
        names = [component.name for component in self.components]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"component names must be unique: {', '.join(duplicates)}")
        return self
