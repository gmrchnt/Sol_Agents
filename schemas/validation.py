from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

CheckStatus = Literal["pass", "fail", "not_configured", "not_run"]


class CheckResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: CheckStatus
    command: str | None = None
    output: str = ""


class ValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    build: CheckResult
    tests: CheckResult
    lint: CheckResult
    static_analysis: CheckResult
    unexpected_files: list[str] = Field(default_factory=list)

    @property
    def failures(self) -> list[str]:
        problems = [
            f"{name} failed:\n{check.output}".rstrip()
            for name, check in self.checks().items()
            if check.status == "fail"
        ]
        problems += [f"unexpected change: {path}" for path in self.unexpected_files]
        return problems

    def checks(self) -> dict[str, CheckResult]:
        return {
            "build": self.build,
            "tests": self.tests,
            "lint": self.lint,
            "static_analysis": self.static_analysis,
        }

    def summary(self) -> str:
        lines = [f"{name}: {check.status}" for name, check in self.checks().items()]
        if self.unexpected_files:
            lines.append(f"unexpected files: {', '.join(self.unexpected_files)}")
        for name, check in self.checks().items():
            if check.output and check.status in {"fail", "pass"}:
                lines.append(f"\n{name.upper()} OUTPUT ({check.command}):\n{check.output}")
        return "\n".join(lines)
