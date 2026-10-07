import ast
from pathlib import Path
from typing import Callable

from schemas.validation import CheckResult, ValidationReport
from tools.git import changed_paths
from tools.project import list_project_files, read_project_file
from tools.dependencies import (
    REQUIREMENTS_FILE,
    DependencyError,
    missing_dependencies,
    read_requirements,
)
from tools.testing import DependenciesNotReady, SandboxUnavailable, run_tests

NOT_CONFIGURED = CheckResult(
    status="not_configured",
    output="No tool for this check is configured for this project.",
)


def check_syntax() -> CheckResult:
    """Parse every .py file without running it - a safe stand-in for a build step."""
    python_files = [path for path in list_project_files() if path.endswith(".py")]
    if not python_files:
        return CheckResult(status="not_configured", output="No Python files in the project.")

    problems = []
    for path in python_files:
        try:
            ast.parse(read_project_file(Path(path)), filename=path)
        except SyntaxError as exc:
            problems.append(f"{path}:{exc.lineno}: {exc.msg}")
        except ValueError as exc:
            problems.append(f"{path}: could not be checked ({exc})")

    return CheckResult(
        status="fail" if problems else "pass",
        command="ast.parse on every .py file",
        output="\n".join(problems),
    )


def check_dependencies() -> CheckResult:
    """Every third-party import must be declared, pinned, in requirements.txt."""
    command = f"imports in .py files vs {REQUIREMENTS_FILE}"
    try:
        requirements = read_requirements()
    except DependencyError as exc:
        return CheckResult(status="fail", command=command, output=str(exc))

    problems = missing_dependencies(requirements)
    if not requirements and not problems:
        return CheckResult(status="not_configured", output="No third-party packages used.")
    return CheckResult(
        status="fail" if problems else "pass",
        command=command,
        output="\n".join(problems),
    )


def check_tests(approve: Callable[[], bool]) -> CheckResult:
    if not approve():
        return CheckResult(status="not_run", output="The human declined to run the tests.")

    try:
        run = run_tests()
    except (SandboxUnavailable, DependenciesNotReady, DependencyError) as exc:
        return CheckResult(status="not_run", output=str(exc))

    command = " ".join(run.command)
    if run.no_tests:
        return CheckResult(status="not_configured", command=command, output=run.output)
    return CheckResult(
        status="pass" if run.passed else "fail",
        command=command,
        output=run.output,
    )


def unexpected_changes(base: str) -> list[str]:
    # The Developer has no delete tool, so a deleted file means something
    # outside its tools touched the project.
    return [path for letter, path in changed_paths(base) if letter == "D"]


def run_validation(base: str, approve_tests: Callable[[], bool]) -> ValidationReport:
    return ValidationReport(
        build=check_syntax(),
        dependencies=check_dependencies(),
        tests=check_tests(approve_tests),
        lint=NOT_CONFIGURED,
        static_analysis=NOT_CONFIGURED,
        unexpected_files=unexpected_changes(base),
    )
