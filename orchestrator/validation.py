import ast
from pathlib import Path
from typing import Callable

from schemas.toolchain import Component, Toolchain
from schemas.validation import CheckResult, ValidationReport
from tools.checks import command_for, is_default_pytest, run_in_component
from tools.toolchain import load_toolchain
from tools.git import changed_paths
from tools.project import (
    ProjectAccessError,
    get_project_path,
    list_project_files,
    read_project_file,
)
from tools.dependencies import (
    REQUIREMENTS_FILE,
    DependencyError,
    missing_dependencies,
    read_requirements,
)
from tools.testing import (
    DependenciesNotReady,
    SandboxUnavailable,
    SuiteRunError,
    import_modules,
    run_tests,
)

# Code at module level that is fine to run just by importing the module.
IMPORT_SAFE = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef,
               ast.ClassDef, ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Try)
NOT_MODULES = {"setup.py", "conftest.py", "__main__.py"}
# Output of package managers and builds; never part of a commit.
GENERATED_DIRS = {"node_modules", "__pycache__", ".venv", "venv", ".pytest_cache",
                  ".mypy_cache", ".gradle", ".next", ".nuxt", ".parcel-cache", ".turbo",
                  ".tox", ".build"}
LARGE_FILE_BYTES = 20_000_000

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


def check_dependencies(skip: tuple[str, ...] = ()) -> CheckResult:
    """Every third-party import must be declared, pinned, in requirements.txt.

    `skip` names folders that are separate components with their own manifests."""
    command = f"imports in .py files vs {REQUIREMENTS_FILE}"
    try:
        requirements = read_requirements()
    except DependencyError as exc:
        return CheckResult(status="fail", command=command, output=str(exc))

    problems = missing_dependencies(requirements, skip)
    if not requirements and not problems:
        return CheckResult(status="not_configured", output="No third-party packages used.")
    return CheckResult(
        status="fail" if problems else "pass",
        command=command,
        output="\n".join(problems),
    )


def _is_main_guard(node: ast.stmt) -> bool:
    return isinstance(node, ast.If) and "__main__" in ast.unparse(node.test)


def _waits_for_input(node: ast.AST) -> bool:
    return any(
        isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
        and call.func.id == "input"
        for call in ast.walk(node)
    )


def _safe_to_import(tree: ast.Module) -> bool:
    """A plain script runs its whole program when imported; leave those to the tests."""
    for node in tree.body:
        if _is_main_guard(node):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue  # docstring
        if not isinstance(node, IMPORT_SAFE):
            return False
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
                and _waits_for_input(node):
            return False
    return True


def importable_modules(skip: tuple[str, ...] = ()) -> list[str]:
    """Dotted names of the project's library modules, skipping tests and scripts."""
    modules = []
    for path in list_project_files():
        file = Path(path)
        if file.suffix != ".py" or file.name in NOT_MODULES:
            continue
        if any(path.startswith(folder + "/") for folder in skip):
            continue
        if file.name.startswith("test_") or file.name.endswith("_test.py"):
            continue
        parts = list(file.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        if not parts or not all(part.isidentifier() for part in parts):
            continue  # tests/, .sol/, folders with dashes...
        if "tests" in parts:
            continue
        try:
            tree = ast.parse(read_project_file(file), filename=path)
        except (SyntaxError, ValueError):
            continue  # check_syntax reports these
        if _safe_to_import(tree):
            modules.append(".".join(parts))
    return sorted(modules)


def check_imports(approve: Callable[[], bool], skip: tuple[str, ...] = ()) -> CheckResult:
    """Backbone: run the code far enough to catch errors parsing cannot."""
    modules = importable_modules(skip)
    if not modules:
        return CheckResult(status="not_configured", output="No importable project modules.")
    if not approve():
        return CheckResult(status="not_run", output="The human declined to run project code.")

    command = f"import {', '.join(modules)}"
    try:
        run = import_modules(modules)
    except (SandboxUnavailable, DependenciesNotReady, DependencyError) as exc:
        return CheckResult(status="not_run", command=command, output=str(exc))
    except SuiteRunError as exc:
        # Importing should be instant; hanging is itself the defect.
        return CheckResult(status="fail", command=command, output=str(exc))
    return CheckResult(
        status="pass" if run.passed else "fail",
        command=command,
        output=run.output,
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


def check_command(component: Component, kind: str, approve: Callable[[], bool]) -> CheckResult:
    """One of the component's own commands - its build, tests, linter."""
    command = command_for(component, kind)
    if command is None:
        return NOT_CONFIGURED
    if not approve():
        return CheckResult(status="not_run", command=command,
                           output="The human declined to run project code.")
    network = kind == "test" and component.test_network
    try:
        run = run_in_component(component, command, network=network)
    except (SandboxUnavailable, DependenciesNotReady, DependencyError) as exc:
        return CheckResult(status="not_run", command=command, output=str(exc))
    except (SuiteRunError, OSError, ProjectAccessError) as exc:
        # The command could not start: a broken Dockerfile, a missing folder.
        return CheckResult(status="fail", command=command, output=str(exc))
    output = run.output if not run.timed_out else f"{run.output}\n(stopped: took too long)"
    return CheckResult(status="pass" if run.passed else "fail", command=command,
                       output=output.strip())


def combine(results: list[tuple[str, CheckResult]]) -> CheckResult:
    """One check across components: any failure fails it; anything unrun leaves it unrun."""
    configured = [(name, result) for name, result in results
                  if result.status != "not_configured"]
    if not configured:
        return results[0][1] if results else NOT_CONFIGURED
    if len(configured) == 1:
        return configured[0][1]
    statuses = {result.status for _, result in configured}
    status = "fail" if "fail" in statuses else "not_run" if "not_run" in statuses else "pass"
    return CheckResult(
        status=status,
        command="; ".join(f"[{name}] {result.command}" for name, result in configured),
        output="\n\n".join(f"[{name}] {result.status}\n{result.output}".rstrip()
                           for name, result in configured),
    )


def deleted_files(base: str) -> list[str]:
    return [path for letter, path in changed_paths(base) if letter == "D"]


def unexpected_changes(base: str) -> list[str]:
    """Added files that should never be committed: package and build output, or huge files."""
    root = get_project_path().resolve()
    problems = []
    for letter, path in changed_paths(base):
        if letter == "D":
            continue
        if any(part in GENERATED_DIRS for part in path.split("/")[:-1]):
            problems.append(f"{path} (generated output - add its folder to .gitignore)")
            continue
        try:
            size = (root / path).stat().st_size
        except OSError:
            continue
        if size > LARGE_FILE_BYTES:
            problems.append(f"{path} ({size} bytes - too large to commit)")
    return problems


def run_validation(base: str, approve_tests: Callable[[], bool],
                   toolchain: Toolchain | None = None) -> ValidationReport:
    """`approve_tests` is asked once and covers every check that runs project code."""
    if toolchain is None:
        toolchain = load_toolchain()
    answer: list[bool] = []

    def approve_once() -> bool:
        if not answer:
            answer.append(approve_tests())
        return answer[0]

    components = toolchain.components
    elsewhere = tuple(c.path for c in components if c.path != ".")
    root_python = any(is_default_pytest(c) for c in components)

    def per_component(kind: str) -> list[tuple[str, CheckResult]]:
        results = []
        for component in components:
            if kind == "test" and is_default_pytest(component):
                results.append((component.name, check_tests(approve_once)))
            elif command_for(component, kind) is not None:
                results.append((component.name, check_command(component, kind, approve_once)))
        return results

    return ValidationReport(
        build=combine([("python syntax", check_syntax())] + per_component("build")),
        dependencies=check_dependencies(elsewhere),
        imports=check_imports(approve_once, elsewhere) if root_python else CheckResult(
            status="not_configured", output="No root Python component."),
        tests=combine(per_component("test")),
        lint=combine(per_component("lint")),
        static_analysis=combine(per_component("static_analysis")),
        unexpected_files=unexpected_changes(base),
        deleted_files=deleted_files(base),
    )
