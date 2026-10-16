"""Let the human run the program for real before deciding on it.

A Python file or a page runs directly on this machine, deliberately NOT sandboxed:
a game or GUI needs the real screen, keyboard and sound. A component's run command
(npm start, cargo run, ...) can run in the sandbox with the terminal attached and
its ports opened, or on this machine if its tools are installed here. The human is
warned and confirms every time.
"""

import os
import subprocess
import sys
import webbrowser
from dataclasses import dataclass
from pathlib import Path

from orchestrator.console import ConsoleUI
from orchestrator.runlog import RunLog
from schemas.toolchain import Component
from tools import sandbox
from tools.checks import describe_target, run_in_component
from tools.dependencies import DependencyError, build_local_env, local_python, read_requirements
from tools.git import changed_paths, restore_paths, status
from tools.project import (
    ProjectAccessError,
    get_project_path,
    list_project_files,
    read_project_file,
    resolve_project_path,
)
from tools.sandbox import SuiteRunError
from tools.toolchain import ToolchainError, load_toolchain

MAX_CHOICES = 9
MAIN_GUARDS = ("__name__ == \"__main__\"", "__name__ == '__main__'")


@dataclass(frozen=True)
class Choice:
    label: str
    kind: str  # "python", "command" or "page"
    path: str = ""
    component: Component | None = None


def entry_points(base: str) -> list[str]:
    """Likely programs to run: files changed in this run, and files with a main guard."""
    changed = {path for letter, path in changed_paths(base) if letter != "D"}
    ranked = []
    for path in list_project_files():
        parts = Path(path).parts
        name = parts[-1]
        if (not path.endswith(".py") or name.startswith("test_")
                or name == "conftest.py" or "tests" in parts):
            continue
        try:
            source = read_project_file(Path(path))
        except (ProjectAccessError, OSError, ValueError):
            continue
        has_main = any(guard in source for guard in MAIN_GUARDS)
        if path in changed or has_main:
            ranked.append(((path not in changed, not has_main, path), path))
    return [path for _, path in sorted(ranked)][:MAX_CHOICES]


def _choices(base: str) -> list[Choice]:
    choices = [Choice(path, "python", path=path) for path in entry_points(base)]
    try:
        components = load_toolchain().components
    except (ToolchainError, OSError, ProjectAccessError):
        components = []
    for component in components:
        if component.run:
            choices.append(Choice(f"[{component.name}] {component.run}", "command",
                                  component=component))
    pages = sorted(path for path in list_project_files()
                   if path.endswith(".html") and Path(path).name == "index.html")
    choices += [Choice(f"open {path} in your browser", "page", path=path)
                for path in pages[:3]]
    return choices


def _pick(ui: ConsoleUI, base: str) -> Choice | None:
    options = _choices(base)
    if options:
        ui.say("\nWhat should run?")
        for number, option in enumerate(options, start=1):
            ui.say(f"  {number}. {option.label}")
        answer = ui.ask("Number or a path in the project (Enter = 1, c = cancel): ").strip()
        if answer.lower() == "c":
            return None
        answer = answer or "1"
    else:
        answer = ui.ask("Path of the file to run (Enter = cancel): ").strip()
        if not answer:
            return None

    if answer.isdigit():
        index = int(answer) - 1
        if 0 <= index < len(options):
            return options[index]
        ui.say(f"There is no option {answer}.")
        return None

    try:
        target = resolve_project_path(Path(answer))
    except ProjectAccessError as exc:
        ui.say(str(exc))
        return None
    if target.suffix not in {".py", ".html", ".htm"} or not target.is_file():
        ui.say(f"{answer} is not a Python file or a page in the project.")
        return None
    path = target.relative_to(get_project_path().resolve()).as_posix()
    return Choice(path, "python" if target.suffix == ".py" else "page", path=path)


def _interpreter(ui: ConsoleUI) -> Path | None:
    try:
        requirements = read_requirements()
    except DependencyError as exc:
        ui.say(f"Cannot use requirements.txt: {exc}")
        return None
    if not requirements:
        return Path(sys.executable)

    python = local_python(requirements)
    if python is not None:
        return python

    packages = ", ".join(req.line for req in requirements)
    where = ("only inside the Docker sandbox image" if sandbox.SANDBOX == "docker"
             else "not installed yet")
    if not ui.confirm(
        f"The program needs {packages}, which are {where}. Install them into a "
        f"local environment so you can try it? (pre-built packages only)",
        default=True,
    ):
        return None
    try:
        return build_local_env(requirements)
    except DependencyError as exc:
        ui.say(f"Install failed: {exc}")
        return None


def _changes() -> list[str]:
    return status().stdout.splitlines()[1:]


def _offer_restore(ui: ConsoleUI, log: RunLog, before: set[str]) -> None:
    new = [line for line in _changes() if line not in before]
    if not new:
        return

    ui.say("\nThe program changed files in the project:")
    for line in new:
        ui.say(f"  {line}")
    choice = ui.choose(
        "[r]estore them so they stay out of the result  [k]eep them  (Enter = restore): ",
        {"r": "restore", "k": "keep"},
        default="r",
    )
    if choice == "k":
        log.event("try_it_files", kept=new)
        return

    tracked = [line[3:] for line in new if not line.startswith("??")]
    created = [line[3:] for line in new if line.startswith("??")]
    restore_paths(tracked)
    for path in created:
        resolve_project_path(Path(path)).unlink(missing_ok=True)
    log.event("try_it_files", restored=new)
    ui.say("Restored.")


def _run_python(ui: ConsoleUI, log: RunLog, path: str, python: Path) -> int | None:
    ui.say(f"\nRunning {path}. Close it or press Ctrl+C to come back here.\n")
    log.event("try_it_started", file=path, python=str(python))
    try:
        done = subprocess.run(
            [str(python), path],
            cwd=get_project_path(),
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        return done.returncode
    except KeyboardInterrupt:
        return None


def _run_command(ui: ConsoleUI, log: RunLog, component: Component) -> int | None:
    command = component.run or ""
    ports = tuple(component.ports)
    if sandbox.SANDBOX == "docker":
        where = ui.choose(
            f"Run it [s]andboxed ({describe_target(component)}) or [h]ere on this machine "
            f"(needs its tools installed here)  (Enter = sandbox): ",
            {"s": "sandbox", "h": "here"},
            default="s",
        )
    else:
        where = "h"
    if where == "s" and not ports:
        answer = ui.ask("Port it serves on, to open from your browser (Enter = none): ").strip()
        if answer.isdigit() and 1 <= int(answer) <= 65535:
            ports = (int(answer),)
    if where == "s" and ports:
        ui.say(f"Open http://localhost:{ports[0]} once it is up. Serving a port gives the "
               f"container network access.")

    ui.say(f"\nRunning `{command}`. Press Ctrl+C to come back here.\n")
    log.event("try_it_started", command=command, component=component.name, where=where)
    if where == "s":
        try:
            run = run_in_component(component, command, writable=True, interactive=True,
                                   ports=ports)
        except (SuiteRunError, ValueError, OSError) as exc:
            ui.say(f"Could not start it: {exc}")
            return None
        return None if run.returncode == -1 else run.returncode
    try:
        done = subprocess.run(command, shell=True, cwd=resolve_project_path(Path(component.path)))
        return done.returncode
    except KeyboardInterrupt:
        return None


def try_program(ui: ConsoleUI, log: RunLog, base: str) -> None:
    choice = _pick(ui, base)
    if choice is None:
        return

    if choice.kind == "page":
        ui.say(f"\nOpening {choice.path} in your browser. Scripts on the page run "
               f"there with your browser's usual protections.")
        webbrowser.open(resolve_project_path(Path(choice.path)).as_uri())
        log.event("try_it_page", file=choice.path)
        return

    if choice.kind == "python" or sandbox.SANDBOX != "docker":
        ui.say(
            f"\nThis runs {choice.label} directly on this machine WITHOUT a sandbox, so it "
            f"can open windows and read your input - and anything else your account can do."
        )
    ui.say(f"Read the code first:  git -C \"{get_project_path()}\" diff {base[:8]}")
    if not ui.confirm(f"Run {choice.label} now?", default=True):
        return

    python = _interpreter(ui) if choice.kind == "python" else None
    if choice.kind == "python" and python is None:
        return

    before = set(_changes())
    if python is not None:
        exit_code = _run_python(ui, log, choice.path, python)
        name = choice.path
    else:
        exit_code = _run_command(ui, log, choice.component)
        name = choice.label
    ended = f"with exit code {exit_code}" if exit_code is not None else "(stopped)"
    ui.say(f"\n{name} ended {ended}.")
    log.event("try_it_finished", what=name, exit_code=exit_code)
    _offer_restore(ui, log, before)
