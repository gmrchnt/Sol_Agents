"""Let the human run the program for real before deciding on it.

This is deliberately NOT sandboxed: a game or GUI needs the real screen,
keyboard and sound. The human is warned and confirms every time.
"""

import os
import subprocess
import sys
from pathlib import Path

from orchestrator.console import ConsoleUI
from orchestrator.runlog import RunLog
from tools import testing
from tools.dependencies import DependencyError, build_local_env, local_python, read_requirements
from tools.git import changed_paths, restore_paths, status
from tools.project import (
    ProjectAccessError,
    get_project_path,
    list_project_files,
    read_project_file,
    resolve_project_path,
)

MAX_CHOICES = 9
MAIN_GUARDS = ("__name__ == \"__main__\"", "__name__ == '__main__'")


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


def _pick_file(ui: ConsoleUI, base: str) -> str | None:
    options = entry_points(base)
    if options:
        ui.say("\nWhich file should run?")
        for number, path in enumerate(options, start=1):
            ui.say(f"  {number}. {path}")
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
    if target.suffix != ".py" or not target.is_file():
        ui.say(f"{answer} is not a Python file in the project.")
        return None
    return target.relative_to(get_project_path().resolve()).as_posix()


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
    where = ("only inside the Docker sandbox image" if testing.SANDBOX == "docker"
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


def try_program(ui: ConsoleUI, log: RunLog, base: str) -> None:
    path = _pick_file(ui, base)
    if path is None:
        return

    ui.say(
        f"\nThis runs {path} directly on this machine WITHOUT a sandbox, so it can "
        f"open windows and read your input - and anything else your account can do."
    )
    ui.say(f"Read the code first:  git -C \"{get_project_path()}\" diff {base[:8]}")
    if not ui.confirm(f"Run {path} now?", default=True):
        return

    python = _interpreter(ui)
    if python is None:
        return

    before = set(_changes())
    ui.say(f"\nRunning {path}. Close it or press Ctrl+C to come back here.\n")
    log.event("try_it_started", file=path, python=str(python))
    try:
        done = subprocess.run(
            [str(python), path],
            cwd=get_project_path(),
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        exit_code: int | None = done.returncode
    except KeyboardInterrupt:
        exit_code = None

    ended = f"with exit code {exit_code}" if exit_code is not None else "(stopped with Ctrl+C)"
    ui.say(f"\n{path} ended {ended}.")
    log.event("try_it_finished", file=path, exit_code=exit_code)
    _offer_restore(ui, log, before)
