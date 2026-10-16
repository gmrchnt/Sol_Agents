import base64
from pathlib import Path

from schemas.toolchain import Toolchain
from tools.checks import (
    command_target,
    format_check_runs,
    pending_setups,
    run_checks,
    run_in_component,
    run_setup,
)
from tools.git import (
    GitResult,
    diff,
    recent_commits,
    status,
)
from tools.project import (
    delete_project_file,
    edit_project_file,
    get_project_path,
    list_project_files,
    move_project_file,
    read_project_file,
    resolve_project_path,
    search_project_files,
    write_project_bytes,
    write_project_file,
)
from tools import sandbox
from tools.review_tests import write_review_file, write_review_test
from tools.sandbox import CommandRun
from tools.testing import SuiteRun, install_dependencies, run_tests
from tools.toolchain import (
    default_images,
    describe_toolchain,
    read_explicit,
    load_toolchain,
    save_toolchain,
)

# Larger files are shown in parts; the model asks for the lines it needs.
READ_CHARS = 40_000
MAX_LISTED_CHANGES = 50


def _snapshot() -> dict[str, tuple[int, int]]:
    root = get_project_path().resolve()
    state = {}
    for path in list_project_files():
        try:
            stat = (root / path).stat()
        except OSError:
            continue
        state[path] = (stat.st_size, stat.st_mtime_ns)
    return state


def _describe_changes(before: dict, after: dict) -> str:
    lines = [f"  + {path}" for path in sorted(after.keys() - before.keys())]
    lines += [f"  ~ {path}" for path in sorted(after.keys() & before.keys())
              if after[path] != before[path]]
    lines += [f"  - {path}" for path in sorted(before.keys() - after.keys())]
    if not lines:
        return "No project files changed."
    shown = lines[:MAX_LISTED_CHANGES]
    more = f"\n  ... and {len(lines) - len(shown)} more" if len(lines) > len(shown) else ""
    return "Project files changed by the command:\n" + "\n".join(shown) + more


def _report(run: CommandRun) -> str:
    verdict = "stopped: took too long" if run.timed_out else f"exit code {run.returncode}"
    return f"{verdict}\n{run.output}".rstrip()


class ReadOnlyToolset:
    """Project inspection only. Nothing here can modify the project."""

    def __init__(self, base: str = "HEAD") -> None:
        # Diffs compare against the run's starting commit, so checkpoint
        # commits made during the run don't hide earlier changes.
        self.base = base

    def list_files(self, directory: str | None = None) -> list[str]:
        files = list_project_files()
        if directory in {None, "", "."}:
            return files
        folder = resolve_project_path(Path(directory)).relative_to(
            get_project_path().resolve()).as_posix()
        return [path for path in files if path.startswith(folder + "/")]

    def read_file(self, relative_path: str | Path, start_line: int | None = None,
                  end_line: int | None = None) -> str:
        try:
            text = read_project_file(Path(relative_path))
        except UnicodeDecodeError:
            size = resolve_project_path(Path(relative_path)).stat().st_size
            return f"{relative_path} is a binary file ({size} bytes); it cannot be shown as text."
        if start_line is None and end_line is None and len(text) <= READ_CHARS:
            return text

        lines = text.splitlines(keepends=True)
        start = max(1, start_line or 1)
        last = min(len(lines), end_line or len(lines))
        if start > len(lines):
            return f"{relative_path} has only {len(lines)} lines."
        shown: list[str] = []
        size = 0
        for number in range(start, last + 1):
            line = lines[number - 1]
            if shown and size + len(line) > READ_CHARS:
                last = number - 1
                break
            shown.append(line)
            size += len(line)
        return (f"[{relative_path}: lines {start}-{last} of {len(lines)}. "
                f"Pass start_line and end_line to read other parts.]\n" + "".join(shown))

    def search_files(self, query: str) -> list[dict[str, object]]:
        return search_project_files(query)

    def git_status(self) -> GitResult:
        return status()

    def git_diff(self) -> GitResult:
        return diff(self.base)

    def git_recent_commits(self, count: int = 10) -> GitResult:
        return recent_commits(count)

    def show_toolchain(self) -> str:
        explicit = read_explicit()
        toolchain = explicit if explicit is not None else load_toolchain()
        text = describe_toolchain(toolchain, explicit is not None)
        if toolchain.components:
            text += "\nDefault images for new components:\n" + default_images()
        return text


class ReviewerToolset(ReadOnlyToolset):
    """Read-only project access, plus a private area outside the project for adversarial tests."""

    def __init__(self, review_dir: Path, base: str = "HEAD") -> None:
        super().__init__(base)
        root = get_project_path().resolve()
        resolved = review_dir.resolve()
        if resolved == root or root in resolved.parents:
            raise ValueError(f"The review area must be outside the project: {review_dir}")
        self.review_dir = review_dir

    def write_review_test(self, filename: str, content: str) -> str:
        return write_review_test(self.review_dir, filename, content)

    def run_review_tests(self) -> SuiteRun:
        self.review_dir.mkdir(parents=True, exist_ok=True)
        return run_tests(target=self.review_dir)

    def write_review_file(self, path: str, content: str) -> str:
        return write_review_file(self.review_dir, path, content)

    def run_review_command(self, command: str, component: str | None = None,
                           timeout_seconds: int | None = None) -> str:
        """The project stays read-only; the review area is at $REVIEW_DIR."""
        self.review_dir.mkdir(parents=True, exist_ok=True)
        target = command_target(component)
        where = "/review" if sandbox.SANDBOX == "docker" else str(self.review_dir.resolve())
        target = target.model_copy(update={"env": {**target.env, "REVIEW_DIR": where}})
        run = run_in_component(target, command, review_dir=self.review_dir,
                               timeout=timeout_seconds)
        return _report(run)


class DeveloperToolset(ReadOnlyToolset):
    """Everything the Reviewer can inspect, plus changing files and running commands."""

    def write_file(self, relative_path: str | Path, content: str,
                   encoding: str = "utf-8") -> str:
        if encoding == "base64":
            return write_project_bytes(Path(relative_path),
                                       base64.b64decode(content, validate=True))
        if encoding.lower() not in {"utf-8", "utf8"}:
            raise ValueError("encoding must be utf-8 or base64")
        return write_project_file(Path(relative_path), content)

    def edit_file(self, relative_path: str, old_text: str, new_text: str) -> str:
        return edit_project_file(Path(relative_path), old_text, new_text)

    def delete_file(self, relative_path: str) -> str:
        return delete_project_file(Path(relative_path))

    def move_file(self, source: str, destination: str) -> str:
        return move_project_file(Path(source), Path(destination))

    def run_command(self, command: str, component: str | None = None,
                    image: str | None = None, workdir: str | None = None,
                    network: bool = False, timeout_seconds: int | None = None) -> str:
        target = command_target(component, image, workdir)
        before = _snapshot()
        run = run_in_component(target, command, writable=True, network=network,
                               timeout=timeout_seconds)
        return f"{_report(run)}\n\n{_describe_changes(before, _snapshot())}"

    def run_tests(self) -> SuiteRun | str:
        runs = run_checks("test")
        if len(runs) == 1 and isinstance(runs[0].result, SuiteRun):
            return runs[0].result
        return (format_check_runs(runs)
                or "No component has tests yet. Add them, and a test command if Sol "
                   "does not detect it (see show_toolchain).")

    def install_dependencies(self, allow_source_builds: bool = False) -> str:
        reports = [install_dependencies(allow_source_builds)]
        for component in pending_setups():
            run = run_setup(component)
            reports.append(f"[{component.name}] {' && '.join(component.setup)}: {_report(run)}")
        return "\n\n".join(reports)

    def set_toolchain(self, toolchain: dict) -> str:
        chosen = Toolchain.model_validate(toolchain)
        save_toolchain(chosen)
        return "Saved.\n" + describe_toolchain(chosen, explicit=True)
