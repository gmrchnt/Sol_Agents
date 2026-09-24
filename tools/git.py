import subprocess
from dataclasses import dataclass

from tools.project import get_project_path


GIT_TIMEOUT = 30
MAX_OUTPUT_BYTES = 200_000


class GitError(Exception):
    """Raised when Git cannot be run against the project."""


@dataclass(frozen=True)
class GitResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    stdout_truncated: bool = False
    stderr_truncated: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    @property
    def truncated(self) -> bool:
        return self.stdout_truncated or self.stderr_truncated


def _truncate(text: str) -> tuple[str, bool]:
    encoded = text.encode("utf-8")

    if len(encoded) <= MAX_OUTPUT_BYTES:
        return text, False

    clipped = encoded[:MAX_OUTPUT_BYTES].decode(
        "utf-8",
        errors="ignore",
    )

    omitted = len(encoded) - MAX_OUTPUT_BYTES

    return (
        f"{clipped}\n\n"
        f"... [output truncated, {omitted} bytes omitted]",
        True,
    )


def _run_git(*args: str) -> GitResult:
    command = [
        "git",
        "--no-pager",
        "-C",
        str(get_project_path()),
        *args,
    ]

    try:
        done = subprocess.run(
            command,
            capture_output=True,
            timeout=GIT_TIMEOUT,
            encoding="utf-8",
            errors="replace",
        )

    except FileNotFoundError as exc:
        raise GitError(
            "Git executable not found on PATH."
        ) from exc

    except subprocess.TimeoutExpired as exc:
        raise GitError(
            f"Git timed out after {GIT_TIMEOUT}s: "
            f"{' '.join(command)}"
        ) from exc

    stdout, stdout_truncated = _truncate(
        done.stdout or ""
    )

    stderr, stderr_truncated = _truncate(
        done.stderr or ""
    )

    return GitResult(
        command=command,
        returncode=done.returncode,
        stdout=stdout,
        stderr=stderr,
        stdout_truncated=stdout_truncated,
        stderr_truncated=stderr_truncated,
    )


def is_git_repository() -> bool:
    result = _run_git(
        "rev-parse",
        "--is-inside-work-tree",
    )

    return (
        result.ok
        and result.stdout.strip() == "true"
    )


def require_git_repository() -> None:
    if not is_git_repository():
        raise GitError(
            f"Not a Git repository: "
            f"{get_project_path()}"
        )


def status() -> GitResult:
    require_git_repository()

    return _run_git(
        "status",
        "--porcelain=v1",
        "--branch",
    )


def diff(
    base: str | None = "HEAD",
) -> GitResult:
    require_git_repository()

    args = [
        "diff",
        "--no-color",
        "--no-ext-diff",
    ]

    if base is not None:
        if base.startswith("-"):
            raise GitError(
                f"Invalid base ref: {base!r}"
            )

        args.append(base)

    return _run_git(*args)


def recent_commits(
    count: int = 10,
) -> GitResult:
    if count < 1 or count > 100:
        raise GitError(
            "count must be between 1 and 100"
        )

    require_git_repository()

    return _run_git(
        "log",
        f"-{count}",
        "--oneline",
        "--no-color",
        "--no-decorate",
    )


def current_commit() -> str:
    require_git_repository()

    result = _run_git(
        "rev-parse",
        "HEAD",
    )

    if not result.ok:
        raise GitError(
            f"Could not read HEAD: "
            f"{result.stderr.strip()}"
        )

    return result.stdout.strip()
