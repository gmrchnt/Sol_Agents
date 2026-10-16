import subprocess
from dataclasses import dataclass
from pathlib import Path
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
    result = _run_git("rev-parse", "--show-toplevel")
    if not result.ok:
        return False
    toplevel = Path(result.stdout.strip()).resolve()
    return toplevel == get_project_path().resolve()

def require_git_repository() -> None:
    if not is_git_repository():
        raise GitError(
            f"PROJECT_PATH must be the root of a Git repository: "
            f"{get_project_path()}"
        )
    
def status() -> GitResult:
    require_git_repository()

    return _run_git(
        "status",
        "--porcelain=v1",
        "--branch",
        "--untracked-files=all",
    )

def diff(
    base: str | None = "HEAD",
    exclude: tuple[str, ...] = (),
) -> GitResult:
    """`exclude` holds glob patterns of files to leave out, e.g. "**/package-lock.json"."""
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

    if exclude:
        args += ["--", ".", *(f":(glob,exclude){pattern}" for pattern in exclude)]

    return _run_git(*args)

def changed_paths(base: str) -> list[tuple[str, str]]:
    """(status letter, path) for every file that differs from `base`."""
    require_git_repository()
    if base.startswith("-"):
        raise GitError(f"Invalid base ref: {base!r}")

    result = _run_git("diff", "--name-status", "--no-renames", "--no-color", base)
    if not result.ok:
        raise GitError(f"Could not list changes since {base}: {result.stderr.strip()}")

    changes = []
    for line in result.stdout.splitlines():
        letter, _, path = line.partition("\t")
        if path:
            changes.append((letter, path))
    return changes


def untracked_files() -> set[str]:
    """New files git would add - ignored ones (node_modules, a .gitignore'd dist/) excluded."""
    require_git_repository()
    result = _must(_run_git("ls-files", "-z", "--others", "--exclude-standard"),
                   "list untracked files")
    return {path for path in result.stdout.split("\0") if path}


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


# --- Orchestrator-only operations -------------------------------------------
# These change the repository, so they are deliberately absent from every
# toolset: only the orchestrator branches, commits and merges.

SOL_IDENTITY = ("-c", "user.name=Sol_Agents", "-c", "user.email=sol-agents@localhost")


def _check_ref_name(name: str) -> None:
    if not name or name.startswith("-"):
        raise GitError(f"Invalid branch name: {name!r}")


def _must(result: GitResult, what: str) -> GitResult:
    if not result.ok:
        raise GitError(f"Could not {what}: {result.stderr.strip() or result.stdout.strip()}")
    return result


def current_branch() -> str:
    require_git_repository()
    result = _must(_run_git("rev-parse", "--abbrev-ref", "HEAD"), "read the current branch")
    branch = result.stdout.strip()
    if branch == "HEAD":
        raise GitError("The project is in detached HEAD state; check out a branch first.")
    return branch


def create_branch(name: str) -> None:
    """Create `name` at the current commit and switch to it."""
    require_git_repository()
    _check_ref_name(name)
    _must(_run_git("switch", "-c", name), f"create branch {name}")


def switch_branch(name: str) -> None:
    require_git_repository()
    _check_ref_name(name)
    _must(_run_git("switch", name), f"switch to {name}")


def commit_all(message: str) -> str | None:
    """Stage everything and commit it. Returns the new commit, or None if nothing changed."""
    require_git_repository()
    _must(_run_git("add", "-A"), "stage changes")
    if not _run_git("status", "--porcelain").stdout.strip():
        return None
    _must(_run_git(*SOL_IDENTITY, "commit", "-q", "-m", message), "commit")
    return current_commit()


def merge_fast_forward(branch: str) -> None:
    """Merge `branch` into the current branch only if no merge commit is needed."""
    require_git_repository()
    _check_ref_name(branch)
    _must(_run_git("merge", "--ff-only", branch), f"fast-forward to {branch}")


def restore_paths(paths: list[str]) -> None:
    """Put tracked files back exactly as they are in the last commit."""
    require_git_repository()
    if not paths:
        return
    _must(
        _run_git("restore", "--source=HEAD", "--staged", "--worktree", "--", *paths),
        "restore files",
    )


# Written to .git/info/exclude (this clone only, never committed) at the start of a
# run, so `git add -A` at a checkpoint can never sweep up installed packages.
EXCLUDE_MARKER = "# Sol_Agents: generated folders that never belong in a commit"
DEFAULT_EXCLUDES = [
    "node_modules/", "__pycache__/", "*.pyc", ".pytest_cache/", ".mypy_cache/",
    ".ruff_cache/", ".gradle/", ".next/", ".nuxt/", ".parcel-cache/", ".turbo/",
    ".build/", ".tox/", ".DS_Store",
]


def ensure_default_excludes() -> None:
    require_git_repository()
    where = _must(_run_git("rev-parse", "--git-path", "info/exclude"),
                  "find .git/info/exclude").stdout.strip()
    path = Path(where)
    if not path.is_absolute():
        path = get_project_path() / path
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    if EXCLUDE_MARKER in text:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    block = "\n".join([EXCLUDE_MARKER, *DEFAULT_EXCLUDES]) + "\n"
    path.write_text((text.rstrip("\n") + "\n\n" if text.strip() else "") + block,
                    encoding="utf-8")
