import subprocess
from pathlib import Path

import config
from config import PROJECT_PATH

class ProjectAccessError(Exception):
    """Raised when the project path is not accessible."""
    pass

MAX_FILE_SIZE = config.MAX_FILE_BYTES
MAX_SEARCH_HITS = 100
MAX_SEARCH_LINE = 300
LISTING_TIMEOUT = 30
# Skipped even when git cannot tell us what is ignored.
SKIPPED_DIRS = {".git", ".venv", "__pycache__", "node_modules"}

def get_project_path() -> Path:
    return PROJECT_PATH

def resolve_project_path(relative_path: Path) -> Path:
    root_path = get_project_path().resolve()
    requested_path = (root_path / relative_path).resolve()
    if requested_path == root_path:
        return requested_path

    if root_path not in requested_path.parents:
        raise ProjectAccessError(
            f"Access to {requested_path} is denied. "
            f"Path must be within the project root: {root_path}"
        )

    return requested_path


def _git_listing(root: Path) -> list[str] | None:
    """Tracked files plus untracked ones git would add - so ignored dependency and
    build folders (node_modules, target, dist) never flood the listing."""
    try:
        done = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", "--cached", "--others",
             "--exclude-standard"],
            capture_output=True, timeout=LISTING_TIMEOUT,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if done.returncode != 0:
        return None
    paths = {path for path in done.stdout.decode("utf-8", "replace").split("\0") if path}
    return [path for path in paths if (root / path).is_file()]


def list_project_files() -> list[str]:
    root_path = get_project_path()
    if not root_path.is_dir():
        raise ProjectAccessError(f"PROJECT_PATH is not a folder: {root_path}")

    listed = _git_listing(root_path)
    if listed is None:
        listed = [
            path.relative_to(root_path).as_posix()
            for path in root_path.rglob("*") if path.is_file()
        ]
    return sorted(
        path for path in listed
        if not any(part in SKIPPED_DIRS for part in Path(path).parts)
    )

def read_project_file(relative_path: Path) -> str:
    file_path = resolve_project_path(relative_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    size = file_path.stat().st_size

    if size > MAX_FILE_SIZE:
        raise ValueError(
            f"File {file_path} is too large ({size} bytes). "
            f"Maximum allowed size is {MAX_FILE_SIZE} bytes."
        )

    return file_path.read_text(
        encoding="utf-8"
    )

PROTECTED_PARTS = {".git"}
# The project document belongs to the orchestrator, not the Developer.
SOL_DIR = ".sol"


def _writable_path(relative_path: Path) -> Path:
    """Where a Developer change may go: inside the project, never .git or .sol."""
    file_path = resolve_project_path(relative_path)
    root = get_project_path().resolve()

    if file_path == root:
        raise ProjectAccessError("Cannot write to the project root itself.")

    parts = file_path.relative_to(root).parts
    if any(part.lower() in PROTECTED_PARTS for part in parts):
        raise ProjectAccessError(f"Writing inside .git is not allowed: {relative_path}")
    if parts[0].lower() == SOL_DIR:
        raise ProjectAccessError(
            f"{SOL_DIR}/ holds the orchestrator's project files; you cannot write there."
        )

    if file_path.is_dir():
        raise ProjectAccessError(f"{relative_path} is a directory.")
    return file_path


def _relative(file_path: Path) -> str:
    return file_path.relative_to(get_project_path().resolve()).as_posix()


def write_project_bytes(relative_path: Path, data: bytes) -> str:
    file_path = _writable_path(relative_path)
    if len(data) > MAX_FILE_SIZE:
        raise ValueError(
            f"Refusing to write {len(data)} bytes; maximum is {MAX_FILE_SIZE} bytes."
        )

    existed = file_path.exists()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_bytes(data)

    action = "updated" if existed else "created"
    return f"{action} {_relative(file_path)} ({len(data)} bytes)"


def write_project_file(relative_path:Path,content:str) -> str:
    # bytes, so Windows doesn't silently rewrite \n as \r\n
    return write_project_bytes(relative_path, content.encode("utf-8"))


def edit_project_file(relative_path: Path, old_text: str, new_text: str) -> str:
    """Replace the one place `old_text` occurs, leaving the rest of the file alone."""
    file_path = _writable_path(relative_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {relative_path}")
    if not old_text:
        raise ValueError("old_text is empty; use write_file to create a file.")

    # Bytes, so the file's own line endings survive.
    text = file_path.read_bytes().decode("utf-8")
    if "\r\n" in text:
        old_text = old_text.replace("\r\n", "\n").replace("\n", "\r\n")
        new_text = new_text.replace("\r\n", "\n").replace("\n", "\r\n")

    count = text.count(old_text)
    if count == 0:
        raise ValueError(
            f"old_text was not found in {relative_path}. Read the file again and copy "
            f"the text exactly, including indentation."
        )
    if count > 1:
        raise ValueError(
            f"old_text appears {count} times in {relative_path}; include more of the "
            f"surrounding lines so it matches only once."
        )
    write_project_bytes(relative_path, text.replace(old_text, new_text).encode("utf-8"))
    return f"edited {_relative(file_path)}"


def delete_project_file(relative_path: Path) -> str:
    file_path = _writable_path(relative_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {relative_path}")
    file_path.unlink()
    return f"deleted {_relative(file_path)}"


def move_project_file(source: Path, destination: Path) -> str:
    source_path = _writable_path(source)
    target_path = _writable_path(destination)
    if not source_path.is_file():
        raise FileNotFoundError(f"File not found: {source}")
    if target_path.exists():
        raise ValueError(f"{destination} already exists; delete it first if it should go.")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.rename(target_path)
    return f"moved {_relative(source_path)} -> {_relative(target_path)}"

def search_project_files(query: str) -> list[dict[str, object]]:
    if not query.strip():
        return []

    query_lower = query.lower()
    results:list[dict[str, object]] = []
    for relative_path in list_project_files():
        try:
            content = read_project_file(relative_path)
        except (ValueError, OSError, ProjectAccessError):
            # Undecodable, too large, or gone since the listing.
            continue

        for line_number, line in enumerate(content.splitlines(), start=1):
            if query_lower in line.lower():
                text = line.strip()
                if len(text) > MAX_SEARCH_LINE:
                    text = text[:MAX_SEARCH_LINE] + f"... [{len(text)} chars]"
                results.append({
                    "file": relative_path,
                    "line": line_number,
                    "content": text,
                })
                if len(results) >= MAX_SEARCH_HITS:
                    return results
    return results
