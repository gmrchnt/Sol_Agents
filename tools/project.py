from pathlib import Path
from config import PROJECT_PATH

class ProjectAccessError(Exception):
    """Raised when the project path is not accessible."""
    pass

MAX_FILE_SIZE = 200_000  

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

def list_project_files() -> list[str]:
    root_path = get_project_path()
    ignore_patterns = {".git", ".venv", "__pycache__"}

    files:list[str] = []
    for path in root_path.rglob("*"):
        if not path.is_file():
            continue
        relative_path = path.relative_to(root_path)
        if any(part in ignore_patterns for part in relative_path.parts):
            continue
        files.append(relative_path.as_posix())
    return sorted(files)

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


def write_project_file(relative_path:Path,content:str) -> str:
    file_path = resolve_project_path(relative_path)
    root = get_project_path().resolve()

    if file_path == root:
        raise ProjectAccessError("Cannot write to the project root itself.")

    parts = file_path.relative_to(root).parts
    if any(part.lower() in PROTECTED_PARTS for part in parts):
        raise ProjectAccessError(f"Writing inside .git is not allowed: {relative_path}")

    if file_path.is_dir():
        raise ProjectAccessError(f"{relative_path} is a directory.")

    data = content.encode("utf-8")
    if len(data) > MAX_FILE_SIZE:
        raise ValueError(
            f"Refusing to write {len(data)} bytes; maximum is {MAX_FILE_SIZE} bytes."
        )

    existed = file_path.exists()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    # bytes, so Windows doesn't silently rewrite \n as \r\n
    file_path.write_bytes(data)

    action = "updated" if existed else "created"
    return f"{action} {file_path.relative_to(root).as_posix()} ({len(data)} bytes)"

def search_project_files(query: str) -> list[dict[str, object]]:
    if not query.strip():
        return []

    query_lower = query.lower()
    results:list[dict[str, object]] = []
    for relative_path in list_project_files():
        try:
            content = read_project_file(relative_path)
        except (UnicodeDecodeError, ValueError):
            continue

        for line_number, line in enumerate(content.splitlines(), start=1):
            if query_lower in line.lower():
                results.append({
                    "file": relative_path,
                    "line": line_number,
                    "content": line.strip(),
                })
                if len(results) >= 100:
                    return results
    return results