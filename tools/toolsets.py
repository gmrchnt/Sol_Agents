from pathlib import Path

from tools.project import (
    list_project_files,
    read_project_file,
    search_project_files,
)

from tools.git import (
    status,
    diff,
    recent_commits,
    GitResult,
)


class ReadOnlyToolset:
    """Project inspection only. Nothing here can modify the project."""

    def list_files(self) -> list[str]:
        return list_project_files()

    def read_file(self, relative_path: str | Path) -> str:
        return read_project_file(Path(relative_path))

    def search_files(self, query: str) -> list[dict[str, object]]:
        return search_project_files(query)

    def git_status(self) -> GitResult:
        return status()

    def git_diff(self,base: str | None = "HEAD") -> GitResult:
        return diff(base)

    def git_recent_commits(self,count: int = 10) -> GitResult:
        return recent_commits(count)



class DeveloperToolset(ReadOnlyToolset):
    """Everything the Reviewer can do. Write access is added here in M3."""
