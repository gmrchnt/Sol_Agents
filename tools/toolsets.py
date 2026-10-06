from pathlib import Path

from tools.project import (
    list_project_files,
    read_project_file,
    search_project_files,
    write_project_file,
)
from tools.testing import SuiteRun, run_tests

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
    """Everything the Reviewer can do, plus writing files and running tests."""

    def write_file(self, relative_path: str | Path, content: str) -> str:
        return write_project_file(Path(relative_path), content)

    def run_tests(self) -> SuiteRun:
        return run_tests()
