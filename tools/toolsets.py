from pathlib import Path

from tools.git import (
    GitResult,
    diff,
    recent_commits,
    status,
)
from tools.project import (
    get_project_path,
    list_project_files,
    read_project_file,
    search_project_files,
    write_project_file,
)
from tools.review_tests import write_review_test
from tools.testing import SuiteRun, run_tests


class ReadOnlyToolset:
    """Project inspection only. Nothing here can modify the project."""

    def __init__(self, base: str = "HEAD") -> None:
        # Diffs compare against the run's starting commit, so checkpoint
        # commits made during the run don't hide earlier changes.
        self.base = base

    def list_files(self) -> list[str]:
        return list_project_files()

    def read_file(self, relative_path: str | Path) -> str:
        return read_project_file(Path(relative_path))

    def search_files(self, query: str) -> list[dict[str, object]]:
        return search_project_files(query)

    def git_status(self) -> GitResult:
        return status()

    def git_diff(self) -> GitResult:
        return diff(self.base)

    def git_recent_commits(self, count: int = 10) -> GitResult:
        return recent_commits(count)


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


class DeveloperToolset(ReadOnlyToolset):
    """Everything the Reviewer can inspect, plus writing files and running tests."""

    def write_file(self, relative_path: str | Path, content: str) -> str:
        return write_project_file(Path(relative_path), content)

    def run_tests(self) -> SuiteRun:
        return run_tests()
