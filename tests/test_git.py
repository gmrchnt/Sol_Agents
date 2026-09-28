import subprocess
from pathlib import Path

import pytest

import tools.project
from tools import git
from tools.git import GitError


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", *args],
        check=True,
        capture_output=True,
    )


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    _git(tmp_path, "init", "-q")
    (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")
    _git(tmp_path, "add", "app.py")
    _git(tmp_path, "commit", "-qm", "initial")
    monkeypatch.setattr(tools.project, "PROJECT_PATH", tmp_path)
    return tmp_path


def test_repo_root_is_accepted(repo: Path) -> None:
    assert git.is_git_repository()


def test_plain_folder_is_rejected(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path_factory.mktemp("plain")
    monkeypatch.setattr(tools.project, "PROJECT_PATH", folder)
    assert not git.is_git_repository()
    with pytest.raises(GitError):
        git.status()


def test_subdirectory_of_a_repo_is_rejected(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (repo / "sub").mkdir()
    monkeypatch.setattr(tools.project, "PROJECT_PATH", repo / "sub")
    assert not git.is_git_repository()
    with pytest.raises(GitError):
        git.diff()


def test_status_lists_modified_file(repo: Path) -> None:
    (repo / "app.py").write_text("x = 2\n", encoding="utf-8")
    assert " M app.py" in git.status().stdout


def test_diff_shows_changed_lines(repo: Path) -> None:
    (repo / "app.py").write_text("x = 2\n", encoding="utf-8")
    out = git.diff().stdout
    assert "-x = 1" in out
    assert "+x = 2" in out


def test_diff_against_head_includes_staged_changes(repo: Path) -> None:
    (repo / "app.py").write_text("x = 2\n", encoding="utf-8")
    _git(repo, "add", "app.py")
    assert "+x = 2" in git.diff().stdout
    assert git.diff(None).stdout == ""


def test_diff_rejects_base_that_looks_like_an_option(repo: Path) -> None:
    with pytest.raises(GitError):
        git.diff("--upload-pack=evil")


@pytest.mark.parametrize("count", [0, 101])
def test_recent_commits_rejects_out_of_range_count(repo: Path, count: int) -> None:
    with pytest.raises(GitError):
        git.recent_commits(count)


def test_recent_commits_lists_history(repo: Path) -> None:
    assert "initial" in git.recent_commits(5).stdout


def test_current_commit_is_a_full_sha(repo: Path) -> None:
    sha = git.current_commit()
    assert len(sha) == 40
    assert all(c in "0123456789abcdef" for c in sha)


def test_result_records_target_repository(repo: Path) -> None:
    result = git.status()
    assert result.command[:4] == ["git", "--no-pager", "-C", str(repo)]
    assert result.ok


def test_truncation_is_flagged_per_stream(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (repo / "app.py").write_text("x = 2\n" * 200, encoding="utf-8")
    monkeypatch.setattr(git, "MAX_OUTPUT_BYTES", 300)
    result = git.diff()
    assert result.stdout_truncated
    assert not result.stderr_truncated
    assert "output truncated" in result.stdout