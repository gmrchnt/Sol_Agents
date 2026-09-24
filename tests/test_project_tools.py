import subprocess
import sys
from pathlib import Path

import pytest

import tools.project
from tools.project import (
    ProjectAccessError,
    get_project_path,
    list_project_files,
    read_project_file,
    resolve_project_path,
    search_project_files,
)


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "calculator.py").write_text(
        "def add(a, b):\n    return a + b\n", encoding="utf-8"
    )
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "util.py").write_text("VALUE = 1\n", encoding="utf-8")

    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("[core]\n", encoding="utf-8")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "stale.pyc").write_bytes(b"\x00\x01")

    monkeypatch.setattr(tools.project, "PROJECT_PATH", tmp_path)
    return tmp_path


ESCAPES = [
    "../outside.txt",
    "../../outside.txt",
    "pkg/../../outside.txt",
    "./../outside.txt",
    "pkg/../pkg/../../outside.txt",
]


@pytest.mark.parametrize("escape", ESCAPES)
def test_resolve_rejects_traversal_out_of_root(project: Path, escape: str) -> None:
    with pytest.raises(ProjectAccessError):
        resolve_project_path(Path(escape))


def test_resolve_rejects_absolute_path_outside_root(
    project: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    outside = tmp_path_factory.mktemp("outside") / "secret.txt"
    outside.write_text("secret\n", encoding="utf-8")
    with pytest.raises(ProjectAccessError):
        resolve_project_path(Path(str(outside)))


def _link_dir(link: Path, target: Path) -> bool:
    """Link a directory by whatever means the platform allows."""
    try:
        link.symlink_to(target, target_is_directory=True)
        return True
    except (OSError, NotImplementedError):
        pass
    if sys.platform == "win32":
        # Junctions need no elevated privileges, unlike symlinks
        done = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True,
        )
        return done.returncode == 0
    return False


def test_resolve_rejects_link_escaping_root(
    project: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    outside_dir = tmp_path_factory.mktemp("outside")
    (outside_dir / "secret.txt").write_text("secret\n", encoding="utf-8")

    if not _link_dir(project / "link", outside_dir):
        pytest.skip("this platform allows neither symlinks nor junctions here")

    with pytest.raises(ProjectAccessError):
        resolve_project_path(Path("link/secret.txt"))


def test_read_file_cannot_escape_root_through_link(
    project: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    outside_dir = tmp_path_factory.mktemp("outside")
    (outside_dir / "secret.txt").write_text("secret\n", encoding="utf-8")

    if not _link_dir(project / "link", outside_dir):
        pytest.skip("this platform allows neither symlinks nor junctions here")

    with pytest.raises(ProjectAccessError):
        read_project_file(Path("link/secret.txt"))


def test_resolve_allows_project_root_itself(project: Path) -> None:
    assert resolve_project_path(Path(".")) == project.resolve()


def test_resolve_allows_nested_path(project: Path) -> None:
    expected = (project / "pkg" / "util.py").resolve()
    assert resolve_project_path(Path("pkg/util.py")) == expected


def test_resolve_allows_path_that_does_not_exist_yet(project: Path) -> None:
    expected = (project / "new" / "file.py").resolve()
    assert resolve_project_path(Path("new/file.py")) == expected


def test_get_project_path_returns_configured_root(project: Path) -> None:
    assert get_project_path() == project


def test_list_files_excludes_vcs_and_cache_dirs(project: Path) -> None:
    assert list_project_files() == ["calculator.py", "pkg/util.py"]


def test_read_file_returns_contents(project: Path) -> None:
    assert read_project_file(Path("calculator.py")).startswith("def add(a, b):")


def test_read_file_raises_for_missing_file(project: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_project_file(Path("nope.py"))


def test_read_file_raises_for_directory(project: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_project_file(Path("pkg"))


def test_read_file_raises_for_oversized_file(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (project / "big.txt").write_text("x" * 100, encoding="utf-8")
    monkeypatch.setattr(tools.project, "MAX_FILE_SIZE", 10)
    with pytest.raises(ValueError):
        read_project_file(Path("big.txt"))


def test_read_file_cannot_escape_root(project: Path) -> None:
    with pytest.raises(ProjectAccessError):
        read_project_file(Path("../outside.txt"))


def test_search_finds_matches_with_locations(project: Path) -> None:
    results = search_project_files("def add")
    assert results == [{"file": "calculator.py", "line": 1, "content": "def add(a, b):"}]


def test_search_is_case_insensitive(project: Path) -> None:
    assert search_project_files("DEF ADD")


def test_search_returns_empty_for_blank_query(project: Path) -> None:
    assert search_project_files("   ") == []


def test_search_skips_undecodable_files(project: Path) -> None:
    (project / "blob.bin").write_bytes(b"\xff\xfe\x00binary")
    assert all(r["file"] != "blob.bin" for r in search_project_files("binary"))
