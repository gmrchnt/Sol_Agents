from pathlib import Path

import pytest

import tools.project
from tools.testing import run_tests


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(tools.project, "PROJECT_PATH", tmp_path)
    return tmp_path


def test_passing_suite(project: Path) -> None:
    (project / "test_ok.py").write_text(
        "def test_ok():\n    assert 1 + 1 == 2\n", encoding="utf-8"
    )
    result = run_tests()
    assert result.passed
    assert "1 passed" in result.output


def test_failing_suite(project: Path) -> None:
    (project / "test_bad.py").write_text(
        "def test_bad():\n    assert 1 == 2\n", encoding="utf-8"
    )
    result = run_tests()
    assert not result.passed
    assert "1 failed" in result.output


def test_runs_inside_project_root(project: Path) -> None:
    (project / "test_where.py").write_text(
        "import os\n\n\ndef test_where():\n    assert os.path.exists('test_where.py')\n",
        encoding="utf-8",
    )
    assert run_tests().passed


def test_command_is_fixed(project: Path) -> None:
    (project / "test_ok.py").write_text("def test_ok():\n    pass\n", encoding="utf-8")
    assert run_tests().command[1:4] == ["-m", "pytest", "-q"]
