import subprocess
from pathlib import Path

import pytest

import tools.project
from orchestrator.tool_loop import dispatch, tool_specs_for
from orchestrator.validation import check_syntax, check_tests, run_validation, unexpected_changes
from tools.git import changed_paths
from tools.review_tests import write_review_test
from tools.toolsets import DeveloperToolset, ReadOnlyToolset, ReviewerToolset


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", *args],
        check=True, capture_output=True, text=True,
    )
    return done.stdout.strip()


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    (root / "calculator.py").write_text(
        "def add(a, b):\n    return a + b\n", encoding="utf-8"
    )
    monkeypatch.setattr(tools.project, "PROJECT_PATH", root)
    return root


@pytest.fixture
def repo(project: Path) -> Path:
    _git(project, "init", "-q")
    _git(project, "add", "-A")
    _git(project, "commit", "-qm", "initial")
    return project


@pytest.fixture
def review_dir(tmp_path: Path) -> Path:
    return tmp_path / "review"


def _approve() -> bool:
    return True


def _decline() -> bool:
    return False


# --- review area -----------------------------------------------------------

@pytest.mark.parametrize("name", ["../escape.py", "sub/test_x.py", "helper.py",
                                  "test_x.txt", "test_.py", "TEST_x.py"])
def test_review_test_names_are_restricted(review_dir: Path, name: str) -> None:
    with pytest.raises(ValueError):
        write_review_test(review_dir, name, "x = 1\n")


def test_review_test_is_written_to_review_area(review_dir: Path) -> None:
    write_review_test(review_dir, "test_edge.py", "def test_x():\n    pass\n")
    assert (review_dir / "test_edge.py").is_file()


def test_review_test_size_is_capped(review_dir: Path) -> None:
    with pytest.raises(ValueError):
        write_review_test(review_dir, "test_big.py", "x" * 60_000)


def test_review_area_must_be_outside_project(project: Path) -> None:
    with pytest.raises(ValueError, match="outside the project"):
        ReviewerToolset(project / "review")


def test_review_tests_import_project_but_skip_project_tests(
    project: Path, review_dir: Path
) -> None:
    (project / "test_project.py").write_text(
        "def test_broken():\n    assert False\n", encoding="utf-8"
    )
    tools_ = ReviewerToolset(review_dir)
    tools_.write_review_test(
        "test_add.py",
        "import calculator\n\n\ndef test_add():\n    assert calculator.add(2, 3) == 5\n",
    )
    run = tools_.run_review_tests()
    assert run.passed
    assert "1 passed" in run.output


def test_empty_review_area_reports_no_tests(project: Path, review_dir: Path) -> None:
    assert ReviewerToolset(review_dir).run_review_tests().no_tests


# --- toolset shapes and dispatch -------------------------------------------

def _names(toolset: object) -> set[str]:
    return {spec["function"]["name"] for spec in tool_specs_for(toolset)}


def test_reviewer_gets_review_tools_but_never_project_writes(
    project: Path, review_dir: Path
) -> None:
    names = _names(ReviewerToolset(review_dir))
    assert {"write_review_test", "run_review_tests", "read_file", "git_diff"} <= names
    assert not {"write_file", "run_tests"} & names


def test_developer_never_gets_review_tools() -> None:
    assert not {"write_review_test", "run_review_tests"} & _names(DeveloperToolset())


def test_running_review_tests_needs_approval(project: Path, review_dir: Path) -> None:
    result = dispatch(ReviewerToolset(review_dir), "run_review_tests", "{}",
                      lambda name, args: False)
    assert "declined" in result


def test_no_tests_is_not_reported_as_failure(project: Path, review_dir: Path) -> None:
    result = dispatch(ReviewerToolset(review_dir), "run_review_tests", "{}",
                      lambda name, args: True)
    assert result.startswith("no tests found")


def test_git_diff_compares_against_run_start(repo: Path) -> None:
    start = _git(repo, "rev-parse", "HEAD")
    (repo / "calculator.py").write_text("def add(a, b):\n    return b + a\n",
                                        encoding="utf-8")
    _git(repo, "commit", "-qam", "checkpoint")
    assert "+    return b + a" in ReadOnlyToolset(base=start).git_diff().stdout
    assert ReadOnlyToolset().git_diff().stdout == ""


# --- deterministic validation ----------------------------------------------

def test_syntax_check_passes_valid_code(project: Path) -> None:
    assert check_syntax().status == "pass"


def test_syntax_check_reports_file_and_line(project: Path) -> None:
    (project / "broken.py").write_text("def f(:\n    pass\n", encoding="utf-8")
    result = check_syntax()
    assert result.status == "fail"
    assert "broken.py:1" in result.output


def test_syntax_check_without_python_files(project: Path) -> None:
    (project / "calculator.py").unlink()
    assert check_syntax().status == "not_configured"


def test_syntax_check_does_not_run_code(project: Path) -> None:
    marker = project / "ran.txt"
    (project / "boom.py").write_text(
        f"open({str(marker)!r}, 'w').write('x')\n", encoding="utf-8"
    )
    check_syntax()
    assert not marker.exists()


def test_declined_tests_are_not_run(project: Path) -> None:
    assert check_tests(_decline).status == "not_run"


def test_project_without_tests_is_not_configured(project: Path) -> None:
    assert check_tests(_approve).status == "not_configured"


def test_failing_tests_fail_the_check(project: Path) -> None:
    (project / "test_bad.py").write_text("def test_bad():\n    assert 1 == 2\n",
                                        encoding="utf-8")
    result = check_tests(_approve)
    assert result.status == "fail"
    assert "1 failed" in result.output


def test_changed_paths_lists_status_letters(repo: Path) -> None:
    start = _git(repo, "rev-parse", "HEAD")
    (repo / "new.py").write_text("", encoding="utf-8")
    (repo / "calculator.py").unlink()
    _git(repo, "add", "-A")
    assert sorted(changed_paths(start)) == [("A", "new.py"), ("D", "calculator.py")]


def test_deleted_files_are_reported_not_unexpected(repo: Path) -> None:
    start = _git(repo, "rev-parse", "HEAD")
    (repo / "calculator.py").unlink()
    assert unexpected_changes(start) == []
    assert run_validation(start, _decline).deleted_files == ["calculator.py"]


def test_committed_dependency_folders_are_unexpected(repo: Path) -> None:
    start = _git(repo, "rev-parse", "HEAD")
    (repo / "node_modules" / "left-pad").mkdir(parents=True)
    (repo / "node_modules" / "left-pad" / "index.js").write_text("x", encoding="utf-8")
    _git(repo, "add", "-A", "-f")
    [problem] = unexpected_changes(start)
    assert problem.startswith("node_modules/left-pad/index.js")


def test_run_validation_combines_checks(repo: Path) -> None:
    start = _git(repo, "rev-parse", "HEAD")
    report = run_validation(start, _decline)
    assert report.build.status == "pass"
    assert report.tests.status == "not_run"
    assert report.lint.status == "not_configured"
    assert report.failures == []
