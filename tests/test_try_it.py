import subprocess
import sys
from pathlib import Path

import pytest

import orchestrator.try_it as try_it
import tools.project
from orchestrator.console import ConsoleUI
from orchestrator.runlog import RunLog
from orchestrator.try_it import entry_points, try_program
from tools.dependencies import Requirement

APP = 'def main():\n    print("hi")\n\n\nif __name__ == "__main__":\n    main()\n'
GAME = (
    "from pathlib import Path\n"
    "Path('score.txt').write_text('42')\n"
    "Path('app.py').write_text('changed\\n')\n"
)


class ScriptedUI(ConsoleUI):
    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)
        self.said: list[str] = []

    def say(self, text: str = "") -> None:
        self.said.append(text)

    def ask(self, prompt: str) -> str:
        if not self.answers:
            raise AssertionError(f"Unexpected prompt: {prompt!r}")
        return self.answers.pop(0)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", *args],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "project"
    (root / "tests").mkdir(parents=True)
    (root / "app.py").write_text(APP, encoding="utf-8")
    (root / "helpers.py").write_text("X = 1\n", encoding="utf-8")
    (root / "tests" / "test_app.py").write_text(APP, encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "initial")
    monkeypatch.setattr(tools.project, "PROJECT_PATH", root)
    return root


@pytest.fixture
def base(repo: Path) -> str:
    start = _git(repo, "rev-parse", "HEAD")
    (repo / "game.py").write_text(GAME, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "add game")
    return start


def _log(tmp_path: Path) -> RunLog:
    return RunLog("r", root=tmp_path / "logs")


def test_entry_points_put_this_runs_files_first(repo: Path, base: str) -> None:
    assert entry_points(base) == ["game.py", "app.py"]


def test_enter_runs_the_first_choice_and_restores_its_files(
    repo: Path, base: str, tmp_path: Path
) -> None:
    ui = ScriptedUI(["", "", ""])  # pick 1, confirm, restore
    try_program(ui, _log(tmp_path), base)
    assert not (repo / "score.txt").exists()
    assert (repo / "app.py").read_text(encoding="utf-8") == APP
    assert any("ended with exit code 0" in line for line in ui.said)


def test_files_can_be_kept(repo: Path, base: str, tmp_path: Path) -> None:
    try_program(ScriptedUI(["1", "y", "k"]), _log(tmp_path), base)
    assert (repo / "score.txt").read_text(encoding="utf-8") == "42"


def test_declining_runs_nothing(repo: Path, base: str, tmp_path: Path) -> None:
    try_program(ScriptedUI(["", "n"]), _log(tmp_path), base)
    assert not (repo / "score.txt").exists()


def test_cancel_at_file_choice(repo: Path, base: str, tmp_path: Path) -> None:
    try_program(ScriptedUI(["c"]), _log(tmp_path), base)
    assert not (repo / "score.txt").exists()


def test_a_typed_path_can_be_run(repo: Path, base: str, tmp_path: Path) -> None:
    ui = ScriptedUI(["app.py", "", ""])
    try_program(ui, _log(tmp_path), base)
    assert any("app.py ended with exit code 0" in line for line in ui.said)


@pytest.mark.parametrize("answer", ["../outside.py", "helpers.txt", "9"])
def test_bad_choices_are_refused(repo: Path, base: str, tmp_path: Path, answer: str) -> None:
    ui = ScriptedUI([answer])
    try_program(ui, _log(tmp_path), base)
    assert not (repo / "score.txt").exists()


def test_warning_names_the_missing_sandbox(repo: Path, base: str, tmp_path: Path) -> None:
    ui = ScriptedUI(["", "n"])
    try_program(ui, _log(tmp_path), base)
    assert any("WITHOUT a sandbox" in line for line in ui.said)
    assert any("diff" in line for line in ui.said)


def test_project_environment_is_used_when_installed(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (repo / "requirements.txt").write_text("six==1.17.0\n", encoding="utf-8")
    monkeypatch.setattr(try_it, "local_python", lambda reqs: Path("env/python.exe"))
    assert try_it._interpreter(ScriptedUI([])) == Path("env/python.exe")


@pytest.mark.parametrize("answer,builds", [("", 1), ("n", 0)])
def test_missing_packages_are_installed_only_on_approval(
    repo: Path, monkeypatch: pytest.MonkeyPatch, answer: str, builds: int
) -> None:
    (repo / "requirements.txt").write_text("six==1.17.0\n", encoding="utf-8")
    monkeypatch.setattr(try_it, "local_python", lambda reqs: None)
    built: list[list[Requirement]] = []
    monkeypatch.setattr(try_it, "build_local_env",
                        lambda reqs: built.append(reqs) or Path("env/python.exe"))
    try_it._interpreter(ScriptedUI([answer]))
    assert len(built) == builds


def test_without_requirements_sol_agents_python_is_used(repo: Path) -> None:
    assert try_it._interpreter(ScriptedUI([])) == Path(sys.executable)
