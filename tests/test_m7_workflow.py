import json
import subprocess
from pathlib import Path

import pytest

import tools.project
import tools.testing
from llm import LLMError
from orchestrator.console import ConsoleUI
from orchestrator.runlog import RunLog
from orchestrator.tool_loop import ToolLoopResult
from orchestrator.workflow import ProjectRun, blocking_problems
from schemas.decisions import DeveloperResponse, FindingDecision
from schemas.project import ProjectSummary
from schemas.requirements import Specification
from schemas.review import Review
from schemas.state import ProjectState
from schemas.validation import CheckResult, ValidationReport
from tools import git

SPEC = Specification(
    summary="multiply",
    requirements=[{"id": "REQ-001", "description": "add multiply",
                   "acceptance_criteria": ["multiply(2, 3) == 6"],
                   "source": "explicit", "source_text": "add multiply"}],
)
MULTIPLY = "def add(a, b):\n    return a + b\n\n\ndef multiply(a, b):\n    return a * b\n"


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


class FakeDeveloper:
    def __init__(self, project: Path, edits: list[dict[str, str]],
                 responses: list[DeveloperResponse] | None = None,
                 fail_on_call: int | None = None) -> None:
        self.project = project
        self.edits = list(edits)
        self.responses = list(responses or [])
        self.fail_on_call = fail_on_call
        self.messages: list[str] = []

    def implement(self, message, approve, on_call=None, on_limit=None) -> ToolLoopResult:
        self.messages.append(message)
        for path, content in (self.edits.pop(0) if self.edits else {}).items():
            (self.project / path).write_text(content, encoding="utf-8")
        if self.fail_on_call == len(self.messages):
            raise LLMError("model server went away")
        return ToolLoopResult("done")

    def adjudicate(self, message, review, human_decisions) -> DeveloperResponse:
        self.messages.append(message)
        return self.responses.pop(0)


class FakeReviewer:
    def __init__(self, reviews: list[Review]) -> None:
        self.reviews = list(reviews)
        self.messages: list[str] = []
        self.changes_made: list[bool] = []

    def investigate(self, message, approve, on_call=None, on_limit=None) -> ToolLoopResult:
        self.messages.append(message)
        return ToolLoopResult("looked at calculator.py")

    def review(self, message, specification, *, changes_made) -> Review:
        self.changes_made.append(changes_made)
        return self.reviews.pop(0)

    def summarize(self, message) -> ProjectSummary:
        self.messages.append(message)
        return ProjectSummary(purpose="a calculator",
                              modules=[{"path": "calculator.py", "responsibility": "math"}])


def _review(status: str = "pass", findings: list[dict] | None = None) -> Review:
    verdict = "approved" if status == "pass" and not findings else "changes_required"
    return Review(
        verdict=verdict,
        requirements=[{"requirement_id": "REQ-001", "status": status, "evidence": "diff"}],
        findings=findings or [],
    )


def _finding(fid: str = "F-001", severity: str = "high") -> dict:
    return {"id": fid, "severity": severity, "category": "bug",
            "claim": "multiply ignores negatives", "evidence": "line 6"}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", *args],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    (root / "calculator.py").write_text("def add(a, b):\n    return a + b\n",
                                        encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "initial")
    monkeypatch.setattr(tools.project, "PROJECT_PATH", root)
    return root


def _run(repo: Path, tmp_path: Path, developer, reviewer, answers: list[str],
         max_iterations: int = 5) -> tuple[ProjectRun, ScriptedUI]:
    state = ProjectState(
        run_id="r1", user_prompt="add multiply", specification=SPEC,
        project_path=str(repo), original_branch=git.current_branch(),
        run_branch="sol/run-r1", base_commit=git.current_commit(),
    )
    ui = ScriptedUI(answers)
    run = ProjectRun(state, developer, reviewer, ui, RunLog("r1", root=tmp_path / "logs"),
                     max_iterations=max_iterations)
    return run, ui


def _file_on(repo: Path, branch: str, path: str = "calculator.py") -> str:
    return _git(repo, "show", f"{branch}:{path}")


# Answers in order: decline tests, accept at final review, decline merge.
ACCEPT_WITHOUT_MERGE = ["n", "a", "n"]


def test_accepted_run_keeps_work_on_its_own_branch(repo: Path, tmp_path: Path) -> None:
    original = git.current_branch()
    start = git.current_commit()
    run, ui = _run(repo, tmp_path, FakeDeveloper(repo, [{"calculator.py": MULTIPLY}]),
                   FakeReviewer([_review()]), ACCEPT_WITHOUT_MERGE)
    state = run.execute()

    assert state.status == "complete"
    assert git.current_branch() == original
    assert git.current_commit() == start
    assert "def multiply" in _file_on(repo, "sol/run-r1")
    assert [c.label for c in state.checkpoints] == ["implementation", "project document"]
    assert ui.answers == []


def test_accepting_and_merging_updates_the_original_branch(repo: Path, tmp_path: Path) -> None:
    original = git.current_branch()
    run, _ = _run(repo, tmp_path, FakeDeveloper(repo, [{"calculator.py": MULTIPLY}]),
                  FakeReviewer([_review()]), ["n", "a", "y"])
    state = run.execute()
    assert "merged" in state.outcome
    assert "def multiply" in _file_on(repo, original)


def test_reviewer_sees_changes_made_since_the_run_started(repo: Path, tmp_path: Path) -> None:
    reviewer = FakeReviewer([_review()])
    run, _ = _run(repo, tmp_path, FakeDeveloper(repo, [{"calculator.py": MULTIPLY}]),
                  reviewer, ACCEPT_WITHOUT_MERGE)
    run.execute()
    assert reviewer.changes_made == [True]
    assert "+def multiply(a, b):" in reviewer.messages[0]


def test_failed_requirement_triggers_a_fix_iteration(repo: Path, tmp_path: Path) -> None:
    developer = FakeDeveloper(repo, [{"calculator.py": "x = 1\n"},
                                     {"calculator.py": MULTIPLY}])
    reviewer = FakeReviewer([_review("fail"), _review()])
    run, _ = _run(repo, tmp_path, developer, reviewer, ["n", "n", "a", "n"])
    state = run.execute()

    assert state.status == "complete"
    assert state.iteration == 2
    assert "Requirement REQ-001 is fail" in developer.messages[1]
    assert [c.label for c in state.checkpoints] == ["implementation", "fix 1",
                                                    "project document"]


def test_iteration_limit_stops_the_run(repo: Path, tmp_path: Path) -> None:
    original = git.current_branch()
    run, _ = _run(repo, tmp_path, FakeDeveloper(repo, [{"calculator.py": "x = 1\n"}]),
                  FakeReviewer([_review("fail")]), ["n", "n"], max_iterations=1)
    state = run.execute()
    assert state.status == "failed"
    assert "iteration limit" in state.outcome
    assert git.current_branch() == original


def test_approved_finding_is_adjudicated_and_fixed(repo: Path, tmp_path: Path) -> None:
    response = DeveloperResponse(decisions=[FindingDecision(
        finding_id="F-001", decision="accept", reasoning="true",
        action="handle negative numbers")])
    developer = FakeDeveloper(repo, [{"calculator.py": MULTIPLY},
                                     {"calculator.py": MULTIPLY + "\n# fixed\n"}],
                              responses=[response])
    reviewer = FakeReviewer([_review(findings=[_finding()]), _review()])
    # tests no, approve finding, tests no, accept, no merge
    run, _ = _run(repo, tmp_path, developer, reviewer, ["n", "a", "n", "a", "n"])
    state = run.execute()

    assert state.status == "complete"
    assert "REVIEWER FINDINGS TO DECIDE" in developer.messages[1]
    assert "handle negative numbers" in developer.messages[2]


def test_upheld_rejection_dismisses_the_finding(repo: Path, tmp_path: Path) -> None:
    response = DeveloperResponse(decisions=[FindingDecision(
        finding_id="F-001", decision="reject", reasoning="false",
        evidence="multiply(-2, 3) returns -6")])
    developer = FakeDeveloper(repo, [{"calculator.py": MULTIPLY}], responses=[response])
    reviewer = FakeReviewer([_review(findings=[_finding()]), _review()])
    # tests no, investigate, uphold, tests no, accept, no merge
    run, ui = _run(repo, tmp_path, developer, reviewer, ["n", "", "", "n", "a", "n"])
    state = run.execute()

    assert state.status == "complete"
    assert [f.id for f in state.dismissed_findings] == ["F-001"]
    assert "- multiply ignores negatives" in reviewer.messages[1]
    assert len(developer.messages) == 2
    assert any("Nothing left for the Developer to fix" in line for line in ui.said)


def test_human_rejected_finding_skips_adjudication(repo: Path, tmp_path: Path) -> None:
    developer = FakeDeveloper(repo, [{"calculator.py": MULTIPLY}])
    reviewer = FakeReviewer([_review(findings=[_finding()]), _review()])
    # tests no, reject + reason, tests no, accept, no merge
    run, _ = _run(repo, tmp_path, developer, reviewer, ["n", "r", "noise", "n", "a", "n"])
    state = run.execute()
    assert state.status == "complete"
    assert len(developer.messages) == 1
    assert state.dismissed_findings[0].id == "F-001"


def test_error_mid_run_saves_work_and_restores_branch(repo: Path, tmp_path: Path) -> None:
    original = git.current_branch()
    developer = FakeDeveloper(repo, [{"calculator.py": MULTIPLY}], fail_on_call=1)
    run, _ = _run(repo, tmp_path, developer, FakeReviewer([]), [])
    with pytest.raises(LLMError):
        run.execute()

    assert git.current_branch() == original
    assert run.state.status == "failed"
    assert "def multiply" in _file_on(repo, "sol/run-r1")


def test_state_and_events_are_logged(repo: Path, tmp_path: Path) -> None:
    run, _ = _run(repo, tmp_path, FakeDeveloper(repo, [{"calculator.py": MULTIPLY}]),
                  FakeReviewer([_review()]), ACCEPT_WITHOUT_MERGE)
    run.execute()

    saved = ProjectState.model_validate_json(run.log.state_path.read_text(encoding="utf-8"))
    assert saved.status == "complete"
    kinds = [json.loads(line)["kind"] for line in
             run.log.events_path.read_text(encoding="utf-8").splitlines()]
    assert kinds[0] == "branch_created"
    assert "checkpoint" in kinds and "review" in kinds


def test_trying_it_returns_to_the_final_choice(repo: Path, tmp_path: Path) -> None:
    run, ui = _run(repo, tmp_path, FakeDeveloper(repo, [{"calculator.py": MULTIPLY}]),
                   FakeReviewer([_review()]), ["n", "t", "c", "a", "n"])
    state = run.execute()
    assert state.status == "complete"
    assert ui.answers == []


def test_enter_approves_a_sandboxed_tool_run(repo: Path, tmp_path: Path,
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tools.testing, "SANDBOX", "docker")
    run, _ = _run(repo, tmp_path, FakeDeveloper(repo, []), FakeReviewer([]), [""])
    assert run._approver("Developer")("run_tests", {}) is True


def test_enter_declines_an_unsandboxed_tool_run(repo: Path, tmp_path: Path) -> None:
    run, _ = _run(repo, tmp_path, FakeDeveloper(repo, []), FakeReviewer([]), [""])
    assert run._approver("Developer")("run_tests", {}) is False


def test_enter_accepts_at_final_review_and_merges(repo: Path, tmp_path: Path) -> None:
    original = git.current_branch()
    run, _ = _run(repo, tmp_path, FakeDeveloper(repo, [{"calculator.py": MULTIPLY}]),
                  FakeReviewer([_review()]), ["", "", ""])
    state = run.execute()
    assert state.status == "complete"
    assert "def multiply" in _file_on(repo, original)


def test_low_findings_do_not_block_completion() -> None:
    report = ValidationReport(**{name: CheckResult(status="pass") for name in
                                 ("build", "tests", "lint", "static_analysis")})
    review = _review(findings=[_finding(severity="low")])
    assert blocking_problems(review, report) == []
    assert blocking_problems(_review(findings=[_finding()]), report) != []
