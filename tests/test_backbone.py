"""Sol_Backbone: human feedback, reviewer suggestions, import checks, and the
project document carried between runs."""

import json
from pathlib import Path

import pytest

import main
from orchestrator.project_doc import (
    SummaryRejected,
    changes_since,
    context_for_models,
    has_existing_code,
    load_document,
    save_document,
    validate_summary,
)
from orchestrator.review import ReviewRejected, validate_review
from orchestrator.specification import SpecificationBuilder
from orchestrator.tool_loop import ToolLoopResult
from orchestrator.validation import check_imports, importable_modules, run_validation
from orchestrator.workflow import renumber
from schemas.decisions import DeveloperResponse, FindingDecision
from schemas.project import ProjectDocument, ProjectSummary, RunRecord
from schemas.requirements import Specification
from schemas.review import Review
from tools import git
from test_m7_workflow import (
    MULTIPLY,
    SPEC,
    FakeDeveloper,
    FakeReviewer,
    ScriptedUI,
    _file_on,
    _git,
    _review,
    _run,
    repo,  # noqa: F401 - fixture
)

DIVIDE_SPEC = Specification(
    summary="divide",
    requirements=[{"id": "REQ-001", "description": "add divide",
                   "acceptance_criteria": ["divide(6, 3) == 2"],
                   "source": "explicit", "source_text": "add divide"}],
)


class FakeSpecifier:
    def __init__(self, specs: list[Specification]) -> None:
        self.specs = list(specs)
        self.calls: list[tuple[str, str | None]] = []

    def build(self, user_prompt, context=None) -> Specification:
        self.calls.append((user_prompt, context))
        return self.specs.pop(0)


def _accept(fid: str, action: str = "fix it") -> DeveloperResponse:
    return DeveloperResponse(decisions=[FindingDecision(
        finding_id=fid, decision="accept", reasoning="confirmed", action=action)])


def _with_suggestions(findings: list[dict] | None = None) -> Review:
    return Review.model_validate({**_review(findings=findings).model_dump(), "suggestions": [
        {"id": "S-001", "kind": "improvement", "file": "calculator.py", "line": 1,
         "description": "add type hints", "rationale": "clearer API"},
        {"id": "S-002", "kind": "addition", "description": "add divide",
         "rationale": "natural next operation"},
    ]})


# --- B1: feedback at final review ---------------------------------------------

def test_reported_bug_is_adjudicated_then_fixed(repo: Path, tmp_path: Path) -> None:
    developer = FakeDeveloper(repo, [{"calculator.py": MULTIPLY},
                                     {"calculator.py": MULTIPLY + "# fixed\n"}],
                              responses=[_accept("H-001", "handle negatives")])
    reviewer = FakeReviewer([_review(), _review()])
    # tests no | feedback, bug, text, done | tests no | accept | no merge
    run, ui = _run(repo, tmp_path, developer, reviewer,
                   ["n", "f", "b", "wrong for negatives", "", "n", "a", "n"])
    state = run.execute()

    assert state.status == "complete"
    assert [f.kind for f in state.feedback] == ["bug"]
    assert "H-001" in developer.messages[1] and "wrong for negatives" in developer.messages[1]
    assert "handle negatives" in developer.messages[2]
    assert ui.answers == []


def test_human_can_override_a_rejected_bug_report(repo: Path, tmp_path: Path) -> None:
    rejection = DeveloperResponse(decisions=[FindingDecision(
        finding_id="H-001", decision="reject", reasoning="works for me",
        evidence="multiply(-2, 3) == -6")])
    developer = FakeDeveloper(repo, [{"calculator.py": MULTIPLY}, {}], responses=[rejection])
    reviewer = FakeReviewer([_review(), _review()])
    # ... | override | tests no | accept | no merge
    run, _ = _run(repo, tmp_path, developer, reviewer,
                  ["n", "f", "b", "crashes on zero", "", "o", "n", "a", "n"])
    state = run.execute()

    assert state.status == "complete"
    assert "the human overrode you" in developer.messages[2]


def test_feature_request_amends_the_specification(repo: Path, tmp_path: Path) -> None:
    developer = FakeDeveloper(repo, [{"calculator.py": MULTIPLY}, {}])
    reviewer = FakeReviewer([_review(), _review()])
    run, _ = _run(repo, tmp_path, developer, reviewer,
                  ["n", "f", "n", "add divide", "y", "", "n", "a", "n"])
    run.specifier = FakeSpecifier([DIVIDE_SPEC])
    state = run.execute()

    assert [r.id for r in state.specification.requirements] == ["REQ-001", "REQ-002"]
    assert state.amendments == ["add divide"]
    assert "New requirement REQ-002: add divide" in developer.messages[1]
    assert "LATER REQUESTS FROM THE HUMAN" in reviewer.messages[-1]
    # The builder sees what is already agreed, so it does not repeat it.
    assert "REQ-001" in run.specifier.calls[0][1]


def test_declined_amendment_changes_nothing(repo: Path, tmp_path: Path) -> None:
    run, ui = _run(repo, tmp_path, FakeDeveloper(repo, [{"calculator.py": MULTIPLY}]),
                   FakeReviewer([_review()]), ["n", "f", "n", "add divide", "n", "", "a", "n"])
    run.specifier = FakeSpecifier([DIVIDE_SPEC])
    state = run.execute()
    assert [r.id for r in state.specification.requirements] == ["REQ-001"]
    assert ui.answers == []


def test_feedback_with_nothing_entered_returns_to_the_choice(repo: Path, tmp_path: Path) -> None:
    run, ui = _run(repo, tmp_path, FakeDeveloper(repo, [{"calculator.py": MULTIPLY}]),
                   FakeReviewer([_review()]), ["n", "f", "", "a", "n"])
    assert run.execute().status == "complete"
    assert ui.answers == []


def test_renumber_continues_after_existing_ids() -> None:
    assert [r.id for r in renumber(DIVIDE_SPEC, after=SPEC)] == ["REQ-002"]


# --- B2: reviewer suggestions -------------------------------------------------

def test_adopted_improvement_goes_to_the_developer(repo: Path, tmp_path: Path) -> None:
    developer = FakeDeveloper(repo, [{"calculator.py": MULTIPLY}, {}],
                              responses=[_accept("S-001", "annotate add")])
    reviewer = FakeReviewer([_with_suggestions(), _review()])
    # tests no | pick, bad id, S-001 | tests no | accept | no merge
    run, ui = _run(repo, tmp_path, developer, reviewer,
                   ["n", "p", "S-009", "s-001", "n", "a", "n"])
    state = run.execute()

    assert state.status == "complete"
    assert "add type hints" in developer.messages[1]
    assert "annotate add" in developer.messages[2]
    assert any("Unknown: S-009" in line for line in ui.said)


def test_adopted_addition_becomes_a_requirement(repo: Path, tmp_path: Path) -> None:
    developer = FakeDeveloper(repo, [{"calculator.py": MULTIPLY}, {}])
    reviewer = FakeReviewer([_with_suggestions(), _review()])
    run, _ = _run(repo, tmp_path, developer, reviewer, ["n", "p", "S-002", "y", "n", "a", "n"])
    run.specifier = FakeSpecifier([DIVIDE_SPEC])
    state = run.execute()
    assert [r.id for r in state.specification.requirements] == ["REQ-001", "REQ-002"]


def test_suggestions_never_block() -> None:
    from orchestrator.workflow import blocking_problems
    from schemas.validation import CheckResult, ValidationReport
    report = ValidationReport(**{name: CheckResult(status="pass") for name in
                                 ("build", "tests", "lint", "static_analysis")})
    assert blocking_problems(_with_suggestions(), report) == []


def test_reviewer_cannot_number_findings_as_human_reports() -> None:
    review = _review(findings=[{"id": "H-001", "severity": "low", "category": "bug",
                                "claim": "c", "evidence": "e"}])
    with pytest.raises(ReviewRejected, match="F-001"):
        validate_review(review, SPEC, changes_made=True)


def test_suggestion_may_not_repeat_a_finding() -> None:
    review = _with_suggestions(findings=[{"id": "F-001", "severity": "low",
                                          "category": "quality", "claim": "Add type hints",
                                          "evidence": "line 1"}])
    with pytest.raises(ReviewRejected, match="S-001 repeats F-001"):
        validate_review(review, SPEC, changes_made=True)


# --- B3: import check ---------------------------------------------------------

@pytest.fixture
def modules(repo: Path) -> Path:
    (repo / "pkg").mkdir()
    (repo / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "pkg" / "shapes.py").write_text(
        '"""Shapes."""\nimport math\n\nPI = math.pi\n\n\ndef area(r):\n    return PI * r * r\n',
        encoding="utf-8")
    (repo / "script.py").write_text("print('hello')\n", encoding="utf-8")
    (repo / "asks.py").write_text("name = input('name? ')\n", encoding="utf-8")
    (repo / "guarded.py").write_text(
        "def main():\n    print('hi')\n\n\nif __name__ == '__main__':\n    main()\n",
        encoding="utf-8")
    (repo / "test_calc.py").write_text("def test_x():\n    pass\n", encoding="utf-8")
    return repo


def test_only_library_modules_are_imported(modules: Path) -> None:
    assert importable_modules() == ["calculator", "guarded", "pkg", "pkg.shapes"]


def test_import_check_passes_for_working_modules(modules: Path) -> None:
    result = check_imports(lambda: True)
    assert result.status == "pass", result.output
    assert "ok   pkg.shapes" in result.output


def test_import_check_catches_errors_at_load_time(modules: Path) -> None:
    (modules / "broken.py").write_text("RATIO = 1 / 0\n", encoding="utf-8")
    result = check_imports(lambda: True)
    assert result.status == "fail"
    assert "FAIL broken: ZeroDivisionError" in result.output


def test_import_check_needs_approval(modules: Path) -> None:
    assert check_imports(lambda: False).status == "not_run"


def test_sandboxed_import_keeps_the_lockdown(modules: Path) -> None:
    from tools.testing import IMPORT_SCRIPT, docker_command
    command = docker_command("c1", image="img", argv=["-c", IMPORT_SCRIPT, "pkg"])
    assert command[command.index("--network") + 1] == "none"
    assert "--read-only" in command
    assert command[-5:] == ["img", "python", "-c", IMPORT_SCRIPT, "pkg"]
    assert "pytest" not in command


def test_validation_asks_once_for_imports_and_tests(modules: Path) -> None:
    asked: list[bool] = []
    report = run_validation(git.current_commit(), lambda: asked.append(True) or False)
    assert asked == [True]
    assert report.imports.status == "not_run" and report.tests.status == "not_run"


# --- B4/B5: survey and project document ---------------------------------------

SUMMARY = ProjectSummary(purpose="a calculator", entry_points=["calculator.py"],
                         modules=[{"path": "calculator.py", "responsibility": "math"}])


def test_summary_may_cite_only_real_files() -> None:
    validate_summary(SUMMARY, {"calculator.py"})
    with pytest.raises(SummaryRejected, match="calculator.py"):
        validate_summary(SUMMARY, {"other.py"})


def test_accepted_run_writes_the_project_document(repo: Path, tmp_path: Path) -> None:
    run, _ = _run(repo, tmp_path, FakeDeveloper(repo, [{"calculator.py": MULTIPLY}]),
                  FakeReviewer([_review()]), ["n", "a", "n"])
    state = run.execute()

    document = ProjectDocument.model_validate_json(_file_on(repo, "sol/run-r1",
                                                            ".sol/project.json"))
    assert document.summary.purpose == "a calculator"
    assert [r.run_id for r in document.history] == ["r1"]
    assert document.history[0].outcome == "accepted by the human"
    # It describes the code commit just before it was written.
    assert document.commit == state.checkpoints[-2].commit
    assert "# Project document" in _file_on(repo, "sol/run-r1", ".sol/PROJECT.md")


def test_stopped_run_writes_no_document(repo: Path, tmp_path: Path) -> None:
    run, _ = _run(repo, tmp_path, FakeDeveloper(repo, [{"calculator.py": MULTIPLY}]),
                  FakeReviewer([_review()]), ["n", "s"])
    run.execute()
    with pytest.raises(Exception):
        _file_on(repo, "sol/run-r1", ".sol/project.json")


def test_history_accumulates_across_runs(repo: Path, tmp_path: Path) -> None:
    save_document(ProjectDocument(commit=git.current_commit(), summary=SUMMARY, history=[
        RunRecord(run_id="r0", request="add add", outcome="accepted by the human")]))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "doc")
    run, _ = _run(repo, tmp_path, FakeDeveloper(repo, [{"calculator.py": MULTIPLY}]),
                  FakeReviewer([_review()]), ["n", "a", "n"])
    run.execute()
    document = ProjectDocument.model_validate_json(_file_on(repo, "sol/run-r1",
                                                            ".sol/project.json"))
    assert [r.run_id for r in document.history] == ["r0", "r1"]


def test_document_is_current_until_code_changes(repo: Path) -> None:
    save_document(ProjectDocument(commit=git.current_commit(), summary=SUMMARY))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "doc")
    document = load_document()
    assert changes_since(document) == []

    (repo / "calculator.py").write_text(MULTIPLY, encoding="utf-8")
    _git(repo, "commit", "-qam", "edit outside sol")
    assert changes_since(document) == ["calculator.py"]


def test_unreadable_document_is_ignored(repo: Path) -> None:
    (repo / ".sol").mkdir()
    (repo / ".sol" / "project.json").write_text("{not json", encoding="utf-8")
    assert load_document() is None


def test_scaffolding_is_not_existing_code(tmp_path: Path, monkeypatch) -> None:
    import tools.project
    monkeypatch.setattr(tools.project, "PROJECT_PATH", tmp_path)
    (tmp_path / ".gitkeep").write_text("", encoding="utf-8")
    assert not has_existing_code()
    (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")
    assert has_existing_code()


class FakeSurveyor:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def investigate(self, message, approve, on_call=None, on_limit=None) -> ToolLoopResult:
        self.messages.append(message)
        return ToolLoopResult("it is a calculator")

    def summarize(self, message) -> ProjectSummary:
        self.messages.append(message)
        return SUMMARY


def test_existing_project_is_surveyed_first(repo: Path, tmp_path: Path) -> None:
    from orchestrator.runlog import RunLog
    surveyor = FakeSurveyor()
    summary, history = main.understand_project(ScriptedUI([""]), surveyor,
                                               RunLog("r", root=tmp_path / "logs"))
    assert summary == SUMMARY and history == []
    assert "it is a calculator" in surveyor.messages[1]


def test_current_document_skips_the_survey(repo: Path, tmp_path: Path) -> None:
    from orchestrator.runlog import RunLog
    save_document(ProjectDocument(commit=git.current_commit(), summary=SUMMARY))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "doc")
    surveyor = FakeSurveyor()
    summary, _ = main.understand_project(ScriptedUI([]), surveyor,
                                         RunLog("r", root=tmp_path / "logs"))
    assert summary == SUMMARY and surveyor.messages == []


def test_stale_document_survey_names_the_changed_files(repo: Path, tmp_path: Path) -> None:
    from orchestrator.runlog import RunLog
    save_document(ProjectDocument(commit=git.current_commit(), summary=SUMMARY))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "doc")
    (repo / "calculator.py").write_text(MULTIPLY, encoding="utf-8")
    _git(repo, "commit", "-qam", "edit outside sol")
    surveyor = FakeSurveyor()
    main.understand_project(ScriptedUI([""]), surveyor, RunLog("r", root=tmp_path / "logs"))
    assert "- calculator.py" in surveyor.messages[0]


def test_context_lists_project_and_history() -> None:
    text = context_for_models(SUMMARY, [RunRecord(run_id="r0", request="add add",
                                                  requirements=["REQ-001: add"],
                                                  outcome="accepted")])
    assert "Purpose: a calculator" in text and "run r0: 'add add'" in text
    assert context_for_models(None, []) is None


class FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.messages: list = []

    def chat(self, messages, **kwargs) -> str:
        self.messages.append(messages)
        return self.reply


def test_spec_builder_gets_context_but_cites_only_the_request() -> None:
    builder = SpecificationBuilder.__new__(SpecificationBuilder)
    builder.llm = FakeLLM(json.dumps({
        "summary": "s", "requirements": [{
            "id": "REQ-001", "description": "d", "acceptance_criteria": ["x"],
            "source": "explicit", "source_text": "a calculator"}]}))
    # source_text quotes the context, not the request: every attempt is rejected.
    with pytest.raises(Exception, match="does not appear"):
        builder.build("add a power function", context="EXISTING PROJECT:\nPurpose: a calculator")
    assert "BACKGROUND" in builder.llm.messages[0][1]["content"]
