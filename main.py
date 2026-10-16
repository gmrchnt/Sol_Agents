import shutil
import sys
import tempfile
from pathlib import Path

from agents.developer import DeveloperAgent
from agents.reviewer import ReviewerAgent
from config import PROJECT_PATH
from llm import LLMError
from orchestrator.console import ConsoleUI
from orchestrator.project_doc import (
    changes_since,
    context_for_models,
    has_existing_code,
    load_document,
)
from orchestrator.runlog import RunLog, new_run_id
from orchestrator.specification import SpecificationBuilder
from orchestrator.structured import StructuredOutputError
from orchestrator.survey import survey_project
from orchestrator.tool_loop import ToolLoopLimitError
from orchestrator.workflow import ProjectRun
from schemas.project import ProjectSummary, RunRecord
from schemas.requirements import Specification
from schemas.state import ProjectState
from tools.git import GitError, current_branch, current_commit, status
from tools.dependencies import DependencyError
from tools.project import ProjectAccessError
from tools.testing import SuiteRunError
from tools.toolchain import ToolchainError
from tools.toolsets import DeveloperToolset, ReviewerToolset


def agree_on_specification(ui: ConsoleUI, builder: SpecificationBuilder, user_request: str,
                           context: str | None) -> Specification | None:
    while True:
        ui.heading("GENERATING SPECIFICATION")
        specification = builder.build(user_request, context=context)
        ui.say(specification.model_dump_json(indent=2))
        ui.say()
        # No default: Enter must not approve a specification nobody read.
        choice = ui.choose(
            "Approve specification? [y]es / [n]o / [r]egenerate: ",
            {"y": "yes", "yes": "yes", "n": "no", "no": "no", "r": "regenerate",
             "regenerate": "regenerate"},
        )
        if choice in {"y", "yes"}:
            return specification
        if choice in {"n", "no"}:
            return None


def understand_project(ui: ConsoleUI, reviewer: ReviewerAgent,
                       log: RunLog) -> tuple[ProjectSummary | None, list[RunRecord]]:
    """Backbone: for an existing project, read it before taking the request."""
    document = load_document()
    history = document.history if document else []
    changed = changes_since(document) if document else None

    if document and changed == []:
        ui.say("Using the project document from the last run (.sol/PROJECT.md).")
        return document.summary, history
    if not has_existing_code():
        return None, history

    ui.heading("EXISTING PROJECT")
    if document:
        ui.say("The project changed since Sol last described it.")
    if not ui.confirm("Let the Reviewer read the project first, so the specification "
                      "builds on what exists?", default=True):
        return (document.summary if document else None), history
    ui.heading("READING THE PROJECT")
    return survey_project(reviewer, ui, log, document, changed), history


def main() -> None:
    ui = ConsoleUI()

    uncommitted = status().stdout.splitlines()[1:]
    if uncommitted:
        ui.say(f"{PROJECT_PATH} has uncommitted changes:")
        for line in uncommitted:
            ui.say(f"  {line}")
        ui.say("Commit or stash them first, so the review sees only this run's work.")
        return
    original_branch = current_branch()
    base_commit = current_commit()

    run_id = new_run_id()
    log = RunLog(run_id)
    review_dir = Path(tempfile.mkdtemp(prefix=f"sol-review-{run_id}-"))
    try:
        reviewer_tools = ReviewerToolset(review_dir, base=base_commit)
        reviewer = ReviewerAgent(reviewer_tools)
        summary, history = understand_project(ui, reviewer, log)
        if has_existing_code():
            ui.say(f"\n{reviewer_tools.show_toolchain()}")

        ui.heading("USER REQUEST")
        user_request = ui.ask_text(
            "Describe what to build or change - any language or kind of project.\n"
            "Finish with an empty line, or type @path to read the request from a file."
        )

        builder = SpecificationBuilder()
        specification = agree_on_specification(
            ui, builder, user_request, context_for_models(summary, history)
        )
        if specification is None:
            ui.say("\nSpecification rejected. Stopping before implementation.")
            return

        state = ProjectState(
            run_id=run_id,
            user_prompt=user_request,
            specification=specification,
            project_path=str(PROJECT_PATH),
            original_branch=original_branch,
            run_branch=f"sol/run-{run_id}",
            base_commit=base_commit,
            project_summary=summary,
        )
        log.save_state(state)
        log.event("specification_approved")

        developer = DeveloperAgent(DeveloperToolset(base=base_commit))
        ProjectRun(state, developer, reviewer, ui, log, specifier=builder).execute()
    finally:
        shutil.rmtree(review_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        main()
    except (LLMError, GitError, SuiteRunError, DependencyError, ToolLoopLimitError,
            StructuredOutputError, ProjectAccessError, ToolchainError) as exc:
        raise SystemExit(f"Error: {exc}")
