import shutil
import sys
import tempfile
from pathlib import Path

from agents.developer import DeveloperAgent
from agents.reviewer import ReviewerAgent
from config import PROJECT_PATH
from llm import LLMError
from orchestrator.console import ConsoleUI
from orchestrator.runlog import RunLog, new_run_id
from orchestrator.specification import SpecificationBuilder
from orchestrator.structured import StructuredOutputError
from orchestrator.tool_loop import ToolLoopLimitError
from orchestrator.workflow import ProjectRun
from schemas.requirements import Specification
from schemas.state import ProjectState
from tools.git import GitError, current_branch, current_commit, status
from tools.testing import SuiteRunError
from tools.toolsets import DeveloperToolset, ReviewerToolset


def agree_on_specification(ui: ConsoleUI, user_request: str) -> Specification | None:
    builder = SpecificationBuilder()
    while True:
        ui.heading("GENERATING SPECIFICATION")
        specification = builder.build(user_request)
        ui.say(specification.model_dump_json(indent=2))
        ui.say()
        choice = ui.choose(
            "Approve specification? [Y]es / [N]o / [R]egenerate: ",
            {"y": "yes", "yes": "yes", "n": "no", "no": "no", "r": "regenerate",
             "regenerate": "regenerate"},
            default="y",
        )
        if choice in {"y", "yes"}:
            return specification
        if choice in {"n", "no"}:
            return None


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

    ui.heading("USER REQUEST")
    user_request = ui.ask_required("Enter a simple software request: ")

    specification = agree_on_specification(ui, user_request)
    if specification is None:
        ui.say("\nSpecification rejected. Stopping before implementation.")
        return

    run_id = new_run_id()
    state = ProjectState(
        run_id=run_id,
        user_prompt=user_request,
        specification=specification,
        project_path=str(PROJECT_PATH),
        original_branch=original_branch,
        run_branch=f"sol/run-{run_id}",
        base_commit=base_commit,
    )
    log = RunLog(run_id)
    log.save_state(state)
    log.event("specification_approved")

    review_dir = Path(tempfile.mkdtemp(prefix=f"sol-review-{run_id}-"))
    try:
        developer = DeveloperAgent(DeveloperToolset(base=base_commit))
        reviewer = ReviewerAgent(ReviewerToolset(review_dir, base=base_commit))
        ProjectRun(state, developer, reviewer, ui, log).execute()
    finally:
        shutil.rmtree(review_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        main()
    except (LLMError, GitError, SuiteRunError, ToolLoopLimitError, StructuredOutputError) as exc:
        raise SystemExit(f"Error: {exc}")
