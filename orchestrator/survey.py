"""Backbone: for an existing project, read the repo and build understanding first."""

from typing import Protocol

import config
from orchestrator.console import ConsoleUI
from orchestrator.runlog import RunLog
from orchestrator.tool_loop import Approve, MoreSteps, ToolCall, ToolLoopResult
from schemas.project import ProjectDocument, ProjectSummary
from tools.testing import is_sandboxed, sandbox_description

TRANSCRIPT_RESULT_CHARS = 1_500

SURVEY_TASK = """
Survey this existing project so that later work builds on it correctly.

Read the files that matter: entry points, the main modules, tests,
requirements.txt and any README. You are describing the project, not
reviewing it. Only write and run a review test if that is the only way to
learn how something behaves.

When you are finished, reply with plain-text notes: what the project does,
how it is run, each module and what it is responsible for, its tests and
dependencies, and any problems a tool result showed you.
""".strip()


class Surveyor(Protocol):
    def investigate(self, message: str, approve: Approve, on_call=None,
                    on_limit: MoreSteps | None = None) -> ToolLoopResult: ...

    def summarize(self, message: str) -> ProjectSummary: ...


def survey_message(previous: ProjectDocument | None, changed: list[str] | None) -> str:
    if previous is None:
        return SURVEY_TASK
    files = "\n".join(f"- {path}" for path in changed or []) or "(unknown - read what you need)"
    return (
        f"{SURVEY_TASK}\n\n"
        f"A SUMMARY FROM AN EARLIER RUN (commit {previous.commit[:8]}):\n"
        f"{previous.summary.model_dump_json(indent=2)}\n\n"
        f"FILES CHANGED OUTSIDE SOL SINCE THEN:\n{files}\n\n"
        "Focus on the changed files and correct the summary wherever they make it wrong."
    )


def survey_project(reviewer: Surveyor, ui: ConsoleUI, log: RunLog,
                   previous: ProjectDocument | None = None,
                   changed: list[str] | None = None) -> ProjectSummary:
    def approve(name: str, arguments: dict) -> bool:
        command = arguments.get("command")
        what = f"run, with the project read-only:\n  $ {command}\n" if command else f"run {name}."
        allowed = ui.confirm(
            f"\nThe Reviewer wants to {what} This executes code "
            f"{sandbox_description()}. Allow?",
            default=is_sandboxed(),
        )
        log.event("approval", agent="Reviewer", tool=name, allowed=allowed)
        return allowed

    def report(call: ToolCall) -> None:
        arguments = call.arguments if len(call.arguments) <= 80 else call.arguments[:77] + "..."
        ui.say(f"  -> {call.name}({arguments})")
        log.event("tool_call", agent="surveyor", tool=call.name,
                  arguments=call.arguments, result=call.result)

    def more(used: int) -> int:
        if ui.confirm(f"\nThe Reviewer has made {used} rounds of tool calls reading the "
                      f"project. Let it continue?", default=True):
            return config.MAX_REVIEW_STEPS
        return 0

    task = survey_message(previous, changed)
    notes = reviewer.investigate(task, approve, report, more)
    transcript = "\n\n".join(
        f"{call.name}({call.arguments[:200]}) ->\n{call.result[:TRANSCRIPT_RESULT_CHARS]}"
        for call in notes.calls
    ) or "(no tools used)"
    summary = reviewer.summarize(
        f"{task}\n\nYOUR INVESTIGATION (tool calls and their real results):\n{transcript}\n\n"
        f"YOUR NOTES:\n{notes.final_text}\n\n"
        "Now give the project summary. Cite only files listed under PROJECT FILES. "
        "List a known issue only when a tool result showed it."
    )
    log.event("survey", summary=summary.model_dump())
    return summary
