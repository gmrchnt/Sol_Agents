import json
from typing import Callable, Protocol

from orchestrator.console import ConsoleUI
from orchestrator.decisions import findings_to_adjudicate
from orchestrator.human import describe_finding, human_review, resolve_disputes
from orchestrator.review import BLOCKING_SEVERITIES
from orchestrator.runlog import RunLog
from orchestrator.tool_loop import Approve, ToolCall, ToolLoopResult
from orchestrator.validation import run_validation
from schemas.decisions import DeveloperResponse, HumanDecision
from schemas.requirements import Specification
from schemas.review import Review
from schemas.state import Checkpoint, ProjectState, Status
from schemas.validation import ValidationReport
from tools.git import (
    GitError,
    commit_all,
    create_branch,
    diff,
    merge_fast_forward,
    switch_branch,
)

MAX_ITERATIONS = 5
TRANSCRIPT_RESULT_CHARS = 1_500


class Developer(Protocol):
    def implement(self, message: str, approve: Approve,
                  on_call: Callable[[ToolCall], None] | None = None) -> ToolLoopResult: ...

    def adjudicate(self, message: str, review: Review,
                   human_decisions: list[HumanDecision]) -> DeveloperResponse: ...


class Reviewer(Protocol):
    def investigate(self, message: str, approve: Approve,
                    on_call: Callable[[ToolCall], None] | None = None) -> ToolLoopResult: ...

    def review(self, message: str, specification: Specification, *,
               changes_made: bool) -> Review: ...


class ProjectRun:
    """Section 14's core loop. Owns the state; the models only ever answer questions."""

    def __init__(
        self,
        state: ProjectState,
        developer: Developer,
        reviewer: Reviewer,
        ui: ConsoleUI,
        log: RunLog,
        max_iterations: int = MAX_ITERATIONS,
    ) -> None:
        self.state = state
        self.developer = developer
        self.reviewer = reviewer
        self.ui = ui
        self.log = log
        self.max_iterations = max_iterations

    # --- top level ----------------------------------------------------------

    def execute(self) -> ProjectState:
        create_branch(self.state.run_branch)
        self.log.event("branch_created", branch=self.state.run_branch,
                       base=self.state.base_commit)
        try:
            self._run()
        except BaseException as exc:
            self.state.status = "failed"
            self.state.outcome = f"stopped by an error: {exc!r}"
            raise
        finally:
            self._close()
        return self.state

    def _run(self) -> None:
        self._set("implement")
        self.ui.heading("DEVELOPER")
        result = self.developer.implement(
            self._implement_message(), self._approver("Developer"), self._reporter("developer")
        )
        self._record_developer_summary(result)
        self._checkpoint("implementation")

        while True:
            self.state.iteration += 1
            self.ui.heading(f"ITERATION {self.state.iteration}")

            validation = self._validate()
            review = self._review(validation)
            problems = blocking_problems(review, validation)

            if not problems:
                choice = self._final_acceptance(review, validation)
                if choice == "accept":
                    self._finish("complete", "accepted by the human")
                    return
                if choice == "stop":
                    self._finish("failed", "stopped by the human at final review")
                    return
            else:
                self.ui.say("\nStill blocking completion:")
                for problem in problems:
                    self.ui.say(f"  - {problem.splitlines()[0]}")
                if self.state.iteration >= self.max_iterations and not self.ui.confirm(
                    f"\nReached the limit of {self.max_iterations} iterations. "
                    f"Run one more iteration?",
                    default=True,
                ):
                    self._finish("failed", f"iteration limit ({self.max_iterations}) reached")
                    return

            work = self._work_from(review, validation)
            if not work:
                self.ui.say("\nNothing left for the Developer to fix; reviewing again.")
                continue

            self._set("fix")
            self.ui.heading("DEVELOPER - FIX")
            result = self.developer.implement(
                self._fix_message(work), self._approver("Developer"), self._reporter("developer")
            )
            self._record_developer_summary(result)
            self._checkpoint(f"fix {self.state.iteration}")

    # --- phases -------------------------------------------------------------

    def _validate(self) -> ValidationReport:
        self._set("validate")
        validation = run_validation(
            self.state.base_commit,
            lambda: self.ui.confirm(
                "\nRun the project's tests to check the work? "
                "This executes project code without a sandbox.",
                default=True,
            ),
        )
        self.state.validation = validation
        self._save()
        self.ui.say("\nValidation:")
        for name, check in validation.checks().items():
            self.ui.say(f"  {name}: {check.status}")
        for failure in validation.failures:
            self.ui.say(f"  ! {failure}")
        return validation

    def _review(self, validation: ValidationReport) -> Review:
        self._set("review")
        self.ui.heading("REVIEWER")
        patch = diff(self.state.base_commit).stdout.strip()

        notes = self.reviewer.investigate(
            self._investigation_message(patch, validation),
            self._approver("Reviewer"),
            self._reporter("reviewer"),
        )
        review = self.reviewer.review(
            self._review_message(patch, validation, notes),
            self.state.specification,
            changes_made=bool(patch),
        )
        self.state.review = review
        self._save()
        self.log.event("review", iteration=self.state.iteration, verdict=review.verdict)
        self._show_review(review)
        return review

    def _work_from(self, review: Review, validation: ValidationReport) -> list[str]:
        work = [
            f"Requirement {r.requirement_id} is {r.status}. Reviewer's evidence: {r.evidence}"
            for r in review.requirements
            if r.status != "pass"
        ]
        work += [f"Validation problem - {failure}" for failure in validation.failures]

        if not review.findings:
            return work

        self._set("human_review")
        self.ui.heading("YOUR REVIEW OF THE FINDINGS")
        decisions = human_review(self.ui, review)
        self.state.human_decisions = decisions
        notes = {d.finding_id: d for d in decisions}
        by_id = {f.id: f for f in review.findings}
        self.state.dismissed_findings += [
            by_id[d.finding_id] for d in decisions if d.action == "reject"
        ]
        self._save()
        self.log.event("human_decisions", decisions=[d.model_dump() for d in decisions])

        to_judge = findings_to_adjudicate(review, decisions)
        if not to_judge:
            return work

        self._set("developer_response")
        self.ui.heading("DEVELOPER - ADJUDICATION")
        response = self.developer.adjudicate(
            self._adjudication_message(review, decisions), review, decisions
        )
        self.state.developer_response = response
        self._save()
        self.log.event("developer_response", decisions=[d.model_dump() for d in response.decisions])
        for decision in response.decisions:
            self.ui.say(f"  {decision.finding_id}: {decision.decision} - {decision.reasoning}")

        overridden, dismissed = resolve_disputes(self.ui, review, response)
        self.state.dismissed_findings += dismissed
        self._save()

        for decision in response.decisions:
            if decision.decision == "reject":
                continue
            work.append(self._finding_work(by_id[decision.finding_id], notes,
                                           f"Your planned action: {decision.action}"))
        for fid in overridden:
            work.append(self._finding_work(by_id[fid], notes,
                                           "You rejected this, but the human overrode you. Fix it."))
        return work

    def _final_acceptance(self, review: Review, validation: ValidationReport) -> str:
        self._set("final_review")
        self.ui.heading("FINAL REVIEW")
        for result in review.requirements:
            self.ui.say(f"  {result.requirement_id}: {result.status}")
        if validation.tests.status == "not_run":
            self.ui.say("  Note: the tests were not run, so they did not verify this work.")
        if review.findings:
            self.ui.say("\nOptional, non-blocking findings:")
            for finding in review.findings:
                self.ui.say(describe_finding(finding))

        options = {"a": "accept", "s": "stop"}
        prompt = "[a]ccept the work  [s]top without accepting  (Enter = accept): "
        if review.findings:
            options["i"] = "iterate"
            prompt = ("[a]ccept the work  [i]terate on the optional findings  [s]top  "
                      "(Enter = accept): ")
        return options[self.ui.choose(prompt, options, default="a")]

    def _close(self) -> None:
        state = self.state
        try:
            leftover = commit_all(f"sol: end of run - {state.outcome or state.status}")
            if leftover:
                state.checkpoints.append(Checkpoint(commit=leftover, label="end of run"))
        finally:
            switch_branch(state.original_branch)
            self._save()

        self.ui.heading("RUN FINISHED")
        self.ui.say(f"Outcome: {state.outcome}")
        self.ui.say(f"All work is on branch {state.run_branch}; you are back on "
                    f"{state.original_branch}.")

        if state.status == "complete" and state.checkpoints:
            if self.ui.confirm(f"Merge {state.run_branch} into {state.original_branch} now?",
                               default=True):
                try:
                    merge_fast_forward(state.run_branch)
                    state.outcome = f"{state.outcome}, merged into {state.original_branch}"
                    self.ui.say("Merged.")
                except GitError as exc:
                    self.ui.say(f"Could not merge: {exc}")
                self._save()

        path = state.project_path
        self.ui.say(f"\nInspect:  git -C \"{path}\" log --oneline {state.original_branch}..{state.run_branch}")
        self.ui.say(f"Diff:     git -C \"{path}\" diff {state.base_commit[:8]} {state.run_branch}")
        self.ui.say(f"Run log:  {self.log.dir}")

    # --- messages -----------------------------------------------------------

    def _request_and_spec(self) -> str:
        return (
            f"ORIGINAL USER REQUEST:\n{self.state.user_prompt}\n\n"
            f"APPROVED SPECIFICATION:\n{self.state.specification.model_dump_json(indent=2)}"
        )

    def _implement_message(self) -> str:
        return (
            f"{self._request_and_spec()}\n\n"
            "Implement the approved specification in the project.\n\n"
            "Rules:\n"
            "- Satisfy every listed requirement.\n"
            "- Do not add requirements that are not in the specification.\n"
            "- Do not change files the specification does not need changed."
        )

    def _evidence(self, patch: str, validation: ValidationReport) -> str:
        dismissed = "\n".join(f"- {f.claim}" for f in self.state.dismissed_findings) or "(none)"
        return (
            f"DEVELOPER SUMMARY (a claim, not evidence):\n{self.state.developer_summary}\n\n"
            f"GIT DIFF since the run started ({self.state.base_commit[:8]}):\n{patch or '(empty)'}\n\n"
            f"VALIDATION (run by the orchestrator, not the Developer):\n{validation.summary()}\n\n"
            f"PREVIOUSLY DISMISSED FINDINGS - do not raise these again without new evidence:\n"
            f"{dismissed}"
        )

    def _investigation_message(self, patch: str, validation: ValidationReport) -> str:
        return (
            f"{self._request_and_spec()}\n\n{self._evidence(patch, validation)}\n\n"
            "Investigate whether the work meets the APPROVED SPECIFICATION. Read the "
            "changed files and, where it helps, write and run review tests that try "
            "to prove the work wrong. When you are finished, reply with plain-text "
            "notes: what you checked, what you found, and the evidence for each point."
        )

    def _review_message(self, patch: str, validation: ValidationReport,
                        notes: ToolLoopResult) -> str:
        transcript = "\n\n".join(
            f"{call.name}({call.arguments[:200]}) ->\n{call.result[:TRANSCRIPT_RESULT_CHARS]}"
            for call in notes.calls
        ) or "(no tools used)"
        return (
            f"{self._request_and_spec()}\n\n{self._evidence(patch, validation)}\n\n"
            f"YOUR INVESTIGATION (tool calls and their real results):\n{transcript}\n\n"
            f"YOUR NOTES:\n{notes.final_text}\n\n"
            "Now give your structured review.\n\n"
            "Rules:\n"
            "- The approved specification defines the required behavior.\n"
            "- Do not invent additional requirements.\n"
            "- Base every status on evidence: the diff, file contents your tools "
            "returned, validation results and your review tests. The Developer "
            "summary proves nothing.\n"
            "- Mark a requirement pass only when the evidence shows it is met. "
            "Otherwise use fail or uncertain.\n"
            "- Only report a finding when you can point to concrete evidence, such "
            "as a line in the diff or a failing test.\n"
            "- Do not report missing evidence as a finding; use uncertain.\n"
            "- Report only defects. If your evidence shows something behaves "
            "acceptably, leave it out entirely - it is not a finding.\n"
            "- Report each issue once.\n"
            "- Use the security category only for a real security problem.\n"
            "- Use critical or high only for defects that break a requirement or are "
            "a real security or correctness problem."
        )

    def _adjudication_message(self, review: Review, decisions: list[HumanDecision]) -> str:
        to_judge = findings_to_adjudicate(review, decisions)
        human = {d.finding_id: d for d in decisions}
        human_lines = "\n".join(
            f"- {f.id}: human chose {human[f.id].action}"
            + (f" - {human[f.id].note}" if human[f.id].note else "")
            for f in to_judge
        )
        findings = json.dumps([f.model_dump(exclude_none=True) for f in to_judge], indent=2)
        patch = diff(self.state.base_commit).stdout.strip()
        return (
            f"{self._request_and_spec()}\n\n"
            f"GIT DIFF since the run started:\n{patch or '(empty)'}\n\n"
            f"REVIEWER FINDINGS TO DECIDE:\n{findings}\n\n"
            f"THE HUMAN'S POSITION ON EACH:\n{human_lines}\n\n"
            "Decide every finding listed above, and only those. Do not assume the "
            "Reviewer is right - check each claim against the diff.\n"
            "- accept: the issue is real. Give the action you will take.\n"
            "- partially_accept: the concern is real but the scope or suggested fix "
            "is not. Give the narrower action you will take.\n"
            "- reject: the claim is false. Give evidence - a counterexample, an "
            "invariant, a passing test or the code path that disproves it.\n"
            "A finding the human approved or gave instructions for cannot be rejected."
        )

    def _fix_message(self, work: list[str]) -> str:
        items = "\n".join(f"{n}. {item}" for n, item in enumerate(work, start=1))
        return (
            f"{self._request_and_spec()}\n\n"
            f"Fix the following, and nothing else:\n{items}\n\n"
            "Rules:\n"
            "- Read the files involved before changing them.\n"
            "- Make the smallest change that resolves each item.\n"
            "- Run the tests afterwards if the project has any."
        )

    @staticmethod
    def _finding_work(finding, notes: dict[str, HumanDecision], instruction: str) -> str:
        text = f"{finding.id}: {finding.claim}\n   Evidence: {finding.evidence}\n   {instruction}"
        human = notes.get(finding.id)
        if human and human.note:
            text += f"\n   The human says: {human.note}"
        return text

    # --- bookkeeping --------------------------------------------------------

    def _set(self, status: Status) -> None:
        self.state.status = status
        self._save()
        self.log.event("status", status=status, iteration=self.state.iteration)

    def _finish(self, status: Status, outcome: str) -> None:
        self.state.outcome = outcome
        self._set(status)

    def _save(self) -> None:
        self.log.save_state(self.state)

    def _checkpoint(self, label: str) -> None:
        commit = commit_all(f"sol: {label} (run {self.state.run_id})")
        if commit:
            self.state.checkpoints.append(Checkpoint(commit=commit, label=label))
            self.log.event("checkpoint", label=label, commit=commit)
            self.ui.say(f"\nCheckpoint {commit[:8]}: {label}")
        else:
            self.ui.say(f"\nNo changes to checkpoint after {label}.")
        self._save()

    def _record_developer_summary(self, result: ToolLoopResult) -> None:
        self.state.developer_summary = result.final_text
        self.ui.say(f"\n{result.final_text}")
        self._save()

    def _approver(self, agent: str) -> Approve:
        def approve(name: str, arguments: dict) -> bool:
            allowed = self.ui.confirm(
                f"\nThe {agent} wants to run {name}. This executes code on this "
                f"machine without a sandbox. Allow?",
                default=True,
            )
            self.log.event("approval", agent=agent, tool=name, allowed=allowed)
            return allowed
        return approve

    def _reporter(self, agent: str) -> Callable[[ToolCall], None]:
        def report(call: ToolCall) -> None:
            arguments = call.arguments if len(call.arguments) <= 80 else call.arguments[:77] + "..."
            first_line = call.result.splitlines()[0] if call.result else ""
            self.ui.say(f"  -> {call.name}({arguments})")
            self.ui.say(f"     {first_line[:100]}")
            self.log.event("tool_call", agent=agent, tool=call.name,
                           arguments=call.arguments, result=call.result)
        return report

    def _show_review(self, review: Review) -> None:
        self.ui.say(f"\nVerdict: {review.verdict}")
        for result in review.requirements:
            self.ui.say(f"  {result.requirement_id}: {result.status} - {result.evidence}")
        for finding in review.findings:
            self.ui.say(describe_finding(finding))


def blocking_problems(review: Review, validation: ValidationReport) -> list[str]:
    """Section 17: what still stands between this run and completion."""
    problems = [
        f"{r.requirement_id} is {r.status}: {r.evidence}"
        for r in review.requirements
        if r.status != "pass"
    ]
    problems += [
        f"{f.id} ({f.severity}): {f.claim}"
        for f in review.findings
        if f.severity in BLOCKING_SEVERITIES
    ]
    problems += validation.failures
    return problems
