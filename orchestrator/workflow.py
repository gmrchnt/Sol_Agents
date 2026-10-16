import json
import re
from fnmatch import fnmatch
from typing import Callable, Protocol

from llm import LLMError
from orchestrator.console import ConsoleUI
from orchestrator.decisions import findings_to_adjudicate
from orchestrator.human import (
    describe_finding,
    describe_suggestion,
    human_review,
    resolve_disputes,
)
from orchestrator.project_doc import (
    MAX_HISTORY,
    describe_summary,
    load_document,
    run_record,
    save_document,
)
from orchestrator.review import BLOCKING_SEVERITIES
from orchestrator.runlog import RunLog
from orchestrator.structured import StructuredOutputError
from orchestrator.tool_loop import (
    Approve,
    MoreSteps,
    ToolCall,
    ToolLoopLimitError,
    ToolLoopResult,
)
from orchestrator.try_it import try_program
from orchestrator.validation import run_validation
import config
from schemas.project import ProjectDocument, ProjectSummary
from schemas.toolchain import Toolchain
from tools.checks import (
    command_target,
    describe_target,
    pending_setups,
    run_setup,
    validation_plan,
)
from tools.dependencies import DependencyError
from tools.project import ProjectAccessError
from tools.testing import (
    SuiteRunError,
    install_dependencies,
    install_description,
    is_sandboxed,
    pending_packages,
    sandbox_description,
)
from tools.toolchain import (
    ToolchainError,
    describe_toolchain,
    detect_toolchain,
    load_toolchain,
    read_explicit,
)
from schemas.decisions import DeveloperResponse, HumanDecision, HumanFeedback
from schemas.requirements import Requirement, Specification
from schemas.review import Finding, Review
from schemas.state import Checkpoint, ProjectState, Status
from schemas.validation import ValidationReport
from tools.git import (
    GitError,
    changed_paths,
    commit_all,
    create_branch,
    current_commit,
    diff,
    ensure_default_excludes,
    merge_fast_forward,
    switch_branch,
)

MAX_ITERATIONS = config.MAX_ITERATIONS
# When an agent's tool budget runs out, the human can grant this many more steps.
EXTRA_STEPS = config.MAX_TOOL_STEPS
TRANSCRIPT_RESULT_CHARS = 1_500
# The diff pasted into a prompt; the agents can call git_diff for the rest.
MAX_PROMPT_DIFF_CHARS = 30_000
# Machine-written, often thousands of lines, and nothing a model needs to read.
LOCKFILES = ("**/package-lock.json", "**/yarn.lock", "**/pnpm-lock.yaml", "**/bun.lockb",
             "**/Cargo.lock", "**/poetry.lock", "**/Pipfile.lock", "**/uv.lock",
             "**/composer.lock", "**/Gemfile.lock", "**/go.sum", "**/mix.lock",
             "**/pubspec.lock", "**/packages.lock.json", "**/deno.lock")
# Tools that run commands; the human can allow them for the rest of the run.
COMMAND_TOOLS = {"run_command", "run_review_command"}
FEEDBACK_KINDS = {"b": "bug", "c": "change", "n": "feature"}
# What the document update and spec amendments may fail with without losing the run.
MODEL_ERRORS = (LLMError, StructuredOutputError, ToolLoopLimitError)


class Developer(Protocol):
    def implement(self, message: str, approve: Approve,
                  on_call: Callable[[ToolCall], None] | None = None,
                  on_limit: MoreSteps | None = None) -> ToolLoopResult: ...

    def adjudicate(self, message: str, review: Review,
                   human_decisions: list[HumanDecision]) -> DeveloperResponse: ...


class Reviewer(Protocol):
    def investigate(self, message: str, approve: Approve,
                    on_call: Callable[[ToolCall], None] | None = None,
                    on_limit: MoreSteps | None = None) -> ToolLoopResult: ...

    def review(self, message: str, specification: Specification, *,
               changes_made: bool) -> Review: ...

    def summarize(self, message: str) -> ProjectSummary: ...


class Specifier(Protocol):
    def build(self, user_prompt: str, context: str | None = None) -> Specification: ...


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
        specifier: Specifier | None = None,
    ) -> None:
        self.state = state
        self.developer = developer
        self.reviewer = reviewer
        self.ui = ui
        self.log = log
        self.max_iterations = max_iterations
        self.specifier = specifier
        # The human's bug reports and adopted suggestions, waiting for adjudication,
        # and work for requirements added since the last fix.
        self._requests: list[Finding] = []
        self._new_work: list[str] = []
        self._next_number: dict[str, int] = {}
        # Command approvals the human granted for the rest of the run.
        self._always_allowed: set[tuple[str, str, bool]] = set()

    # --- top level ----------------------------------------------------------

    def execute(self) -> ProjectState:
        create_branch(self.state.run_branch)
        ensure_default_excludes()
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
            self._implement_message(), self._approver("Developer"),
            self._reporter("developer"), self._more_steps("Developer"),
        )
        self._record_developer_summary(result)
        self._checkpoint("implementation")

        while True:
            self.state.iteration += 1
            self.ui.heading(f"ITERATION {self.state.iteration}")

            validation = self._validate()
            review = self._review(validation)
            problems = blocking_problems(review, validation)

            choice = None
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

            # After feedback or adopted suggestions, the optional findings the human
            # already passed over at final review are not asked about again.
            work = self._work_from(review, validation,
                                   with_reviewer_findings=choice != "requests")
            if not work:
                self.ui.say("\nNothing left for the Developer to fix; reviewing again.")
                continue

            self._set("fix")
            self.ui.heading("DEVELOPER - FIX")
            result = self.developer.implement(
                self._fix_message(work), self._approver("Developer"),
                self._reporter("developer"), self._more_steps("Developer"),
            )
            self._record_developer_summary(result)
            self._checkpoint(f"fix {self.state.iteration}")

    # --- phases -------------------------------------------------------------

    def _sync_dependencies(self) -> None:
        """Section 9: dependency additions need the human's approval before install."""
        self._sync_python_packages()
        self._sync_setups()

    def _sync_python_packages(self) -> None:
        try:
            pending = pending_packages()
        except (SuiteRunError, DependencyError) as exc:
            self.ui.say(f"\nDependencies: {exc}")
            return
        if not pending:
            return

        self.ui.say(f"\nrequirements.txt asks for packages that are not installed yet:")
        for package in pending:
            self.ui.say(f"  + {package}")
        if not self.ui.confirm(f"Install them {install_description()}?", default=True):
            self.log.event("dependencies", installed=False, packages=pending)
            self.ui.say("Not installed. Tests will not run until they are.")
            return

        try:
            summary = install_dependencies()
        except DependencyError as exc:
            self.log.event("dependencies", installed=False, packages=pending, error=str(exc))
            self.ui.say(f"Install failed: {exc}")
            if not self.ui.confirm(
                "Try again, letting packages without a pre-built version build from "
                "source? That runs each package's own setup code.", default=False,
            ):
                return
            try:
                summary = install_dependencies(allow_source=True)
            except DependencyError as retry:
                self.ui.say(f"Install failed again: {retry}")
                return
        self.log.event("dependencies", installed=True, packages=pending)
        self.ui.say(summary)

    def _sync_setups(self) -> None:
        """Other languages: npm ci, cargo fetch, ... run with network, once per manifest."""
        try:
            pending = pending_setups()
        except (ToolchainError, OSError, ProjectAccessError) as exc:
            self.ui.say(f"\nDependencies: {exc}")
            return
        if not pending:
            return

        self.ui.say("\nThese components have dependencies that are not installed yet:")
        for component in pending:
            self.ui.say(f"  {component.name}: {' && '.join(component.setup)}  "
                        f"({describe_target(component)})")
        if not self.ui.confirm("Install them now? This downloads packages, so these "
                               "commands run WITH network access.", default=True):
            self.log.event("setup", installed=False, components=[c.name for c in pending])
            self.ui.say("Not installed. Builds and tests may fail until they are.")
            return
        for component in pending:
            try:
                run = run_setup(component)
            except (SuiteRunError, OSError, ProjectAccessError, ValueError) as exc:
                self.ui.say(f"  {component.name}: could not run setup: {exc}")
                continue
            self.log.event("setup", component=component.name, installed=run.passed,
                           output=run.output)
            verdict = "done" if run.passed else f"FAILED\n{run.output[-2_000:]}"
            self.ui.say(f"  {component.name}: {verdict}")

    def _toolchain(self) -> Toolchain:
        try:
            return load_toolchain()
        except ToolchainError as exc:
            self.ui.say(f"\n{exc}\nUsing the toolchain detected from the project's files.")
            return detect_toolchain()

    def _validate(self) -> ValidationReport:
        self._sync_dependencies()
        self._set("validate")
        toolchain = self._toolchain()
        plan = "\n".join(f"  {line}" for line in validation_plan(toolchain))
        validation = run_validation(
            self.state.base_commit,
            lambda: self.ui.confirm(
                f"\nRun the project's checks on the work?\n{plan}\n"
                f"This executes project code {sandbox_description()}.",
                default=is_sandboxed(),
            ),
            toolchain,
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
        patch = self._patch()

        notes = self.reviewer.investigate(
            self._investigation_message(patch, validation),
            self._approver("Reviewer"),
            self._reporter("reviewer"),
            self._more_steps("Reviewer"),
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

    def _work_from(self, review: Review, validation: ValidationReport,
                   with_reviewer_findings: bool = True) -> list[str]:
        work = [
            f"Requirement {r.requirement_id} is {r.status}. Reviewer's evidence: {r.evidence}"
            for r in review.requirements
            if r.status != "pass"
        ]
        work += [f"Validation problem - {failure}" for failure in validation.failures]
        work += self._new_work
        self._new_work = []

        reviewer_findings = review.findings if with_reviewer_findings else []
        requests = self._requests
        self._requests = []
        if not reviewer_findings and not requests:
            return work

        decisions: list[HumanDecision] = []
        if reviewer_findings:
            self._set("human_review")
            self.ui.heading("YOUR REVIEW OF THE FINDINGS")
            decisions = human_review(self.ui, review.model_copy(
                update={"findings": reviewer_findings}))
        # The human raised these, but the Developer still checks them and may reject
        # them with evidence - the human settles any dispute below.
        decisions += [
            HumanDecision(finding_id=f.id, action="investigate", note=_REQUEST_NOTES[f.id[0]])
            for f in requests
        ]
        review = review.model_copy(update={"findings": reviewer_findings + requests})
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
        if validation.deleted_files:
            self.ui.say(f"  Files deleted in this run: {', '.join(validation.deleted_files)}")
        if review.findings:
            self.ui.say("\nOptional, non-blocking findings:")
            for finding in review.findings:
                self.ui.say(describe_finding(finding))
        if review.suggestions:
            self.ui.say("\nReviewer suggestions (optional):")
            for suggestion in review.suggestions:
                self.ui.say(describe_suggestion(suggestion))

        options = {"a": "accept", "t": "try", "f": "feedback"}
        labels = ["[a]ccept the work", "[t]ry it yourself",
                  "[f]eedback - a bug, a change or something new"]
        if review.suggestions:
            options["p"] = "pick"
            labels.append("[p]ick suggestions to adopt")
        if review.findings:
            options["i"] = "iterate"
            labels.append("[i]terate on the optional findings")
        options["s"] = "stop"
        labels.append("[s]top without accepting")
        prompt = "\n".join(labels) + "\n(Enter = accept): "

        while True:
            choice = options[self.ui.choose(prompt, options, default="a")]
            if choice == "try":
                try_program(self.ui, self.log, self.state.base_commit)
                self.ui.say()
            elif choice == "feedback":
                if self._collect_feedback():
                    return "requests"
            elif choice == "pick":
                if self._adopt_suggestions(review):
                    return "requests"
            else:
                return choice

    def _collect_feedback(self) -> bool:
        """Backbone: the human reports what they found and asks for anything new."""
        queued = False
        while True:
            key = self.ui.choose(
                "\nFeedback: [b]ug  [c]hange to how it works  [n]ew feature  "
                "[d]one  (Enter = done): ",
                {**FEEDBACK_KINDS, "d": "done"},
                default="d",
            )
            if key == "d":
                return queued
            kind = FEEDBACK_KINDS[key]
            text = self.ui.ask_required("Describe it: ")
            feedback = HumanFeedback(kind=kind, text=text)
            self.state.feedback.append(feedback)
            self._save()
            self.log.event("feedback", feedback=feedback.model_dump())

            if kind == "feature":
                queued = self._amend_specification(text) or queued
            else:
                finding = Finding(
                    id=self._next_id("H"),
                    severity="high" if kind == "bug" else "medium",
                    category="bug" if kind == "bug" else "quality",
                    claim=text,
                    evidence="Reported by the human at final review.",
                )
                self._requests.append(finding)
                self.ui.say(f"Recorded as {finding.id}. The Developer will check it.")
                queued = True

    def _adopt_suggestions(self, review: Review) -> bool:
        by_id = {s.id: s for s in review.suggestions}
        while True:
            answer = self.ui.ask(
                "Suggestions to adopt, e.g. S-001,S-003  (Enter = none): "
            ).strip().upper()
            chosen = [part for part in re.split(r"[,\s]+", answer) if part]
            unknown = [sid for sid in chosen if sid not in by_id]
            if not unknown:
                break
            self.ui.say(f"Unknown: {', '.join(unknown)}. Choose from {', '.join(by_id)}.")
        if not chosen:
            return False

        self.log.event("suggestions_adopted", suggestions=chosen)
        queued = False
        for sid in dict.fromkeys(chosen):
            suggestion = by_id[sid]
            if suggestion.kind == "addition":
                # New behavior is a new requirement, agreed like any other.
                queued = self._amend_specification(suggestion.description) or queued
                continue
            self._requests.append(Finding(
                id=suggestion.id, severity="low", category="quality",
                file=suggestion.file, line=suggestion.line,
                claim=suggestion.description, evidence=suggestion.rationale,
            ))
            queued = True
        return queued

    def _amend_specification(self, request: str) -> bool:
        """Add requirements for something new without restarting the run."""
        if self.specifier is None:
            self.ui.say("This run cannot add requirements; start a new run for it.")
            return False

        current = self.state.specification
        context = (f"CURRENT SPECIFICATION - already agreed; do not repeat these:\n"
                   f"{current.model_dump_json(indent=2)}")
        if self.state.project_summary is not None:
            context += f"\n\nEXISTING PROJECT:\n{describe_summary(self.state.project_summary)}"

        while True:
            self.ui.heading("AMENDING THE SPECIFICATION")
            try:
                addition = self.specifier.build(request, context=context)
            except MODEL_ERRORS as exc:
                self.ui.say(f"Could not turn that into requirements: {exc}")
                return False
            added = renumber(addition, after=current)
            for requirement in added:
                self.ui.say(requirement.model_dump_json(indent=2))
            choice = self.ui.choose(
                "Add these requirements? [y]es / [n]o / [r]egenerate: ",
                {"y": "yes", "yes": "yes", "n": "no", "no": "no",
                 "r": "regenerate", "regenerate": "regenerate"},
            )
            if choice in {"n", "no"}:
                return False
            if choice in {"y", "yes"}:
                break

        self.state.specification = current.model_copy(update={
            "requirements": current.requirements + added,
            "assumptions": current.assumptions + addition.assumptions,
            "open_questions": current.open_questions + addition.open_questions,
        })
        self.state.amendments.append(request)
        self._save()
        self.log.event("specification_amended", request=request,
                       requirements=[r.id for r in added])
        self._new_work += [
            f"New requirement {r.id}: {r.description}\n   Acceptance criteria: "
            + "; ".join(r.acceptance_criteria)
            for r in added
        ]
        return True

    def _next_id(self, prefix: str) -> str:
        number = self._next_number.get(prefix, 0) + 1
        self._next_number[prefix] = number
        return f"{prefix}-{number:03d}"

    def _update_document(self) -> None:
        """Backbone: record what was built so the next request starts from it."""
        state = self.state
        previous = load_document()
        before = previous.summary if previous else state.project_summary
        patch = self._patch()
        self.ui.heading("UPDATING THE PROJECT DOCUMENT")
        try:
            summary = self.reviewer.summarize(
                f"{self._request_and_spec()}\n\n"
                f"SUMMARY BEFORE THIS RUN:\n"
                f"{before.model_dump_json(indent=2) if before else '(none - new project)'}\n\n"
                f"WHAT THIS RUN CHANGED (git diff):\n{patch or '(empty)'}\n\n"
                "Describe the project as it is now, after this run. Keep what is still "
                "true from the earlier summary and correct what the changes made wrong. "
                "Cite only files listed under PROJECT FILES."
            )
        except MODEL_ERRORS as exc:
            self.ui.say(f"Could not update the project document: {exc}")
            self.log.event("project_document", updated=False, error=str(exc))
            return

        history = (previous.history if previous else []) + [run_record(state)]
        save_document(ProjectDocument(commit=current_commit(), summary=summary,
                                      history=history[-MAX_HISTORY:]))
        self.ui.say(describe_summary(summary))
        commit = commit_all(f"sol: update project document (run {state.run_id})")
        if commit:
            state.checkpoints.append(Checkpoint(commit=commit, label="project document"))
        self.log.event("project_document", updated=True, commit=commit)

    def _close(self) -> None:
        state = self.state
        try:
            leftover = commit_all(f"sol: end of run - {state.outcome or state.status}")
            if leftover:
                state.checkpoints.append(Checkpoint(commit=leftover, label="end of run"))
            if state.status == "complete":
                try:
                    self._update_document()
                except (OSError, ProjectAccessError) as exc:
                    self.ui.say(f"Could not write the project document: {exc}")
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
        text = ""
        if self.state.project_summary is not None:
            text += (f"EXISTING PROJECT (before this run):\n"
                     f"{describe_summary(self.state.project_summary)}\n\n")
        text += f"ORIGINAL USER REQUEST:\n{self.state.user_prompt}\n\n"
        if self.state.amendments:
            later = "\n".join(f"- {request}" for request in self.state.amendments)
            text += f"LATER REQUESTS FROM THE HUMAN (already in the specification):\n{later}\n\n"
        return text + (
            f"APPROVED SPECIFICATION:\n{self.state.specification.model_dump_json(indent=2)}"
        )

    def _patch(self) -> str:
        """The run's diff as the models see it: lockfiles named but not shown, and
        cut to fit the context window."""
        base = self.state.base_commit
        patch = diff(base, exclude=LOCKFILES).stdout.strip()
        if len(patch) > MAX_PROMPT_DIFF_CHARS:
            omitted = len(patch) - MAX_PROMPT_DIFF_CHARS
            patch = (f"{patch[:MAX_PROMPT_DIFF_CHARS]}\n... [{omitted} more characters of "
                     f"diff omitted - read the files or call git_diff for the rest]")
        locks = sorted({path for _, path in changed_paths(base)
                        if any(fnmatch(path, pattern) or fnmatch(path, pattern[3:])
                               for pattern in LOCKFILES)})
        if locks:
            patch += f"\n(lockfiles changed, not shown: {', '.join(locks)})"
        return patch.strip()

    def _toolchain_text(self) -> str:
        try:
            explicit = read_explicit()
            return describe_toolchain(explicit or detect_toolchain(), explicit is not None)
        except (ToolchainError, OSError, ProjectAccessError) as exc:
            return str(exc)

    def _implement_message(self) -> str:
        return (
            f"{self._request_and_spec()}\n\n"
            f"PROJECT TOOLCHAIN:\n{self._toolchain_text()}\n\n"
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
            "- Put optional improvements to existing code, and ideas for new "
            "additions, in suggestions - never in findings. Suggestions are "
            "optional and never change a requirement status or the verdict.\n"
            "- Use the security category only for a real security problem.\n"
            "- If requirements.txt changed, check every new package is actually "
            "needed, is a well-known package with the correct name, and is pinned. "
            "A misspelled or unnecessary package is a security finding.\n"
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
        patch = self._patch()
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
            f"PROJECT TOOLCHAIN:\n{self._toolchain_text()}\n\n"
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

    @staticmethod
    def _install_question(agent: str, allow_source: bool = False) -> str:
        try:
            pending = pending_packages()
            setups = pending_setups()
        except (SuiteRunError, DependencyError, ToolchainError, OSError) as exc:
            return f"\nThe {agent} wants to install dependencies ({exc}). Allow?"
        packages = "\n".join(f"  + {p}" for p in pending) or "  (nothing new)"
        question = (f"\nThe {agent} wants to install these packages "
                    f"{install_description()}:\n{packages}")
        if allow_source:
            question += ("\n  ...letting packages build from source, which runs their "
                         "own setup code")
        for component in setups:
            question += (f"\nand run, WITH network, for {component.name}: "
                         f"{' && '.join(component.setup)}")
        return question + "\nAllow?"

    @staticmethod
    def _command_question(agent: str, name: str, arguments: dict) -> str:
        try:
            if name == "run_command":
                target = command_target(arguments.get("component"), arguments.get("image"),
                                        arguments.get("workdir"))
            else:
                target = command_target(arguments.get("component"))
            where = describe_target(target)
        except (ValueError, OSError) as exc:
            where = f"(cannot tell where: {exc})"
        if name == "run_review_command":
            access = "the project read-only, no network"
        elif arguments.get("network"):
            access = "the project writable, WITH network access"
        else:
            access = "the project writable, no network"
        return (f"\nThe {agent} wants to run, in {where} ({access}):\n"
                f"  $ {arguments.get('command', '')}")

    @staticmethod
    def _toolchain_question(agent: str, arguments: dict) -> str:
        try:
            explicit = read_explicit()
            current = describe_toolchain(explicit or detect_toolchain(), explicit is not None)
        except (ToolchainError, OSError) as exc:
            current = str(exc)
        proposed = arguments.get("toolchain")
        try:
            shown = describe_toolchain(Toolchain.model_validate(proposed), explicit=True)
        except ValueError:
            shown = json.dumps(proposed, indent=2)
        return (f"\nThe {agent} wants to change how the project is set up, built, tested "
                f"and run. These commands decide what 'the checks pass' means.\n"
                f"NOW:\n{current}\n\nPROPOSED:\n{shown}")

    def _record_developer_summary(self, result: ToolLoopResult) -> None:
        self.state.developer_summary = result.final_text
        self.ui.say(f"\n{result.final_text}")
        self._save()

    def _approver(self, agent: str) -> Approve:
        def approve(name: str, arguments: dict) -> bool:
            # Enter means yes only where a mistake is contained by the sandbox.
            if name == "install_dependencies":
                allowed = self.ui.confirm(
                    self._install_question(agent, bool(arguments.get("allow_source_builds"))),
                    default=True)
            elif name in COMMAND_TOOLS:
                allowed = self._approve_command(agent, name, arguments)
            elif name == "set_toolchain":
                # No default: nobody should change what "passing" means by pressing Enter.
                allowed = self.ui.choose(
                    self._toolchain_question(agent, arguments) + "\nAllow? [y]es / [n]o: ",
                    {"y": "yes", "yes": "yes", "n": "no", "no": "no"},
                ) in {"y", "yes"}
            else:
                allowed = self.ui.confirm(
                    f"\nThe {agent} wants to run {name}. This executes code "
                    f"{sandbox_description()}. Allow?",
                    default=is_sandboxed())
            self.log.event("approval", agent=agent, tool=name, allowed=allowed,
                           arguments=arguments if name in COMMAND_TOOLS else None)
            return allowed
        return approve

    def _approve_command(self, agent: str, name: str, arguments: dict) -> bool:
        network = bool(arguments.get("network"))
        key = (agent, name, network)
        if key in self._always_allowed:
            self.ui.say(f"\n  $ {arguments.get('command', '')}  (allowed for this run)")
            return True
        contained = is_sandboxed() and not network
        choice = self.ui.choose(
            self._command_question(agent, name, arguments)
            + f"\nAllow? [y]es  [n]o  [a]lways for this run "
              f"(Enter = {'yes' if contained else 'no'}): ",
            {"y": "yes", "n": "no", "a": "always"},
            default="y" if contained else "n",
        )
        if choice == "a":
            self._always_allowed.add(key)
        return choice in {"y", "a"}

    def _more_steps(self, agent: str) -> MoreSteps:
        def more(used: int) -> int:
            if not self.ui.confirm(
                f"\nThe {agent} has made {used} rounds of tool calls without finishing. "
                f"Let it continue for {EXTRA_STEPS} more?", default=True,
            ):
                return 0
            self.log.event("more_steps", agent=agent, used=used)
            return EXTRA_STEPS
        return more

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
        for suggestion in review.suggestions:
            self.ui.say(describe_suggestion(suggestion))


_REQUEST_NOTES = {
    "H": "The human reported this after reviewing the work.",
    "S": "The human adopted this Reviewer suggestion.",
}


def renumber(addition: Specification, after: Specification) -> list[Requirement]:
    """Give new requirements the IDs that follow the existing ones."""
    start = max(int(r.id.split("-")[1]) for r in after.requirements) + 1
    return [
        requirement.model_copy(update={"id": f"REQ-{number:03d}"})
        for number, requirement in enumerate(addition.requirements, start=start)
    ]


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
