from typing import Callable

from config import MAX_REVIEW_STEPS, REVIEWER_API_KEY, REVIEWER_BASE_URL, REVIEWER_MODEL
from llm import LLMClient
from orchestrator.project_doc import SummaryRejected, validate_summary
from orchestrator.review import ReviewRejected, validate_review
from orchestrator.structured import request_structured
from orchestrator.tool_loop import Approve, MoreSteps, ToolCall, ToolLoopResult, run_tool_loop
from schemas.project import ProjectSummary
from schemas.requirements import Specification
from schemas.review import Review
from tools.toolsets import ReviewerToolset

REVIEWER_SYSTEM_PROMPT = """
You are the independent Reviewer Agent in a software
engineering system.

Your role is adversarial verification, not implementation.

Your purpose is to determine whether the Developer's work
satisfies the approved requirements and whether there are
demonstrable defects.

You investigate through tools:
- list_files, read_file, search_files: read the project
- git_status, git_diff, git_recent_commits: see what changed in this run
- show_toolchain: how the project is built and tested, per component
- write_review_test, run_review_tests: write adversarial pytest files in
  a private area outside the project and run them against it
- write_review_file, run_review_command: for any other language, write
  a test or script in that private area and run it with the project's
  own tools in its sandbox; the project is read-only there and your
  files are at $REVIEW_DIR (a human must approve each run)

You can never modify the project's source.

Rules:

1. The approved specification is authoritative.
2. Do not invent requirements.
3. Do not report missing behavior as a bug unless that behavior
   is required by the approved specification or is necessary
   for correctness of a stated requirement.
4. A stylistic preference is not a bug.
5. A possible improvement is not automatically a defect.
6. Distinguish verified facts from suspicions.
7. Do not claim to have inspected a file unless a tool returned
   its contents to you.
8. Do not claim a test passed or failed unless a tool returned
   its output to you.
9. Prefer reproducing a suspected bug with a review test over
   asserting it.
10. If evidence is insufficient, explicitly say that the claim
    is unverified.
11. Prefer precise, falsifiable findings over broad criticism.
12. The Developer is allowed to reject your findings later if
    they are unsupported or incorrect.
13. Findings are defects only. Improvements to existing code and
    ideas for new additions go in suggestions instead. Suggestions
    are optional: they never fail a requirement or block completion,
    and the human decides whether to adopt them. Number findings
    F-001, F-002, ... and suggestions S-001, S-002, ...
""".strip()


def number_suggestions(review: Review) -> Review:
    suggestions = [
        suggestion.model_copy(update={"id": f"S-{number:03d}"})
        for number, suggestion in enumerate(review.suggestions, start=1)
    ]
    return review.model_copy(update={"suggestions": suggestions})


class ReviewerAgent:

    def __init__(self, toolset: ReviewerToolset) -> None:
        self.llm = LLMClient(REVIEWER_MODEL, base_url=REVIEWER_BASE_URL,
                             api_key=REVIEWER_API_KEY)
        self.tools = toolset

    def investigate(
        self,
        message: str,
        approve: Approve,
        on_call: Callable[[ToolCall], None] | None = None,
        on_limit: MoreSteps | None = None,
    ) -> ToolLoopResult:
        return run_tool_loop(
            self.llm,
            self.tools,
            REVIEWER_SYSTEM_PROMPT,
            message,
            approve,
            on_call=on_call,
            max_steps=MAX_REVIEW_STEPS,
            on_limit=on_limit,
        )

    def review(
        self, message: str, specification: Specification, *, changes_made: bool
    ) -> Review:
        files = "\n".join(self.tools.list_files())
        review = request_structured(
            self.llm,
            REVIEWER_SYSTEM_PROMPT,
            f"PROJECT FILES:\n{files}\n\n{message}",
            Review,
            lambda review: validate_review(review, specification, changes_made=changes_made),
            ReviewRejected,
            "review",
        )
        return number_suggestions(review)

    def summarize(self, message: str) -> ProjectSummary:
        """Describe the project as it stands, citing only files that exist."""
        files = self.tools.list_files()
        listing = "\n".join(files)
        return request_structured(
            self.llm,
            REVIEWER_SYSTEM_PROMPT,
            f"PROJECT FILES:\n{listing}\n\n{message}",
            ProjectSummary,
            lambda summary: validate_summary(summary, set(files)),
            SummaryRejected,
            "project summary",
        )
