from typing import Callable

from config import REVIEWER_MODEL
from llm import LLMClient
from orchestrator.review import ReviewRejected, validate_review
from orchestrator.structured import request_structured
from orchestrator.tool_loop import Approve, ToolCall, ToolLoopResult, run_tool_loop
from schemas.requirements import Specification
from schemas.review import Review
from tools.toolsets import ReviewerToolset

MAX_REVIEW_STEPS = 15

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
- write_review_test: write your own adversarial test in a private area
  outside the project
- run_review_tests: run those tests against the project (a human must
  approve each run)

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
""".strip()


class ReviewerAgent:

    def __init__(self, toolset: ReviewerToolset) -> None:
        self.llm = LLMClient(REVIEWER_MODEL)
        self.tools = toolset

    def investigate(
        self,
        message: str,
        approve: Approve,
        on_call: Callable[[ToolCall], None] | None = None,
    ) -> ToolLoopResult:
        return run_tool_loop(
            self.llm,
            self.tools,
            REVIEWER_SYSTEM_PROMPT,
            message,
            approve,
            on_call=on_call,
            max_steps=MAX_REVIEW_STEPS,
        )

    def review(
        self, message: str, specification: Specification, *, changes_made: bool
    ) -> Review:
        files = "\n".join(self.tools.list_files())
        return request_structured(
            self.llm,
            REVIEWER_SYSTEM_PROMPT,
            f"PROJECT FILES:\n{files}\n\n{message}",
            Review,
            lambda review: validate_review(review, specification, changes_made=changes_made),
            ReviewRejected,
            "review",
        )
