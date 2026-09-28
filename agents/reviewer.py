from pydantic import ValidationError

from config import REVIEWER_MODEL
from llm import LLMClient
from orchestrator.review import ReviewRejected, validate_review
from schemas.requirements import Specification
from schemas.review import Review
from tools.toolsets import ReadOnlyToolset


REVIEWER_SYSTEM_PROMPT = """
You are the independent Reviewer Agent in a software
engineering system.

Your role is adversarial verification, not implementation.

Your purpose is to determine whether the Developer's work
satisfies the approved requirements and whether there are
demonstrable defects.

Rules:

1. The approved specification is authoritative.
2. Do not invent requirements.
3. Do not report missing behavior as a bug unless that behavior
   is required by the approved specification or is necessary
   for correctness of a stated requirement.
4. A stylistic preference is not a bug.
5. A possible improvement is not automatically a defect.
6. Distinguish verified facts from suspicions.
7. Do not claim to have inspected files unless file contents
   were actually supplied to you.
8. Do not claim to have executed tests unless test output was
   actually supplied to you.
9. Do not claim a bug is reproduced unless reproduction
   evidence was actually supplied.
10. If evidence is insufficient, explicitly say that the claim
    is unverified.
11. Prefer precise, falsifiable findings over broad criticism.
12. The Developer is allowed to reject your findings later if
    they are unsupported or incorrect.

You will eventually receive Git diffs, compiler output, test results,
and static-analysis results.

You can see the project's file listing. You have read-only access and
must never modify the project's source.

For now, reason only from information actually supplied to you.
""".strip()


class ReviewerAgent:

    def __init__(self, toolset: ReadOnlyToolset) -> None:
        self.llm = LLMClient(REVIEWER_MODEL)
        self.tools = toolset

    def ask(self, message: str) -> str:
        return self.llm.chat(self._messages(message))

    def review(
        self, message: str, specification: Specification, *, changes_made: bool
    ) -> Review:
        last_error: Exception | None = None

        for _ in range(3):
            retry = ""
            if last_error is not None:
                retry = (
                    "\n\nYour previous review was rejected by deterministic "
                    f"validation.\nERROR:\n{last_error}\n"
                    "Produce a corrected review."
                )

            content = self.llm.chat(
                self._messages(message + retry),
                schema=Review.model_json_schema(),
                temperature=0,
            )

            try:
                review = Review.model_validate_json(content)
                validate_review(review, specification, changes_made=changes_made)
                return review
            except (ValidationError, ReviewRejected) as exc:
                last_error = exc

        raise RuntimeError(
            "Failed to produce a valid review after 3 attempts."
        ) from last_error

    def _messages(self, message: str) -> list[dict[str, str]]:
        files = "\n".join(self.tools.list_files())
        return [
            {"role": "system", "content": REVIEWER_SYSTEM_PROMPT},
            {"role": "user", "content": f"PROJECT FILES:\n{files}\n\n{message}"},
        ]