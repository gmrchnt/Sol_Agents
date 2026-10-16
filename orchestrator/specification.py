from pydantic import ValidationError

from config import DEVELOPER_API_KEY, DEVELOPER_BASE_URL, DEVELOPER_MODEL
from llm import LLMClient
from orchestrator.structured import StructuredOutputError
from schemas.requirements import Specification

SPECIFICATION_SYSTEM_PROMPT = """
You are the requirements-extraction component of a software
development orchestrator.

Your job is to convert the user's request into a minimal,
testable contract.

The specification contains ONLY requirements explicitly stated
by the user.

DO NOT CREATE DERIVED REQUIREMENTS.

Do not add behavior merely because it would be:
- useful
- sensible
- conventional
- safer
- more robust
- cleaner
- easier to test
- considered best practice

If the user did not request it, it is not a requirement.

Every requirement must:
1. Have a unique ID such as REQ-001.
2. Describe behavior explicitly requested by the user.
3. Have the smallest set of acceptance criteria necessary to
   test that behavior.
4. Include source_text copied verbatim from the user's prompt.

Acceptance criteria may make an explicit requirement testable,
but MUST NOT introduce additional behavior.

For example, given the request "store uploads in S3":

Valid acceptance criterion:
"An uploaded file is written to S3"

Invalid acceptance criterion:
"An uploaded file is written to S3 and the write is retried on failure"

because the user did not request retry behavior.

Preserve exact constraints named by the user, including exception
types, return formats, status codes, APIs, file formats, and other
observable behavior.

If the request contains important ambiguity, record it in
open_questions.

If a harmless interpretation is necessary, record it in
assumptions.

Assumptions and open questions are NOT requirements.

Do not write implementation code.
Do not propose improvements.
Do not add future features.
""".strip()
class SpecificationRejected(ValueError):
    """The model produced a specification that failed deterministic validation."""

MIN_SOURCE_TEXT_WORDS = 2


def _normalize(text: str) -> str:
    return " ".join(text.split()).lower()


def _cited_on_word_boundaries(prompt: str, source_text: str) -> bool:
    """True if some occurrence of source_text in prompt does not split a word."""
    start = prompt.find(source_text)
    while start != -1:
        end = start + len(source_text)
        starts_clean = start == 0 or not (
            prompt[start - 1].isalnum() and source_text[0].isalnum()
        )
        ends_clean = end == len(prompt) or not (
            prompt[end].isalnum() and source_text[-1].isalnum()
        )
        if starts_clean and ends_clean:
            return True
        start = prompt.find(source_text, start + 1)
    return False


class SpecificationBuilder:

    def __init__(self) -> None:
        self.llm = LLMClient(DEVELOPER_MODEL, base_url=DEVELOPER_BASE_URL,
                             api_key=DEVELOPER_API_KEY)

    def build(self, user_prompt: str, context: str | None = None) -> Specification:
        """`context` describes what already exists. It is background only: every
        requirement must still be quoted from `user_prompt`."""
        background = ""
        if context:
            background = (
                "BACKGROUND - what already exists. Use it to understand the request; "
                "never take requirements or source_text from it:\n"
                f"{context}\n\n"
            )
        last_error: Exception | None = None
        for attempt in range(3):
            extra_instruction = ""

            if last_error is not None:
                extra_instruction = f"""
                Your previous specification was rejected by
                deterministic validation.
                ERROR:
                {last_error}
                Generate a corrected specification.

                Every requirement must be explicitly supported by the
                user's request.
                source_text MUST be an exact substring copied verbatim
                from the user's request.
                """
            content = self.llm.chat(
                [
                    {
                        "role": "system",
                        "content": SPECIFICATION_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": f"""
                            Create the minimal specification for this request.

                            Do not add capabilities beyond what is requested.

                            For every explicit requirement, source_text must be copied
                            verbatim from the user's request.

                            {background}USER REQUEST:
                            {user_prompt}
                            {extra_instruction}
                            """.strip()
                    },
                ],
                schema=Specification.model_json_schema(),
                temperature=0,
            )

            if not content:
                last_error = RuntimeError("Specification model returned no content.")
                continue

            try:
                specification = Specification.model_validate_json(content)
                self._validate_specification(specification, user_prompt)
                return specification
            except (ValidationError, SpecificationRejected) as exc:
                last_error = exc
        raise StructuredOutputError(
            "Failed to generate a valid specification after 3 attempts. "
            f"Last problem: {last_error}"
        ) from last_error

    def _validate_specification(self, specification: Specification, user_prompt: str) -> None:
        prompt = _normalize(user_prompt)
        seen_ids: set[str] = set()

        for req in specification.requirements:
            if req.id in seen_ids:
                raise SpecificationRejected(f"Duplicate requirement ID: {req.id}")
            seen_ids.add(req.id)
            source_text = _normalize(req.source_text)

            if not source_text:
                raise SpecificationRejected(
                    f"{req.id} has no "
                    f"source_text."
                )
            if len(source_text.split()) < MIN_SOURCE_TEXT_WORDS:
                raise SpecificationRejected(
                    f"{req.id} cites only {req.source_text!r}. source_text "
                    f"must quote at least {MIN_SOURCE_TEXT_WORDS} words from "
                    f"the user's request that state this requirement."
                )
            if source_text not in prompt:
                raise SpecificationRejected(
                    f"{req.id} claims to be "
                    f"explicit, but its source_text "
                    f"does not appear in the original "
                    f"user prompt:\n"
                    f"{req.source_text!r}"
                )
            if not _cited_on_word_boundaries(prompt, source_text):
                raise SpecificationRejected(
                    f"{req.id} cites {req.source_text!r}, which appears in the "
                    f"user's request only inside longer words. Quote whole "
                    f"words."
                )