from ollama import Client

from config import DEVELOPER_MODEL, OLLAMA_HOST
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

For example:

User:
"raise ValueError when the denominator is zero"

Valid acceptance criterion:
"A zero denominator raises ValueError"

Invalid acceptance criterion:
"A zero denominator raises ValueError with a meaningful message"

because the user did not request an error-message requirement.

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

class SpecificationBuilder:

    def __init__(self) -> None:
        self.client = Client(
            host=OLLAMA_HOST
        )

        self.model = DEVELOPER_MODEL

    def build(self, user_prompt: str) -> Specification:
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
                For explicit requirements, source_text MUST be an exact
                substring copied verbatim from the user's request.
                """
            response = self.client.chat(
                model=self.model,
                messages=[
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

                            USER REQUEST:
                            {user_prompt}
                            {extra_instruction}
                            """.strip()
                    },
                ],
                format=Specification.model_json_schema(),
                options={
                    "temperature": 0,
                }
            )
            content = response.message.content

            if not content:
                last_error = RuntimeError("Specification model returned no content.")
                continue

            try:
                specification = Specification.model_validate_json(content)
                self._validate_specification(specification)
                return specification
            except Exception as exc:
                last_error = exc
        raise RuntimeError(
            "Failed to generate a valid specification after 3 attempts. "
        ) from last_error

    def _validate_specification(self, specification: Specification) -> None:
        prompt = specification.original_prompt.lower()
        seen_ids: set[str] = set()

        for req in specification.requirements:
            if req.id in seen_ids:
                raise ValueError(f"Duplicate requirement ID: {req.id}")
            seen_ids.add(req.id)
            source_text = req.source_text.strip().lower()

            if not source_text:
                raise ValueError(
                    f"{req.id} has no "
                    f"source_text."
                )
            if source_text not in prompt:
                raise ValueError(
                    f"{req.id} claims to be "
                    f"explicit, but its source_text "
                    f"does not appear in the original "
                    f"user prompt:\n"
                    f"{req.source_text!r}"
                )