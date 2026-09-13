from ollama import Client

from config import DEVELOPER_MODEL, OLLAMA_HOST
from schemas.requirements import Specification

SPECIFICATION_SYSTEM_PROMPT = """
You are the requirements-analysis component of a software
development orchestrator.

Your job is to convert a user's software request into a precise,
minimal specification.

Rules:

1. Preserve the user's intent.
2. Do not invent features the user did not request.
3. Every requirement must have a unique ID:
   REQ-001, REQ-002, REQ-003, etc.
4. Every requirement must have testable acceptance criteria.
5. Mark a requirement as "explicit" when it directly comes from
   the user's request.
6. Mark it as "derived" only when it is genuinely necessary to
   make an explicit requirement implementable.
7. Do not treat stylistic preferences as requirements unless the
   user explicitly requested them.
8. Put uncertain interpretations in "assumptions".
9. Put ambiguities that require user clarification in
   "open_questions".
10. Do not silently resolve important ambiguities.
11. Keep the specification minimal.
12. Do not write implementation code.

The specification will later be used by both a Developer Agent
and an independent Reviewer Agent.
""".strip()


class SpecificationBuilder:

    def __init__(self) -> None:
        self.client = Client(
            host=OLLAMA_HOST
        )

        self.model = DEVELOPER_MODEL

    def build(self, user_prompt: str) -> Specification:
        response = self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": SPECIFICATION_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            format=Specification.model_json_schema(),
            options={
                "temperature": 0.0,
            }
        )
        content = response.message.content

        if not content:
            raise RuntimeError("Specification model returned no content.")
        return Specification.model_validate_json(content)