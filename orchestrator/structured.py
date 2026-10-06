from typing import Callable, TypeVar

from pydantic import BaseModel, ValidationError

from llm import LLMClient

MAX_ATTEMPTS = 3

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(RuntimeError):
    """The model could not produce output that passed validation."""


def request_structured(
    llm: LLMClient,
    system_prompt: str,
    user_message: str,
    model: type[T],
    validate: Callable[[T], None],
    rejection: type[Exception],
    what: str,
) -> T:
    """Ask for JSON matching `model`, retrying with the exact rejection reason."""
    last_error: Exception | None = None

    for _ in range(MAX_ATTEMPTS):
        retry = ""
        if last_error is not None:
            retry = (
                f"\n\nYour previous {what} was rejected by deterministic "
                f"validation.\nERROR:\n{last_error}\nProduce a corrected {what}."
            )

        content = llm.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message + retry},
            ],
            schema=model.model_json_schema(),
            temperature=0,
        )

        try:
            result = model.model_validate_json(content)
            validate(result)
            return result
        except (ValidationError, rejection) as exc:
            last_error = exc

    raise StructuredOutputError(
        f"Failed to produce a valid {what} after {MAX_ATTEMPTS} attempts. "
        f"Last problem: {last_error}"
    ) from last_error
