import pytest
from pydantic import ValidationError

from orchestrator.specification import SpecificationBuilder, SpecificationRejected
from schemas.requirements import Specification

HEALTH = "add a fastapi /health endpoint"
DIVIDE = "divides two numbers and raises ValueError when the denominator is zero"


def _requirement(source_text: str, req_id: str = "REQ-001") -> dict:
    return {
        "id": req_id,
        "description": "d",
        "acceptance_criteria": ["x"],
        "source": "explicit",
        "source_text": source_text,
    }


def _validate(prompt: str, *requirements: dict) -> None:
    # __new__ skips __init__ so the validator runs without a model server
    builder = SpecificationBuilder.__new__(SpecificationBuilder)
    builder._validate_specification(
        Specification(summary="s", requirements=list(requirements)), prompt
    )


ACCEPTED = {
    "full prompt quoted verbatim": (HEALTH, HEALTH),
    "clause starting with punctuation": (HEALTH, "/health endpoint"),
    "clause from the middle": (HEALTH, "fastapi /health"),
    "different capitalization": (HEALTH, "ADD A FASTAPI /HEALTH ENDPOINT"),
    "extra spaces inside the citation": (HEALTH, "fastapi   /health"),
    "newline inside the prompt": ("add a\nfastapi /health endpoint", "a fastapi"),
    "shortest legitimate citation": (DIVIDE, "raises ValueError"),
    "second clause of a compound prompt": (DIVIDE, "when the denominator is zero"),
    "valid occurrence follows a mid-word one": (
        "prelogin and login endpoint",
        "login endpoint",
    ),
}

REJECTED = {
    "single letter": (HEALTH, "a"),
    "single common word": (HEALTH, "the"),
    "mid-word fragment": (HEALTH, "ealth endp"),
    "text the user never wrote": (HEALTH, "add rate limiting"),
    "whitespace only": (HEALTH, "   "),
}


@pytest.mark.parametrize(
    "prompt,source_text", list(ACCEPTED.values()), ids=list(ACCEPTED)
)
def test_accepts_genuine_citation(prompt: str, source_text: str) -> None:
    _validate(prompt, _requirement(source_text))


@pytest.mark.parametrize(
    "prompt,source_text", list(REJECTED.values()), ids=list(REJECTED)
)
def test_rejects_unsupported_citation(prompt: str, source_text: str) -> None:
    with pytest.raises(SpecificationRejected):
        _validate(prompt, _requirement(source_text))


def test_rejects_duplicate_requirement_ids() -> None:
    with pytest.raises(SpecificationRejected):
        _validate(
            HEALTH,
            _requirement(HEALTH, "REQ-001"),
            _requirement("/health endpoint", "REQ-001"),
        )


def test_schema_rejects_specification_with_no_requirements() -> None:
    with pytest.raises(ValidationError):
        Specification(summary="s", requirements=[])


def test_schema_rejects_malformed_requirement_id() -> None:
    with pytest.raises(ValidationError):
        Specification(summary="s", requirements=[_requirement(HEALTH, "REQ-1")])


def test_schema_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        Specification(summary="s", requirements=[_requirement(HEALTH)], priority="high")
