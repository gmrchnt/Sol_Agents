import re
from pathlib import Path

REVIEW_TEST_NAME = re.compile(r"test_[A-Za-z0-9_]{1,60}\.py")
MAX_REVIEW_TEST_BYTES = 50_000


def write_review_test(review_dir: Path, filename: str, content: str) -> str:
    if not REVIEW_TEST_NAME.fullmatch(filename):
        raise ValueError(
            f"{filename!r} is not allowed. Review tests must be named like "
            f"test_something.py, with no folders."
        )

    data = content.encode("utf-8")
    if len(data) > MAX_REVIEW_TEST_BYTES:
        raise ValueError(
            f"Review test is {len(data)} bytes; maximum is {MAX_REVIEW_TEST_BYTES}."
        )

    review_dir.mkdir(parents=True, exist_ok=True)
    (review_dir / filename).write_bytes(data)
    return f"wrote review test {filename} ({len(data)} bytes)"
