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


REVIEW_PATH = re.compile(r"[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+){0,4}")


def write_review_file(review_dir: Path, path: str, content: str) -> str:
    """Any file the Reviewer needs for a check in another language: a test, a fixture,
    a script. It lives in the review area, never in the project."""
    if not REVIEW_PATH.fullmatch(path) or ".." in path.split("/") or len(path) > 120:
        raise ValueError(
            f"{path!r} is not allowed. Use a relative path of letters, digits, '.', '_' "
            f"and '-', at most five folders deep, e.g. checks/edge_case.test.js."
        )

    data = content.encode("utf-8")
    if len(data) > MAX_REVIEW_TEST_BYTES:
        raise ValueError(
            f"Review file is {len(data)} bytes; maximum is {MAX_REVIEW_TEST_BYTES}."
        )

    target = review_dir / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return (f"wrote review file {path} ({len(data)} bytes); "
            f"run_review_command sees it at $REVIEW_DIR/{path}")
