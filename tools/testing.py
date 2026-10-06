import os
import subprocess
import sys
from dataclasses import dataclass

from tools.git import _truncate
from tools.project import get_project_path

TEST_TIMEOUT = 120


class SuiteRunError(Exception):
    """Raised when the test command cannot be run at all."""


@dataclass(frozen=True)
class SuiteRun:
    command: list[str]
    returncode: int
    output: str
    truncated: bool = False

    @property
    def passed(self) -> bool:
        return self.returncode == 0


def run_tests() -> SuiteRun:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--no-header",
        "-p",
        "no:cacheprovider",
    ]
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}

    try:
        done = subprocess.run(
            command,
            cwd=get_project_path(),
            capture_output=True,
            timeout=TEST_TIMEOUT,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise SuiteRunError(f"Tests did not finish within {TEST_TIMEOUT}s.") from exc

    output, truncated = _truncate((done.stdout or "") + (done.stderr or ""))
    return SuiteRun(command, done.returncode, output, truncated)
