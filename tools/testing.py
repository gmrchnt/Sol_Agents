import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

import config
from tools.git import _truncate
from tools.project import get_project_path

TEST_TIMEOUT = 120
DOCKER_CHECK_TIMEOUT = 15
NO_TESTS_COLLECTED = 5

PYTEST_ARGS = ["-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider"]

SANDBOX = config.SANDBOX
SANDBOX_IMAGE = config.SANDBOX_IMAGE


class SuiteRunError(Exception):
    """Raised when the test command cannot be run at all."""


class SandboxUnavailable(SuiteRunError):
    """The sandbox is required but cannot be used, so nothing was run."""


@dataclass(frozen=True)
class SuiteRun:
    command: list[str]
    returncode: int
    output: str
    truncated: bool = False

    @property
    def passed(self) -> bool:
        return self.returncode == 0

    @property
    def no_tests(self) -> bool:
        return self.returncode == NO_TESTS_COLLECTED


def sandbox_description() -> str:
    if SANDBOX == "docker":
        return "inside the Docker sandbox (no network, read-only project)"
    return "directly on this machine WITHOUT a sandbox"


def run_tests(target: Path | None = None) -> SuiteRun:
    """Run pytest against the project; `target` limits it to one folder."""
    if SANDBOX == "docker":
        return _run_in_docker(target)
    return _run_locally(target)


def _run_locally(target: Path | None) -> SuiteRun:
    command = [sys.executable, *PYTEST_ARGS]
    if target is not None:
        command.append(str(target))
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

    return _result(command, done)


def docker_command(container: str, target: Path | None = None) -> list[str]:
    root = get_project_path().resolve()
    command = [
        "docker", "run", "--rm",
        "--name", container,
        "--network", "none",
        "--memory", "1g",
        "--cpus", "2",
        "--pids-limit", "256",
        "--read-only",
        "--tmpfs", "/tmp:rw,size=64m",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--user", "65534:65534",
        "-e", "PYTHONDONTWRITEBYTECODE=1",
        "-e", "HOME=/tmp",
        "--mount", f"type=bind,source={root},target=/project,readonly",
        "-w", "/project",
    ]
    if target is not None:
        command += ["--mount", f"type=bind,source={target.resolve()},target=/review,readonly"]
    command += [SANDBOX_IMAGE, "python", *PYTEST_ARGS]
    if target is not None:
        command.append("/review")
    return command


def require_docker() -> None:
    if shutil.which("docker") is None:
        raise SandboxUnavailable(
            "Docker is not installed, so the tests were not run. Install Docker "
            "Desktop, or set SANDBOX=local in .env to run them without a sandbox."
        )

    try:
        info = subprocess.run(["docker", "info"], capture_output=True,
                              timeout=DOCKER_CHECK_TIMEOUT)
        if info.returncode != 0:
            raise SandboxUnavailable(
                "Docker is installed but not running, so the tests were not run. "
                "Start Docker Desktop and try again."
            )
        image = subprocess.run(["docker", "image", "inspect", SANDBOX_IMAGE],
                               capture_output=True, timeout=DOCKER_CHECK_TIMEOUT)
    except subprocess.TimeoutExpired as exc:
        raise SandboxUnavailable("Docker did not respond, so the tests were not run.") from exc

    if image.returncode != 0:
        raise SandboxUnavailable(
            f"The sandbox image {SANDBOX_IMAGE} has not been built, so the tests "
            f"were not run. Build it with: docker build -t {SANDBOX_IMAGE} sandbox"
        )


def _run_in_docker(target: Path | None) -> SuiteRun:
    require_docker()
    container = f"sol-tests-{uuid.uuid4().hex[:12]}"
    command = docker_command(container, target)

    try:
        done = subprocess.run(
            command,
            capture_output=True,
            timeout=TEST_TIMEOUT,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired as exc:
        # Killing the docker CLI leaves the container running; stop it directly.
        subprocess.run(["docker", "kill", container], capture_output=True,
                       timeout=DOCKER_CHECK_TIMEOUT)
        raise SuiteRunError(f"Tests did not finish within {TEST_TIMEOUT}s.") from exc

    return _result(command, done)


def _result(command: list[str], done: subprocess.CompletedProcess) -> SuiteRun:
    output, truncated = _truncate((done.stdout or "") + (done.stderr or ""))
    return SuiteRun(command, done.returncode, output, truncated)
