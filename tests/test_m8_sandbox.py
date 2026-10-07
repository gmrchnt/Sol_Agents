import shutil
import subprocess
from pathlib import Path

import pytest

import tools.project
import tools.testing
from orchestrator.tool_loop import dispatch
from orchestrator.validation import check_tests
from tools.testing import (
    SandboxUnavailable,
    docker_command,
    require_docker,
    run_tests,
    sandbox_description,
)
from tools.toolsets import DeveloperToolset


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    (root / "test_ok.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    monkeypatch.setattr(tools.project, "PROJECT_PATH", root)
    return root


@pytest.fixture
def docker_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tools.testing, "SANDBOX", "docker")


@pytest.fixture
def no_docker(monkeypatch: pytest.MonkeyPatch, docker_mode: None) -> None:
    monkeypatch.setattr(tools.testing.shutil, "which", lambda name: None)


def _flag(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


# --- the container is locked down ------------------------------------------

def test_container_has_no_network(project: Path) -> None:
    assert _flag(docker_command("c1"), "--network") == "none"


def test_project_is_mounted_read_only(project: Path) -> None:
    mount = _flag(docker_command("c1"), "--mount")
    assert f"source={project.resolve()}" in mount
    assert mount.endswith("target=/project,readonly")


def test_container_runs_unprivileged_with_limits(project: Path) -> None:
    command = docker_command("c1")
    assert _flag(command, "--user") == "65534:65534"
    assert _flag(command, "--cap-drop") == "ALL"
    assert _flag(command, "--security-opt") == "no-new-privileges"
    assert "--read-only" in command
    assert {"--memory", "--cpus", "--pids-limit"} <= set(command)


def test_container_is_named_so_it_can_be_killed(project: Path) -> None:
    assert _flag(docker_command("sol-tests-abc"), "--name") == "sol-tests-abc"


def test_review_area_is_mounted_read_only_and_targeted(project: Path, tmp_path: Path) -> None:
    review = tmp_path / "review"
    review.mkdir()
    command = docker_command("c1", review)
    mounts = [command[i + 1] for i, part in enumerate(command) if part == "--mount"]
    assert any(m.endswith("target=/review,readonly") for m in mounts)
    assert command[-1] == "/review"


def test_command_runs_pytest_in_the_sandbox_image(project: Path) -> None:
    command = docker_command("c1")
    image = command.index(tools.testing.SANDBOX_IMAGE)
    assert command[image + 1:image + 4] == ["python", "-m", "pytest"]


# --- no silent fallback ------------------------------------------------------

def test_missing_docker_refuses_to_run(project: Path, no_docker: None) -> None:
    with pytest.raises(SandboxUnavailable, match="Docker is not installed"):
        run_tests()


def test_missing_docker_never_runs_code_locally(
    project: Path, no_docker: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("project code must not run outside the sandbox")
    monkeypatch.setattr(tools.testing, "_run_locally", fail)
    with pytest.raises(SandboxUnavailable):
        run_tests()


def test_validation_reports_not_run_with_the_reason(project: Path, no_docker: None) -> None:
    result = check_tests(lambda: True)
    assert result.status == "not_run"
    assert "Docker is not installed" in result.output


def test_agent_is_told_why_tests_did_not_run(project: Path, no_docker: None) -> None:
    result = dispatch(DeveloperToolset(), "run_tests", "{}", lambda name, args: True)
    assert result.startswith("ERROR: Docker is not installed")


def test_docker_not_running_is_reported(
    project: Path, docker_mode: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tools.testing.shutil, "which", lambda name: "docker")
    monkeypatch.setattr(tools.testing.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 1, b"", b""))
    with pytest.raises(SandboxUnavailable, match="not running"):
        require_docker()


def test_missing_image_says_how_to_build_it(
    project: Path, docker_mode: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tools.testing.shutil, "which", lambda name: "docker")
    results = iter([0, 1])  # docker info succeeds, image inspect fails
    monkeypatch.setattr(tools.testing.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, next(results), b"", b""))
    with pytest.raises(SandboxUnavailable, match="docker build -t"):
        require_docker()


def test_prompts_say_whether_code_is_sandboxed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tools.testing, "SANDBOX", "docker")
    assert "Docker sandbox" in sandbox_description()
    monkeypatch.setattr(tools.testing, "SANDBOX", "local")
    assert "WITHOUT a sandbox" in sandbox_description()


# --- real container, only where Docker is set up -----------------------------

def _docker_ready() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        require_docker()
    except SandboxUnavailable:
        return False
    return True


@pytest.mark.skipif(not _docker_ready(), reason="Docker and the sandbox image are not available")
def test_tests_really_run_in_a_container(project: Path, docker_mode: None) -> None:
    (project / "test_escape.py").write_text(
        "import socket, pathlib\n\n"
        "def test_no_network():\n"
        "    try:\n"
        "        socket.create_connection(('1.1.1.1', 53), timeout=2)\n"
        "    except OSError:\n"
        "        return\n"
        "    raise AssertionError('network was reachable')\n\n"
        "def test_project_is_read_only():\n"
        "    try:\n"
        "        pathlib.Path('/project/x.txt').write_text('x')\n"
        "    except OSError:\n"
        "        return\n"
        "    raise AssertionError('project was writable')\n",
        encoding="utf-8",
    )
    run = run_tests()
    assert run.passed, run.output
    assert not (project / "x.txt").exists()
