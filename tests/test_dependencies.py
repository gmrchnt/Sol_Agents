import subprocess
import sys
from pathlib import Path

import pytest

import tools.dependencies as deps
import tools.project
import tools.testing
from orchestrator.console import ConsoleUI
from orchestrator.runlog import RunLog
from orchestrator.tool_loop import dispatch, tool_specs_for
from orchestrator.validation import check_dependencies, check_tests
from orchestrator.workflow import ProjectRun
from schemas.requirements import Specification
from schemas.state import ProjectState
from tools.dependencies import (
    DependencyError,
    Requirement,
    docker_tag,
    dockerfile,
    missing_dependencies,
    parse_requirements,
    third_party_imports,
)
from tools.testing import DependenciesNotReady, install_dependencies, pending_packages, run_tests
from tools.toolsets import DeveloperToolset, ReviewerToolset


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    monkeypatch.setattr(tools.project, "PROJECT_PATH", root)
    monkeypatch.setattr(deps, "ENV_ROOT", tmp_path / "envs")
    return root


def _write(project: Path, path: str, text: str) -> None:
    target = project / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


# --- requirements.txt rules ---------------------------------------------------

def test_pinned_requirements_are_accepted() -> None:
    reqs = parse_requirements("# images\nPillow==11.0.0\nrequests[socks]==2.32.3  # http\n\n")
    assert [(r.line, r.name) for r in reqs] == [
        ("Pillow==11.0.0", "pillow"),
        ("requests[socks]==2.32.3", "requests"),
    ]


@pytest.mark.parametrize("line", [
    "pillow",                                     # unpinned
    "pillow>=11",                                 # range, not exact
    "--index-url https://evil.example/simple",    # another package index
    "--extra-index-url https://evil.example",
    "-r other.txt",
    "-e .",
    "git+https://github.com/x/y.git",
    "https://example.com/pkg.whl",
    "./local/pkg",
    "pkg @ https://example.com/pkg.tar.gz",
])
def test_unsafe_or_unpinned_lines_are_rejected(line: str) -> None:
    with pytest.raises(DependencyError, match="line 1 is not allowed"):
        parse_requirements(line)


def test_package_count_is_capped() -> None:
    text = "\n".join(f"pkg{n}==1.0" for n in range(51))
    with pytest.raises(DependencyError, match="limit is 50"):
        parse_requirements(text)


# --- finding undeclared imports -------------------------------------------------

def test_third_party_imports_skip_stdlib_project_and_pytest(project: Path) -> None:
    _write(project, "app.py",
           "import os, json\nfrom pathlib import Path\nimport helpers\n"
           "from . import sibling\nimport numpy as np\nfrom PIL import Image\n")
    _write(project, "helpers.py", "")
    _write(project, "tests/test_app.py", "import pytest\nimport app\nimport requests.adapters\n")
    assert sorted(third_party_imports()) == ["PIL", "numpy", "requests"]


def test_missing_dependency_names_the_package_to_install(project: Path) -> None:
    _write(project, "app.py", "from PIL import Image\n")
    assert missing_dependencies([]) == [
        "'PIL' is imported in app.py but not declared in requirements.txt "
        "(the package is called pillow)"
    ]


@pytest.mark.parametrize("module,declared", [
    ("PIL", "Pillow==11.0.0"),
    ("yaml", "PyYAML==6.0.2"),
    ("dateutil", "python_dateutil==2.9.0"),
    ("requests", "requests==2.32.3"),
])
def test_declared_packages_cover_their_imports(project: Path, module: str, declared: str) -> None:
    _write(project, "app.py", f"import {module}\n")
    assert missing_dependencies(parse_requirements(declared)) == []


def test_files_with_syntax_errors_do_not_crash_the_scan(project: Path) -> None:
    _write(project, "broken.py", "def f(:\n")
    assert third_party_imports() == {}


# --- validation -------------------------------------------------------------------

def test_dependency_check_not_configured_without_packages(project: Path) -> None:
    _write(project, "app.py", "import os\n")
    assert check_dependencies().status == "not_configured"


def test_dependency_check_fails_on_undeclared_import(project: Path) -> None:
    _write(project, "app.py", "import numpy\n")
    result = check_dependencies()
    assert result.status == "fail"
    assert "'numpy' is imported" in result.output


def test_dependency_check_passes_when_declared(project: Path) -> None:
    _write(project, "app.py", "import numpy\n")
    _write(project, "requirements.txt", "numpy==2.1.3\n")
    assert check_dependencies().status == "pass"


def test_dependency_check_fails_on_unsafe_requirements(project: Path) -> None:
    _write(project, "requirements.txt", "--index-url https://evil.example\n")
    assert check_dependencies().status == "fail"


# --- tests wait for installs ------------------------------------------------------

def test_uninstalled_packages_are_pending(project: Path) -> None:
    _write(project, "requirements.txt", "numpy==2.1.3\n")
    assert pending_packages() == ["numpy==2.1.3"]


def test_no_requirements_means_nothing_pending(project: Path) -> None:
    assert pending_packages() == []


def test_tests_refuse_to_run_until_packages_are_installed(project: Path) -> None:
    _write(project, "requirements.txt", "numpy==2.1.3\n")
    with pytest.raises(DependenciesNotReady, match="numpy==2.1.3"):
        run_tests()
    assert check_tests(lambda: True).status == "not_run"


def test_tests_use_the_installed_environment(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(project, "requirements.txt", "numpy==2.1.3\n")
    _write(project, "test_ok.py", "def test_ok():\n    pass\n")
    monkeypatch.setattr(tools.testing, "local_python", lambda reqs: Path(sys.executable))
    run = run_tests()
    assert run.passed
    assert run.command[0] == sys.executable


def test_install_builds_a_separate_environment(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[list[str]] = []
    monkeypatch.setattr(tools.testing, "build_local_env",
                        lambda reqs: built.append([r.line for r in reqs]))
    _write(project, "requirements.txt", "numpy==2.1.3\n")
    assert install_dependencies() == "Installed: numpy==2.1.3"
    assert built == [["numpy==2.1.3"]]


def test_install_with_no_requirements_does_nothing(project: Path) -> None:
    assert "nothing to install" in install_dependencies()


def test_local_environment_is_never_sol_agents_own(project: Path) -> None:
    env = deps.local_env_dir([Requirement("numpy==2.1.3", "numpy")])
    assert Path(sys.prefix).resolve() not in [env.resolve(), *env.resolve().parents]


# --- docker images ----------------------------------------------------------------

def test_docker_image_tag_follows_the_requirements() -> None:
    a = [Requirement("numpy==2.1.3", "numpy")]
    b = [Requirement("numpy==2.2.0", "numpy")]
    assert docker_tag(a, "base") == docker_tag(a, "base")
    assert docker_tag(a, "base") != docker_tag(b, "base")
    assert docker_tag(a, "base") != docker_tag(a, "other-base")


def test_docker_install_uses_prebuilt_packages_only() -> None:
    text = dockerfile("sol-agents-sandbox:latest")
    assert text.startswith("FROM sol-agents-sandbox:latest")
    assert "--only-binary=:all:" in text


def test_docker_tests_run_in_the_dependency_image(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(project, "requirements.txt", "numpy==2.1.3\n")
    monkeypatch.setattr(tools.testing, "SANDBOX", "docker")
    monkeypatch.setattr(tools.testing, "require_docker", lambda: None)
    monkeypatch.setattr(tools.testing, "docker_image_exists", lambda tag: True)
    seen: list[list[str]] = []
    monkeypatch.setattr(tools.testing.subprocess, "run",
                        lambda cmd, **k: seen.append(cmd) or subprocess.CompletedProcess(cmd, 0, "", ""))
    run_tests()
    expected = docker_tag(parse_requirements("numpy==2.1.3"), tools.testing.SANDBOX_IMAGE)
    assert expected in seen[0]


# --- the tool and the approval -------------------------------------------------------

def _names(toolset: object) -> set[str]:
    return {spec["function"]["name"] for spec in tool_specs_for(toolset)}


def test_only_the_developer_can_install(project: Path, tmp_path: Path) -> None:
    assert "install_dependencies" in _names(DeveloperToolset())
    assert "install_dependencies" not in _names(ReviewerToolset(tmp_path / "review"))


def test_installing_needs_approval(project: Path) -> None:
    _write(project, "requirements.txt", "numpy==2.1.3\n")
    assert "declined" in dispatch(DeveloperToolset(), "install_dependencies", "{}",
                                  lambda name, args: False)


def test_bad_requirements_are_reported_to_the_developer(project: Path) -> None:
    _write(project, "requirements.txt", "numpy\n")
    result = dispatch(DeveloperToolset(), "install_dependencies", "{}", lambda n, a: True)
    assert result.startswith("ERROR: requirements.txt line 1 is not allowed")


class ScriptedUI(ConsoleUI):
    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)
        self.said: list[str] = []
        self.asked: list[str] = []

    def say(self, text: str = "") -> None:
        self.said.append(text)

    def ask(self, prompt: str) -> str:
        self.asked.append(prompt)
        return self.answers.pop(0)


def _project_run(tmp_path: Path, ui: ScriptedUI) -> ProjectRun:
    spec = Specification(summary="s", requirements=[
        {"id": "REQ-001", "description": "d", "acceptance_criteria": ["x"],
         "source": "explicit", "source_text": "do it"}])
    state = ProjectState(run_id="r", user_prompt="do it", specification=spec,
                         project_path="p", original_branch="master",
                         run_branch="sol/run-r", base_commit="abc")
    return ProjectRun(state, None, None, ui, RunLog("r", root=tmp_path / "logs"))


@pytest.mark.parametrize("answer,installs", [("", 1), ("n", 0)])
def test_workflow_asks_before_installing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answer: str, installs: int
) -> None:
    import orchestrator.workflow as workflow
    calls: list[int] = []
    monkeypatch.setattr(workflow, "pending_packages", lambda: ["numpy==2.1.3"])
    monkeypatch.setattr(workflow, "install_dependencies",
                        lambda: calls.append(1) or "Installed: numpy==2.1.3")
    ui = ScriptedUI([answer])
    _project_run(tmp_path, ui)._sync_dependencies()
    assert len(calls) == installs
    assert "  + numpy==2.1.3" in ui.said


def test_install_approval_lists_the_packages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import orchestrator.workflow as workflow
    monkeypatch.setattr(workflow, "pending_packages", lambda: ["pillow==11.0.0"])
    ui = ScriptedUI([""])
    assert _project_run(tmp_path, ui)._approver("Developer")("install_dependencies", {})
    assert "+ pillow==11.0.0" in ui.asked[0]
