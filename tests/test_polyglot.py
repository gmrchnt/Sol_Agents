"""Any language, any kind of project: toolchain detection, the general sandbox,
file tools beyond whole-file text writes, long sessions and big files."""

import base64
import json
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

import orchestrator.try_it as try_it
import tools.project
from llm import LLMClient
from orchestrator.console import ConsoleUI
from orchestrator.runlog import RunLog
from orchestrator.tool_loop import STUB, ToolLoopLimitError, compact, dispatch, run_tool_loop
from orchestrator.try_it import try_program
from orchestrator.validation import combine, run_validation
from orchestrator.workflow import EXTRA_STEPS, ProjectRun
from schemas.state import ProjectState
from schemas.toolchain import Component, Service, Toolchain
from schemas.validation import CheckResult
from tools import git, sandbox
from tools.checks import pending_setups, run_setup, validation_plan
from tools.project import ProjectAccessError, list_project_files
from tools.review_tests import write_review_file
from tools.toolchain import (
    PYTEST_COMMAND,
    TOOLCHAIN_FILE,
    ToolchainError,
    describe_toolchain,
    detect_toolchain,
    load_toolchain,
    save_toolchain,
)
from tools.toolsets import DeveloperToolset, ReadOnlyToolset, ReviewerToolset
from test_m7_workflow import SPEC

# Commands below run in the host shell (SANDBOX=local in tests): cmd.exe or sh.
PY = sys.executable if " " not in sys.executable else "python"
SETUP = f'{PY} -c "import pathlib; pathlib.Path(\'installed\').mkdir(exist_ok=True)"'
FAIL = f'{PY} -c "import sys; sys.exit(3)"'
PRINT_42 = f'{PY} -c "print(42)"'
MAKE_FILE = f'{PY} -c "open(\'made.txt\', \'w\').write(\'hi\')"'


class ScriptedUI(ConsoleUI):
    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)
        self.said: list[str] = []
        self.asked: list[str] = []

    def say(self, text: str = "") -> None:
        self.said.append(text)

    def ask(self, prompt: str) -> str:
        self.asked.append(prompt)
        if not self.answers:
            raise AssertionError(f"Unexpected prompt: {prompt!r}")
        return self.answers.pop(0)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", *args],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def _write(root: Path, path: str, text: str = "") -> None:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    monkeypatch.setattr(tools.project, "PROJECT_PATH", root)
    return root


@pytest.fixture
def repo(project: Path) -> Path:
    _git(project, "init", "-q")
    _write(project, "README.md", "x")
    _git(project, "add", "-A")
    _git(project, "commit", "-qm", "initial")
    return project


def _explicit(**fields) -> None:
    save_toolchain(Toolchain(components=[Component(name="app", **fields)]))


# --- detection ------------------------------------------------------------------

def test_node_project_uses_its_scripts_and_lockfile(project: Path) -> None:
    _write(project, "package.json", json.dumps({"scripts": {
        "build": "vite build", "test": "vitest run", "lint": "eslint .", "start": "vite"}}))
    _write(project, "package-lock.json", "{}")
    [node] = detect_toolchain().components
    assert (node.language, node.path, node.setup) == ("node", ".", ["npm ci"])
    assert (node.build, node.test, node.lint, node.run) == (
        "npm run build", "npm test", "npm run lint", "npm run start")
    assert node.setup_creates == "node_modules"


def test_npm_placeholder_test_script_is_not_a_test(project: Path) -> None:
    _write(project, "package.json", json.dumps(
        {"scripts": {"test": 'echo "Error: no test specified" && exit 1'}}))
    assert detect_toolchain().components[0].test is None


@pytest.mark.parametrize("files,language,test", [
    ({"Cargo.toml": ""}, "rust", "cargo test"),
    ({"go.mod": "module x"}, "go", "go test ./..."),
    ({"pom.xml": ""}, "java-maven", "mvn -B -q test"),
    ({"build.gradle.kts": "", "gradlew": ""}, "java-gradle", "sh ./gradlew --no-daemon -q test"),
    ({"App.csproj": ""}, "dotnet", "dotnet test --no-restore"),
    ({"CMakeLists.txt": ""}, "c-cpp", "ctest --test-dir .build --output-on-failure"),
    ({"Makefile": "all:\n\tcc main.c\n\ntest: all\n\t./a.out\n"}, "c-cpp", "make test"),
    ({"Gemfile": "", "spec/x_spec.rb": ""}, "ruby", "bundle exec rspec"),
    ({"composer.json": json.dumps({"require-dev": {"phpunit/phpunit": "^11"}})},
     "php", "vendor/bin/phpunit"),
    ({"mix.exs": ""}, "elixir", "mix test"),
    ({"Package.swift": ""}, "swift", "swift test"),
    ({"pubspec.yaml": ""}, "dart", "dart test"),
    ({"stack.yaml": ""}, "haskell", "stack test"),
    ({"deno.json": "{}"}, "deno", "deno test -A"),
])
def test_each_language_gets_its_own_tools(project: Path, files: dict, language: str,
                                          test: str) -> None:
    for path, text in files.items():
        _write(project, path, text)
    [component] = detect_toolchain().components
    assert component.language == language
    assert component.test == test


def test_monorepo_gets_a_component_per_folder(project: Path) -> None:
    _write(project, "frontend/package.json", json.dumps({"scripts": {"test": "jest"}}))
    _write(project, "backend/requirements.txt", "flask==3.0.3\n")
    _write(project, "backend/app.py", "import flask\n")
    components = {c.name: c for c in detect_toolchain().components}
    assert set(components) == {"frontend", "backend"}
    assert components["frontend"].path == "frontend"
    assert components["backend"].test == PYTEST_COMMAND
    assert components["backend"].setup[0].startswith("pip install --user")


def test_workspace_members_belong_to_the_root(project: Path) -> None:
    _write(project, "Cargo.toml", "[workspace]\n")
    _write(project, "crates/core/Cargo.toml", "")
    assert [c.path for c in detect_toolchain().components] == ["."]


def test_python_gui_code_gets_a_virtual_display(project: Path) -> None:
    _write(project, "game.py", "import tkinter\n")
    [python] = detect_toolchain().components
    assert python.name == "python" and python.display


def test_plain_site_can_be_served(project: Path) -> None:
    _write(project, "index.html", "<h1>hi</h1>")
    [web] = detect_toolchain().components
    assert web.run.startswith("python -m http.server") and web.ports == [8000]


def test_empty_project_explains_how_to_start(project: Path) -> None:
    toolchain = detect_toolchain()
    assert toolchain.components == []
    assert "rust:1-bookworm" in describe_toolchain(toolchain, explicit=False)


def test_explicit_toolchain_wins(project: Path) -> None:
    _write(project, "package.json", "{}")
    _explicit(language="go", test="go test ./...")
    assert [c.name for c in load_toolchain().components] == ["app"]


def test_broken_toolchain_file_is_reported(project: Path) -> None:
    _write(project, TOOLCHAIN_FILE, '{"components": [{"name": "x", "path": "../out"}]}')
    with pytest.raises(ToolchainError, match="inside the project"):
        load_toolchain()


@pytest.mark.parametrize("path", ["../x", "/etc", "C:/Windows", ".git/hooks", ".sol"])
def test_component_paths_stay_inside_the_project(path: str) -> None:
    with pytest.raises(ValueError):
        Component(name="x", path=path)


# --- the sandbox ----------------------------------------------------------------

def _flags(args: list[str], name: str) -> list[str]:
    return [args[i + 1] for i, part in enumerate(args) if part == name]


def test_writable_runs_still_protect_git_and_sol(project: Path) -> None:
    (project / ".git").mkdir()
    (project / ".sol").mkdir()
    args = sandbox.docker_run_args("npm ci", image="node:22", container="c", writable=True,
                                   network="bridge")
    mounts = _flags(args, "--mount")
    assert mounts[0] == f"type=bind,source={project.resolve()},target=/project"
    assert (f"type=bind,source={project.resolve() / '.git'},target=/project/.git,readonly"
            in mounts)
    assert any(m.endswith("target=/project/.sol,readonly") for m in mounts)
    assert any(m.endswith("target=/cache") for m in mounts)


def test_checks_run_read_only_and_offline(project: Path) -> None:
    args = sandbox.docker_run_args("cargo test", image="rust:1", container="c")
    assert _flags(args, "--mount")[0].endswith("target=/project,readonly")
    assert _flags(args, "--network") == ["none"]
    env = _flags(args, "-e")
    assert "CARGO_NET_OFFLINE=true" in env and "HOME=/cache/home" in env
    assert "CARGO_NET_OFFLINE=true" not in _flags(
        sandbox.docker_run_args("npm ci", image="node", container="c", network="bridge",
                                internet=True), "-e")


def test_container_is_locked_down(project: Path) -> None:
    args = sandbox.docker_run_args("make", image="gcc", container="c")
    assert "--read-only" in args
    assert _flags(args, "--cap-drop") == ["ALL"]
    assert _flags(args, "--security-opt") == ["no-new-privileges"]
    assert {"--memory", "--cpus", "--pids-limit", "--user"} <= set(args)
    assert args[-5:] == ["--entrypoint", "sh", "gcc", "-c", "make"]


def test_display_ports_and_folder(project: Path) -> None:
    args = sandbox.docker_run_args("python app.py", image="i", container="c", display=True,
                                   ports=(8000,), workdir="web")
    assert args[-7:] == ["--entrypoint", "xvfb-run", "i", "-a", "sh", "-c", "python app.py"]
    assert _flags(args, "-p") == ["127.0.0.1:8000:8000"]
    assert _flags(args, "-w") == ["/project/web"]
    assert "SDL_VIDEODRIVER=dummy" not in _flags(args, "-e")


def test_local_command_runs_in_its_folder(project: Path) -> None:
    (project / "sub").mkdir()
    run = sandbox.run_command(f'{PY} -c "import os; print(os.getcwd())"', workdir="sub")
    assert run.passed
    assert run.output.strip().endswith("sub")


def test_slow_command_is_stopped_with_its_children(project: Path) -> None:
    run = sandbox.run_command(f'{PY} -c "import time; time.sleep(60)"', timeout=1)
    assert run.timed_out and not run.passed


def test_workdir_must_be_inside_the_project(project: Path) -> None:
    with pytest.raises(ProjectAccessError):
        sandbox.run_command("echo hi", workdir="../elsewhere")


def test_long_output_keeps_its_start_and_end() -> None:
    short, cut = sandbox.truncate_output("start" + "x" * 50_000 + "end")
    assert cut and short.startswith("start") and short.endswith("end")
    assert len(short) < 21_000


def _docker_calls(monkeypatch: pytest.MonkeyPatch, exists: bool = False) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake(args, **kwargs):
        calls.append(args)
        inspect = args[:3] == ["docker", "image", "inspect"]
        return subprocess.CompletedProcess(args, 1 if inspect and not exists else 0, "", "")

    monkeypatch.setattr(sandbox.subprocess, "run", fake)
    return calls


def test_missing_public_image_is_pulled(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _docker_calls(monkeypatch)
    sandbox.ensure_image("node:22-bookworm")
    assert calls[-1] == ["docker", "pull", "node:22-bookworm"]


def test_missing_sol_image_is_built(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _docker_calls(monkeypatch)
    sandbox.ensure_image("sol-agents-cpp:latest")
    assert calls[-1][:4] == ["docker", "build", "-t", "sol-agents-cpp:latest"]
    assert calls[-1][5].endswith("cpp.Dockerfile")


def test_project_dockerfile_builds_from_the_project(project: Path,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    _write(project, "Dockerfile.sol", "FROM debian\n")
    tag = sandbox.project_image("Dockerfile.sol")
    calls = _docker_calls(monkeypatch)
    sandbox.ensure_image(tag, "Dockerfile.sol")
    assert calls[-1][:4] == ["docker", "build", "-t", tag]
    assert calls[-1][-1] == str(project.resolve())


def test_present_image_is_left_alone(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _docker_calls(monkeypatch, exists=True)
    sandbox.ensure_image("node:22")
    assert len(calls) == 1


def test_services_share_a_private_network(project: Path,
                                          monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _docker_calls(monkeypatch, exists=True)
    db = Service(name="db", image="postgres:16", env={"POSTGRES_PASSWORD": "x"},
                 ready_command="pg_isready")
    network, started = sandbox._start_services([db], internet=False)
    assert calls[0][:4] == ["docker", "network", "create", "--internal"]
    run = next(call for call in calls if call[:3] == ["docker", "run", "-d"])
    assert run[run.index("--network-alias") + 1] == "db"
    assert started == [f"{network}-db"]


# --- the project's own commands ----------------------------------------------------

def test_component_commands_drive_validation(repo: Path) -> None:
    _explicit(build=FAIL, test=PRINT_42)
    report = run_validation(git.current_commit(), lambda: True)
    assert report.build.status == "fail"
    assert report.tests.status == "pass" and "42" in report.tests.output
    assert report.lint.status == "not_configured"
    assert any(failure.startswith("build failed") for failure in report.failures)


def test_declined_component_checks_do_not_run(repo: Path) -> None:
    _explicit(test=MAKE_FILE)
    report = run_validation(git.current_commit(), lambda: False)
    assert report.tests.status == "not_run"
    assert not (repo / "made.txt").exists()


def test_a_non_python_project_runs_end_to_end(repo: Path, tmp_path: Path) -> None:
    from test_m7_workflow import FakeDeveloper, FakeReviewer, _review, _run
    _explicit(language="other", test=f'{PY} -c "import pathlib, sys; '
                                     f'sys.exit(0 if pathlib.Path(\'hello.txt\').exists() else 1)"')
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "toolchain")
    # approve the checks, accept, do not merge
    run, ui = _run(repo, tmp_path, FakeDeveloper(repo, [{"hello.txt": "hi"}]),
                   FakeReviewer([_review()]), ["y", "a", "n"])
    state = run.execute()

    assert state.status == "complete"
    assert state.validation.tests.status == "pass"
    assert ui.answers == []


def test_validation_plan_lists_every_command() -> None:
    toolchain = Toolchain(components=[Component(name="web", build="npm run build",
                                                test="npm test", test_network=True)])
    assert validation_plan(toolchain) == ["web: npm run build, npm test (WITH network)"]


def test_one_failing_component_fails_the_check() -> None:
    result = combine([("web", CheckResult(status="pass", command="a", output="fine")),
                      ("api", CheckResult(status="fail", command="b", output="boom"))])
    assert result.status == "fail"
    assert "[api] fail" in result.output


def test_setup_runs_once_per_manifest(project: Path) -> None:
    _write(project, "deps.txt", "a")
    _explicit(setup=[SETUP], setup_inputs=["deps.txt"], setup_creates="installed")
    [pending] = pending_setups()
    assert run_setup(pending).passed
    assert (project / "installed").is_dir()
    assert pending_setups() == []

    _write(project, "deps.txt", "b")
    assert [c.name for c in pending_setups()] == ["app"]
    run_setup(pending)
    (project / "installed").rmdir()
    assert [c.name for c in pending_setups()] == ["app"]


# --- file tools -------------------------------------------------------------------

def test_edit_replaces_exactly_one_match(project: Path) -> None:
    _write(project, "a.py", "x = 1\ny = 2\n")
    assert DeveloperToolset().edit_file("a.py", "y = 2", "y = 3") == "edited a.py"
    assert (project / "a.py").read_text(encoding="utf-8") == "x = 1\ny = 3\n"


def test_edit_refuses_ambiguous_or_missing_text(project: Path) -> None:
    _write(project, "a.py", "x = 1\nx = 1\n")
    with pytest.raises(ValueError, match="2 times"):
        DeveloperToolset().edit_file("a.py", "x = 1", "z")
    with pytest.raises(ValueError, match="not found"):
        DeveloperToolset().edit_file("a.py", "nope", "z")


def test_edit_keeps_windows_line_endings(project: Path) -> None:
    (project / "a.txt").write_bytes(b"one\r\ntwo\r\n")
    DeveloperToolset().edit_file("a.txt", "one\ntwo", "one\nthree")
    assert (project / "a.txt").read_bytes() == b"one\r\nthree\r\n"


def test_files_can_be_moved_and_deleted(project: Path) -> None:
    _write(project, "old.py", "x")
    toolset = DeveloperToolset()
    toolset.move_file("old.py", "pkg/new.py")
    assert (project / "pkg" / "new.py").read_text(encoding="utf-8") == "x"
    assert not (project / "old.py").exists()
    toolset.delete_file("pkg/new.py")
    assert not (project / "pkg" / "new.py").exists()


@pytest.mark.parametrize("change", [
    lambda t: t.delete_file(".git/config"),
    lambda t: t.move_file("a.py", ".sol/a.py"),
    lambda t: t.edit_file(".sol/project.json", "a", "b"),
])
def test_protected_folders_stay_protected(project: Path, change) -> None:
    for path in (".git/config", "a.py", ".sol/project.json"):
        _write(project, path, "a")
    with pytest.raises(ProjectAccessError):
        change(DeveloperToolset())


def test_move_never_overwrites(project: Path) -> None:
    _write(project, "a.py", "a")
    _write(project, "b.py", "b")
    with pytest.raises(ValueError, match="already exists"):
        DeveloperToolset().move_file("a.py", "b.py")


def test_binary_files_are_written_and_not_dumped(project: Path) -> None:
    png = b"\x89PNG\r\n\x1a\n" + bytes(range(256))
    DeveloperToolset().write_file("logo.png", base64.b64encode(png).decode(), encoding="base64")
    assert (project / "logo.png").read_bytes() == png
    assert "binary file" in DeveloperToolset().read_file("logo.png")


def test_large_files_are_read_in_parts(project: Path) -> None:
    _write(project, "big.txt", "".join(f"line {n}\n" for n in range(1, 20_001)))
    first = DeveloperToolset().read_file("big.txt")
    assert first.startswith("[big.txt: lines 1-") and "of 20000" in first
    part = DeveloperToolset().read_file("big.txt", start_line=100, end_line=102)
    assert part.splitlines()[1:] == ["line 100", "line 101", "line 102"]


def test_listing_can_focus_on_a_folder(project: Path) -> None:
    for path in ("src/a.rs", "src/b/c.rs", "README.md"):
        _write(project, path)
    assert ReadOnlyToolset().list_files("src") == ["src/a.rs", "src/b/c.rs"]


def test_listing_follows_gitignore(project: Path) -> None:
    _git(project, "init", "-q")
    _write(project, ".gitignore", "dist/\n")
    for path in ("dist/app.js", "src/app.ts", "node_modules/x/index.js"):
        _write(project, path, "x")
    assert list_project_files() == [".gitignore", "src/app.ts"]


def test_default_excludes_are_written_once(project: Path) -> None:
    _git(project, "init", "-q")
    git.ensure_default_excludes()
    git.ensure_default_excludes()
    text = (project / ".git" / "info" / "exclude").read_text(encoding="utf-8")
    assert text.count("node_modules/") == 1
    _write(project, "node_modules/a.js", "x")
    assert "node_modules" not in _git(project, "status", "--porcelain")


# --- tools through dispatch -----------------------------------------------------

def _allow(name: str, arguments: dict) -> bool:
    return True


def test_run_command_reports_the_files_it_changed(project: Path) -> None:
    result = dispatch(DeveloperToolset(), "run_command", json.dumps({"command": MAKE_FILE}),
                      _allow)
    assert result.startswith("exit code 0")
    assert "+ made.txt" in result


def test_commands_need_approval(project: Path) -> None:
    result = dispatch(DeveloperToolset(), "run_command", '{"command": "echo hi"}',
                      lambda name, arguments: False)
    assert "declined" in result


def test_network_must_be_a_real_boolean(project: Path) -> None:
    result = dispatch(DeveloperToolset(), "run_command",
                      '{"command": "echo hi", "network": "yes"}', _allow)
    assert result == "ERROR: network must be a boolean"


def test_null_optional_arguments_are_ignored(project: Path) -> None:
    _write(project, "a.txt", "hello")
    result = dispatch(DeveloperToolset(), "read_file",
                      '{"relative_path": "a.txt", "start_line": null}', _allow)
    assert result == "hello"


def test_toolchain_can_be_set(project: Path) -> None:
    proposal = {"toolchain": {"components": [{"name": "api", "language": "go",
                                              "test": "go test ./..."}]}}
    result = dispatch(DeveloperToolset(), "set_toolchain", json.dumps(proposal), _allow)
    assert result.startswith("Saved.")
    assert load_toolchain().components[0].test == "go test ./..."


def test_invalid_toolchain_is_refused(project: Path) -> None:
    proposal = {"toolchain": {"components": [{"name": "../x"}]}}
    result = dispatch(DeveloperToolset(), "set_toolchain", json.dumps(proposal), _allow)
    assert result.startswith("ERROR")
    assert not (project / ".sol" / "toolchain.json").exists()


def test_both_agents_can_see_the_toolchain(project: Path, tmp_path: Path) -> None:
    _write(project, "go.mod", "module x")
    assert "go test ./..." in ReviewerToolset(tmp_path / "review").show_toolchain()


@pytest.mark.parametrize("path", ["../x.js", "/abs.js", "a/../../x", "bad name.js"])
def test_review_files_stay_in_the_review_area(tmp_path: Path, path: str) -> None:
    with pytest.raises(ValueError):
        write_review_file(tmp_path, path, "x")


def test_review_commands_find_review_files(project: Path, tmp_path: Path) -> None:
    toolset = ReviewerToolset(tmp_path / "review")
    toolset.write_review_file("checks/hello.py", "print('from review')")
    result = toolset.run_review_command(
        f'{PY} -c "import os, runpy; runpy.run_path(os.environ[\'REVIEW_DIR\'] '
        f'+ \'/checks/hello.py\')"')
    assert result.startswith("exit code 0")
    assert "from review" in result


# --- long sessions ----------------------------------------------------------------

class FakeLLM:
    def __init__(self, replies: list[dict]) -> None:
        self.replies = list(replies)

    def complete(self, messages, *, tools, temperature=None):
        return self.replies.pop(0)


def _list_call() -> dict:
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": "c", "type": "function", "function": {"name": "list_files", "arguments": "{}"}}]}


def test_budget_can_be_extended(project: Path) -> None:
    llm = FakeLLM([_list_call() for _ in range(3)] + [{"role": "assistant", "content": "done"}])
    asked: list[int] = []
    result = run_tool_loop(llm, DeveloperToolset(), "s", "t", _allow, max_steps=2,
                           on_limit=lambda used: asked.append(used) or 2)
    assert result.final_text == "done"
    assert asked == [2]


def test_refusing_more_steps_stops_the_loop(project: Path) -> None:
    llm = FakeLLM([_list_call() for _ in range(3)])
    with pytest.raises(ToolLoopLimitError):
        run_tool_loop(llm, DeveloperToolset(), "s", "t", _allow, max_steps=2,
                      on_limit=lambda used: 0)


def test_old_results_are_compacted() -> None:
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "task"}]
    for n in range(20):
        arguments = json.dumps({"relative_path": f"f{n}.py", "content": "x" * 5_000})
        messages.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": str(n), "function": {"name": "write_file", "arguments": arguments}}]})
        messages.append({"role": "tool", "tool_call_id": str(n), "content": "y" * 5_000})
    compact(messages, budget=60_000)

    assert messages[1]["content"] == "task"
    assert messages[3]["content"] == STUB
    assert json.loads(messages[2]["tool_calls"][0]["function"]["arguments"]) == {
        "relative_path": "f0.py", "content": "[omitted]"}
    assert messages[-1]["content"] == "y" * 5_000


# --- the human's side -------------------------------------------------------------

def _project_run(tmp_path: Path, answers: list[str]) -> ProjectRun:
    state = ProjectState(run_id="r", user_prompt="x", specification=SPEC, project_path="p",
                         original_branch="main", run_branch="sol/run-r", base_commit="abc")
    return ProjectRun(state, None, None, ScriptedUI(answers),
                      RunLog("r", root=tmp_path / "logs"))


def test_always_allows_the_same_kind_of_command(project: Path, tmp_path: Path) -> None:
    approve = _project_run(tmp_path, ["a"])._approver("Developer")
    assert approve("run_command", {"command": "npm test"})
    assert approve("run_command", {"command": "npm run build"})


def test_network_commands_default_to_no(project: Path, tmp_path: Path) -> None:
    run = _project_run(tmp_path, ["a", ""])
    approve = run._approver("Developer")
    approve("run_command", {"command": "npm test"})
    assert not approve("run_command", {"command": "npm ci", "network": True})
    assert "WITH network" in run.ui.asked[-1]


def test_toolchain_change_needs_an_explicit_yes(project: Path, tmp_path: Path) -> None:
    run = _project_run(tmp_path, ["", "y"])
    proposal = {"toolchain": {"components": [{"name": "api", "test": "go test ./..."}]}}
    assert run._approver("Developer")("set_toolchain", proposal)
    assert len(run.ui.asked) == 2
    assert "PROPOSED" in run.ui.asked[0] and "go test ./..." in run.ui.asked[0]


def test_more_steps_are_the_humans_call(project: Path, tmp_path: Path) -> None:
    more = _project_run(tmp_path, ["", "n"])._more_steps("Developer")
    assert more(40) == EXTRA_STEPS
    assert more(80) == 0


def test_workflow_installs_pending_setups(project: Path, tmp_path: Path) -> None:
    _explicit(setup=[SETUP], setup_creates="installed")
    _project_run(tmp_path, [""])._sync_setups()
    assert (project / "installed").is_dir()


def test_any_component_can_be_tried(repo: Path, tmp_path: Path) -> None:
    _explicit(run=MAKE_FILE)
    ui = ScriptedUI(["", "", ""])  # pick 1, confirm, restore
    try_program(ui, RunLog("r", root=tmp_path / "logs"), git.current_commit())
    assert any("ended with exit code 0" in line for line in ui.said)
    assert not (repo / "made.txt").exists()


def test_pages_open_in_the_browser(repo: Path, tmp_path: Path,
                                   monkeypatch: pytest.MonkeyPatch) -> None:
    _write(repo, "index.html", "<h1>hi</h1>")
    opened: list[str] = []
    monkeypatch.setattr(try_it.webbrowser, "open", opened.append)
    try_program(ScriptedUI(["2"]), RunLog("r", root=tmp_path / "logs"), git.current_commit())
    assert opened and opened[0].endswith("index.html")


def test_requests_can_span_lines() -> None:
    assert ScriptedUI(["first line", "  second", ""]).ask_text("p") == "first line\n  second"


def test_request_can_come_from_a_file(tmp_path: Path) -> None:
    (tmp_path / "request.md").write_text("Build a Rust CLI\n", encoding="utf-8")
    assert ScriptedUI([f"@{tmp_path / 'request.md'}"]).ask_text("p") == "Build a Rust CLI"


def test_api_key_is_sent_when_set() -> None:
    seen: list[httpx.Request] = []

    def reply(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    LLMClient("m", base_url="http://x/v1", api_key="sk-test",
              transport=httpx.MockTransport(reply)).chat([{"role": "user", "content": "hi"}])
    LLMClient("m", base_url="http://x/v1",
              transport=httpx.MockTransport(reply)).chat([{"role": "user", "content": "hi"}])
    assert seen[0].headers["Authorization"] == "Bearer sk-test"
    assert "Authorization" not in seen[1].headers


def test_models_see_lockfiles_named_not_shown(repo: Path, tmp_path: Path) -> None:
    base = git.current_commit()
    _write(repo, "web/package-lock.json", '{"lockfileVersion": 3}\n' * 500)
    _write(repo, "web/app.js", "console.log('hi')\n")
    _git(repo, "add", "-A")
    patch = _patch_run(base, tmp_path)._patch()
    assert "console.log('hi')" in patch
    assert "lockfileVersion" not in patch
    assert patch.endswith("(lockfiles changed, not shown: web/package-lock.json)")


def test_huge_diffs_are_cut_for_the_prompt(repo: Path, tmp_path: Path) -> None:
    base = git.current_commit()
    _write(repo, "big.txt", "line\n" * 20_000)
    _git(repo, "add", "-A")
    patch = _patch_run(base, tmp_path)._patch()
    assert len(patch) < 31_000
    assert "call git_diff for the rest" in patch


def test_build_checks_may_write_into_the_project(repo: Path,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    import orchestrator.validation as validation
    seen: dict = {}

    def fake(component, command, **kwargs):
        seen.update(kwargs)
        return sandbox.CommandRun(command, 0, "")

    monkeypatch.setattr(validation, "run_in_component", fake)
    validation.check_command(Component(name="web", build="vite build"), "build", lambda: True)
    assert seen["writable"] is True


def _patch_run(base: str, tmp_path: Path) -> ProjectRun:
    run = _project_run(tmp_path, [])
    run.state.base_commit = base
    return run


def test_files_made_by_checks_are_cleaned_up(repo: Path) -> None:
    _write(repo, "notes.txt", "mine")  # untracked before the checks: left alone
    build = f'{PY} -c "import os; os.makedirs(\'dist/assets\'); open(\'dist/assets/a.js\', \'w\')"'
    _explicit(build=build)
    report = run_validation(git.current_commit(), lambda: True)
    assert report.build.status == "pass"
    assert not (repo / "dist").exists()
    assert (repo / "notes.txt").exists()
