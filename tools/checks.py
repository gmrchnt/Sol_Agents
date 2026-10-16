"""Run the project's own commands - dependency setup, build, test, lint - per component."""

import fnmatch
import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from schemas.toolchain import Component, Toolchain
from tools import sandbox
from tools.dependencies import DependencyError
from tools.project import ProjectAccessError, list_project_files, resolve_project_path
from tools.sandbox import CommandRun, SandboxUnavailable, SuiteRunError, cache_dir
from tools.testing import DependenciesNotReady, SuiteRun, run_tests
from tools.toolchain import PYTEST_COMMAND, component_image, load_toolchain

CHECK_KINDS = ("build", "test", "lint", "static_analysis")


def is_default_pytest(component: Component) -> bool:
    """The root Python component without its own test command keeps Sol's pytest
    runner, which installs requirements.txt into the sandbox image."""
    return component.language == "python" and component.path == "." and component.test is None


def command_for(component: Component, kind: str) -> str | None:
    if kind == "test" and component.language == "python" and component.test is None:
        return PYTEST_COMMAND
    return getattr(component, kind)


def run_in_component(component: Component, command: str, *, writable: bool = False,
                     network: bool = False, review_dir: Path | None = None,
                     timeout: int | None = None, interactive: bool = False,
                     ports: tuple[int, ...] = ()) -> CommandRun:
    image, dockerfile = component_image(component)
    return sandbox.run_command(
        command, image=image, dockerfile=dockerfile, workdir=component.path,
        writable=writable, network=network, env=component.env, review_dir=review_dir,
        display=component.display, services=component.services, ports=ports,
        timeout=timeout, interactive=interactive,
    )


def describe_target(component: Component) -> str:
    where = "the project root" if component.path == "." else f"{component.path}/"
    if sandbox.SANDBOX != "docker":
        return f"{where}, directly on this machine WITHOUT a sandbox"
    image, dockerfile = component_image(component)
    built = f", built from {dockerfile}" if dockerfile else ""
    return f"{where}, in the {image} container{built}"


def command_target(component: str | None = None, image: str | None = None,
                   workdir: str | None = None) -> Component:
    """Where a requested command runs: a named component, the root one, or a bare image."""
    toolchain = load_toolchain()
    if component is not None:
        chosen = next((c for c in toolchain.components if c.name == component), None)
        if chosen is None:
            names = ", ".join(c.name for c in toolchain.components) or "none yet"
            raise ValueError(f"No component named {component!r}. Components: {names}.")
    elif image is None and toolchain.components:
        chosen = next((c for c in toolchain.components if c.path == "."),
                      toolchain.components[0])
    else:
        chosen = Component(name="shell")

    fields = chosen.model_dump()
    if image is not None:
        fields.update(image=image, dockerfile=None, language="other")
    if workdir is not None:
        fields["path"] = workdir
    return Component.model_validate(fields)


# --- checks -----------------------------------------------------------------------

@dataclass(frozen=True)
class CheckRun:
    component: str
    command: str
    result: CommandRun | SuiteRun | None = None
    skipped: str | None = None


def run_check(component: Component, kind: str) -> CheckRun | None:
    command = command_for(component, kind)
    if command is None:
        return None
    try:
        if kind == "test" and is_default_pytest(component):
            return CheckRun(component.name, "pytest", run_tests(display=component.display))
        network = kind == "test" and component.test_network
        return CheckRun(component.name, command,
                        run_in_component(component, command, network=network))
    except SandboxUnavailable:
        raise
    except (DependenciesNotReady, DependencyError, SuiteRunError, OSError,
            ProjectAccessError) as exc:
        return CheckRun(component.name, command, skipped=str(exc))


def run_checks(kind: str, toolchain: Toolchain | None = None) -> list[CheckRun]:
    toolchain = toolchain or load_toolchain()
    return [run for run in (run_check(c, kind) for c in toolchain.components) if run]


def describe_result(result: CommandRun | SuiteRun) -> str:
    if isinstance(result, SuiteRun):
        if result.no_tests:
            return f"no tests found (exit code {result.returncode})"
        verdict = "PASSED" if result.passed else "FAILED"
        return f"tests {verdict} (exit code {result.returncode})"
    if result.timed_out:
        return "TIMED OUT"
    return f"{'passed' if result.passed else 'FAILED'} (exit code {result.returncode})"


def format_check_runs(runs: list[CheckRun]) -> str:
    blocks = []
    for run in runs:
        if run.result is None:
            blocks.append(f"[{run.component}] {run.command}: not run - {run.skipped}")
        else:
            blocks.append(f"[{run.component}] {run.command}: {describe_result(run.result)}\n"
                          f"{run.result.output}".rstrip())
    return "\n\n".join(blocks)


def validation_plan(toolchain: Toolchain) -> list[str]:
    """What validation will run, for the human to approve."""
    lines = []
    for component in toolchain.components:
        commands = []
        if is_default_pytest(component):
            commands.append("import check")
        for kind in CHECK_KINDS:
            command = command_for(component, kind)
            if command is not None:
                network = kind == "test" and component.test_network
                commands.append(command + (" (WITH network)" if network else ""))
        if commands:
            lines.append(f"{component.name}: {', '.join(commands)}")
    return lines


# --- dependency setup -------------------------------------------------------------

def _matches(path: str, folder: str, patterns: list[str]) -> bool:
    if folder != "." and not path.startswith(folder + "/"):
        return False
    relative = path if folder == "." else path[len(folder) + 1:]
    name = PurePosixPath(relative).name
    return any(fnmatch.fnmatch(relative if "/" in pattern else name, pattern)
               for pattern in patterns)


def setup_fingerprint(component: Component) -> str:
    """Changes when the setup commands or the files they read change."""
    digest = hashlib.sha256("\n".join(component.setup).encode("utf-8"))
    for path in list_project_files():
        if _matches(path, component.path, component.setup_inputs):
            digest.update(path.encode("utf-8"))
            digest.update(resolve_project_path(Path(path)).read_bytes())
    return digest.hexdigest()[:16]


def _marker(component: Component) -> Path:
    return cache_dir() / "setup" / f"{component.name}.done"


def pending_setups(toolchain: Toolchain | None = None) -> list[Component]:
    """Components whose dependencies have not been installed for their current manifests."""
    toolchain = toolchain or load_toolchain()
    pending = []
    for component in toolchain.components:
        if not component.setup:
            continue
        marker = _marker(component)
        done = marker.is_file() and marker.read_text(encoding="utf-8") == setup_fingerprint(component)
        if done and component.setup_creates:
            done = resolve_project_path(Path(component.path) / component.setup_creates).exists()
        if not done:
            pending.append(component)
    return pending


def run_setup(component: Component) -> CommandRun:
    """Install a component's dependencies: with network, allowed to write the project."""
    run = None
    for command in component.setup:
        run = run_in_component(component, command, writable=True, network=True)
        if not run.passed:
            return run
    marker = _marker(component)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(setup_fingerprint(component), encoding="utf-8")
    return run
