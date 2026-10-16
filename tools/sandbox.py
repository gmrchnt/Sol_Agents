"""Run any command for the project: in a locked-down container, or with SANDBOX=local
directly on this machine.

Containers get the project at /project (read-only unless the command is meant to
change it), a per-project cache at /cache, no network unless asked for, and no way
to touch .git or .sol: a git hook written there would run on this machine.
"""

import hashlib
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import config
from schemas.toolchain import Service
from tools.project import SOL_DIR, get_project_path, resolve_project_path

SANDBOX = config.SANDBOX
SANDBOX_IMAGE = config.SANDBOX_IMAGE
CACHE_ROOT = config.CACHE_ROOT
COMMAND_TIMEOUT = config.COMMAND_TIMEOUT
MAX_COMMAND_TIMEOUT = 3_600
DOCKER_CHECK_TIMEOUT = 15
IMAGE_TIMEOUT = 1_800
SERVICE_READY_TIMEOUT = 90
MAX_OUTPUT_CHARS = 20_000
OUTPUT_HEAD_CHARS = 6_000

# Images Sol builds itself from sandbox/, the first time they are needed.
LOCAL_DOCKERFILES = {
    SANDBOX_IMAGE: "Dockerfile",
    "sol-agents-cpp:latest": "cpp.Dockerfile",
}

# Tools keep their caches under /cache, outside the project, so a dependency
# downloaded once (with network) is still there for offline builds and tests.
CACHE_ENV = {
    "HOME": "/cache/home",
    "XDG_CACHE_HOME": "/cache/home/.cache",
    "CARGO_HOME": "/cache/cargo",
    "GOPATH": "/cache/go",
    "GRADLE_USER_HOME": "/cache/gradle",
    "BUNDLE_PATH": "/cache/bundle",
    "COMPOSER_HOME": "/cache/composer",
    "NUGET_PACKAGES": "/cache/nuget",
    "DOTNET_CLI_HOME": "/cache/home",
    "DENO_DIR": "/cache/deno",
    "STACK_ROOT": "/cache/stack",
    "CABAL_DIR": "/cache/cabal",
    "MIX_HOME": "/cache/mix",
    "HEX_HOME": "/cache/hex",
    "PUB_CACHE": "/cache/pub",
    "npm_config_cache": "/cache/npm",
    "PIP_CACHE_DIR": "/cache/pip",
}
# Non-interactive, quiet, and headless wherever a tool allows it.
QUIET_ENV = {
    "CI": "true",
    "npm_config_yes": "true",
    "DOTNET_CLI_TELEMETRY_OPTOUT": "1",
    "DOTNET_NOLOGO": "1",
    "NEXT_TELEMETRY_DISABLED": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "MPLBACKEND": "Agg",
    "SDL_AUDIODRIVER": "dummy",
}
# Fail fast with a clear message instead of hanging on a network that is not there.
OFFLINE_ENV = {
    "CARGO_NET_OFFLINE": "true",
    "MAVEN_ARGS": "--offline",
    "GOPROXY": "off",
}


class SuiteRunError(Exception):
    """Raised when the test command cannot be run at all."""


class SandboxUnavailable(SuiteRunError):
    """The sandbox is required but cannot be used, so nothing was run."""


@dataclass(frozen=True)
class CommandRun:
    command: str
    returncode: int
    output: str
    truncated: bool = False
    timed_out: bool = False

    @property
    def passed(self) -> bool:
        return self.returncode == 0 and not self.timed_out


def truncate_output(text: str) -> tuple[str, bool]:
    """Keep the start (first errors) and the end (summary) of long output."""
    if len(text) <= MAX_OUTPUT_CHARS:
        return text, False
    tail = MAX_OUTPUT_CHARS - OUTPUT_HEAD_CHARS
    omitted = len(text) - MAX_OUTPUT_CHARS
    return f"{text[:OUTPUT_HEAD_CHARS]}\n\n... [{omitted} characters omitted] ...\n\n{text[-tail:]}", True


def container_user() -> str:
    # Files a container writes into the project must belong to whoever runs Sol.
    # Docker Desktop (Windows, macOS) maps ownership itself.
    if hasattr(os, "getuid"):
        return f"{os.getuid()}:{os.getgid()}"
    return "65534:65534"


def cache_dir() -> Path:
    root = get_project_path().resolve()
    key = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:12]
    folder = CACHE_ROOT / f"{root.name}-{key}"
    (folder / "home").mkdir(parents=True, exist_ok=True)
    return folder


# --- docker ---------------------------------------------------------------------

def require_daemon() -> None:
    if shutil.which("docker") is None:
        raise SandboxUnavailable(
            "Docker is not installed, so nothing was run. Install Docker Desktop, "
            "or set SANDBOX=local in .env to run project code without a sandbox."
        )
    try:
        info = subprocess.run(["docker", "info"], capture_output=True,
                              timeout=DOCKER_CHECK_TIMEOUT)
    except subprocess.TimeoutExpired as exc:
        raise SandboxUnavailable("Docker did not respond, so nothing was run.") from exc
    if info.returncode != 0:
        raise SandboxUnavailable(
            "Docker is installed but not running, so nothing was run. "
            "Start Docker Desktop and try again."
        )


def image_exists(image: str) -> bool:
    try:
        done = subprocess.run(["docker", "image", "inspect", image], capture_output=True,
                              timeout=DOCKER_CHECK_TIMEOUT)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return done.returncode == 0


def project_image(dockerfile: str) -> str:
    """The tag for an image built from a Dockerfile in the project."""
    data = resolve_project_path(Path(dockerfile)).read_bytes()
    root = str(get_project_path().resolve()).encode("utf-8")
    return f"sol-project:{hashlib.sha256(data + root).hexdigest()[:12]}"


def _docker(args: list[str], what: str, error: type[Exception] = SandboxUnavailable) -> None:
    try:
        done = subprocess.run(args, capture_output=True, timeout=IMAGE_TIMEOUT,
                              encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired as exc:
        raise error(f"{what} took longer than {IMAGE_TIMEOUT}s.") from exc
    if done.returncode != 0:
        output, _ = truncate_output(((done.stdout or "") + (done.stderr or "")).strip())
        raise error(f"{what} failed:\n{output}")


def ensure_image(image: str, dockerfile: str | None = None) -> None:
    """Pull the image, or build it from Sol's or the project's Dockerfile, if missing."""
    if image_exists(image):
        return
    if dockerfile is not None:
        # A broken Dockerfile is the project's defect, not a missing sandbox.
        root = get_project_path().resolve()
        _docker(["docker", "build", "-t", image, "-f",
                 str(resolve_project_path(Path(dockerfile))), str(root)],
                f"Building {image} from {dockerfile}", SuiteRunError)
    elif image in LOCAL_DOCKERFILES:
        folder = config.BASE_DIR / "sandbox"
        _docker(["docker", "build", "-t", image, "-f",
                 str(folder / LOCAL_DOCKERFILES[image]), str(folder)],
                f"Building the sandbox image {image}")
    else:
        _docker(["docker", "pull", image], f"Pulling {image}")


def docker_run_args(
    command: str,
    *,
    image: str,
    container: str,
    workdir: str = ".",
    writable: bool = False,
    network: str = "none",
    internet: bool = False,
    env: dict[str, str] | None = None,
    review_dir: Path | None = None,
    display: bool = False,
    ports: tuple[int, ...] = (),
    interactive: bool = False,
) -> list[str]:
    root = get_project_path().resolve()
    args = [
        "docker", "run", "--rm",
        "--name", container,
        "--network", network,
        "--memory", config.SANDBOX_MEMORY,
        "--cpus", config.SANDBOX_CPUS,
        "--pids-limit", str(config.SANDBOX_PIDS),
        "--read-only",
        "--tmpfs", "/tmp:rw,exec,size=2g",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--user", container_user(),
    ]
    variables = {**CACHE_ENV, **QUIET_ENV}
    if not internet:
        variables.update(OFFLINE_ENV)
    if not display:
        variables["SDL_VIDEODRIVER"] = "dummy"
    variables.update(env or {})
    for key, value in variables.items():
        args += ["-e", f"{key}={value}"]

    project = f"type=bind,source={root},target=/project"
    args += ["--mount", project if writable else project + ",readonly"]
    if writable:
        for protected in (".git", SOL_DIR):
            if (root / protected).exists():
                args += ["--mount", f"type=bind,source={root / protected},"
                                    f"target=/project/{protected},readonly"]
    args += ["--mount", f"type=bind,source={cache_dir()},target=/cache"]
    if review_dir is not None:
        args += ["--mount", f"type=bind,source={review_dir.resolve()},target=/review,readonly"]
    for port in ports:
        args += ["-p", f"127.0.0.1:{port}:{port}"]
    if interactive:
        args += ["-i"] + (["-t"] if sys.stdin.isatty() else [])

    folder = "/project" if workdir in {"", "."} else f"/project/{workdir}"
    args += ["-w", folder]
    if display:
        return args + ["--entrypoint", "xvfb-run", image, "-a", "sh", "-c", command]
    return args + ["--entrypoint", "sh", image, "-c", command]


def _start_services(services: list[Service], internet: bool) -> tuple[str, list[str]]:
    """A private network where the command reaches each service by its name."""
    network = f"sol-net-{uuid.uuid4().hex[:12]}"
    create = ["docker", "network", "create"] + ([] if internet else ["--internal"])
    _docker(create + [network], f"Creating network {network}")
    started: list[str] = []
    try:
        for service in services:
            ensure_image(service.image)
            name = f"{network}-{service.name}"
            run = ["docker", "run", "-d", "--rm", "--name", name, "--network", network,
                   "--network-alias", service.name]
            for key, value in service.env.items():
                run += ["-e", f"{key}={value}"]
            _docker(run + [service.image], f"Starting service {service.name}")
            started.append(name)
            if service.ready_command:
                _wait_until_ready(name, service)
    except BaseException:
        _stop_services(network, started)
        raise
    return network, started


def _wait_until_ready(container: str, service: Service) -> None:
    deadline = time.monotonic() + SERVICE_READY_TIMEOUT
    while time.monotonic() < deadline:
        done = subprocess.run(["docker", "exec", container, "sh", "-c", service.ready_command],
                              capture_output=True, timeout=DOCKER_CHECK_TIMEOUT)
        if done.returncode == 0:
            return
        time.sleep(1)
    raise SuiteRunError(
        f"Service {service.name} was not ready within {SERVICE_READY_TIMEOUT}s "
        f"({service.ready_command!r} kept failing)."
    )


def _stop_services(network: str, started: list[str]) -> None:
    for name in started:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True,
                       timeout=DOCKER_CHECK_TIMEOUT)
    subprocess.run(["docker", "network", "rm", network], capture_output=True,
                   timeout=DOCKER_CHECK_TIMEOUT)


# --- running --------------------------------------------------------------------

def run_command(
    command: str,
    *,
    image: str | None = None,
    dockerfile: str | None = None,
    workdir: str = ".",
    writable: bool = False,
    network: bool = False,
    env: dict[str, str] | None = None,
    review_dir: Path | None = None,
    display: bool = False,
    services: list[Service] | None = None,
    ports: tuple[int, ...] = (),
    timeout: int | None = None,
    interactive: bool = False,
) -> CommandRun:
    """Run `command` with `sh -c` in the project folder `workdir`.

    `interactive` connects the human's terminal instead of capturing output.
    """
    timeout = min(timeout or COMMAND_TIMEOUT, MAX_COMMAND_TIMEOUT)
    folder = resolve_project_path(Path(workdir))
    if not folder.is_dir():
        raise ValueError(f"{workdir} is not a folder in the project.")
    if SANDBOX != "docker":
        return _run_locally(command, folder, env or {}, timeout, interactive)

    require_daemon()
    image = image or SANDBOX_IMAGE
    ensure_image(image, dockerfile)
    if writable:
        # Mounted read-only below, so nothing in a container can create Sol's files.
        resolve_project_path(Path(SOL_DIR)).mkdir(exist_ok=True)

    # Publishing a port needs a real network, which also reaches the internet.
    internet = network or bool(ports)
    network_name = "bridge" if internet else "none"
    started: list[str] = []
    if services:
        network_name, started = _start_services(services, internet=network)
    container = f"sol-run-{uuid.uuid4().hex[:12]}"
    args = docker_run_args(
        command, image=image, container=container, workdir=folder.relative_to(
            get_project_path().resolve()).as_posix(),
        writable=writable, network=network_name, internet=internet, env=env,
        review_dir=review_dir,
        display=display, ports=ports, interactive=interactive,
    )
    try:
        return _execute(command, args, timeout, interactive, container)
    finally:
        if services:
            _stop_services(network_name, started)


def _kill_tree(process: subprocess.Popen) -> None:
    # Killing only the shell would leave its children (a dev server, say) running
    # and holding the output pipe open, so the wait below would never end.
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)],
                       capture_output=True)
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _run_locally(command: str, folder: Path, env: dict[str, str], timeout: int,
                 interactive: bool) -> CommandRun:
    shell = config.LOCAL_SHELL
    args: list[str] | str = [shell, "-c", command] if shell else command
    environment = {**os.environ, **QUIET_ENV, **env}
    if interactive:
        try:
            done = subprocess.run(args, shell=not shell, cwd=folder, env=environment)
        except KeyboardInterrupt:
            return CommandRun(command, -1, "stopped with Ctrl+C")
        return CommandRun(command, done.returncode, "")

    group = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
             else {"start_new_session": True})
    process = subprocess.Popen(args, shell=not shell, cwd=folder, env=environment,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL, encoding="utf-8",
                               errors="replace", **group)
    try:
        output, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_tree(process)
        partial, _ = process.communicate()
        output, _ = truncate_output(partial or "")
        return CommandRun(command, -1, f"{output}\nStopped: still running after {timeout}s."
                          .strip(), timed_out=True)
    output, truncated = truncate_output(output or "")
    return CommandRun(command, process.returncode, output, truncated)


def _execute(command: str, args: list[str], timeout: int, interactive: bool,
             container: str) -> CommandRun:
    if interactive:
        try:
            done = subprocess.run(args)
        except KeyboardInterrupt:
            subprocess.run(["docker", "kill", container], capture_output=True,
                           timeout=DOCKER_CHECK_TIMEOUT)
            return CommandRun(command, -1, "stopped with Ctrl+C")
        return CommandRun(command, done.returncode, "")
    try:
        done = subprocess.run(args, capture_output=True, stdin=subprocess.DEVNULL,
                              timeout=timeout, encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        # Killing the docker CLI leaves the container running; stop it directly.
        subprocess.run(["docker", "kill", container], capture_output=True,
                       timeout=DOCKER_CHECK_TIMEOUT)
        return CommandRun(command, -1, f"Stopped: still running after {timeout}s.",
                          timed_out=True)
    output, truncated = truncate_output((done.stdout or "") + (done.stderr or ""))
    return CommandRun(command, done.returncode, output, truncated)
