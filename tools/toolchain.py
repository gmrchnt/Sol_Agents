"""How the project is set up, built, tested and run, for any language.

Detected from the project's own files (package.json, Cargo.toml, go.mod, pom.xml,
...) unless .sol/toolchain.json says otherwise. The Developer proposes that file
with set_toolchain, the human approves it, and only the orchestrator writes it:
the commands in it decide what "the tests pass" means.
"""

import json
import re
from pathlib import Path, PurePosixPath

from pydantic import ValidationError

import config
from schemas.toolchain import Component, Toolchain
from tools import sandbox
from tools.dependencies import imported_modules
from tools.project import (
    SOL_DIR,
    ProjectAccessError,
    list_project_files,
    read_project_file,
    resolve_project_path,
)
from tools.sandbox import project_image
from tools.testing import python_image

TOOLCHAIN_FILE = f"{SOL_DIR}/toolchain.json"
# How deep below the root a folder can hold its own component (a monorepo's
# frontend/, backend/, services/api/).
SCAN_DEPTH = 2
PYTEST_COMMAND = "python -m pytest -q --no-header -p no:cacheprovider"
PYTHON_MARKERS = {"requirements.txt", "pyproject.toml", "setup.py", "setup.cfg", "Pipfile"}
# Python GUI toolkits that need an X display; pygame works with SDL's dummy driver.
GUI_MODULES = {"tkinter", "turtle", "PyQt5", "PyQt6", "PySide2", "PySide6", "wx", "kivy",
               "pyglet", "arcade"}

DEFAULT_IMAGES = {
    "python": config.SANDBOX_IMAGE,
    "node": "node:22-bookworm",
    "rust": "rust:1-bookworm",
    "go": "golang:1-bookworm",
    "java-maven": "maven:3-eclipse-temurin-21",
    "java-gradle": "gradle:8-jdk21",
    "dotnet": "mcr.microsoft.com/dotnet/sdk:8.0",
    "c-cpp": "sol-agents-cpp:latest",
    "ruby": "ruby:3.3-bookworm",
    "php": "composer:2",
    "elixir": "elixir:1.17",
    "deno": "denoland/deno:debian",
    "swift": "swift:6.0",
    "dart": "dart:stable",
    "haskell": "haskell:9",
    "web": config.SANDBOX_IMAGE,
}


class ToolchainError(ValueError):
    """.sol/toolchain.json exists but is not a valid toolchain."""


def _join(folder: str, name: str) -> Path:
    return Path(name) if folder == "." else Path(folder) / name


def _read_text(folder: str, name: str) -> str:
    try:
        return read_project_file(_join(folder, name))
    except (OSError, ValueError, ProjectAccessError):
        return ""


def _read_json(folder: str, name: str) -> dict:
    try:
        data = json.loads(_read_text(folder, name) or "{}")
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _component(language: str, **fields) -> Component:
    return Component(name=language, language=language, **fields)


# --- one detector per kind of project; each sees one folder's file names --------

def _node(folder: str, names: set[str], files: list[str]) -> Component | None:
    if "package.json" not in names:
        return None
    manifest = _read_json(folder, "package.json")
    scripts = manifest.get("scripts") if isinstance(manifest.get("scripts"), dict) else {}
    if "pnpm-lock.yaml" in names:
        runner, setup = "npx --yes pnpm@9", "npx --yes pnpm@9 install --frozen-lockfile"
    elif "yarn.lock" in names:
        runner, setup = "yarn", "yarn install --frozen-lockfile"
    elif "package-lock.json" in names:
        runner, setup = "npm", "npm ci"
    else:
        runner, setup = "npm", "npm install"

    def script(name: str) -> str | None:
        return f"{runner} run {name}" if name in scripts else None

    test = str(scripts.get("test", ""))
    return _component(
        "node", setup=[setup], setup_creates="node_modules",
        setup_inputs=["package.json", "package-lock.json", "yarn.lock", "pnpm-lock.yaml"],
        build=script("build"), lint=script("lint"), static_analysis=script("typecheck"),
        test=f"{runner} test" if test and "no test specified" not in test else None,
        run=script("start") or script("dev"),
    )


def _python(folder: str, names: set[str], files: list[str]) -> Component | None:
    # The root is handled after the other components: see _root_python.
    if folder == "." or not PYTHON_MARKERS & names:
        return None
    if "requirements.txt" in names:
        setup = ["pip install --user --no-warn-script-location -r requirements.txt"]
    elif {"pyproject.toml", "setup.py"} & names:
        setup = ["pip install --user --no-warn-script-location -e ."]
    else:
        setup = []
    return _component("python", setup=setup, setup_inputs=sorted(PYTHON_MARKERS),
                      test=PYTEST_COMMAND)


def _rust(folder: str, names: set[str], files: list[str]) -> Component | None:
    if "Cargo.toml" not in names:
        return None
    return _component("rust", setup=["cargo fetch"], setup_inputs=["Cargo.toml", "Cargo.lock"],
                      build="cargo build --all-targets", test="cargo test", run="cargo run")


def _go(folder: str, names: set[str], files: list[str]) -> Component | None:
    if "go.mod" not in names:
        return None
    return _component("go", setup=["go mod download"], setup_inputs=["go.mod", "go.sum"],
                      build="go build ./...", test="go test ./...",
                      static_analysis="go vet ./...", run="go run .")


def _maven(folder: str, names: set[str], files: list[str]) -> Component | None:
    if "pom.xml" not in names:
        return None
    return _component("java-maven", setup=["mvn -B -q dependency:go-offline"],
                      setup_inputs=["pom.xml"], build="mvn -B -q -DskipTests package",
                      test="mvn -B -q test")


def _gradle(folder: str, names: set[str], files: list[str]) -> Component | None:
    if not {"build.gradle", "build.gradle.kts"} & names:
        return None
    gradle = "sh ./gradlew" if "gradlew" in names else "gradle"
    return _component(
        "java-gradle", setup=[f"{gradle} --no-daemon -q dependencies"],
        setup_inputs=["*.gradle", "*.gradle.kts", "gradle/libs.versions.toml",
                      "gradle/wrapper/gradle-wrapper.properties"],
        build=f"{gradle} --no-daemon -q assemble", test=f"{gradle} --no-daemon -q test",
    )


def _dotnet(folder: str, names: set[str], files: list[str]) -> Component | None:
    projects = [name for name in names if name.endswith((".csproj", ".fsproj"))]
    if not projects and not any(name.endswith(".sln") for name in names):
        return None
    return _component(
        "dotnet", setup=["dotnet restore"],
        setup_inputs=["*.sln", "*.csproj", "*.fsproj", "Directory.*.props", "nuget.config"],
        build="dotnet build --no-restore", test="dotnet test --no-restore",
        run="dotnet run" if len(projects) == 1 else None,
    )


def _cmake(folder: str, names: set[str], files: list[str]) -> Component | None:
    if "CMakeLists.txt" not in names:
        return None
    return _component("c-cpp", build="cmake -S . -B .build -G Ninja && cmake --build .build",
                      test="ctest --test-dir .build --output-on-failure")


def _make(folder: str, names: set[str], files: list[str]) -> Component | None:
    if "Makefile" not in names or "CMakeLists.txt" in names:
        return None
    makefile = _read_text(folder, "Makefile")
    target = next((name for name in ("test", "check")
                   if re.search(rf"^{name}\s*:", makefile, re.MULTILINE)), None)
    return _component("c-cpp", build="make", test=f"make {target}" if target else None)


def _ruby(folder: str, names: set[str], files: list[str]) -> Component | None:
    if "Gemfile" not in names:
        return None
    specs = f"{folder}/spec/" if folder != "." else "spec/"
    if any(path.startswith(specs) for path in files):
        test = "bundle exec rspec"
    else:
        test = "bundle exec rake test" if "Rakefile" in names else None
    return _component("ruby", setup=["bundle install"],
                      setup_inputs=["Gemfile", "Gemfile.lock"], test=test)


def _php(folder: str, names: set[str], files: list[str]) -> Component | None:
    if "composer.json" not in names:
        return None
    manifest = _read_json(folder, "composer.json")
    dev = manifest.get("require-dev") if isinstance(manifest.get("require-dev"), dict) else {}
    phpunit = "phpunit/phpunit" in dev or {"phpunit.xml", "phpunit.xml.dist"} & names
    return _component("php", setup=["composer install --no-interaction"],
                      setup_inputs=["composer.json", "composer.lock"], setup_creates="vendor",
                      test="vendor/bin/phpunit" if phpunit else None)


def _elixir(folder: str, names: set[str], files: list[str]) -> Component | None:
    if "mix.exs" not in names:
        return None
    return _component(
        "elixir", setup=["mix local.hex --force && mix local.rebar --force && mix deps.get"],
        setup_inputs=["mix.exs", "mix.lock"], build="mix compile", test="mix test",
    )


def _deno(folder: str, names: set[str], files: list[str]) -> Component | None:
    if not {"deno.json", "deno.jsonc"} & names:
        return None
    return _component("deno", setup=["deno install"],
                      setup_inputs=["deno.json", "deno.jsonc", "deno.lock"], test="deno test -A")


def _swift(folder: str, names: set[str], files: list[str]) -> Component | None:
    if "Package.swift" not in names:
        return None
    return _component("swift", setup=["swift package resolve"],
                      setup_inputs=["Package.swift", "Package.resolved"],
                      build="swift build", test="swift test")


def _dart(folder: str, names: set[str], files: list[str]) -> Component | None:
    if "pubspec.yaml" not in names:
        return None
    return _component("dart", setup=["dart pub get"],
                      setup_inputs=["pubspec.yaml", "pubspec.lock"], test="dart test")


def _haskell(folder: str, names: set[str], files: list[str]) -> Component | None:
    if "stack.yaml" in names:
        return _component("haskell", setup=["stack build --only-dependencies --test --no-run-tests"],
                          setup_inputs=["stack.yaml", "package.yaml", "*.cabal"],
                          build="stack build", test="stack test")
    if any(name.endswith(".cabal") for name in names):
        return _component(
            "haskell", setup=["cabal update && cabal build --only-dependencies --enable-tests"],
            setup_inputs=["*.cabal", "cabal.project"], build="cabal build", test="cabal test",
        )
    return None


def _web(folder: str, names: set[str], files: list[str]) -> Component | None:
    # A plain site with no build step: serve it so a person can look at it.
    if folder != "." or "index.html" not in names or "package.json" in names:
        return None
    return _component("web", run="python -m http.server 8000", ports=[8000])


DETECTORS = [_node, _python, _rust, _go, _maven, _gradle, _dotnet, _cmake, _make, _ruby,
             _php, _elixir, _deno, _swift, _dart, _haskell, _web]


def _folders(files: list[str]) -> dict[str, set[str]]:
    folders: dict[str, set[str]] = {}
    for path in files:
        parts = PurePosixPath(path).parts
        if len(parts) - 1 > SCAN_DEPTH or any(part.startswith(".") for part in parts[:-1]):
            continue
        folders.setdefault("/".join(parts[:-1]) or ".", set()).add(parts[-1])
    return folders


def _inside(path: str, folder: str) -> bool:
    return folder == "." or path == folder or path.startswith(folder + "/")


def _root_python(files: list[str], names: set[str],
                 components: list[Component]) -> Component | None:
    """Python at the root: a manifest there, or .py files no other component owns."""
    others = [c.path for c in components if c.path != "."]
    loose = any(path.endswith(".py") and not any(_inside(path, other) for other in others)
                for path in files)
    if not (PYTHON_MARKERS & names or loose):
        return None
    gui = GUI_MODULES & set(imported_modules(tuple(others)))
    return _component("python", display=bool(gui))


def detect_toolchain() -> Toolchain:
    files = list_project_files()
    folders = _folders(files)
    components: list[Component] = []
    for folder in sorted(folders, key=lambda f: (f != ".", f.count("/"), f)):
        for detect in DETECTORS:
            found = detect(folder, folders[folder], files)
            if found is None:
                continue
            # A workspace member (a crate in a Cargo workspace, a package in an npm
            # workspace) is built by the component above it.
            if any(c.language == found.language and _inside(folder, c.path)
                   for c in components):
                continue
            name = found.language if folder == "." else PurePosixPath(folder).name
            if any(c.name == name for c in components):
                name = f"{name}-{found.language}"
            components.append(found.model_copy(update={"name": name, "path": folder}))

    python = _root_python(files, folders.get(".", set()), components)
    if python is not None and not any(c.name == "python" for c in components):
        components.insert(0, python)
    return Toolchain(components=components)


# --- the explicit file -----------------------------------------------------------

def read_explicit() -> Toolchain | None:
    path = resolve_project_path(Path(TOOLCHAIN_FILE))
    if not path.is_file():
        return None
    try:
        return Toolchain.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise ToolchainError(f"{TOOLCHAIN_FILE} is not valid: {exc}") from exc


def load_toolchain() -> Toolchain:
    explicit = read_explicit()
    return explicit if explicit is not None else detect_toolchain()


def save_toolchain(toolchain: Toolchain) -> None:
    resolve_project_path(Path(SOL_DIR)).mkdir(exist_ok=True)
    text = toolchain.model_dump_json(indent=2, exclude_defaults=True) + "\n"
    resolve_project_path(Path(TOOLCHAIN_FILE)).write_bytes(text.encode("utf-8"))


def component_image(component: Component) -> tuple[str, str | None]:
    """The image the component's commands run in, and the Dockerfile it is built from."""
    if component.dockerfile:
        return project_image(component.dockerfile), component.dockerfile
    if component.image:
        return component.image, None
    if component.language == "python" and sandbox.SANDBOX == "docker":
        return python_image(), None
    return DEFAULT_IMAGES.get(component.language, config.SANDBOX_IMAGE), None


def default_images() -> str:
    return "\n".join(f"  {language}: {image}" for language, image in DEFAULT_IMAGES.items())


def describe_toolchain(toolchain: Toolchain, explicit: bool) -> str:
    source = (f"from {TOOLCHAIN_FILE}" if explicit
              else "detected from the project's files; set_toolchain can change it")
    if not toolchain.components:
        return (
            "No toolchain detected: the project has no file Sol recognises (package.json, "
            "Cargo.toml, go.mod, pom.xml, *.csproj, CMakeLists.txt, requirements.txt, ...).\n"
            "For a new project, pass `image` to run_command to scaffold it, e.g. "
            "`cargo init` in rust:1-bookworm.\nDefault images:\n" + default_images()
        )
    lines = [f"Toolchain ({source}):"]
    for component in toolchain.components:
        where = "the project root" if component.path == "." else f"{component.path}/"
        image = component.dockerfile and f"built from {component.dockerfile}"
        lines.append(f"- {component.name} ({component.language}) in {where} "
                     f"[{image or component.image or DEFAULT_IMAGES.get(component.language, config.SANDBOX_IMAGE)}]")
        if component.setup:
            lines.append(f"    setup (with network): {' && '.join(component.setup)}")
        for kind in ("build", "test", "lint", "static_analysis", "run"):
            command = getattr(component, kind)
            if kind == "test" and command is None and component.language == "python":
                command = "pytest"
            if command:
                lines.append(f"    {kind}: {command}")
        if component.display:
            lines.append("    runs under a virtual display (xvfb)")
        for service in component.services:
            lines.append(f"    service {service.name}: {service.image}")
    return "\n".join(lines)
