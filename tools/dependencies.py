import ast
import hashlib
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import config
from tools.project import get_project_path, list_project_files, read_project_file

REQUIREMENTS_FILE = "requirements.txt"
MAX_PACKAGES = 50
INSTALL_TIMEOUT = 900
DOCKER_CHECK_TIMEOUT = 15
ENV_ROOT = config.BASE_DIR / ".sandbox_envs"
PYTEST_PIN = "pytest==9.1.1"

# Always present wherever tests run.
PROVIDED = {"pytest", "_pytest"}

# Where the import name differs from the package you install.
IMPORT_TO_PACKAGE = {
    "PIL": "pillow",
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
    "skimage": "scikit-image",
    "yaml": "pyyaml",
    "bs4": "beautifulsoup4",
    "dateutil": "python-dateutil",
    "dotenv": "python-dotenv",
    "Crypto": "pycryptodome",
    "attr": "attrs",
    "jwt": "pyjwt",
    "serial": "pyserial",
    "OpenGL": "pyopengl",
    "docx": "python-docx",
    "pptx": "python-pptx",
    "magic": "python-magic",
}

_REQUIREMENT = re.compile(
    r"(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[A-Za-z0-9,._-]+\])?"
    r"==(?P<version>[A-Za-z0-9.+!_-]+)"
)


class DependencyError(Exception):
    """requirements.txt is unacceptable, or installing it failed."""


@dataclass(frozen=True)
class Requirement:
    line: str
    name: str


def normalize(name: str) -> str:
    """PEP 503: Pillow, pillow and PILLOW are the same package; so are a_b and a-b."""
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_requirements(text: str) -> list[Requirement]:
    requirements: list[Requirement] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        match = _REQUIREMENT.fullmatch(line)
        if match is None:
            raise DependencyError(
                f"{REQUIREMENTS_FILE} line {number} is not allowed: {raw.strip()!r}. "
                f"Each line must be one package pinned to an exact version, like "
                f"'pillow==11.0.0'. Options, URLs, file paths and other package "
                f"indexes are not accepted."
            )
        requirements.append(Requirement(line=line, name=normalize(match["name"])))

    if len(requirements) > MAX_PACKAGES:
        raise DependencyError(
            f"{REQUIREMENTS_FILE} lists {len(requirements)} packages; the limit is {MAX_PACKAGES}."
        )
    return requirements


def read_requirements() -> list[Requirement]:
    path = get_project_path() / REQUIREMENTS_FILE
    if not path.is_file():
        return []
    return parse_requirements(read_project_file(Path(REQUIREMENTS_FILE)))


def _outside(path: str, skip: tuple[str, ...]) -> bool:
    return not any(path.startswith(folder.rstrip("/") + "/") for folder in skip)


def imported_modules(skip: tuple[str, ...] = ()) -> dict[str, list[str]]:
    """Top-level module -> project files importing it. Parses code without running it.

    `skip` names folders that are separate components with their own dependencies."""
    found: dict[str, list[str]] = {}
    for path in list_project_files():
        if not path.endswith(".py") or not _outside(path, skip):
            continue
        try:
            tree = ast.parse(read_project_file(Path(path)), filename=path)
        except (SyntaxError, ValueError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                found.setdefault(name.split(".")[0], []).append(path)
    return {module: sorted(set(files)) for module, files in found.items()}


def project_modules() -> set[str]:
    """Names the project itself provides: its .py files and its folders."""
    names = set()
    for path in list_project_files():
        parts = Path(path).parts
        names.update(parts[:-1])
        if path.endswith(".py"):
            names.add(Path(path).stem)
    return names


def third_party_imports(skip: tuple[str, ...] = ()) -> dict[str, list[str]]:
    local = project_modules()
    return {
        module: files
        for module, files in imported_modules(skip).items()
        if module not in sys.stdlib_module_names
        and module not in local
        and module not in PROVIDED
    }


def missing_dependencies(requirements: list[Requirement],
                         skip: tuple[str, ...] = ()) -> list[str]:
    declared = {req.name for req in requirements}
    problems = []
    for module, files in sorted(third_party_imports(skip).items()):
        package = IMPORT_TO_PACKAGE.get(module, module)
        if normalize(package) in declared or normalize(module) in declared:
            continue
        hint = f" (the package is called {package})" if package != module else ""
        problems.append(
            f"'{module}' is imported in {', '.join(files)} but not declared in "
            f"{REQUIREMENTS_FILE}{hint}"
        )
    return problems


def requirements_digest(requirements: list[Requirement], salt: str) -> str:
    text = "\n".join(sorted(req.line for req in requirements)) + "\n" + salt
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _install_output(done: subprocess.CompletedProcess) -> str:
    output = ((done.stdout or "") + (done.stderr or "")).strip()
    return output[-3_000:]


# --- Docker: a per-requirements image built on top of the sandbox image -------

def docker_tag(requirements: list[Requirement], base_image: str) -> str:
    return f"sol-agents-sandbox-deps:{requirements_digest(requirements, base_image)}"


def docker_image_exists(tag: str) -> bool:
    done = subprocess.run(["docker", "image", "inspect", tag], capture_output=True,
                          timeout=DOCKER_CHECK_TIMEOUT)
    return done.returncode == 0


def _binary_only(allow_source: bool) -> list[str]:
    # Pre-built wheels only, unless the human allowed it: building from source
    # runs each package's own setup code.
    return [] if allow_source else ["--only-binary=:all:"]


def dockerfile(base_image: str, allow_source: bool = False) -> str:
    # Network is available only while this image builds; tests run with none.
    flags = " ".join(["--no-cache-dir", *_binary_only(allow_source)])
    return (
        f"FROM {base_image}\n"
        "COPY requirements.txt /tmp/requirements.txt\n"
        f"RUN pip install {flags} -r /tmp/requirements.txt\n"
    )


def build_docker_image(requirements: list[Requirement], base_image: str,
                       allow_source: bool = False) -> str:
    tag = docker_tag(requirements, base_image)
    with tempfile.TemporaryDirectory(prefix="sol-deps-") as context:
        root = Path(context)
        (root / "requirements.txt").write_text(
            "\n".join(req.line for req in requirements) + "\n", encoding="utf-8"
        )
        (root / "Dockerfile").write_text(dockerfile(base_image, allow_source),
                                         encoding="utf-8")
        try:
            done = subprocess.run(
                ["docker", "build", "-t", tag, str(root)],
                capture_output=True, timeout=INSTALL_TIMEOUT,
                encoding="utf-8", errors="replace",
            )
        except subprocess.TimeoutExpired as exc:
            raise DependencyError(f"Installing dependencies took longer than {INSTALL_TIMEOUT}s.") from exc
    if done.returncode != 0:
        raise DependencyError(f"Installing dependencies failed:\n{_install_output(done)}")
    return tag


# --- Local: a separate virtual environment per requirements, never .venv -----

def local_env_dir(requirements: list[Requirement]) -> Path:
    salt = f"{sys.version_info.major}.{sys.version_info.minor}"
    return ENV_ROOT / requirements_digest(requirements, salt)


def _env_python(env_dir: Path) -> Path:
    if sys.platform == "win32":
        return env_dir / "Scripts" / "python.exe"
    return env_dir / "bin" / "python"


def local_python(requirements: list[Requirement]) -> Path | None:
    env_dir = local_env_dir(requirements)
    if (env_dir / ".ready").is_file():
        return _env_python(env_dir)
    return None


def build_local_env(requirements: list[Requirement], allow_source: bool = False) -> Path:
    env_dir = local_env_dir(requirements)
    shutil.rmtree(env_dir, ignore_errors=True)
    env_dir.parent.mkdir(parents=True, exist_ok=True)

    try:
        made = subprocess.run([sys.executable, "-m", "venv", str(env_dir)],
                              capture_output=True, timeout=INSTALL_TIMEOUT,
                              encoding="utf-8", errors="replace")
        if made.returncode != 0:
            raise DependencyError(f"Could not create an environment:\n{_install_output(made)}")

        with tempfile.TemporaryDirectory(prefix="sol-deps-") as folder:
            listing = Path(folder) / "requirements.txt"
            listing.write_text("\n".join(req.line for req in requirements) + "\n",
                               encoding="utf-8")
            done = subprocess.run(
                [str(_env_python(env_dir)), "-m", "pip", "install", "--disable-pip-version-check",
                 *_binary_only(allow_source), PYTEST_PIN, "-r", str(listing)],
                capture_output=True, timeout=INSTALL_TIMEOUT,
                encoding="utf-8", errors="replace",
            )
        if done.returncode != 0:
            raise DependencyError(f"Installing dependencies failed:\n{_install_output(done)}")
    except subprocess.TimeoutExpired as exc:
        shutil.rmtree(env_dir, ignore_errors=True)
        raise DependencyError(f"Installing dependencies took longer than {INSTALL_TIMEOUT}s.") from exc
    except BaseException:
        shutil.rmtree(env_dir, ignore_errors=True)
        raise

    (env_dir / ".ready").write_text("ok\n", encoding="utf-8")
    return _env_python(env_dir)
