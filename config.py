import os
import httpx
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

PROJECT_PATH = Path(
    os.getenv(
        "PROJECT_PATH",
        str(BASE_DIR / "workspace"),
    )
).resolve()



DEVELOPER_MODEL = os.getenv(
    "DEVELOPER_MODEL",
    "llama3.2:latest",
)

REVIEWER_MODEL = os.getenv(
    "REVIEWER_MODEL",
    "llama3.2:latest",
)

LLM_BASE_URL = os.getenv(
    "LLM_BASE_URL",
    "http://localhost:8080/v1",
)
LLM_TIMEOUT = httpx.Timeout(connect=2.0, read=300.0, write=10.0, pool=5.0)

# "docker" runs project code in a locked-down container; "local" runs it
# directly on this machine and must be chosen explicitly.
SANDBOX = os.getenv("SANDBOX", "docker").strip().lower()
if SANDBOX not in {"docker", "local"}:
    raise ValueError(f"SANDBOX must be 'docker' or 'local', not {SANDBOX!r}")

SANDBOX_IMAGE = os.getenv("SANDBOX_IMAGE", "sol-agents-sandbox:latest")

LOG_DIR = BASE_DIR / "logs"


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    return int(value) if value else default


# Each role can use its own server - e.g. a strong hosted model for the
# Developer. A key is sent as "Authorization: Bearer <key>" when set.
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
DEVELOPER_BASE_URL = os.getenv("DEVELOPER_BASE_URL", LLM_BASE_URL)
REVIEWER_BASE_URL = os.getenv("REVIEWER_BASE_URL", LLM_BASE_URL)
DEVELOPER_API_KEY = os.getenv("DEVELOPER_API_KEY", LLM_API_KEY)
REVIEWER_API_KEY = os.getenv("REVIEWER_API_KEY", LLM_API_KEY)

# Budgets for big projects. The human is asked before a budget is exceeded.
MAX_TOOL_STEPS = _int_env("SOL_MAX_TOOL_STEPS", 40)
MAX_REVIEW_STEPS = _int_env("SOL_MAX_REVIEW_STEPS", 25)
MAX_ITERATIONS = _int_env("SOL_MAX_ITERATIONS", 5)
MAX_FILE_BYTES = _int_env("SOL_MAX_FILE_BYTES", 2_000_000)
CONTEXT_CHARS = _int_env("SOL_CONTEXT_CHARS", 150_000)

# Commands (builds, installs, tests) run in containers with these limits.
COMMAND_TIMEOUT = _int_env("SOL_COMMAND_TIMEOUT", 900)
SANDBOX_MEMORY = os.getenv("SOL_SANDBOX_MEMORY", "4g")
SANDBOX_CPUS = os.getenv("SOL_SANDBOX_CPUS", "2")
SANDBOX_PIDS = _int_env("SOL_SANDBOX_PIDS", 4096)
# Package-manager caches, one folder per project, kept between runs so that
# dependencies downloaded once (with network) are there for offline builds.
CACHE_ROOT = BASE_DIR / ".sandbox_cache"
# With SANDBOX=local, commands run in this shell (e.g. a path to bash);
# empty means the system shell (cmd.exe on Windows, /bin/sh elsewhere).
LOCAL_SHELL = os.getenv("SOL_SHELL", "")