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