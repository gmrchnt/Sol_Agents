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

WORKSPACE_DIR = BASE_DIR / "workspace"
LOG_DIR = BASE_DIR / "logs"