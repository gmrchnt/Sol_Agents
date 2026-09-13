import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")


DEVELOPER_MODEL = os.getenv(
    "DEVELOPER_MODEL",
    "llama3.2:latest",
)

REVIEWER_MODEL = os.getenv(
    "REVIEWER_MODEL",
    "llama3.2:latest",
)

OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://localhost:11434",
)

WORKSPACE_DIR = BASE_DIR / "workspace"
LOG_DIR = BASE_DIR / "logs"