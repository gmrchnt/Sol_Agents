import json
from datetime import datetime
from pathlib import Path

from config import LOG_DIR
from schemas.state import ProjectState

MAX_LOGGED_RESULT = 4_000


def new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


class RunLog:
    """Audit trail for one run: logs/<run_id>/state.json and events.jsonl."""

    def __init__(self, run_id: str, root: Path = LOG_DIR) -> None:
        self.dir = root / run_id
        self.dir.mkdir(parents=True, exist_ok=True)

    @property
    def state_path(self) -> Path:
        return self.dir / "state.json"

    @property
    def events_path(self) -> Path:
        return self.dir / "events.jsonl"

    def save_state(self, state: ProjectState) -> None:
        self.state_path.write_text(state.model_dump_json(indent=2), encoding="utf-8")

    def event(self, kind: str, **data: object) -> None:
        record = {"time": datetime.now().isoformat(timespec="seconds"), "kind": kind, **data}
        for key, value in record.items():
            if isinstance(value, str) and len(value) > MAX_LOGGED_RESULT:
                record[key] = value[:MAX_LOGGED_RESULT] + f"... [{len(value)} chars]"
        with self.events_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
