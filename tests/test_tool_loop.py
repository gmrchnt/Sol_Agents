import json
from pathlib import Path

import pytest

import tools.project
from orchestrator.tool_loop import (
    ToolLoopLimitError,
    dispatch,
    run_tool_loop,
    tool_specs_for,
)
from tools.toolsets import DeveloperToolset, ReadOnlyToolset


class FakeLLM:
    def __init__(self, replies: list[dict]) -> None:
        self.replies = list(replies)
        self.seen: list[list[dict]] = []

    def complete(self, messages, *, tools, temperature=None):
        self.seen.append(list(messages))
        return self.replies.pop(0)


def call(name: str, arguments: dict, call_id: str = "c1") -> dict:
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": call_id, "type": "function",
         "function": {"name": name, "arguments": json.dumps(arguments)}}]}


def say(text: str) -> dict:
    return {"role": "assistant", "content": text}


def allow(name: str, arguments: dict) -> bool:
    return True


def deny(name: str, arguments: dict) -> bool:
    return False


def never_asked(name: str, arguments: dict) -> bool:
    raise AssertionError(f"approval should not be requested for {name}")


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "calculator.py").write_text(
        "def add(a, b):\n    return a + b\n", encoding="utf-8"
    )
    monkeypatch.setattr(tools.project, "PROJECT_PATH", tmp_path)
    return tmp_path


def _names(toolset: object) -> set[str]:
    return {spec["function"]["name"] for spec in tool_specs_for(toolset)}


def test_reviewer_is_never_offered_write_or_test_tools() -> None:
    names = _names(ReadOnlyToolset())
    assert "read_file" in names
    assert "write_file" not in names
    assert "run_tests" not in names


def test_developer_is_offered_write_and_test_tools() -> None:
    assert {"write_file", "run_tests"} <= _names(DeveloperToolset())


def test_plain_reply_ends_the_loop(project: Path) -> None:
    result = run_tool_loop(FakeLLM([say("done")]), DeveloperToolset(), "s", "t", allow)
    assert result.final_text == "done"
    assert result.calls == []


def test_tool_result_is_fed_back_to_the_model(project: Path) -> None:
    llm = FakeLLM([call("read_file", {"relative_path": "calculator.py"}), say("ok")])
    result = run_tool_loop(llm, DeveloperToolset(), "s", "t", never_asked)
    assert result.calls[0].result.startswith("def add")
    assert llm.seen[1][-1] == {
        "role": "tool", "tool_call_id": "c1", "content": result.calls[0].result
    }


def test_write_file_changes_disk(project: Path) -> None:
    llm = FakeLLM([
        call("write_file", {"relative_path": "calculator.py", "content": "x = 1\n"}),
        say("ok"),
    ])
    run_tool_loop(llm, DeveloperToolset(), "s", "t", never_asked)
    assert (project / "calculator.py").read_text(encoding="utf-8") == "x = 1\n"


def test_reviewer_cannot_write_even_if_it_asks(project: Path) -> None:
    llm = FakeLLM([
        call("write_file", {"relative_path": "calculator.py", "content": "x"}),
        say("ok"),
    ])
    result = run_tool_loop(llm, ReadOnlyToolset(), "s", "t", allow)
    assert result.calls[0].result.startswith("ERROR")
    assert "def add" in (project / "calculator.py").read_text(encoding="utf-8")


def test_unknown_tool_is_refused() -> None:
    assert dispatch(DeveloperToolset(), "delete_everything", "{}", allow).startswith("ERROR")


def test_invalid_json_is_reported() -> None:
    assert "not valid JSON" in dispatch(DeveloperToolset(), "read_file", "{nope", allow)


def test_missing_argument_is_reported() -> None:
    assert "missing" in dispatch(DeveloperToolset(), "read_file", "{}", allow)


def test_unexpected_argument_is_reported() -> None:
    assert "does not take" in dispatch(DeveloperToolset(), "list_files", '{"x": 1}', allow)


def test_wrong_argument_type_is_reported() -> None:
    result = dispatch(DeveloperToolset(), "git_recent_commits", '{"count": "5"}', allow)
    assert "must be a integer" in result


def test_tool_errors_are_returned_not_raised(project: Path) -> None:
    result = dispatch(DeveloperToolset(), "read_file", '{"relative_path": "../x"}', allow)
    assert result.startswith("ERROR")


def test_declined_approval_blocks_the_tool(project: Path) -> None:
    assert "declined" in dispatch(DeveloperToolset(), "run_tests", "{}", deny)


def test_step_limit_stops_a_runaway_model(project: Path) -> None:
    llm = FakeLLM([call("list_files", {}) for _ in range(5)])
    with pytest.raises(ToolLoopLimitError):
        run_tool_loop(llm, DeveloperToolset(), "s", "t", allow, max_steps=3)


def test_each_call_is_reported(project: Path) -> None:
    seen = []
    llm = FakeLLM([call("list_files", {}), say("ok")])
    run_tool_loop(llm, DeveloperToolset(), "s", "t", allow, on_call=seen.append)
    assert [c.name for c in seen] == ["list_files"]
