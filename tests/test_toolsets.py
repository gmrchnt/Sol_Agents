import pytest

from tools.toolsets import DeveloperToolset, ReadOnlyToolset

WRITE_NAMES = ["write_file", "apply_patch", "delete_file", "run_command", "run_tests"]

READ_NAMES = ["list_files", "read_file", "search_files", "git_status", "git_diff", "git_recent_commits"]


@pytest.mark.parametrize("name", WRITE_NAMES)
def test_readonly_toolset_exposes_no_mutating_tool(name: str) -> None:
    assert not hasattr(ReadOnlyToolset(), name)


@pytest.mark.parametrize("name", READ_NAMES)
def test_readonly_toolset_exposes_inspection_tools(name: str) -> None:
    assert callable(getattr(ReadOnlyToolset(), name))


@pytest.mark.parametrize("name", READ_NAMES)
def test_developer_toolset_keeps_inspection_tools(name: str) -> None:
    assert callable(getattr(DeveloperToolset(), name))


def test_developer_toolset_is_a_readonly_toolset() -> None:
    assert isinstance(DeveloperToolset(), ReadOnlyToolset)


def test_readonly_toolset_is_not_a_developer_toolset() -> None:
    assert not isinstance(ReadOnlyToolset(), DeveloperToolset)


@pytest.mark.parametrize("name", ["write_file", "run_tests"])
def test_developer_toolset_has_write_and_test_tools(name: str) -> None:
    assert callable(getattr(DeveloperToolset(), name))
