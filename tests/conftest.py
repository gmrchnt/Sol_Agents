import pytest

import tools.project
import tools.sandbox
import tools.testing


@pytest.fixture(autouse=True)
def run_project_tests_locally(monkeypatch: pytest.MonkeyPatch) -> None:
    # Tests build throwaway projects in temp folders; the sandbox has its own tests.
    monkeypatch.setattr(tools.testing, "SANDBOX", "local")
    monkeypatch.setattr(tools.sandbox, "SANDBOX", "local")


@pytest.fixture(autouse=True)
def isolated_project(tmp_path_factory: pytest.TempPathFactory,
                     monkeypatch: pytest.MonkeyPatch) -> None:
    # Never the PROJECT_PATH from .env: a test that forgets to set its own project
    # sees an empty folder, and package caches stay out of Sol_Agents.
    monkeypatch.setattr(tools.project, "PROJECT_PATH", tmp_path_factory.mktemp("empty"))
    monkeypatch.setattr(tools.sandbox, "CACHE_ROOT", tmp_path_factory.mktemp("cache"))
