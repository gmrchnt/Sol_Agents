import pytest

import tools.testing


@pytest.fixture(autouse=True)
def run_project_tests_locally(monkeypatch: pytest.MonkeyPatch) -> None:
    # Tests build throwaway projects in temp folders; the sandbox has its own tests.
    monkeypatch.setattr(tools.testing, "SANDBOX", "local")
