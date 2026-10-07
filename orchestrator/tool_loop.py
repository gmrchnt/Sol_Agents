import json
from dataclasses import dataclass, field
from typing import Callable

from llm import LLMClient
from tools.git import GitError, GitResult
from tools.project import ProjectAccessError
from tools.dependencies import DependencyError
from tools.testing import SuiteRun, SuiteRunError

MAX_TOOL_STEPS = 20
NEEDS_APPROVAL = {"run_tests", "run_review_tests", "install_dependencies"}
RECOVERABLE_ERRORS = (
    ProjectAccessError, OSError, ValueError, GitError, SuiteRunError, DependencyError,
)

Approve = Callable[[str, dict], bool]


def _tool(
    name: str,
    description: str,
    properties: dict | None = None,
    required: list[str] | None = None,
) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": required or [],
            },
        },
    }


_PATH = {
    "type": "string",
    "description": "Path relative to the project root, e.g. calculator.py",
}

TOOL_SPECS: dict[str, dict] = {
    spec["function"]["name"]: spec
    for spec in [
        _tool("list_files", "List every file in the project."),
        _tool(
            "read_file",
            "Return the full text of one project file.",
            {"relative_path": _PATH},
            ["relative_path"],
        ),
        _tool(
            "search_files",
            "Find lines containing some text, case-insensitive. "
            "Returns the file, line number and line.",
            {"query": {"type": "string"}},
            ["query"],
        ),
        _tool("git_status", "Show which files have changed since the last commit."),
        _tool("git_diff", "Show the line-by-line changes made since this run started."),
        _tool(
            "git_recent_commits",
            "Show recent commit messages.",
            {"count": {"type": "integer", "description": "How many, 1 to 100."}},
        ),
        _tool(
            "write_file",
            "Replace a file's entire contents, creating it if needed. "
            "Read the file first and write its complete new contents.",
            {
                "relative_path": _PATH,
                "content": {
                    "type": "string",
                    "description": "The complete new contents of the file.",
                },
            },
            ["relative_path", "content"],
        ),
        _tool(
            "run_tests",
            "Run the project's pytest suite and return its output. "
            "A human must approve each run.",
        ),
        _tool(
            "install_dependencies",
            "Install the packages listed in the project's requirements.txt where "
            "tests run. Tests have no network, so this is the only way to get a "
            "third-party package. A human must approve each install.",
        ),
        _tool(
            "write_review_test",
            "Create or replace an adversarial pytest file in your private review "
            "area, outside the project. Name it like test_something.py. It can "
            "import the project's modules by name, e.g. `import calculator`.",
            {
                "filename": {"type": "string"},
                "content": {"type": "string", "description": "The complete test file."},
            },
            ["filename", "content"],
        ),
        _tool(
            "run_review_tests",
            "Run the tests in your review area against the project and return "
            "the output. A human must approve each run.",
        ),
    ]
}

_JSON_TYPES = {"string": str, "integer": int}


@dataclass
class ToolCall:
    name: str
    arguments: str
    result: str


@dataclass
class ToolLoopResult:
    final_text: str
    calls: list[ToolCall] = field(default_factory=list)


class ToolLoopLimitError(RuntimeError):
    """The model kept calling tools without ever giving a final answer."""


def tool_specs_for(toolset: object) -> list[dict]:
    return [
        spec
        for name, spec in TOOL_SPECS.items()
        if callable(getattr(toolset, name, None))
    ]


def _argument_problem(name: str, arguments: dict, spec: dict) -> str | None:
    params = spec["function"]["parameters"]
    properties = params["properties"]

    unknown = sorted(set(arguments) - set(properties))
    if unknown:
        return f"{name} does not take: {', '.join(unknown)}"

    missing = sorted(set(params["required"]) - set(arguments))
    if missing:
        return f"{name} is missing: {', '.join(missing)}"

    for key, value in arguments.items():
        expected = properties[key]["type"]
        if isinstance(value, bool) or not isinstance(value, _JSON_TYPES[expected]):
            return f"{key} must be a {expected}"

    return None


def _format(result: object) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, GitResult):
        text = result.stdout if result.ok else result.stderr
        return f"exit code {result.returncode}\n{text}".rstrip()
    if isinstance(result, SuiteRun):
        if result.no_tests:
            return f"no tests found (exit code {result.returncode})\n{result.output}".rstrip()
        verdict = "PASSED" if result.passed else "FAILED"
        return f"tests {verdict} (exit code {result.returncode})\n{result.output}".rstrip()
    return json.dumps(result, indent=2)


def dispatch(toolset: object, name: str, raw_arguments: str, approve: Approve) -> str:
    spec = TOOL_SPECS.get(name)
    method = getattr(toolset, name, None)
    if spec is None or not callable(method):
        return f"ERROR: tool {name!r} is not available to you."

    try:
        arguments = json.loads(raw_arguments or "{}")
    except json.JSONDecodeError:
        return f"ERROR: arguments for {name} were not valid JSON: {raw_arguments!r}"

    if not isinstance(arguments, dict):
        return f"ERROR: arguments for {name} must be a JSON object."

    problem = _argument_problem(name, arguments, spec)
    if problem:
        return f"ERROR: {problem}"

    if name in NEEDS_APPROVAL and not approve(name, arguments):
        return f"ERROR: the human declined to let you run {name}."

    try:
        result = method(**arguments)
    except RECOVERABLE_ERRORS as exc:
        return f"ERROR: {exc}"

    return _format(result)


def run_tool_loop(
    llm: LLMClient,
    toolset: object,
    system_prompt: str,
    user_message: str,
    approve: Approve,
    *,
    on_call: Callable[[ToolCall], None] | None = None,
    max_steps: int = MAX_TOOL_STEPS,
) -> ToolLoopResult:
    specs = tool_specs_for(toolset)
    messages: list[dict[str, object]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    calls: list[ToolCall] = []

    for _ in range(max_steps):
        reply = llm.complete(messages, tools=specs)
        messages.append(reply)

        requested = reply.get("tool_calls") or []
        if not requested:
            return ToolLoopResult(str(reply.get("content") or ""), calls)

        for request in requested:
            function = request.get("function", {})
            name = function.get("name", "")
            raw = function.get("arguments", "")
            if isinstance(raw, dict):
                raw = json.dumps(raw)

            result = dispatch(toolset, name, raw, approve)
            call = ToolCall(name, raw, result)
            calls.append(call)
            if on_call:
                on_call(call)

            messages.append(
                {"role": "tool", "tool_call_id": request.get("id", ""), "content": result}
            )

    raise ToolLoopLimitError(
        f"The agent made {max_steps} rounds of tool calls without finishing."
    )
