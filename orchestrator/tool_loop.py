import json
from dataclasses import dataclass, field
from typing import Callable

import config
from llm import LLMClient
from tools.git import GitError, GitResult
from tools.project import ProjectAccessError
from tools.dependencies import DependencyError
from tools.sandbox import CommandRun
from tools.testing import SuiteRun, SuiteRunError

MAX_TOOL_STEPS = config.MAX_TOOL_STEPS
NEEDS_APPROVAL = {"run_tests", "run_review_tests", "install_dependencies", "run_command",
                  "run_review_command", "set_toolchain"}
RECOVERABLE_ERRORS = (
    ProjectAccessError, OSError, ValueError, GitError, SuiteRunError, DependencyError,
)
# A single tool result never takes more of the model's context than this.
MAX_RESULT_CHARS = 24_000
# Once the conversation grows past this, old tool results are shortened.
CONTEXT_CHARS = config.CONTEXT_CHARS
KEEP_RECENT_MESSAGES = 8
STUB = "[earlier result removed to save space - call the tool again if you need it]"

Approve = Callable[[str, dict], bool]
# Asked when the step budget runs out; returns how many more steps to allow.
MoreSteps = Callable[[int], int]


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
_COMPONENT = {
    "type": "string",
    "description": "Which toolchain component's folder and image to use (see show_toolchain).",
}
_TIMEOUT = {"type": "integer", "description": "Seconds before the command is stopped."}

TOOL_SPECS: dict[str, dict] = {
    spec["function"]["name"]: spec
    for spec in [
        _tool(
            "list_files",
            "List the project's files (git-ignored ones, like node_modules, are left out).",
            {"directory": {"type": "string", "description": "Only list files under this folder."}},
        ),
        _tool(
            "read_file",
            "Return the text of one project file. Large files come back in parts; "
            "pass start_line and end_line to read a range.",
            {"relative_path": _PATH,
             "start_line": {"type": "integer"},
             "end_line": {"type": "integer"}},
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
            "show_toolchain",
            "Show how the project is set up, built, tested and run - per component, with "
            "the sandbox image each uses - and the default images for each language.",
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
                "encoding": {
                    "type": "string",
                    "description": "utf-8 (the default), or base64 for a binary file.",
                },
            },
            ["relative_path", "content"],
        ),
        _tool(
            "edit_file",
            "Replace one exact piece of text in a file with new text. old_text must "
            "appear exactly once. Best for small changes to large files.",
            {"relative_path": _PATH,
             "old_text": {"type": "string", "description": "Copied exactly from the file."},
             "new_text": {"type": "string"}},
            ["relative_path", "old_text", "new_text"],
        ),
        _tool("delete_file", "Delete one file from the project.",
              {"relative_path": _PATH}, ["relative_path"]),
        _tool("move_file", "Move or rename one file in the project.",
              {"source": _PATH, "destination": _PATH}, ["source", "destination"]),
        _tool(
            "run_command",
            "Run a shell command (sh -c) in the project's sandbox and return its output "
            "and the files it changed. The project is writable; .git is not. There is no "
            "network unless network is true. Use it to scaffold projects, install "
            "dependencies, build, run generators and check behavior. Never start a "
            "program that keeps running (a server, a watcher, a GUI). A human must "
            "approve each command.",
            {
                "command": {"type": "string"},
                "component": _COMPONENT,
                "image": {"type": "string",
                          "description": "A Docker image to run in instead, e.g. "
                                         "rust:1-bookworm to scaffold a new Rust project."},
                "workdir": {"type": "string",
                            "description": "Folder in the project to run in."},
                "network": {"type": "boolean",
                            "description": "Allow internet access, e.g. to install packages."},
                "timeout_seconds": _TIMEOUT,
            },
            ["command"],
        ),
        _tool(
            "run_tests",
            "Run every component's tests and return the output. "
            "A human must approve each run.",
        ),
        _tool(
            "install_dependencies",
            "Install the project's dependencies where its code runs: Python packages "
            "from requirements.txt, and each component's setup (npm ci, cargo fetch, "
            "...). Builds and tests have no network, so this is how packages arrive. "
            "A human must approve each install.",
            {"allow_source_builds": {
                "type": "boolean",
                "description": "Let Python packages without a pre-built wheel build from "
                               "source. Only if an install failed for that reason."}},
        ),
        _tool(
            "set_toolchain",
            "Set how the project is set up, built, tested and run, when show_toolchain "
            "is wrong or incomplete. Replaces the whole toolchain. A human must approve.",
            {"toolchain": {
                "type": "object",
                "description": (
                    'JSON like {"components": [{"name": "web", "language": "node", '
                    '"path": "frontend", "image": "node:22-bookworm", "setup": ["npm ci"], '
                    '"setup_inputs": ["package.json", "package-lock.json"], '
                    '"setup_creates": "node_modules", "build": "npm run build", '
                    '"test": "npm test", "lint": null, "static_analysis": null, '
                    '"run": "npm start", "ports": [3000], "display": false, '
                    '"test_network": false, "env": {}, "services": [{"name": "db", '
                    '"image": "postgres:16", "env": {"POSTGRES_PASSWORD": "test"}, '
                    '"ready_command": "pg_isready -U postgres"}]}]}. Use "dockerfile" '
                    "instead of image to build one from a Dockerfile in the project."
                ),
            }},
            ["toolchain"],
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
            "Run the pytest files in your review area against the project and return "
            "the output. A human must approve each run.",
        ),
        _tool(
            "write_review_file",
            "Create or replace any file in your private review area, outside the "
            "project - a test in the project's own language, a fixture, a script.",
            {"path": {"type": "string", "description": "e.g. checks/edge.test.js"},
             "content": {"type": "string"}},
            ["path", "content"],
        ),
        _tool(
            "run_review_command",
            "Run a shell command (sh -c) in a component's sandbox with the project "
            "read-only and your review area at $REVIEW_DIR, e.g. to run a test you "
            "wrote with the project's own test runner. No network. Copy files to /tmp "
            "if a tool needs to write. A human must approve each command.",
            {"command": {"type": "string"}, "component": _COMPONENT,
             "timeout_seconds": _TIMEOUT},
            ["command"],
        ),
    ]
}

_JSON_TYPES = {"string": str, "integer": int, "boolean": bool, "object": dict}


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
        if expected == "boolean":
            valid = isinstance(value, bool)
        else:
            valid = not isinstance(value, bool) and isinstance(value, _JSON_TYPES[expected])
        if not valid:
            return f"{key} must be a {expected}"

    return None


def _cap(text: str) -> str:
    if len(text) <= MAX_RESULT_CHARS:
        return text
    omitted = len(text) - MAX_RESULT_CHARS
    return text[:MAX_RESULT_CHARS] + f"\n... [{omitted} characters omitted]"


def _format(result: object) -> str:
    if isinstance(result, str):
        return _cap(result)
    if isinstance(result, GitResult):
        text = result.stdout if result.ok else result.stderr
        return _cap(f"exit code {result.returncode}\n{text}".rstrip())
    if isinstance(result, SuiteRun):
        if result.no_tests:
            return f"no tests found (exit code {result.returncode})\n{result.output}".rstrip()
        verdict = "PASSED" if result.passed else "FAILED"
        return f"tests {verdict} (exit code {result.returncode})\n{result.output}".rstrip()
    if isinstance(result, CommandRun):
        verdict = "stopped: took too long" if result.timed_out else f"exit code {result.returncode}"
        return f"{verdict}\n{result.output}".rstrip()
    return _cap(json.dumps(result, indent=2))


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

    # Models often send null for an optional argument they mean to leave out.
    required = set(spec["function"]["parameters"]["required"])
    arguments = {key: value for key, value in arguments.items()
                 if value is not None or key in required}

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


def _size(message: dict) -> int:
    size = len(str(message.get("content") or ""))
    for call in message.get("tool_calls") or []:
        size += len(str(call.get("function", {}).get("arguments", "")))
    return size


def _shrunk_calls(calls: list[dict]) -> list[dict]:
    """Old tool calls keep their short arguments; long ones (whole files) are dropped."""
    shrunk = []
    for call in calls:
        function = dict(call.get("function", {}))
        raw = function.get("arguments", "")
        if len(str(raw)) > len(STUB):
            try:
                values = json.loads(raw) if isinstance(raw, str) else dict(raw)
            except (ValueError, TypeError):
                values = {}
            if not isinstance(values, dict):
                values = {}
            function["arguments"] = json.dumps({
                key: value if len(str(value)) <= 200 else "[omitted]"
                for key, value in values.items()
            })
        shrunk.append({**call, "function": function})
    return shrunk


def compact(messages: list[dict], budget: int = CONTEXT_CHARS) -> None:
    """Keep a long tool session inside the model's context: the system prompt, the
    task and the latest exchanges stay whole; older results and file contents go."""
    total = sum(_size(message) for message in messages)
    for index in range(2, max(2, len(messages) - KEEP_RECENT_MESSAGES)):
        if total <= budget:
            return
        message = messages[index]
        before = _size(message)
        if message.get("role") == "tool" and len(str(message.get("content") or "")) > len(STUB):
            messages[index] = {**message, "content": STUB}
        elif message.get("tool_calls"):
            messages[index] = {**message, "tool_calls": _shrunk_calls(message["tool_calls"])}
        total -= before - _size(messages[index])


def run_tool_loop(
    llm: LLMClient,
    toolset: object,
    system_prompt: str,
    user_message: str,
    approve: Approve,
    *,
    on_call: Callable[[ToolCall], None] | None = None,
    max_steps: int = MAX_TOOL_STEPS,
    on_limit: MoreSteps | None = None,
) -> ToolLoopResult:
    specs = tool_specs_for(toolset)
    messages: list[dict[str, object]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    calls: list[ToolCall] = []
    steps_left = max_steps
    used = 0

    while True:
        while steps_left > 0:
            steps_left -= 1
            used += 1
            compact(messages)
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

        steps_left = on_limit(used) if on_limit else 0
        if steps_left <= 0:
            raise ToolLoopLimitError(
                f"The agent made {used} rounds of tool calls without finishing."
            )
