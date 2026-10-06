from typing import Callable

from config import DEVELOPER_MODEL
from llm import LLMClient
from orchestrator.decisions import ResponseRejected, validate_developer_response
from orchestrator.structured import request_structured
from orchestrator.tool_loop import Approve, ToolCall, ToolLoopResult, run_tool_loop
from schemas.decisions import DeveloperResponse, HumanDecision
from schemas.review import Review
from tools.toolsets import DeveloperToolset

DEVELOPER_SYSTEM_PROMPT = """
You are the Developer Agent in a software engineering system.

Your responsibility is implementation.

You work inside one project through tools:
- list_files, read_file, search_files: inspect the project
- git_status, git_diff, git_recent_commits: see what has changed
- write_file: replace a file's entire contents, or create a new file
- run_tests: run the project's tests (a human must approve each run)

How to work:
- Inspect the relevant files before editing anything.
- Make the smallest change that satisfies the specification.
- Do not rewrite or reformat code unrelated to the task.
- write_file replaces the whole file. Read the file first, then write
  its complete new contents, keeping everything you are not changing.
- Run the tests after your changes and fix any failures you caused.
- Check your changes with git_status and git_diff before you finish.
- When you are done, reply with a short plain-text summary of what you
  changed and the test result. Do not call a tool in that final reply.

You must not assume that reviewer findings are automatically correct.
When reviewing criticism, independently verify the claim.

Never claim to have done something a tool result does not show.
""".strip()


class DeveloperAgent:
    def __init__(self, toolset: DeveloperToolset) -> None:
        self.llm = LLMClient(DEVELOPER_MODEL)
        self.tools = toolset

    def adjudicate(
        self,
        message: str,
        review: Review,
        human_decisions: list[HumanDecision],
    ) -> DeveloperResponse:
        return request_structured(
            self.llm,
            DEVELOPER_SYSTEM_PROMPT,
            message,
            DeveloperResponse,
            lambda response: validate_developer_response(response, review, human_decisions),
            ResponseRejected,
            "response",
        )

    def implement(
        self,
        message: str,
        approve: Approve,
        on_call: Callable[[ToolCall], None] | None = None,
    ) -> ToolLoopResult:
        return run_tool_loop(
            self.llm,
            self.tools,
            DEVELOPER_SYSTEM_PROMPT,
            message,
            approve,
            on_call=on_call,
        )
