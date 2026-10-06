import sys
from pathlib import Path

from agents.developer import DeveloperAgent
from agents.reviewer import ReviewerAgent
from config import PROJECT_PATH
from llm import LLMError
from orchestrator.specification import SpecificationBuilder
from orchestrator.tool_loop import ToolCall, ToolLoopLimitError
from tools.git import GitError, current_commit, diff, status
from tools.project import ProjectAccessError, read_project_file
from tools.testing import SuiteRun, SuiteRunError, run_tests
from tools.toolsets import DeveloperToolset, ReadOnlyToolset


def banner(title: str) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def confirm(question: str) -> bool:
    return input(f"{question} [y/N]: ").strip().lower() in {"y", "yes"}


def approve_tool(name: str, arguments: dict) -> bool:
    return confirm(
        f"\nThe Developer wants to run {name}. This executes project code "
        f"on this machine without a sandbox. Allow?"
    )


def show_call(call: ToolCall) -> None:
    arguments = call.arguments if len(call.arguments) <= 80 else call.arguments[:77] + "..."
    first_line = call.result.splitlines()[0] if call.result else ""
    print(f"  -> {call.name}({arguments})")
    print(f"     {first_line[:100]}")


def changed_files() -> list[str]:
    return status().stdout.splitlines()[1:]


def collect_evidence(base_commit: str, changes: list[str], tests: SuiteRun | None) -> str:
    sections = [
        f"GIT STATUS since {base_commit[:8]}:\n" + ("\n".join(changes) or "(no changes)")
    ]

    patch = diff(base_commit)
    sections.append("GIT DIFF:\n" + (patch.stdout.strip() or "(empty)"))

    for line in changes:
        if line.startswith("?? "):
            path = line[3:]
            try:
                content = read_project_file(Path(path))
            except (ProjectAccessError, OSError, ValueError) as exc:
                content = f"(could not read: {exc})"
            sections.append(f"NEW FILE {path}:\n{content}")

    if tests is None:
        sections.append("TEST RESULTS: not run")
    else:
        verdict = "PASSED" if tests.passed else "FAILED"
        sections.append(
            f"TEST RESULTS: {verdict} (exit code {tests.returncode})\n{tests.output}"
        )

    return "\n\n".join(sections)


def main() -> None:

    developer = DeveloperAgent(DeveloperToolset())
    reviewer = ReviewerAgent(ReadOnlyToolset())
    builder = SpecificationBuilder()

    uncommitted = changed_files()
    if uncommitted:
        print(f"{PROJECT_PATH} has uncommitted changes:")
        for line in uncommitted:
            print(f"  {line}")
        print("Commit or stash them first, so the review sees only this run's work.")
        return
    base_commit = current_commit()

    banner("USER REQUEST")
    user_request = input("Enter a simple software request: ")

    while True:
        banner("GENERATING SPECIFICATION")
        specification = builder.build(user_request)
        specification_json = specification.model_dump_json(indent=2)
        print(specification_json)
        print()
        approval = input(
            "Approve specification? "
            "[Y]es / [N]o / [R]egenerate: "
        ).strip().lower()
        if approval in {"", "y", "yes"}:
            break
        if approval in {"r", "regenerate"}:
            continue

        print()
        print("Specification rejected. Stopping before implementation.")
        return

    banner("DEVELOPER")
    print(f"Working in {PROJECT_PATH}, starting from commit {base_commit[:8]}.")

    result = developer.implement(
        f"""
        ORIGINAL USER REQUEST:
        {user_request}
        APPROVED SPECIFICATION:
        {specification_json}
        Implement the approved specification in the project.

        Rules:
        - Satisfy every listed requirement.
        - Do not add requirements that are not in the specification.
        - Do not change files the specification does not need changed.
        """.strip(),
        approve=approve_tool,
        on_call=show_call,
    )
    print()
    print(result.final_text)

    banner("VALIDATION")
    changes = changed_files()
    if changes:
        for line in changes:
            print(f"  {line}")
    else:
        print("  No files changed.")

    tests = None
    if changes and confirm("\nRun the tests independently to check the Developer's work?"):
        tests = run_tests()
        print(f"\nTests {'PASSED' if tests.passed else 'FAILED'} (exit code {tests.returncode})")
        print(tests.output)

    banner("REVIEWER")
    evidence = collect_evidence(base_commit, changes, tests)

    review = reviewer.review(
        f"""
        ORIGINAL USER REQUEST:
        {user_request}
        APPROVED SPECIFICATION:
        {specification_json}
        DEVELOPER SUMMARY (a claim, not evidence):
        {result.final_text}
        EVIDENCE (collected by the orchestrator, not by the Developer):
        {evidence}

        Review the Developer's work against the APPROVED SPECIFICATION.

        Important rules:
        - The approved specification defines the required behavior.
        - Do not invent additional requirements.
        - Something is not a defect merely because you personally
        would implement it differently.
        - Separate actual requirement violations from optional
        improvements.
        - Base every status on the EVIDENCE section. The Developer
          summary is a claim and proves nothing.
        - Mark a requirement pass only when the diff, new files or test
          results show it is met. Otherwise use fail or uncertain.
        - Only report a finding when you can point to concrete evidence,
          such as a line in the diff or a failing test.
        - Do not report missing evidence as a finding; use uncertain.
        """.strip(),
        specification,
        changes_made=bool(changes),
    )
    print(review.model_dump_json(indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        main()
    except (LLMError, GitError, SuiteRunError, ToolLoopLimitError) as exc:
        raise SystemExit(f"Error: {exc}")
