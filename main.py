from agents.developer import DeveloperAgent
from agents.reviewer import ReviewerAgent
from llm import LLMError
from orchestrator.specification import SpecificationBuilder
from tools.toolsets import DeveloperToolset, ReadOnlyToolset

def main() -> None:

    developer = DeveloperAgent(DeveloperToolset())
    reviewer = ReviewerAgent(ReadOnlyToolset())
    builder = SpecificationBuilder()

    print()
    print("=" * 60)
    print("USER REQUEST")
    print("=" * 60)

    user_request = input(
        "Enter a simple software request: "
    )

    while True:
        print()
        print("=" * 60)
        print("GENERATING SPECIFICATION")
        print("=" * 60)
        specification = (
            builder.build(
                user_request
            )
        )
        specification_json = (
            specification.model_dump_json(
                indent=2
            )
        )
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
        print(
            "Specification rejected. "
            "Stopping before implementation."
        )
        return
    print()
    print("=" * 60)
    print("DEVELOPER")
    print("=" * 60)

    implementation = developer.ask(
        f"""
        ORIGINAL USER REQUEST:
        {user_request}
        APPROVED SPECIFICATION:
        {specification_json}
        You can see the project's file listing but cannot read file
        contents, write files, or run commands yet.
        Describe how you would implement the approved specification.
                
        Rules:
        - Satisfy every listed requirement.
        - Do not add requirements that are not in the specification.
        - Do not claim to have modified files.
        - Do not claim to have run tests.
        - Keep the response concise.
        """.strip()
    )

    print(implementation)

    print()
    print("=" * 60)
    print("REVIEWER")
    print("=" * 60)

    review = reviewer.ask(
        f"""
        ORIGINAL USER REQUEST:
        {user_request}
        APPROVED SPECIFICATION:
        {specification_json}
        DEVELOPER PROPOSAL:
        {implementation}
        Review the Developer proposal against the APPROVED SPECIFICATION.

        Important rules:
        - The approved specification defines the required behavior.
        - Do not invent additional requirements.
        - Something is not a defect merely because you personally
        would implement it differently.
        - Separate actual requirement violations from optional
        improvements.
        - You can see the project's file listing, nothing more.
        - You have not read any file contents.
        - You have not executed any tests.
        - Clearly state what cannot currently be verified.
        """.strip()
    )
    print(review)

if __name__ == "__main__":
    try:
        main()
    except LLMError as exc:
        raise SystemExit(f"Error: {exc}")