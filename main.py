from agents.developer import DeveloperAgent
from agents.reviewer import ReviewerAgent
from orchestrator.specification import SpecificationBuilder

def main() -> None:

    developer = DeveloperAgent()
    reviewer = ReviewerAgent()
    builder = SpecificationBuilder()

    print()
    print("=" * 60)
    print("USER REQUEST")
    print("=" * 60)

    user_request = input(
        "Enter a simple software request: "
    )

    print()
    print("=" * 60)
    print("GENERATING SPECIFICATION")
    print("=" * 60)

    specification = builder.build(user_request)
    specification_json = specification.model_dump_json(indent=2)

    print(specification_json)
    print()
    approval = input("Approve specification? (y/n): ").strip().lower()

    if approval not in ("", "y", "yes"):
        print()
        print(
            "SPECIFICATION REJECTED. "
            "STOPPING IMPLEMENTATION."
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
        You do not currently have filesystem access.
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
        - You do not have access to the repository yet.
        - You have not executed any tests.
        - Clearly state what cannot currently be verified.
        """.strip()
    )
    print(review)

if __name__ == "__main__":
    main()