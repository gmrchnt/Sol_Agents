from config import DEVELOPER_MODEL, LLM_BASE_URL, REVIEWER_MODEL
from llm import LLMClient


def check_model(role: str, model: str) -> None:
    print()
    print(f"Testing {role} \nModel: {model} @ {LLM_BASE_URL}")
    print("-" * 50)

    reply = LLMClient(model).chat(
        [
            {
                "role": "system",
                "content": f"You are the {role} in a software development system.",
            },
            {
                "role": "user",
                "content": "Reply with one short sentence confirming that you are available",
            },
        ]
    )

    print(reply)


def main() -> None:
    check_model(role="Developer", model=DEVELOPER_MODEL)
    check_model(role="Reviewer", model=REVIEWER_MODEL)


if __name__ == "__main__":
    main()
