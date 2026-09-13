from ollama import Client
from config import OLLAMA_HOST, DEVELOPER_MODEL, REVIEWER_MODEL

client = Client(host=OLLAMA_HOST)

def test_model(role:str,model:str) -> None:
    print()
    print(f"Testing {role} \nModel: {model}")
    print("-" * 50)

    response = client.chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": f"YOu are the {role} in a software development system."
            },
            {
                "role": "user",
                "content": ("Reply with one short sentence confirming that you are available")
            }
        ]
    )

    print(response.message.content)

def main() -> None:
    test_model(role="Developer", model=DEVELOPER_MODEL)
    test_model(role="Reviewer", model=REVIEWER_MODEL)


if __name__ == "__main__":
    main()