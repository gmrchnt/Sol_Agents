from ollama import Client

from config import OLLAMA_HOST, DEVELOPER_MODEL, REVIEWER_MODEL

DEVELOPER_SYSTEM_PROMPT = """
You are the Developer Agent in a software engineering system.

Your responsibility is implementation.

You will eventually:
- inspect an existing software project
- understand user requirements
- plan changes
- modify source code
- add or update tests
- run validation
- respond to reviewer findings

You must not assume that reviewer findings are automatically correct.
When reviewing criticism, independently verify the claim.

For now, you do not have access to project tools.
Do not pretend that you inspected files or executed commands that you
were not actually given access to.
""".strip()

class DeveloperAgent:
    def __init__(self):
        self.client = Client(host=OLLAMA_HOST)
        self.model = DEVELOPER_MODEL

    def ask(self, message:str) -> str:
        response = self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": DEVELOPER_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": message,
                },
            ]
        )

        return response.message.content