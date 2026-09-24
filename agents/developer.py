from config import DEVELOPER_MODEL
from llm import LLMClient
from tools.toolsets import DeveloperToolset

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

IMPORTANT: You are only allowed to output code.

You can see the project's file listing. You cannot yet read file
contents on demand, write files, or run commands.
Do not pretend that you inspected files or executed commands that you
were not actually given access to.
""".strip()

class DeveloperAgent:
    def __init__(self, toolset: DeveloperToolset) -> None:
        self.llm = LLMClient(DEVELOPER_MODEL)
        self.tools = toolset

    def ask(self, message:str) -> str:
        files = "\n".join(self.tools.list_files())
        return self.llm.chat(
            [
                {
                    "role": "system",
                    "content": DEVELOPER_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": f"PROJECT FILES:\n{files}\n\n{message}",
                },
            ]
        )