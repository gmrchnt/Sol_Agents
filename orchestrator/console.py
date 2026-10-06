class ConsoleUI:
    """Terminal input/output. Tests swap in a scripted subclass."""

    def say(self, text: str = "") -> None:
        print(text)

    def ask(self, prompt: str) -> str:
        return input(prompt)

    def heading(self, title: str) -> None:
        self.say()
        self.say("=" * 60)
        self.say(title)
        self.say("=" * 60)

    def confirm(self, question: str, default: bool = False) -> bool:
        hint = "[Y/n]" if default else "[y/N]"
        while True:
            answer = self.ask(f"{question} {hint}: ").strip().lower()
            if not answer:
                return default
            if answer in {"y", "yes"}:
                return True
            if answer in {"n", "no"}:
                return False
            self.say("Please answer y or n.")

    def choose(self, prompt: str, options: dict[str, str], default: str | None = None) -> str:
        """Return the key the user picked; re-ask until it is valid."""
        while True:
            answer = self.ask(prompt).strip().lower()
            if not answer and default is not None:
                return default
            if answer in options:
                return answer
            self.say(f"Please choose one of: {', '.join(options)}.")

    def ask_required(self, prompt: str) -> str:
        while True:
            answer = self.ask(prompt).strip()
            if answer:
                return answer
            self.say("This can't be empty.")
