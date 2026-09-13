from orchestrator.specification import (
    SpecificationBuilder,
)

def main() -> None:
    builder = SpecificationBuilder()
    prompt = (
        "Write a Python function that divides two numbers "
        "and raises ValueError when the denominator is zero."
    )
    specification = builder.build(prompt)
    print(specification.model_dump_json(indent=2))

if __name__ == "__main__":
    main()