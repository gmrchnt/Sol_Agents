"""The project document in .sol/: what Sol has learned about a project, kept between runs.

The Reviewer writes it at the end of an accepted run; the next run hands it to the
specification builder and both agents, so they know what has already been built.
"""

from pathlib import Path

from schemas.project import ProjectDocument, ProjectSummary, RunRecord
from schemas.state import ProjectState
from tools.git import GitError, changed_paths
from tools.project import SOL_DIR, list_project_files, resolve_project_path

DOCUMENT_JSON = f"{SOL_DIR}/project.json"
DOCUMENT_MD = f"{SOL_DIR}/PROJECT.md"
MAX_HISTORY = 20
CONTEXT_HISTORY = 10
# Files that make a folder a repository without making it a project.
SCAFFOLDING = {".gitkeep", ".gitignore", ".gitattributes"}


class SummaryRejected(ValueError):
    """The model produced a project summary that failed deterministic validation."""


def validate_summary(summary: ProjectSummary, files: set[str]) -> None:
    cited = summary.entry_points + [module.path for module in summary.modules]
    unknown = sorted({path for path in cited if path not in files})
    if unknown:
        raise SummaryRejected(
            f"These files are not in the project: {', '.join(unknown)}. "
            f"Cite only paths listed under PROJECT FILES, exactly as written."
        )
    paths = [module.path for module in summary.modules]
    duplicates = sorted({path for path in paths if paths.count(path) > 1})
    if duplicates:
        raise SummaryRejected(f"Modules listed more than once: {', '.join(duplicates)}")


def is_sol_path(path: str) -> bool:
    return path.split("/")[0].lower() == SOL_DIR


def has_existing_code() -> bool:
    return any(
        not is_sol_path(path) and Path(path).name not in SCAFFOLDING
        for path in list_project_files()
    )


def load_document() -> ProjectDocument | None:
    path = resolve_project_path(Path(DOCUMENT_JSON))
    if not path.is_file():
        return None
    try:
        return ProjectDocument.model_validate_json(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        # Unreadable or from an older format: survey again rather than trust it.
        return None


def changes_since(document: ProjectDocument) -> list[str] | None:
    """Files outside .sol/ changed since the document was written; None if unknown."""
    try:
        changes = changed_paths(document.commit)
    except GitError:
        return None
    return [path for _, path in changes if not is_sol_path(path)]


def save_document(document: ProjectDocument) -> None:
    folder = resolve_project_path(Path(SOL_DIR))
    folder.mkdir(exist_ok=True)
    # bytes, so Windows doesn't silently rewrite \n as \r\n
    resolve_project_path(Path(DOCUMENT_JSON)).write_bytes(
        (document.model_dump_json(indent=2) + "\n").encode("utf-8")
    )
    resolve_project_path(Path(DOCUMENT_MD)).write_bytes(
        render_markdown(document).encode("utf-8")
    )


def run_record(state: ProjectState) -> RunRecord:
    return RunRecord(
        run_id=state.run_id,
        request=state.user_prompt,
        requirements=[f"{r.id}: {r.description}" for r in state.specification.requirements],
        outcome=state.outcome or state.status,
    )


def describe_summary(summary: ProjectSummary) -> str:
    lines = [f"Purpose: {summary.purpose}"]
    if summary.entry_points:
        lines.append(f"Entry points: {', '.join(summary.entry_points)}")
    if summary.how_to_run:
        lines.append(f"How to run: {summary.how_to_run}")
    if summary.modules:
        lines.append("Modules:")
        lines += [f"  - {m.path}: {m.responsibility}" for m in summary.modules]
    if summary.tests:
        lines.append(f"Tests: {summary.tests}")
    if summary.dependencies:
        lines.append(f"Dependencies: {', '.join(summary.dependencies)}")
    if summary.known_issues:
        lines.append("Known issues:")
        lines += [f"  - {issue}" for issue in summary.known_issues]
    return "\n".join(lines)


def describe_history(history: list[RunRecord]) -> str:
    lines = []
    for record in history[-CONTEXT_HISTORY:]:
        lines.append(f"- run {record.run_id}: {record.request!r} -> {record.outcome}")
        lines += [f"    {requirement}" for requirement in record.requirements]
    return "\n".join(lines)


def context_for_models(summary: ProjectSummary | None,
                       history: list[RunRecord]) -> str | None:
    parts = []
    if summary is not None:
        parts.append(f"EXISTING PROJECT:\n{describe_summary(summary)}")
    if history:
        parts.append(f"EARLIER RUNS (oldest first):\n{describe_history(history)}")
    return "\n\n".join(parts) or None


def render_markdown(document: ProjectDocument) -> str:
    sections = [
        "# Project document",
        "Written by Sol_Agents at the end of each accepted run. Do not edit by hand; "
        "`project.json` is the copy Sol reads.",
        f"Describes commit `{document.commit[:8]}`.",
        "## Summary",
        describe_summary(document.summary),
    ]
    if document.history:
        sections += ["## Runs", describe_history(document.history)]
    return "\n\n".join(sections) + "\n"
