"""Safe, bounded loading of project-local coding instructions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_ROOT_FILENAMES = (
    "REPOPILOT.md",
    "REPOPILOT.local.md",
    "AGENTS.md",
    # Read existing Claude-style instructions for interoperability, but keep
    # them untrusted and bounded just like RepoPilot's native files.
    "CLAUDE.md",
    "CLAUDE.local.md",
)
_MAX_FILES = 16
_MAX_FILE_CHARACTERS = 8_000
_MAX_TOTAL_CHARACTERS = 24_000
_MAX_IMPORT_DEPTH = 4


@dataclass(frozen=True, slots=True)
class ProjectInstructions:
    """Project-authored guidance retained as untrusted model context."""

    documents: tuple[tuple[str, str], ...] = ()

    def render(self) -> str:
        """Render source-labelled instructions within the bounded load budget."""
        return "\n\n".join(f"[{path}]\n{content}" for path, content in self.documents)


def load_project_instructions(
    root: Path, *, user_instructions_root: Path | None = None
) -> ProjectInstructions:
    """Load bounded user and repository instructions from explicit trusted locations.

    Project instructions may inherit from a repository root down to the opened
    subdirectory. User instructions are opt-in through the caller-supplied
    RepoPilot state root; imports never cross either approved boundary.
    """
    resolved_root = root.resolve(strict=True)
    loaded: list[tuple[str, str]] = []
    remaining = _MAX_TOTAL_CHARACTERS
    seen: set[Path] = set()

    def append_scope(scope: Path, *, boundary: Path, label_prefix: str) -> None:
        nonlocal remaining
        candidates = [scope / filename for filename in _ROOT_FILENAMES]
        candidates.extend(
            (scope / ".claude" / filename) for filename in ("CLAUDE.md", "CLAUDE.local.md")
        )
        for rules_root in (scope / ".repopilot" / "rules", scope / ".claude" / "rules"):
            if rules_root.is_dir():
                candidates.extend(sorted(rules_root.rglob("*.md")))
        for candidate in candidates:
            if remaining <= 0 or len(loaded) >= _MAX_FILES:
                return
            for resolved, content in _load_document_with_imports(candidate, boundary, seen=seen):
                if len(loaded) >= _MAX_FILES or remaining <= 0:
                    return
                bounded = content[: min(_MAX_FILE_CHARACTERS, remaining)].strip()
                if not bounded:
                    continue
                relative = resolved.relative_to(boundary).as_posix()
                loaded.append((f"{label_prefix}{relative}", bounded))
                remaining -= len(bounded)

    if user_instructions_root is not None:
        user_root = user_instructions_root.resolve()
        if user_root.is_dir():
            append_scope(user_root, boundary=user_root, label_prefix="~/.repopilot/")
    project_boundary = _repository_boundary(resolved_root)
    hierarchy = tuple(
        candidate
        for candidate in reversed((resolved_root, *resolved_root.parents))
        if candidate == project_boundary or project_boundary in candidate.parents
    )
    for scope in hierarchy:
        append_scope(scope, boundary=project_boundary, label_prefix="")
    return ProjectInstructions(tuple(loaded))


def _repository_boundary(root: Path) -> Path:
    for candidate in (root, *root.parents):
        if (candidate / ".git").exists():
            return candidate
    return root


def project_instruction_template(root: Path) -> str:
    """Create a small, reviewable starting point for ``/init``."""
    resolved = root.resolve(strict=True)
    markers = [
        marker
        for marker in ("pyproject.toml", "package.json", "Cargo.toml", "go.mod", "pom.xml")
        if (resolved / marker).is_file()
    ]
    marker_text = ", ".join(markers) if markers else "no standard build marker discovered"
    return (
        "# RepoPilot project instructions\n\n"
        "## Working agreement\n"
        "- Inspect the relevant code before editing.\n"
        "- Keep changes focused and explain verification results.\n"
        "- Do not read or expose secrets.\n\n"
        "## Project facts\n"
        f"- Detected build markers: {marker_text}.\n"
        "- Add the project's preferred build, test, style, and review commands here.\n"
    )


def _load_document_with_imports(
    candidate: Path,
    root: Path,
    *,
    seen: set[Path],
    depth: int = 0,
) -> tuple[tuple[Path, str], ...]:
    if depth > _MAX_IMPORT_DEPTH:
        return ()
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError:
        return ()
    if resolved in seen or (resolved != root and root not in resolved.parents):
        return ()
    if not resolved.is_file() or resolved.stat().st_size > _MAX_FILE_CHARACTERS * 4:
        return ()
    try:
        content = resolved.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ()
    seen.add(resolved)
    documents: list[tuple[Path, str]] = [(resolved, content)]
    for line in content.splitlines():
        requested = line.strip()
        if not requested.startswith("@") or requested.startswith("@@"):
            continue
        raw_relative = requested[1:].strip()
        if not raw_relative or Path(raw_relative).is_absolute() or ".." in Path(raw_relative).parts:
            continue
        documents.extend(
            _load_document_with_imports(
                resolved.parent / raw_relative, root, seen=seen, depth=depth + 1
            )
        )
    return tuple(documents)
