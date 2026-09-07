"""Project discovery for language-neutral interactive workspaces."""

from dataclasses import dataclass
from pathlib import Path

from repopilot.workspace.instructions import ProjectInstructions, load_project_instructions

_MARKERS: tuple[tuple[str, str], ...] = (
    ("pyproject.toml", "python"),
    ("requirements.txt", "python"),
    ("package.json", "javascript"),
    ("Cargo.toml", "rust"),
    ("go.mod", "go"),
    ("pom.xml", "java"),
    ("build.gradle", "java"),
    ("CMakeLists.txt", "cpp"),
)


@dataclass(frozen=True, slots=True)
class ProjectWorkspace:
    """Discovered facts about one user-selected project root."""

    root: Path
    git_root: Path | None
    languages: tuple[str, ...]
    build_markers: tuple[str, ...]
    instructions: ProjectInstructions

    @classmethod
    def discover(
        cls, root: Path, *, user_instructions_root: Path | None = None
    ) -> "ProjectWorkspace":
        resolved = root.resolve()
        if not resolved.is_dir():
            raise ValueError(f"project root does not exist: {resolved}")
        markers = tuple(marker for marker, _ in _MARKERS if (resolved / marker).exists())
        languages = tuple(
            dict.fromkeys(language for marker, language in _MARKERS if marker in markers)
        )
        return cls(
            resolved,
            cls._find_git_root(resolved),
            languages,
            markers,
            load_project_instructions(resolved, user_instructions_root=user_instructions_root),
        )

    @staticmethod
    def _find_git_root(root: Path) -> Path | None:
        for candidate in (root, *root.parents):
            if (candidate / ".git").exists():
                return candidate
        return None
