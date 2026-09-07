from __future__ import annotations

from pathlib import Path

from repopilot.context.builder import ContextBuilder
from repopilot.core.contracts import AgentState, Message
from repopilot.memory.store import Episode, EpisodicMemoryStore, SessionMemory
from repopilot.retrieval.bm25 import BM25CodeIndex
from repopilot.skills.registry import SkillRegistry
from repopilot.task import PublicTaskSpec


def test_bm25_ranks_relevant_file(tmp_path: Path) -> None:
    (tmp_path / "calculator.py").write_text(
        "def subtract(left, right): return left + right", encoding="utf-8"
    )
    (tmp_path / "unrelated.py").write_text("def render_page(): return 'html'", encoding="utf-8")

    hits = BM25CodeIndex.build(tmp_path).search("subtract calculator bug")

    assert hits[0].path == "calculator.py"


def test_session_and_episodic_memory_are_bounded_and_searchable(tmp_path: Path) -> None:
    session = SessionMemory()
    for index in range(25):
        session.remember(f"fact {index}")
    store = EpisodicMemoryStore(tmp_path / "episodes.jsonl")
    store.append(Episode("t1", "subtract bug", "inspect calculator", "completed"))

    assert len(session.facts) == 20
    assert store.search("calculator subtract")[0].task_id == "t1"


def test_skill_progressive_disclosure_and_context_trimming(tmp_path: Path) -> None:
    skill_dir = tmp_path / "skills" / "python-bugfix"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: python-bugfix\ndescription: fix Python bugs\n---\nInspect, patch, verify.",
        encoding="utf-8",
    )
    registry = SkillRegistry(tmp_path / "skills")
    selected = registry.select("Fix a Python bug")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    task = PublicTaskSpec("t", workspace, "Fix a Python bug", trusted_fixture=True)
    state = AgentState("r", "t")
    state.messages.append(Message("tool", "x" * 100))
    request = ContextBuilder(max_message_chars=20).build(
        task=task, state=state, tools=(), skills=selected
    )

    assert registry.descriptors() == (("python-bugfix", "fix Python bugs"),)
    assert "Inspect, patch, verify" in request.messages[1].content
    assert "Never claim that a file was inspected" in request.messages[0].content
    assert request.messages[-1].content.endswith("[context compressed]")


def test_project_skill_overrides_same_named_user_skill(tmp_path: Path) -> None:
    user_skill = tmp_path / "user" / "skills" / "review" / "SKILL.md"
    project_skill = tmp_path / "project" / "skills" / "review" / "SKILL.md"
    user_skill.parent.mkdir(parents=True)
    project_skill.parent.mkdir(parents=True)
    user_skill.write_text(
        "---\nname: review\ndescription: user review\n---\nUser procedure.",
        encoding="utf-8",
    )
    project_skill.write_text(
        "---\nname: review\ndescription: project review\n---\nProject procedure.",
        encoding="utf-8",
    )

    registry = SkillRegistry(
        project_skill.parent.parent,
        additional_roots=(user_skill.parent.parent,),
    )
    selected = registry.select("Please review this patch.")

    review = registry.get("review")
    assert review is not None
    assert review.source == "project"
    assert review.load_instructions() == "Project procedure."
    assert selected[0].source == "project"
