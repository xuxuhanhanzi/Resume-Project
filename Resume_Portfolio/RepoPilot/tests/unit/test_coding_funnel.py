from __future__ import annotations

from pathlib import Path

import pytest

from repopilot.evaluation.coding_funnel import diagnose_coding_funnel


def test_coding_funnel_reports_each_stage_without_echoing_patch_text(tmp_path: Path) -> None:
    records = tmp_path / "development.jsonl"
    records.write_text(
        "\n".join(
            (
                '{"task_id":"not-found","localized":false,"patch":"","patch_applied":false,"verification_passed":false}',
                '{"task_id":"empty","localized":true,"patch":"","patch_applied":false,"verification_passed":false}',
                '{"task_id":"conflict","localized":true,"patch":"secret patch body",'
                '"patch_applied":false,"verification_passed":false}',
                '{"task_id":"test-failure","localized":true,"patch":"diff","patch_applied":true,"verification_passed":false}',
                '{"task_id":"fixed","localized":true,"patch":"diff","patch_applied":true,"verification_passed":true}',
            )
        )
        + "\n",
        encoding="utf-8",
    )

    summary = diagnose_coding_funnel(records)

    assert summary.tasks == 5
    assert summary.localized == 4
    assert summary.nonempty_patch == 3
    assert summary.patch_applied == 2
    assert summary.verification_passed == 1
    assert summary.categories == {
        "empty_patch": 1,
        "localization_failed": 1,
        "patch_apply_failed": 1,
        "verification_failed": 1,
        "verified": 1,
    }
    assert "secret patch body" not in str(summary.to_dict())


def test_coding_funnel_rejects_impossible_stage_order(tmp_path: Path) -> None:
    records = tmp_path / "invalid.jsonl"
    records.write_text(
        '{"task_id":"invalid","localized":false,"patch":"diff","patch_applied":false,"verification_passed":false}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="before localization"):
        diagnose_coding_funnel(records)
