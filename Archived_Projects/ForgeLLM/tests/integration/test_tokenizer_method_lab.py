"""Integration test for the unified G1-B learning experiment."""

from pathlib import Path
from typing import cast

from forgellm.structured_logging import JsonValue
from forgellm.tokenization.method_lab import run_method_lab

FIXTURES = Path(__file__).parents[1] / "fixtures" / "tokenizer"


def test_method_lab_compares_all_algorithms_and_writes_report(tmp_path: Path) -> None:
    output_path = tmp_path / "method_lab.json"

    report = run_method_lab(
        FIXTURES / "train.jsonl",
        FIXTURES / "method_lab_evaluation.jsonl",
        output_path,
    )

    assert output_path.exists()
    assert set(report["evaluation"]) == {
        "classic_bpe_global",
        "classic_bpe_unicode_boundaries",
        "picky_bpe_educational",
        "raw_utf8_bytes",
        "super_bpe_curriculum",
        "unigram_em",
    }
    assert report["fixed_settings"]["no_vocabulary_search"] is True
    picky = report["structure"]["picky_bpe_educational"]
    dropout = cast(dict[str, JsonValue], report["sampling"]["bpe_dropout"])
    unigram_sampling = cast(dict[str, JsonValue], report["sampling"]["unigram_sampling"])
    assert cast(int, picky["removal_events"]) > 0
    assert dropout["round_trip_all"] is True
    assert unigram_sampling["round_trip_all"] is True
    assert report["entropy_patching"]["exact_round_trip"] is True
    for result in report["evaluation"].values():
        all_metrics = cast(dict[str, JsonValue], result["all"])
        assert all_metrics["round_trip_rate"] == 1.0
        subsets = cast(dict[str, JsonValue], result["subsets"])
        assert set(subsets) == {
            "chinese",
            "code",
            "emoji",
            "english",
            "mixed",
            "numbers",
            "whitespace",
        }
