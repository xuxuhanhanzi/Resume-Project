from __future__ import annotations

import json

import pytest

from forgellm.alignment.preference_data import (
    adversarial_responses,
    build_preference_splits,
    verify_response,
)
from forgellm.alignment.schema import (
    AlignmentSchemaError,
    PreferenceRecord,
    RolloutRecord,
    assert_split_disjoint,
)


def test_preference_data_is_deterministic_disjoint_and_strict() -> None:
    first = build_preference_splits()
    second = build_preference_splits()
    assert {name: len(records) for name, records in first.items()} == {
        "train": 384,
        "validation": 64,
        "test": 64,
    }
    assert_split_disjoint(first)
    assert [record.fingerprint() for record in first["train"]] == [
        record.fingerprint() for record in second["train"]
    ]
    reasons = {record.rejection_reason for record in first["train"]}
    assert reasons == {
        "wrong_answer",
        "format_error",
        "extra_explanation",
        "truncated",
        "repetition",
        "length_gaming",
    }
    for record in first["test"][:10]:
        assert verify_response(record, record.chosen)["total"] == 1.0
        assert verify_response(record, record.rejected)["total"] == 0.0
        assert all(
            verify_response(record, attack)["total"] == 0.0
            for attack in adversarial_responses(record).values()
        )


def test_preference_round_trip_and_tampering_detection() -> None:
    record = build_preference_splits()["validation"][0]
    restored = PreferenceRecord.from_dict(json.loads(json.dumps(record.as_dict())))
    assert restored == record
    tampered = record.as_dict()
    tampered["chosen"] = "tampered"
    with pytest.raises(AlignmentSchemaError, match="fingerprint"):
        PreferenceRecord.from_dict(tampered)


def test_rollout_contract_binds_tokens_log_probs_and_revision() -> None:
    rollout = RolloutRecord(
        prompt_id="p0",
        policy_revision="policy-v1",
        adapter_sha256="a" * 64,
        generation_config={"temperature": 0.8},
        seed=41,
        token_ids=(1, 2),
        old_log_probs=(-0.2, -0.3),
        reward_components={"exact": 1.0, "total": 1.0},
        verifier_version="v1",
        response="ok",
        timestamp_utc="2026-07-28T00:00:00Z",
    )
    assert rollout.response_length == 2
    assert len(rollout.fingerprint()) == 64
    with pytest.raises(AlignmentSchemaError, match="align"):
        RolloutRecord(
            prompt_id="p0",
            policy_revision="policy-v1",
            adapter_sha256="a" * 64,
            generation_config={},
            seed=41,
            token_ids=(1,),
            old_log_probs=(-0.2, -0.3),
            reward_components={"total": 1.0},
            verifier_version="v1",
            response="ok",
            timestamp_utc="2026-07-28T00:00:00Z",
        )
