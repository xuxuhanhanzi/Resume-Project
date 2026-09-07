from __future__ import annotations

import json
from pathlib import Path

import pytest

from repopilot.configuration import (
    ModelProfile,
    ProjectShortcut,
    UserConfiguration,
    UserConfigurationStore,
)
from repopilot.credentials import CredentialStore


def test_user_configuration_persists_profiles_and_project_shortcuts_without_secrets(
    tmp_path: Path,
) -> None:
    store = UserConfigurationStore(tmp_path / "state")
    configuration = store.set_profile(
        UserConfiguration(),
        ModelProfile("deepseek-main", "deepseek", "deepseek-v4-flash"),
        make_default=True,
    )
    project = tmp_path / "project"
    project.mkdir()
    configuration = store.set_project(
        configuration,
        ProjectShortcut("repopilot", project, "deepseek-main"),
    )

    reloaded = store.load()
    payload = json.loads(store.path.read_text(encoding="utf-8"))

    assert reloaded.default_profile == "deepseek-main"
    assert reloaded.profile("deepseek-main") == ModelProfile(
        "deepseek-main", "deepseek", "deepseek-v4-flash"
    )
    assert store.resolve_project(reloaded, "repopilot").project_root == project.resolve()
    assert "api_key" not in json.dumps(payload)
    assert "secret" not in json.dumps(payload).lower()


def test_user_configuration_rejects_plaintext_credentials_and_unknown_profile_keys(
    tmp_path: Path,
) -> None:
    store = UserConfigurationStore(tmp_path / "state")
    store.path.parent.mkdir()
    store.path.write_text(
        json.dumps(
            {
                "version": 1,
                "default_profile": None,
                "profiles": {
                    "deepseek-main": {
                        "provider": "deepseek",
                        "model": "deepseek-v4-flash",
                        "api_key": "must-not-be-stored",
                    }
                },
                "projects": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="only provider and model"):
        store.load()


def test_user_configuration_requires_known_project_profile(tmp_path: Path) -> None:
    store = UserConfigurationStore(tmp_path / "state")
    project = tmp_path / "project"
    project.mkdir()

    with pytest.raises(ValueError, match="unknown model profile"):
        store.set_project(
            UserConfiguration(),
            ProjectShortcut("repopilot", project, "missing"),
        )


def test_credential_store_uses_a_provider_scoped_system_account(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    observed: dict[str, str] = {}

    class _Backend:
        @staticmethod
        def get_password(service: str, account: str) -> str | None:
            observed["read_service"] = service
            observed["read_account"] = account
            return " saved-key "

        @staticmethod
        def set_password(service: str, account: str, secret: str) -> None:
            observed["write_service"] = service
            observed["write_account"] = account
            observed["secret"] = secret

    monkeypatch.setattr(CredentialStore, "_backend", staticmethod(lambda: _Backend))
    store = CredentialStore()

    assert store.get("deepseek") == "saved-key"
    store.set("qwen", "new-key")
    assert observed["read_account"] == "provider:deepseek"
    assert observed["write_account"] == "provider:qwen"
    assert observed["read_service"] == observed["write_service"]
