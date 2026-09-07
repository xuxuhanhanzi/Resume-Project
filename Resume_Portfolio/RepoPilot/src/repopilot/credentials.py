"""Small operating-system credential-store boundary for cloud provider keys."""

from __future__ import annotations

from typing import Any, Protocol

from repopilot.configuration import cloud_providers

_SERVICE_NAME = "RepoPilot CLI model credentials"


class CredentialStoreError(RuntimeError):
    """The system credential store is unavailable or rejected an operation."""


class CredentialReader(Protocol):
    """Minimal credential lookup boundary used by provider-profile resolution."""

    def get(self, provider: str) -> str | None:
        """Return one provider secret if a secure backend holds it."""


class CredentialStore:
    """Store provider API keys in the OS keychain, never in RepoPilot JSON files."""

    def get(self, provider: str) -> str | None:
        credential_backend = self._backend()
        try:
            value = credential_backend.get_password(_SERVICE_NAME, self._account(provider))
        except Exception as error:  # pragma: no cover - backend-specific implementations
            raise CredentialStoreError("could not read the system credential store") from error
        return value.strip() if isinstance(value, str) and value.strip() else None

    def set(self, provider: str, secret: str) -> None:
        if not secret.strip():
            raise ValueError("API key must not be empty")
        credential_backend = self._backend()
        try:
            credential_backend.set_password(_SERVICE_NAME, self._account(provider), secret)
        except Exception as error:  # pragma: no cover - backend-specific implementations
            raise CredentialStoreError("could not save to the system credential store") from error

    def delete(self, provider: str) -> bool:
        credential_backend = self._backend()
        if self.get(provider) is None:
            return False
        try:
            credential_backend.delete_password(_SERVICE_NAME, self._account(provider))
        except Exception as error:  # pragma: no cover - backend-specific implementations
            raise CredentialStoreError("could not remove the system credential") from error
        return True

    @staticmethod
    def _account(provider: str) -> str:
        if provider not in cloud_providers():
            allowed = ", ".join(cloud_providers())
            raise ValueError(f"provider must be one of: {allowed}")
        return f"provider:{provider}"

    @staticmethod
    def _backend() -> Any:
        try:
            import keyring
        except ImportError as error:
            raise CredentialStoreError(
                "credential support is unavailable; reinstall RepoPilot with its dependencies"
            ) from error
        return keyring
