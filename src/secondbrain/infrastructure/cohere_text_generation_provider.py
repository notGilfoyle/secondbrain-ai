"""Cohere v2 Chat API adapter for application text generation."""

from typing import Any, Self

import httpx

from secondbrain.application.rag import TextGenerationProvider
from secondbrain.config.settings import Settings, get_settings


class CohereProviderError(RuntimeError):
    """Raised when Cohere cannot return a usable generated response."""


class CohereConfigurationError(ValueError):
    """Raised when required Cohere environment-backed settings are missing."""


class CohereTextGenerationProvider(TextGenerationProvider):
    """Generate text through Cohere's non-streaming v2 Chat API."""

    ENDPOINT = "https://api.cohere.com/v2/chat"

    def __init__(
        self,
        settings: Settings | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        settings = settings or get_settings()
        if settings.cohere_api_key is None:
            raise CohereConfigurationError(
                "Configure SB_COHERE_API_KEY before using the Cohere provider"
            )
        if settings.cohere_model is None:
            raise CohereConfigurationError(
                "Configure SB_COHERE_MODEL before using the Cohere provider"
            )

        self._api_key = settings.cohere_api_key.get_secret_value()
        self._model = settings.cohere_model
        self._timeout_seconds = settings.cohere_timeout_seconds
        self._client = client or httpx.Client(timeout=self._timeout_seconds)
        self._owns_client = client is None

    def generate(self, system_instruction: str, user_message: str) -> str:
        """Generate a response without exposing prompt or credential data in errors."""
        try:
            response = self._client.post(
                self.ENDPOINT,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                json={
                    "model": self._model,
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": user_message},
                    ],
                },
                timeout=self._timeout_seconds,
            )
            response.raise_for_status()
        except httpx.TimeoutException:
            raise CohereProviderError("Cohere request timed out") from None
        except httpx.HTTPStatusError as exc:
            raise CohereProviderError(
                f"Cohere request failed with HTTP status {exc.response.status_code}"
            ) from None
        except httpx.RequestError:
            raise CohereProviderError("Could not connect to the Cohere API") from None

        try:
            payload: Any = response.json()
        except ValueError:
            raise CohereProviderError("Cohere returned malformed JSON") from None

        text = self._extract_text(payload)
        if not text.strip():
            raise CohereProviderError("Cohere returned an empty response")
        return text.strip()

    @staticmethod
    def _extract_text(payload: Any) -> str:
        try:
            message = payload["message"]
        except (KeyError, TypeError):
            raise CohereProviderError(
                "Cohere response is missing message data"
            ) from None
        if not isinstance(message, dict):
            raise CohereProviderError("Cohere response message has an invalid format")
        if message.get("role") != "assistant":
            raise CohereProviderError(
                "Cohere response message role is missing or unexpected"
            )
        if "content" not in message:
            raise CohereProviderError("Cohere response is missing message content")
        content = message["content"]

        if not isinstance(content, list):
            raise CohereProviderError("Cohere response content has an invalid format")

        text_parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                raise CohereProviderError("Cohere response content has an invalid item")
            if item.get("type") == "text":
                text = item.get("text")
                if not isinstance(text, str):
                    raise CohereProviderError(
                        "Cohere response text has an invalid format"
                    )
                text_parts.append(text)
        return "".join(text_parts)

    def close(self) -> None:
        """Close the HTTP client when this provider created it."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
