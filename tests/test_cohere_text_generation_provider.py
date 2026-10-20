"""Mocked tests for the Cohere text generation HTTP adapter."""

import json

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from secondbrain.config.settings import Settings
from secondbrain.infrastructure.cohere_text_generation_provider import (
    CohereConfigurationError,
    CohereProviderError,
    CohereTextGenerationProvider,
)

API_KEY = "cohere-test-secret"


def make_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "cohere_api_key": SecretStr(API_KEY),
        "cohere_model": "command-test-model",
        "cohere_timeout_seconds": 12.5,
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def make_provider(
    handler: httpx.MockTransport,
    settings: Settings | None = None,
) -> tuple[CohereTextGenerationProvider, httpx.Client]:
    client = httpx.Client(transport=handler)
    return CohereTextGenerationProvider(settings or make_settings(), client), client


def test_request_construction_and_successful_response_parsing() -> None:
    captured: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "A grounded "},
                        {"type": "text", "text": "answer."},
                    ],
                },
                "finish_reason": "COMPLETE",
            },
        )

    provider, client = make_provider(httpx.MockTransport(handle))
    try:
        answer = provider.generate("System instruction", "Question and context")
    finally:
        client.close()

    assert answer == "A grounded answer."
    request = captured[0]
    assert request.method == "POST"
    assert request.url == "https://api.cohere.com/v2/chat"
    assert request.headers["Authorization"] == f"Bearer {API_KEY}"
    assert request.headers["Content-Type"] == "application/json"
    assert json.loads(request.content) == {
        "model": "command-test-model",
        "stream": False,
        "messages": [
            {"role": "system", "content": "System instruction"},
            {"role": "user", "content": "Question and context"},
        ],
    }
    assert request.extensions["timeout"]["read"] == 12.5


def test_http_errors_do_not_expose_api_key_or_response_body() -> None:
    provider, client = make_provider(
        httpx.MockTransport(
            lambda request: httpx.Response(401, text=f"invalid credential {API_KEY}")
        )
    )
    try:
        with pytest.raises(CohereProviderError, match="HTTP status 401") as error:
            provider.generate("private prompt", "private context")
    finally:
        client.close()

    assert API_KEY not in str(error.value)
    assert "private context" not in str(error.value)


@pytest.mark.parametrize(
    ("network_error", "message"),
    [
        (httpx.ReadTimeout("timed out"), "timed out"),
        (httpx.ConnectError("connection refused"), "connect to the Cohere API"),
    ],
)
def test_timeout_and_connection_errors_are_sanitized(
    network_error: Exception, message: str
) -> None:
    def fail_request(request: httpx.Request) -> httpx.Response:
        raise network_error

    provider, client = make_provider(httpx.MockTransport(fail_request))
    try:
        with pytest.raises(CohereProviderError, match=message) as error:
            provider.generate("private prompt", f"private {API_KEY}")
    finally:
        client.close()

    assert API_KEY not in str(error.value)


def test_malformed_json_and_missing_response_fields_are_rejected() -> None:
    cases = [
        (httpx.Response(200, content=b"not-json"), "malformed JSON"),
        (
            httpx.Response(200, json={"message": {"role": "assistant"}}),
            "missing message content",
        ),
        (
            httpx.Response(
                200,
                json={"message": {"role": "assistant", "content": "answer"}},
            ),
            "invalid format",
        ),
        (
            httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": 123}],
                    }
                },
            ),
            "text has an invalid format",
        ),
    ]
    for response, expected_message in cases:
        provider, client = make_provider(
            httpx.MockTransport(lambda request, item=response: item)
        )
        try:
            with pytest.raises(CohereProviderError, match=expected_message):
                provider.generate("system", "user")
        finally:
            client.close()


@pytest.mark.parametrize(
    "message",
    [
        {"content": [{"type": "text", "text": API_KEY}]},
        {"role": API_KEY, "content": [{"type": "text", "text": API_KEY}]},
    ],
)
def test_missing_or_unexpected_response_role_is_rejected_safely(
    message: dict[str, object],
) -> None:
    provider, client = make_provider(
        httpx.MockTransport(
            lambda request: httpx.Response(200, json={"message": message})
        )
    )
    try:
        with pytest.raises(
            CohereProviderError, match="message role is missing or unexpected"
        ) as error:
            provider.generate("private prompt", f"private context {API_KEY}")
    finally:
        client.close()

    assert API_KEY not in str(error.value)
    assert "private prompt" not in str(error.value)
    assert "private context" not in str(error.value)


@pytest.mark.parametrize(
    "content",
    [[], [{"type": "tool-call", "text": ""}], [{"type": "text", "text": "  "}]],
)
def test_empty_model_output_is_rejected(content: list[dict[str, str]]) -> None:
    provider, client = make_provider(
        httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={"message": {"role": "assistant", "content": content}},
            )
        )
    )
    try:
        with pytest.raises(CohereProviderError, match="empty response"):
            provider.generate("system", "user")
    finally:
        client.close()


def test_environment_backed_settings_configure_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SB_COHERE_API_KEY", API_KEY)
    monkeypatch.setenv("SB_COHERE_MODEL", "command-env-model")
    monkeypatch.setenv("SB_COHERE_TIMEOUT_SECONDS", "7.0")
    settings = Settings(_env_file=None)
    assert settings.cohere_api_key is not None
    assert settings.cohere_api_key.get_secret_value() == API_KEY
    assert settings.cohere_model == "command-env-model"
    assert settings.cohere_timeout_seconds == 7.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cohere_api_key": "   "},
        {"cohere_model": "   "},
        {"cohere_timeout_seconds": 0},
        {"cohere_timeout_seconds": -1},
    ],
)
def test_invalid_cohere_settings_are_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        make_settings(**kwargs)


@pytest.mark.parametrize(
    ("key", "model", "missing_setting"),
    [
        (None, "command-model", "SB_COHERE_API_KEY"),
        (SecretStr(API_KEY), None, "SB_COHERE_MODEL"),
    ],
)
def test_provider_requires_explicit_credentials_and_model(
    key: SecretStr | None, model: str | None, missing_setting: str
) -> None:
    with pytest.raises(CohereConfigurationError, match=missing_setting):
        CohereTextGenerationProvider(
            make_settings(cohere_api_key=key, cohere_model=model)
        )
