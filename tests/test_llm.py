"""Unit tests of the LLM client wrapper, without any provider.

The OpenAI client class is replaced by a fake that records how it was built
and called and answers a scripted reply, so these tests need neither a key
nor a network and run in CI.
"""

from types import SimpleNamespace

import httpx2
import openai
import pytest
from pydantic import BaseModel

from app import llm

LLM_VARIABLES = ("LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY", "LLM_TIMEOUT_S")


class Answer(BaseModel):
    """A small schema standing in for the endpoint schemas."""

    cypher: str
    rows: int


class FakeOpenAI:
    """Stands in for openai.OpenAI: records its arguments, returns `reply`."""

    reply: str | None | Exception = None
    instances: list["FakeOpenAI"] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        FakeOpenAI.instances.append(self)

    def create(self, **kwargs):
        """Record the request and return the scripted reply (or raise it)."""
        self.calls.append(kwargs)
        if isinstance(FakeOpenAI.reply, Exception):
            raise FakeOpenAI.reply
        message = SimpleNamespace(content=FakeOpenAI.reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


@pytest.fixture(name="fake_client")
def fixture_fake_client(monkeypatch):
    """A configured LLM whose client is the fake; the real env is ignored."""
    for name in LLM_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setattr(llm.openai, "OpenAI", FakeOpenAI)
    FakeOpenAI.instances = []
    FakeOpenAI.reply = None
    return FakeOpenAI


@pytest.fixture(name="no_llm_env")
def fixture_no_llm_env(monkeypatch):
    """An environment without any LLM variable."""
    for name in LLM_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_settings_are_off_without_key_or_model(no_llm_env):
    """Both a key and a model are needed; either one alone keeps the LLM off."""
    assert llm.get_settings() is None
    no_llm_env.setenv("LLM_API_KEY", "k")
    assert llm.get_settings() is None
    no_llm_env.delenv("LLM_API_KEY")
    no_llm_env.setenv("LLM_MODEL", "m")
    assert llm.get_settings() is None


def test_settings_treat_blank_values_as_unset(no_llm_env):
    """An empty `LLM_API_KEY=` line from .env.example does not switch the LLM on."""
    no_llm_env.setenv("LLM_API_KEY", "  ")
    no_llm_env.setenv("LLM_MODEL", "m")
    assert llm.get_settings() is None


def test_settings_default_to_groq_and_twenty_seconds(no_llm_env):
    """Only key and model are required; base URL and timeout have defaults."""
    no_llm_env.setenv("LLM_API_KEY", "k")
    no_llm_env.setenv("LLM_MODEL", "m")
    no_llm_env.setenv("LLM_BASE_URL", "")
    settings = llm.get_settings()
    assert settings == llm.LLMSettings(
        base_url=llm.DEFAULT_BASE_URL, model="m", api_key="k", timeout_s=20.0
    )


def test_settings_read_a_custom_provider(no_llm_env):
    """Any OpenAI-compatible provider is reachable through the variables."""
    no_llm_env.setenv("LLM_API_KEY", "k")
    no_llm_env.setenv("LLM_MODEL", "m")
    no_llm_env.setenv("LLM_BASE_URL", "http://localhost:9999/v1")
    no_llm_env.setenv("LLM_TIMEOUT_S", "5")
    settings = llm.get_settings()
    assert (settings.base_url, settings.timeout_s) == ("http://localhost:9999/v1", 5.0)


def test_complete_json_refuses_when_not_configured(no_llm_env):
    """Without settings no client is built and LLMNotConfigured is raised."""
    no_llm_env.setattr(llm.openai, "OpenAI", FakeOpenAI)
    FakeOpenAI.instances = []
    with pytest.raises(llm.LLMNotConfigured):
        llm.complete_json([{"role": "user", "content": "hi"}], Answer)
    assert not FakeOpenAI.instances


def test_complete_json_returns_the_validated_model(fake_client):
    """A reply matching the schema comes back as the Pydantic model."""
    fake_client.reply = '{"cypher": "MATCH (m:Movie) RETURN m LIMIT 1", "rows": 1}'
    answer = llm.complete_json([{"role": "user", "content": "one movie"}], Answer)
    assert answer == Answer(cypher="MATCH (m:Movie) RETURN m LIMIT 1", rows=1)


def test_complete_json_sends_json_mode_and_the_schema(fake_client):
    """The request uses JSON mode, temperature 0, and states the JSON Schema."""
    fake_client.reply = '{"cypher": "RETURN 1", "rows": 0}'
    llm.complete_json([{"role": "user", "content": "q"}], Answer)
    client = fake_client.instances[0]
    assert client.kwargs["api_key"] == "test-key"
    assert client.kwargs["base_url"] == llm.DEFAULT_BASE_URL
    request = client.calls[0]
    assert request["model"] == "test-model"
    assert request["response_format"] == {"type": "json_object"}
    assert request["temperature"] == 0
    assert request["messages"][0] == {"role": "user", "content": "q"}
    assert request["messages"][-1]["role"] == "system"
    assert "cypher" in request["messages"][-1]["content"]


@pytest.mark.parametrize(
    "reply",
    [
        "not json at all",
        '{"cypher": "RETURN 1"}',             # missing field
        '{"cypher": "RETURN 1", "rows": "many"}',  # wrong type
        None,                                 # empty message
    ],
)
def test_complete_json_rejects_answers_off_schema(fake_client, reply):
    """Anything that is not a valid instance of the schema is an LLMError."""
    fake_client.reply = reply
    with pytest.raises(llm.LLMError, match="does not match Answer"):
        llm.complete_json([{"role": "user", "content": "q"}], Answer)


def test_complete_json_wraps_provider_errors_without_leaking_the_key(fake_client):
    """A provider failure becomes LLMError naming the error class, never the key."""
    fake_client.reply = openai.APITimeoutError(
        request=httpx2.Request("POST", "https://llm.invalid/v1/chat/completions")
    )
    with pytest.raises(llm.LLMError) as caught:
        llm.complete_json([{"role": "user", "content": "q"}], Answer)
    assert "APITimeoutError" in str(caught.value)
    assert "test-key" not in str(caught.value)
