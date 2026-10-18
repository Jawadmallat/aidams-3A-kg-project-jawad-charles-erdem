"""The one place the API talks to a language model.

Any OpenAI-compatible provider works (Groq, OpenAI, a local server): the
client is configured by LLM_BASE_URL, LLM_MODEL and LLM_API_KEY, read from the
environment on each call so tests and `.env` changes need no restart logic.
Without a key or a model the LLM features are off: `get_settings` returns None
and the endpoints answer 503 (or omit the AI-written part) instead of failing.

Tests never reach a provider: they replace `complete_json` with a stub.
"""

import json
import os
from dataclasses import dataclass

import openai
from pydantic import BaseModel, ValidationError

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_TIMEOUT_S = 20.0


class LLMNotConfigured(Exception):
    """No API key or model is set: the LLM features are switched off."""


class LLMError(Exception):
    """The provider failed, timed out, or answered something unusable."""


@dataclass(frozen=True)
class LLMSettings:
    """Connection settings of the OpenAI-compatible provider."""

    base_url: str
    model: str
    api_key: str
    timeout_s: float


def get_settings() -> LLMSettings | None:
    """Read the LLM settings from the environment; None when the LLM is off."""
    api_key = os.environ.get("LLM_API_KEY", "").strip()
    model = os.environ.get("LLM_MODEL", "").strip()
    if not api_key or not model:
        return None
    return LLMSettings(
        base_url=os.environ.get("LLM_BASE_URL", "").strip() or DEFAULT_BASE_URL,
        model=model,
        api_key=api_key,
        timeout_s=float(os.environ.get("LLM_TIMEOUT_S", DEFAULT_TIMEOUT_S)),
    )


def complete_json[T: BaseModel](messages: list[dict[str, str]], schema: type[T]) -> T:
    """Ask the model for a JSON object and validate it against `schema`.

    JSON mode (`response_format={"type": "json_object"}`) is the structured
    output every OpenAI-compatible provider supports; the expected JSON Schema
    is appended to the system instructions, and Pydantic validation is what
    actually guarantees the shape. Raises LLMNotConfigured when the LLM is
    off and LLMError for any provider or validation failure.
    """
    settings = get_settings()
    if settings is None:
        raise LLMNotConfigured("set LLM_API_KEY and LLM_MODEL to enable the LLM")
    client = openai.OpenAI(
        api_key=settings.api_key,
        base_url=settings.base_url,
        timeout=settings.timeout_s,
        max_retries=1,
    )
    instructions = {
        "role": "system",
        "content": "Answer with one JSON object matching this JSON Schema, and nothing else:\n"
        + json.dumps(schema.model_json_schema()),
    }
    try:
        completion = client.chat.completions.create(
            model=settings.model,
            messages=[*messages, instructions],
            response_format={"type": "json_object"},
            temperature=0,
        )
    except openai.APIError as error:
        raise LLMError(f"LLM provider error: {type(error).__name__}") from error
    content = completion.choices[0].message.content or ""
    try:
        return schema.model_validate_json(content)
    except ValidationError as error:
        raise LLMError(f"LLM answer does not match {schema.__name__}") from error
