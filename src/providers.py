"""
Provider adapter layer - one interface, three backends, selected by
LLM_PROVIDER (env var, default "gemini").

Why this exists: the project brief specified the Anthropic API
(claude-sonnet-4-6). During the build, Anthropic and OpenAI accounts both
had no usable credits (see reports/llm_provider_blockers.md for the raw
errors), so Tier 3 shipped on Google Gemini instead. Rather than leave that
as a hardcoded `from google import genai` scattered through
llm_adjudicator.py and llm_naive_experiment.py, every provider call in this
project goes through the LLMProvider interface below - so "provider-
agnostic" is demonstrable (swap LLM_PROVIDER, same code runs) rather than
just asserted in a docstring.

VERIFICATION STATUS (be precise about what's actually been tested):
  - GeminiProvider: fully verified. This is what every real run in this
    repo - the naive experiment, the disciplined adjudicator, the cached
    reports/tier3_adjudication_results.json - actually used.
  - AnthropicProvider, OpenAIProvider: written to the same interface,
    using each SDK's real API shape (structured output via
    output_config.format for Anthropic, response_format for OpenAI), but
    NOT run against a live API - both accounts were blocked on billing for
    the entire build (see reports/llm_provider_blockers.md). Treat these
    as "should work, unverified" until someone with credits runs them.
    If you get Anthropic or OpenAI credits, re-running Tier 3 is exactly
    `LLM_PROVIDER=anthropic python src/pipeline.py --rerun-tier3` (or
    `LLM_PROVIDER=openai ...`) - no other code change needed.

Interface: one method, generate_structured(), that every backend
implements. It takes a system instruction, a user prompt, and an optional
JSON schema; returns (parsed_dict_or_None, raw_text). When schema is None,
parsed is None and the caller works with raw_text directly (this is what
the naive experiment uses, deliberately, to reproduce prompt-and-hope
behavior for the "what broke" comparison - see llm_naive_experiment.py).
"""

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class ProviderConfig:
    provider: str
    model: str
    api_key_env_var: str


# Default model per provider. Override with LLM_MODEL if you want a
# different one on the same provider.
DEFAULT_MODELS = {
    "gemini": "gemini-3.5-flash-lite",
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-4o-mini",
}
API_KEY_ENV_VARS = {
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}


class LLMProvider(ABC):
    """One method every backend must implement. See module docstring."""

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    @abstractmethod
    def generate_structured(
        self, system_instruction: str, user_prompt: str, json_schema: Optional[dict] = None
    ) -> tuple:
        """
        Returns (parsed, raw_text):
          - if json_schema is given: parsed is a dict (schema-validated by
            the provider where supported), raw_text is the same content
            serialized.
          - if json_schema is None: parsed is None, raw_text is the raw
            model output (used by the naive experiment, which deliberately
            does NOT use structured output - see llm_naive_experiment.py).
        Must raise the provider SDK's own exception on failure - callers
        are responsible for catching/retrying, this layer doesn't swallow
        errors.
        """
        raise NotImplementedError


def _to_gemini_schema(schema):
    """
    Gemini's response_schema dialect rejects standard JSON Schema's
    `"type": ["string", "null"]` union form (confirmed via a live 400:
    "Input should be 'TYPE_UNSPECIFIED', 'STRING', ... or 'NULL'
    [type=enum, input_value=['string', 'null']"). It wants
    `"type": "string", "nullable": true` instead. Schemas in this project
    are written in standard JSON Schema (see llm_adjudicator.RESPONSE_SCHEMA)
    so they stay portable across providers; this translates ONLY for the
    Gemini call, recursively, without touching the schema object callers
    pass in.
    """
    if isinstance(schema, dict):
        result = {}
        for key, value in schema.items():
            if key == "type" and isinstance(value, list) and "null" in value:
                non_null = [t for t in value if t != "null"]
                result["type"] = non_null[0] if len(non_null) == 1 else non_null
                result["nullable"] = True
            elif isinstance(value, (dict, list)):
                result[key] = _to_gemini_schema(value)
            else:
                result[key] = value
        return result
    if isinstance(schema, list):
        return [_to_gemini_schema(item) for item in schema]
    return schema


class GeminiProvider(LLMProvider):
    """Verified - every real run in this repo used this backend."""

    def generate_structured(
        self, system_instruction: str, user_prompt: str, json_schema: Optional[dict] = None
    ) -> tuple:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)

        config_kwargs = {"system_instruction": system_instruction, "temperature": 0.0}
        if json_schema is not None:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = _to_gemini_schema(json_schema)

        response = client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        raw_text = response.text or ""
        parsed = json.loads(raw_text) if json_schema is not None else None
        return parsed, raw_text


class AnthropicProvider(LLMProvider):
    """
    UNVERIFIED - written to the Anthropic Messages API's documented shape
    (structured output via output_config.format, per the current API - see
    module docstring) but never run against a live account; both attempts
    during this build hit "credit balance too low" (see
    reports/llm_provider_blockers.md). Re-run and report back if you get
    this working - the interface is designed so no other file needs to
    change.
    """

    def generate_structured(
        self, system_instruction: str, user_prompt: str, json_schema: Optional[dict] = None
    ) -> tuple:
        import anthropic

        client = anthropic.Anthropic(api_key=self.api_key)

        kwargs = {
            "model": self.model,
            "max_tokens": 4096,
            "system": system_instruction,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        if json_schema is not None:
            kwargs["output_config"] = {"format": {"type": "json_schema", "schema": json_schema}}

        response = client.messages.create(**kwargs)
        raw_text = "".join(block.text for block in response.content if block.type == "text")
        parsed = json.loads(raw_text) if json_schema is not None else None
        return parsed, raw_text


class OpenAIProvider(LLMProvider):
    """
    UNVERIFIED - written to the OpenAI Chat Completions API's documented
    structured-output shape (response_format json_schema) but never run
    against a live account; the attempt during this build hit
    "insufficient_quota / credit_balance_exhausted" (see
    reports/llm_provider_blockers.md). Same note as AnthropicProvider: the
    interface is designed so getting this working is a credits problem,
    not a code problem.
    """

    def generate_structured(
        self, system_instruction: str, user_prompt: str, json_schema: Optional[dict] = None
    ) -> tuple:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)

        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt},
            ],
        }
        if json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": json_schema, "strict": True},
            }

        response = client.chat.completions.create(**kwargs)
        raw_text = response.choices[0].message.content or ""
        parsed = json.loads(raw_text) if json_schema is not None else None
        return parsed, raw_text


_PROVIDER_CLASSES = {
    "gemini": GeminiProvider,
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
}


def get_provider() -> LLMProvider:
    """
    Selects and constructs a provider from LLM_PROVIDER (default "gemini")
    and LLM_MODEL (default: DEFAULT_MODELS[provider]). Fails loudly if the
    corresponding API key env var is missing - same discipline as the rest
    of this pipeline (see llm_adjudicator.require_api_key), never silently
    falls back to a different provider or a mocked response.
    """
    provider_name = os.environ.get("LLM_PROVIDER", "gemini").lower()
    if provider_name not in _PROVIDER_CLASSES:
        raise ValueError(
            f"Unknown LLM_PROVIDER={provider_name!r}. Must be one of {sorted(_PROVIDER_CLASSES)}."
        )

    api_key_var = API_KEY_ENV_VARS[provider_name]
    api_key = os.environ.get(api_key_var)
    if not api_key:
        raise RuntimeError(
            f"LLM_PROVIDER={provider_name!r} requires {api_key_var} to be set (in .env or the "
            f"environment). Get one and set it, or switch LLM_PROVIDER to a provider you have a "
            f"key for."
        )

    model = os.environ.get("LLM_MODEL", DEFAULT_MODELS[provider_name])
    provider_class = _PROVIDER_CLASSES[provider_name]
    return provider_class(api_key=api_key, model=model)
