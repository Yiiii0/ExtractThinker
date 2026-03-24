import sys
import types

import pytest

sys.modules.setdefault("magic", types.SimpleNamespace())

from extract_thinker.llm import LLM


def test_forge_routes_requests_to_forge_base(monkeypatch):
    monkeypatch.setenv("FORGE_API_KEY", "forge-test-key")
    monkeypatch.setenv("FORGE_API_BASE", "https://forge.example/v1")

    llm = LLM("forge/OpenAI/gpt-4o-mini")
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(llm.client.chat.completions, "create", fake_create)

    llm._request_direct([{"role": "user", "content": "ping"}], None)

    assert captured["model"] == "OpenAI/gpt-4o-mini"
    assert captured["api_key"] == "forge-test-key"
    assert captured["api_base"] == "https://forge.example/v1"


def test_forge_uses_default_base_when_override_missing(monkeypatch):
    monkeypatch.setenv("FORGE_API_KEY", "forge-test-key")
    monkeypatch.delenv("FORGE_API_BASE", raising=False)

    llm = LLM("forge/OpenAI/gpt-4o-mini")
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(llm.client.chat.completions, "create", fake_create)

    llm._request_direct([{"role": "user", "content": "ping"}], None)

    assert captured["api_base"] == llm.FORGE_DEFAULT_API_BASE


def test_forge_requires_api_key(monkeypatch):
    monkeypatch.delenv("FORGE_API_KEY", raising=False)

    llm = LLM("forge/OpenAI/gpt-4o-mini")
    with pytest.raises(ValueError, match="FORGE_API_KEY"):
        llm._request_direct([{"role": "user", "content": "ping"}], None)


def test_forge_requires_provider_model_format(monkeypatch):
    monkeypatch.setenv("FORGE_API_KEY", "forge-test-key")

    llm = LLM("forge/gpt-4o-mini")
    with pytest.raises(ValueError, match="forge/Provider/model-name"):
        llm._request_direct([{"role": "user", "content": "ping"}], None)
