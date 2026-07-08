from unittest.mock import MagicMock, patch

import pytest

from src.rag.llm_provider import AnthropicProvider, OpenAIProvider, get_llm_provider


def test_anthropic_provider_requires_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        AnthropicProvider(model="claude-sonnet-5")


def test_anthropic_provider_generate_calls_sdk_and_joins_text_blocks():
    with patch("anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        block1 = MagicMock(type="text", text="Merhaba ")
        block2 = MagicMock(type="text", text="dunya")
        mock_response = MagicMock(content=[block1, block2])
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_cls.return_value = mock_client

        provider = AnthropicProvider(model="claude-sonnet-5", api_key="fake-key")
        result = provider.generate("test prompt", system="test system")

        assert result == "Merhaba dunya"
        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == "claude-sonnet-5"
        assert call_kwargs["system"] == "test system"
        assert call_kwargs["messages"] == [{"role": "user", "content": "test prompt"}]


def test_openai_provider_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIProvider(model="gpt-4o-mini")


def test_openai_provider_generate_calls_sdk():
    with patch("openai.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_message = MagicMock(content="test yaniti")
        mock_choice = MagicMock(message=mock_message)
        mock_response = MagicMock(choices=[mock_choice])
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_cls.return_value = mock_client

        provider = OpenAIProvider(model="gpt-4o-mini", api_key="fake-key")
        result = provider.generate("test prompt", system="test system")

        assert result == "test yaniti"
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["messages"][0] == {"role": "system", "content": "test system"}
        assert call_kwargs["messages"][1] == {"role": "user", "content": "test prompt"}


def test_get_llm_provider_selects_anthropic_from_config(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    config = {
        "llm": {
            "provider": "anthropic",
            "model_anthropic": "claude-sonnet-5",
            "model_openai": "gpt-4o-mini",
            "max_tokens": 512,
            "temperature": 0.1,
        }
    }
    with patch("anthropic.Anthropic"):
        provider = get_llm_provider(config)
        assert isinstance(provider, AnthropicProvider)
        assert provider.model == "claude-sonnet-5"
        assert provider.max_tokens == 512


def test_get_llm_provider_selects_openai_from_config(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    config = {
        "llm": {
            "provider": "openai",
            "model_anthropic": "claude-sonnet-5",
            "model_openai": "gpt-4o-mini",
            "max_tokens": 512,
            "temperature": 0.1,
        }
    }
    with patch("openai.OpenAI"):
        provider = get_llm_provider(config)
        assert isinstance(provider, OpenAIProvider)
        assert provider.model == "gpt-4o-mini"


def test_get_llm_provider_rejects_unknown_provider():
    config = {"llm": {"provider": "not_a_real_provider"}}
    with pytest.raises(ValueError, match="Bilinmeyen LLM saglayici"):
        get_llm_provider(config)
