"""LLM saglayici soyutlamasi (Faz 4 - RAG generation katmani).

Oneri formu 2.4: "hazir buyuk dil modeli, sadece cikarim (inference) icin kullanilacak,
egitilmeyecek". Kullanici karari (2026-07-06): API tabanli LLM (Anthropic/OpenAI) - 6GB VRAM'i
yerel LLM ile zorlamamak icin. Bu arayuz sayesinde saglayici degistirmek tek bir alt siniftan
baska hicbir cagiran kodu etkilemez.

Tum sayisal hesaplar (esik kiyaslama, risk bayraklari) kural tabanli Python'da yapilir
(src/ocr/risk_engine.py) - LLM sadece retrieval ile saglanan baglami dogal dile donusturur,
kendi basina esik/oran UYDURMAZ (bkz. src/rag/generate.py prompt tasarimi).
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

from dotenv import load_dotenv

load_dotenv()


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, system: str | None = None) -> str:
        """Prompt (+ opsiyonel sistem talimati) icin LLM'den metin yaniti dondurur."""


class AnthropicProvider(LLMProvider):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> None:
        import anthropic

        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY bulunamadi (.env dosyasini kontrol edin)")
        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, prompt: str, system: str | None = None) -> str:
        kwargs = {"system": system} if system else {}
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return "".join(block.text for block in response.content if block.type == "text")


class OpenAIProvider(LLMProvider):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> None:
        import openai

        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY bulunamadi (.env dosyasini kontrol edin)")
        self._client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, prompt: str, system: str | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=messages,
        )
        return response.choices[0].message.content or ""


def get_llm_provider(config: dict | None = None) -> LLMProvider:
    """config.yaml -> llm.provider alanina gore dogru saglayici sinifini kurar (factory)."""
    if config is None:
        from src.common.config import get_config

        config = get_config()

    llm_cfg = config["llm"]
    provider_name = llm_cfg["provider"]

    if provider_name == "anthropic":
        return AnthropicProvider(
            model=llm_cfg["model_anthropic"],
            max_tokens=llm_cfg["max_tokens"],
            temperature=llm_cfg["temperature"],
        )
    if provider_name == "openai":
        return OpenAIProvider(
            model=llm_cfg["model_openai"],
            max_tokens=llm_cfg["max_tokens"],
            temperature=llm_cfg["temperature"],
        )
    raise ValueError(f"Bilinmeyen LLM saglayici: {provider_name}")
