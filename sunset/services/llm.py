"""
LLM service supporting multiple providers (OpenAI, Gemini).

Usage:
    from sunset.services import LLMService
    
    llm = LLMService(openai_api_key="...")
    response = await llm.complete("What is 2+2?")
"""

import logging
from typing import Optional, List, Dict

from openai import AsyncOpenAI


logger = logging.getLogger(__name__)


class LLMService:
    """Multi-provider LLM service."""

    _instance: Optional["LLMService"] = None

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        default_provider: str = "openai",
        default_model: Optional[str] = None,
    ):
        self.openai_client = None
        self.gemini_client = None
        self.default_provider = default_provider

        if openai_api_key:
            self.openai_client = AsyncOpenAI(api_key=openai_api_key)
            logger.info("OpenAI client initialized")

        if gemini_api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=gemini_api_key)
                self.gemini_client = genai
                logger.info("Gemini client initialized")
            except ImportError:
                logger.warning("google-generativeai not installed")

        self.default_model = default_model or self._get_default_model()

    def _get_default_model(self) -> str:
        if self.default_provider == "openai":
            return "gpt-4o-mini"
        elif self.default_provider == "gemini":
            return "gemini-1.5-flash"
        return "gpt-4o-mini"

    @classmethod
    def get_instance(cls, **kwargs) -> "LLMService":
        if cls._instance is None:
            cls._instance = cls(**kwargs)
        return cls._instance

    async def complete(
        self,
        prompt: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        return await self.chat(messages=messages, provider=provider, model=model, temperature=temperature, max_tokens=max_tokens)

    async def chat(
        self,
        messages: List[Dict[str, str]],
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        provider = provider or self.default_provider
        model = model or self.default_model

        if provider == "openai":
            return await self._openai_chat(messages, model, temperature, max_tokens)
        elif provider == "gemini":
            return await self._gemini_chat(messages, model, temperature, max_tokens)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    async def _openai_chat(self, messages: List[Dict[str, str]], model: str, temperature: float, max_tokens: int) -> str:
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")

        response = await self.openai_client.chat.completions.create(
            model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
        )
        return response.choices[0].message.content

    async def _gemini_chat(self, messages: List[Dict[str, str]], model: str, temperature: float, max_tokens: int) -> str:
        if not self.gemini_client:
            raise ValueError("Gemini client not initialized")

        gemini_model = self.gemini_client.GenerativeModel(model)
        
        history = []
        for msg in messages[:-1]:
            role = "user" if msg["role"] == "user" else "model"
            history.append({"role": role, "parts": [msg["content"]]})

        chat = gemini_model.start_chat(history=history)
        
        last_message = messages[-1]["content"]
        response = chat.send_message(last_message, generation_config={"temperature": temperature, "max_output_tokens": max_tokens})
        return response.text
