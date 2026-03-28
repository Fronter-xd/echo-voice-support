"""
Large Language Model Service using Ollama.
Provides conversational AI with low latency.
"""

import asyncio
import time
from typing import Optional, AsyncGenerator
import structlog

logger = structlog.get_logger()


class OllamaLLM:
    """LLM service using Ollama for local inference."""

    SYSTEM_PROMPT = """You are Echo, a helpful and friendly AI voice assistant. 
You provide clear, concise responses suitable for voice interaction.
Keep responses relatively short (1-3 sentences) for natural conversation flow.
Be warm, professional, and helpful."""

    def __init__(self, config):
        self.config = config.ollama
        self.base_url = self.config.base_url
        self.model = self.config.model
        self._session_history = {}

    async def check_health(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/api/tags", timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    return response.status == 200

        except Exception as e:
            logger.error("Ollama health check failed", error=str(e))
            return False

    async def generate(
        self,
        prompt: str,
        session_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate a response from the LLM.

        Args:
            prompt: User input text
            session_id: Optional session ID for conversation context
            system_prompt: Optional custom system prompt

        Returns:
            Generated response text
        """
        start_time = time.perf_counter()

        try:
            import aiohttp

            messages = []

            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            else:
                messages.append({"role": "system", "content": self.SYSTEM_PROMPT})

            if session_id and session_id in self._session_history:
                messages.extend(self._session_history[session_id])

            messages.append({"role": "user", "content": prompt})

            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_ctx": self.config.num_ctx,
                },
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout),
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(
                            "Ollama API error", status=response.status, error=error_text
                        )
                        return self._get_fallback_response()

                    data = await response.json()
                    response_text = data.get("message", {}).get("content", "")

                    if session_id:
                        messages.append({"role": "assistant", "content": response_text})
                        self._session_history[session_id] = messages[1:]

                    process_time = (time.perf_counter() - start_time) * 1000
                    logger.info(
                        "LLM response generated",
                        time_ms=round(process_time, 2),
                        prompt_length=len(prompt),
                    )

                    return response_text.strip()

        except asyncio.TimeoutError:
            logger.error("LLM request timed out")
            return "I'm sorry, that took too long. Could you try again?"

        except Exception as e:
            logger.error("LLM generation failed", error=str(e))
            return self._get_fallback_response()

    async def generate_streaming(
        self,
        prompt: str,
        session_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response from the LLM.
        Yields tokens as they become available.
        """
        try:
            import aiohttp

            messages = []

            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            else:
                messages.append({"role": "system", "content": self.SYSTEM_PROMPT})

            if session_id and session_id in self._session_history:
                messages.extend(self._session_history[session_id])

            messages.append({"role": "user", "content": prompt})

            payload = {"model": self.model, "messages": messages, "stream": True}

            full_response = []

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout),
                ) as response:
                    async for line in response.content:
                        if line:
                            try:
                                import json

                                data = json.loads(line)

                                if "message" in data:
                                    token = data["message"].get("content", "")
                                    if token:
                                        full_response.append(token)
                                        yield token

                                if data.get("done"):
                                    break

                            except json.JSONDecodeError:
                                continue

                    if session_id:
                        messages.append(
                            {"role": "assistant", "content": "".join(full_response)}
                        )
                        self._session_history[session_id] = messages[1:]

        except Exception as e:
            logger.error("Streaming generation failed", error=str(e))
            yield self._get_fallback_response()

    def clear_session(self, session_id: str):
        """Clear conversation history for a session."""
        if session_id in self._session_history:
            del self._session_history[session_id]
            logger.info("Session cleared", session_id=session_id)

    def _get_fallback_response(self) -> str:
        """Get a fallback response when LLM is unavailable."""
        return "I'm having trouble processing that right now. Could you please repeat yourself?"

    async def get_available_models(self) -> list:
        """Get list of available models in Ollama."""
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/api/tags", timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return [m["name"] for m in data.get("models", [])]
                    return []

        except Exception as e:
            logger.error("Failed to get models", error=str(e))
            return []


class ConversationManager:
    """Manages multi-session conversations."""

    def __init__(self, llm: OllamaLLM):
        self.llm = llm
        self.active_sessions = set()
        self.session_contexts = {}

    async def chat(self, session_id: str, message: str) -> str:
        """Chat with context awareness."""
        self.active_sessions.add(session_id)

        response = await self.llm.generate(prompt=message, session_id=session_id)

        return response

    def end_session(self, session_id: str):
        """End a conversation session."""
        self.llm.clear_session(session_id)
        self.active_sessions.discard(session_id)

        if session_id in self.session_contexts:
            del self.session_contexts[session_id]
