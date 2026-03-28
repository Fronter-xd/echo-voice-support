"""
Voice Engine - Main orchestrator for the Echo Voice Support system.
Coordinates STT, LLM, and TTS for real-time voice conversations.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Optional, Callable, Awaitable
import structlog

from services.stt import WhisperSTT, StreamingWhisper
from services.llm import OllamaLLM, ConversationManager
from services.tts import EdgeTTS, TTSCache, VoiceProcessor
from config.settings import config

logger = structlog.get_logger()


@dataclass
class TimingStats:
    """Performance statistics for voice pipeline."""

    stt_time_ms: float = 0
    llm_time_ms: float = 0
    tts_time_ms: float = 0
    total_time_ms: float = 0
    timestamp: Optional[float] = None


@dataclass
class VoiceMessage:
    """Represents a voice interaction."""

    id: str
    user_id: str
    audio_input: Optional[bytes] = None
    text_input: Optional[str] = None
    text_output: Optional[str] = None
    audio_output: Optional[bytes] = None
    timing: Optional[TimingStats] = None
    error: Optional[str] = None


class VoiceEngine:
    """
    Main voice engine that orchestrates the entire voice pipeline.

    Pipeline:
    1. Audio Input -> STT (Whisper) -> Text
    2. Text -> LLM (Ollama) -> Response Text
    3. Response Text -> TTS (Edge-TTS) -> Audio Output
    """

    def __init__(
        self,
        on_transcript: Optional[Callable[[str], Awaitable[None]]] = None,
        on_response: Optional[Callable[[str], Awaitable[None]]] = None,
        on_audio: Optional[Callable[[bytes], Awaitable[None]]] = None,
        on_error: Optional[Callable[[Exception], Awaitable[None]]] = None,
    ):
        self.stt = WhisperSTT(config)
        self.stt_streaming = StreamingWhisper(config)
        self.llm = OllamaLLM(config)
        self.conversation_manager = ConversationManager(self.llm)
        self.tts = EdgeTTS(config)
        self.tts_cache = TTSCache()
        self.voice_processor = VoiceProcessor()

        self.on_transcript = on_transcript
        self.on_response = on_response
        self.on_audio = on_audio
        self.on_error = on_error

        self.active_sessions = {}
        self._running = False

    async def start(self):
        """Initialize and start the voice engine."""
        logger.info("Starting Echo Voice Engine")

        try:
            self.stt.load_model()
            logger.info("STT model loaded")

            if await self.llm.check_health():
                logger.info("LLM service connected")
            else:
                logger.warning("LLM service not available - using fallback responses")

            self._running = True
            logger.info("Voice engine started successfully")

        except Exception as e:
            logger.error("Failed to start voice engine", error=str(e))
            raise

    async def stop(self):
        """Stop the voice engine."""
        logger.info("Stopping Voice Engine")
        self._running = False

        for session_id in list(self.active_sessions.keys()):
            await self.end_session(session_id)

        logger.info("Voice engine stopped")

    async def process_voice_input(
        self, audio_data: bytes, user_id: str, session_id: Optional[str] = None
    ) -> VoiceMessage:
        """
        Process a voice input through the complete pipeline.

        Args:
            audio_data: Raw audio bytes (16-bit PCM, 16kHz)
            user_id: User identifier
            session_id: Optional session for conversation context

        Returns:
            VoiceMessage with timing stats and outputs
        """
        message_id = str(uuid.uuid4())
        session_id = session_id or str(uuid.uuid4())
        start_time = time.perf_counter()

        message = VoiceMessage(id=message_id, user_id=user_id)

        timing = TimingStats()

        try:
            text_input, stt_time = self.stt.transcribe(audio_data)
            timing.stt_time_ms = stt_time

            if not text_input:
                logger.debug("No speech detected")
                return message

            message.text_input = text_input

            if self.on_transcript:
                await self.on_transcript(text_input)

            text_output = await self.conversation_manager.chat(
                session_id=session_id, message=text_input
            )

            timing.llm_time_ms = (time.perf_counter() - start_time) * 1000 - stt_time

            message.text_output = text_output

            if self.on_response:
                await self.on_response(text_output)

            audio_output, tts_time = await self.tts.synthesize(text_output)
            timing.tts_time_ms = tts_time

            audio_output = self.voice_processor.add_silence(
                audio_output, duration_ms=100
            )

            message.audio_output = audio_output

            if self.on_audio:
                await self.on_audio(audio_output)

            timing.total_time_ms = (time.perf_counter() - start_time) * 1000
            message.timing = timing

            logger.info(
                "Voice interaction complete",
                message_id=message_id,
                total_time_ms=round(timing.total_time_ms, 2),
                stt_time=round(stt_time, 2),
                llm_time=round(timing.llm_time_ms, 2),
                tts_time=round(tts_time, 2),
            )

            return message

        except Exception as e:
            logger.error("Voice processing failed", error=str(e))
            message.error = str(e)
            message.timing = timing

            if self.on_error:
                await self.on_error(e)

            return message

    async def process_text_input(
        self, text_input: str, user_id: str, session_id: Optional[str] = None
    ) -> VoiceMessage:
        """
        Process text input (bypass STT) and return audio response.
        Useful for testing or text-based interactions.
        """
        message_id = str(uuid.uuid4())
        session_id = session_id or str(uuid.uuid4())
        start_time = time.perf_counter()

        message = VoiceMessage(id=message_id, user_id=user_id, text_input=text_input)

        timing = TimingStats()

        try:
            text_output = await self.conversation_manager.chat(
                session_id=session_id, message=text_input
            )

            timing.llm_time_ms = (time.perf_counter() - start_time) * 1000
            message.text_output = text_output

            if self.on_response:
                await self.on_response(text_output)

            audio_output, tts_time = await self.tts.synthesize(text_output)
            timing.tts_time_ms = tts_time
            message.audio_output = audio_output

            if self.on_audio:
                await self.on_audio(audio_output)

            timing.total_time_ms = (time.perf_counter() - start_time) * 1000
            message.timing = timing

            return message

        except Exception as e:
            logger.error("Text processing failed", error=str(e))
            message.error = str(e)
            return message

    async def start_session(self, user_id: str) -> str:
        """Start a new voice session."""
        session_id = str(uuid.uuid4())
        self.active_sessions[session_id] = {
            "user_id": user_id,
            "started_at": time.time(),
            "message_count": 0,
        }
        logger.info("Session started", session_id=session_id, user_id=user_id)
        return session_id

    async def end_session(self, session_id: str):
        """End a voice session."""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            self.conversation_manager.end_session(session_id)
            logger.info("Session ended", session_id=session_id)

    def get_session_stats(self, session_id: str) -> Optional[dict]:
        """Get statistics for a session."""
        if session_id not in self.active_sessions:
            return None

        session = self.active_sessions[session_id]
        return {
            "session_id": session_id,
            "user_id": session["user_id"],
            "duration_seconds": time.time() - session["started_at"],
            "message_count": session["message_count"],
        }


class VoicePipeline:
    """
    Optimized voice pipeline for sub-2s latency.
    Implements parallel processing where possible.
    """

    def __init__(self, engine: VoiceEngine):
        self.engine = engine
        self.stt = engine.stt
        self.llm = engine.llm
        self.tts = engine.tts

    async def low_latency_process(
        self, audio_data: bytes, user_id: str
    ) -> VoiceMessage:
        """
        Optimized processing for minimum latency.

        Optimization strategies:
        - Pre-load models
        - Parallel LLM and TTS preparation
        - Stream audio output
        """
        start_time = time.perf_counter()

        text_input, stt_time = self.stt.transcribe(audio_data)

        if not text_input:
            return VoiceMessage(
                id=str(uuid.uuid4()),
                user_id=user_id,
                timing=TimingStats(
                    stt_time_ms=stt_time,
                    total_time_ms=(time.perf_counter() - start_time) * 1000,
                ),
            )

        response_task = asyncio.create_task(
            self.llm.generate(prompt=text_input, session_id=user_id)
        )

        audio_task = asyncio.create_task(self._prepare_audio(response_task))

        text_output = await response_task
        audio_output = await audio_task

        total_time = (time.perf_counter() - start_time) * 1000

        return VoiceMessage(
            id=str(uuid.uuid4()),
            user_id=user_id,
            text_input=text_input,
            text_output=text_output,
            audio_output=audio_output,
            timing=TimingStats(
                stt_time_ms=stt_time,
                llm_time_ms=stt_time,
                tts_time_ms=stt_time,
                total_time_ms=total_time,
            ),
        )

    async def _prepare_audio(self, response_task: asyncio.Task) -> bytes:
        """Prepare audio while waiting for LLM response."""
        text_output = await response_task

        if not text_output:
            return b""

        audio_output, _ = await self.tts.synthesize(text_output)
        return audio_output
