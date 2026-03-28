"""
Text-to-Speech Service using Edge-TTS.
Provides natural-sounding voices with Indian accent support.
"""

import asyncio
import io
import time
from typing import Optional, Tuple
import structlog

logger = structlog.get_logger()


class EdgeTTS:
    """Text-to-Speech using Microsoft Edge's neural voices."""

    VOICES = {
        "en-IN-Neural": "en-IN-NeerjaNeural",
        "en-IN-Neural-Male": "en-IN-PrabhatNeural",
        "hi-IN-Neural": "hi-IN-MadhurNeural",
        "ta-IN-Neural": "ta-IN-PallaviNeural",
    }

    def __init__(self, config):
        self.config = config.tts
        self.voice = self.config.voice
        self.rate = self.config.rate
        self.pitch = self.config.pitch
        self._pool = None

    async def synthesize(
        self, text: str, output_path: Optional[str] = None, voice: Optional[str] = None
    ) -> Tuple[bytes, float]:
        """
        Synthesize speech from text.

        Args:
            text: Input text to synthesize
            output_path: Optional path to save audio file
            voice: Optional voice override

        Returns:
            Tuple of (audio_bytes, synthesis_time_ms)
        """
        start_time = time.perf_counter()

        try:
            from edge_tts import Communicate

            voice_to_use = voice or self.voice

            communicate = Communicate(
                text, voice=voice_to_use, rate=self.rate, pitch=self.pitch
            )

            if output_path:
                await communicate.save(output_path)
                audio_data = open(output_path, "rb").read()
            else:
                audio_buffer = io.BytesIO()
                await communicate.save(audio_buffer)
                audio_data = audio_buffer.getvalue()

            process_time = (time.perf_counter() - start_time) * 1000
            logger.info(
                "TTS synthesis complete",
                voice=voice_to_use,
                text_length=len(text),
                time_ms=round(process_time, 2),
            )

            return audio_data, process_time

        except ImportError:
            logger.warning("Edge-TTS not installed, using fallback")
            return self._fallback_synthesize(text, start_time)

        except Exception as e:
            logger.error("TTS synthesis failed", error=str(e))
            return self._fallback_synthesize(text, start_time)

    async def synthesize_streaming(
        self, text: str, voice: Optional[str] = None
    ) -> AsyncGenerator[bytes, None]:
        """
        Synthesize speech in a streaming fashion.
        Yields audio chunks as they become available.
        """
        try:
            from edge_tts import Communicate

            voice_to_use = voice or self.voice

            async for chunk in Communicate(
                text, voice=voice_to_use, rate=self.rate, pitch=self.pitch
            ).stream():
                if chunk["type"] == "audio":
                    yield chunk["data"]

        except ImportError:
            logger.error("Edge-TTS not available for streaming")

        except Exception as e:
            logger.error("Streaming TTS failed", error=str(e))

    def _fallback_synthesize(self, text, start_time) -> Tuple[bytes, float]:
        """Generate silence as fallback."""
        logger.info("Using silent fallback for TTS")

        import struct

        duration_sec = min(len(text) / 10, 3)
        sample_rate = 16000
        num_samples = int(duration_sec * sample_rate)

        audio_data = struct.pack("<" + "h" * num_samples, *([0] * num_samples))

        return audio_data, (time.perf_counter() - start_time) * 1000

    async def synthesize_ssml(
        self, ssml: str, output_path: Optional[str] = None
    ) -> bytes:
        """Synthesize from SSML markup for fine-tuned control."""
        try:
            from edge_tts import Communicate

            if output_path:
                await Communicate(ssml, voice=self.voice).save(output_path)
                return open(output_path, "rb").read()
            else:
                buffer = io.BytesIO()
                await Communicate(ssml, voice=self.voice).save(buffer)
                return buffer.getvalue()

        except Exception as e:
            logger.error("SSML synthesis failed", error=str(e))
            return b""


class TTSCache:
    """Cache for frequently used TTS outputs."""

    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.max_size = max_size
        self.access_times = {}

    def _make_key(self, text: str, voice: str) -> str:
        """Create cache key from text and voice."""
        return f"{voice}:{hash(text)}"

    async def get_or_synthesize(
        self, tts: EdgeTTS, text: str, voice: Optional[str] = None
    ) -> bytes:
        """Get from cache or synthesize."""
        key = self._make_key(text, voice or tts.voice)

        if key in self.cache:
            return self.cache[key]

        audio, _ = await tts.synthesize(text, voice=voice)

        if len(self.cache) >= self.max_size:
            oldest = min(self.access_times, key=self.access_times.get)
            del self.cache[oldest]
            del self.access_times[oldest]

        self.cache[key] = audio
        self.access_times[key] = time.time()

        return audio

    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.access_times.clear()


class VoiceProcessor:
    """Process and optimize TTS output."""

    @staticmethod
    def convert_sample_rate(audio_data: bytes, from_rate: int, to_rate: int) -> bytes:
        """Convert audio between sample rates (basic implementation)."""
        if from_rate == to_rate:
            return audio_data

        import struct

        num_samples = len(audio_data) // 2
        samples = struct.unpack("<" + "h" * num_samples, audio_data)

        ratio = to_rate / from_rate
        new_num_samples = int(num_samples * ratio)

        indices = [int(i * ratio) for i in range(new_num_samples)]
        resampled = [samples[min(i, num_samples - 1)] for i in indices]

        return struct.pack("<" + "h" * new_num_samples, *resampled)

    @staticmethod
    def add_silence(
        audio_data: bytes, duration_ms: int, sample_rate: int = 16000
    ) -> bytes:
        """Add silence at the beginning of audio."""
        import struct

        silence_samples = int(sample_rate * duration_ms / 1000)
        silence = struct.pack("<" + "h" * silence_samples, *([0] * silence_samples))

        return silence + audio_data

    @staticmethod
    def normalize_volume(audio_data: bytes, target_db: float = -3.0) -> bytes:
        """Normalize audio volume."""
        import struct
        import math

        num_samples = len(audio_data) // 2
        samples = list(struct.unpack("<" + "h" * num_samples, audio_data))

        rms = math.sqrt(sum(s * s for s in samples) / num_samples)
        if rms < 1:
            return audio_data

        current_db = 20 * math.log10(rms / 32768)
        gain = math.pow(10, (target_db - current_db) / 20)

        normalized = [int(max(-32768, min(32767, s * gain))) for s in samples]

        return struct.pack("<" + "h" * num_samples, *normalized)
