"""
Speech-to-Text Service using Faster Whisper.
Provides fast, accurate local speech recognition.
"""

import io
import time
import numpy as np
from typing import Optional, Tuple
import structlog

logger = structlog.get_logger()


class WhisperSTT:
    """Speech-to-Text using Faster Whisper for low-latency transcription."""

    def __init__(self, config):
        self.config = config.whisper
        self.model = None
        self._loaded = False

    def load_model(self):
        """Load the Whisper model."""
        if self._loaded:
            return

        try:
            from faster_whisper import WhisperModel

            logger.info(
                "Loading Whisper model",
                model=self.config.model,
                device=self.config.device,
            )

            self.model = WhisperModel(
                self.config.model,
                device=self.config.device,
                compute_type=self.config.compute_type,
            )

            self._loaded = True
            logger.info("Whisper model loaded successfully")

        except ImportError:
            logger.warning("Faster Whisper not available, using fallback")
            self.model = None

    def transcribe(
        self, audio_data: bytes, sample_rate: int = 16000
    ) -> Tuple[Optional[str], float]:
        """
        Transcribe audio to text.

        Args:
            audio_data: Raw audio bytes (16-bit PCM)
            sample_rate: Audio sample rate (default 16kHz)

        Returns:
            Tuple of (transcribed_text, processing_time_ms)
        """
        start_time = time.perf_counter()

        if not self._loaded:
            self.load_model()

        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        audio_float32 = audio_np.astype(np.float32) / 32768.0

        if self.model is None:
            return self._fallback_transcribe(audio_float32, start_time)

        try:
            segments, info = self.model.transcribe(
                audio_float32,
                language="en",
                task="transcribe",
                beam_size=5,
                vad_filter=self.config.vad_enabled
                if hasattr(self.config, "vad_enabled")
                else False,
            )

            full_text = " ".join([segment.text for segment in segments])
            process_time = (time.perf_counter() - start_time) * 1000

            logger.debug(
                "Transcription complete",
                text=full_text[:50],
                time_ms=round(process_time, 2),
            )

            return full_text.strip(), process_time

        except Exception as e:
            logger.error("Transcription failed", error=str(e))
            return None, (time.perf_counter() - start_time) * 1000

    def transcribe_streaming(self, audio_generator, sample_rate: int = 16000):
        """
        Transcribe from an audio stream (generator).
        Yields partial transcriptions as they become available.
        """
        if not self._loaded:
            self.load_model()

        audio_chunks = []

        for chunk in audio_generator:
            audio_chunks.append(chunk)

            if len(audio_chunks) * 1024 / sample_rate >= 1.0:
                audio_data = b"".join(audio_chunks)
                text, _ = self.transcribe(audio_data, sample_rate)

                if text:
                    yield text

                audio_chunks = []

        if audio_chunks:
            audio_data = b"".join(audio_chunks)
            text, _ = self.transcribe(audio_data, sample_rate)
            if text:
                yield text

    def _fallback_transcribe(
        self, audio_float32, start_time
    ) -> Tuple[Optional[str], float]:
        """Fallback transcription using basic energy detection."""
        logger.info("Using fallback transcription")

        energy = np.abs(audio_float32).mean()

        if energy < 0.01:
            return None, (time.perf_counter() - start_time) * 1000

        return "[Audio detected - transcription unavailable]", (
            time.perf_counter() - start_time
        ) * 1000

    def get_audio_features(self, audio_data: bytes) -> dict:
        """Extract features from audio for VAD or other purposes."""
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        audio_float32 = audio_np.astype(np.float32) / 32768.0

        return {
            "energy": float(np.abs(audio_float32).mean()),
            "rms": float(np.sqrt(np.mean(audio_float32**2))),
            "duration": len(audio_data) / (2 * 16000),
        }

    def is_speech(self, audio_data: bytes, threshold: float = 0.02) -> bool:
        """Simple voice activity detection based on energy."""
        features = self.get_audio_features(audio_data)
        return features["energy"] > threshold


class StreamingWhisper:
    """Optimized streaming transcription for real-time applications."""

    def __init__(self, config):
        self.config = config.whisper
        self.stt = WhisperSTT(config)
        self.sample_rate = 16000
        self.buffer_duration_ms = 500
        self.buffer_samples = int(self.sample_rate * self.buffer_duration_ms / 1000)
        self.audio_buffer = np.array([], dtype=np.int16)
        self.is_speaking = False
        self.speech_threshold = 0.02

    def process_audio(self, audio_chunk: bytes) -> Optional[str]:
        """
        Process incoming audio chunk and return transcription if available.

        Returns transcription when speech ends or buffer is full.
        """
        chunk_np = np.frombuffer(audio_chunk, dtype=np.int16)
        self.audio_buffer = np.concatenate([self.audio_buffer, chunk_np])

        features = self.stt.get_audio_features(audio_chunk)
        current_energy = features["energy"]

        if current_energy > self.speech_threshold:
            self.is_speaking = True

        if len(self.audio_buffer) >= self.buffer_samples * 2:
            if self.is_speaking and current_energy < self.speech_threshold * 0.5:
                self.is_speaking = False
                audio_data = self.audio_buffer.tobytes()
                self.audio_buffer = np.array([], dtype=np.int16)

                text, _ = self.stt.transcribe(audio_data, self.sample_rate)
                return text

            if len(self.audio_buffer) >= self.buffer_samples * 4:
                audio_data = self.audio_buffer[: self.buffer_samples * 2].tobytes()
                self.audio_buffer = self.audio_buffer[self.buffer_samples :]

                text, _ = self.stt.transcribe(audio_data, self.sample_rate)
                return text

        return None

    def reset(self):
        """Reset the streaming state."""
        self.audio_buffer = np.array([], dtype=np.int16)
        self.is_speaking = False
