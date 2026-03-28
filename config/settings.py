import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class WhisperConfig:
    model: str = os.getenv("WHISPER_MODEL", "base")
    device: str = os.getenv("WHISPER_DEVICE", "cpu")
    compute_type: str = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    max_audio_length: int = int(os.getenv("MAX_AUDIO_LENGTH", "30"))


@dataclass
class OllamaConfig:
    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model: str = os.getenv("OLLAMA_MODEL", "llama3")
    temperature: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.7"))
    num_ctx: int = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
    timeout: int = int(os.getenv("LLM_TIMEOUT", "30"))


@dataclass
class TTSConfig:
    voice: str = os.getenv("TTS_VOICE", "en-IN-NeerjaNeural")
    rate: str = os.getenv("TTS_RATE", "+0%")
    pitch: str = os.getenv("TTS_PITCH", "+0Hz")
    volume: str = os.getenv("TTS_VOLUME", "+0%")


@dataclass
class AudioConfig:
    sample_rate: int = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
    channels: int = int(os.getenv("AUDIO_CHANNELS", "1"))
    chunk_size: int = int(os.getenv("AUDIO_CHUNK_SIZE", "1024"))


@dataclass
class WebSocketConfig:
    host: str = os.getenv("WS_HOST", "0.0.0.0")
    port: int = int(os.getenv("WS_PORT", "8765"))
    max_connections: int = int(os.getenv("WS_MAX_CONNECTIONS", "10"))


@dataclass
class VADConfig:
    enabled: bool = os.getenv("VAD_ENABLED", "true").lower() == "true"
    threshold: float = float(os.getenv("VAD_THRESHOLD", "0.5"))
    silence_timeout_ms: int = int(os.getenv("VAD_SILENCE_TIMEOUT_MS", "1000"))


@dataclass
class Config:
    whisper: WhisperConfig = WhisperConfig()
    ollama: OllamaConfig = OllamaConfig()
    tts: TTSConfig = TTSConfig()
    audio: AudioConfig = AudioConfig()
    websocket: WebSocketConfig = WebSocketConfig()
    vad: VADConfig = VADConfig()

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            whisper=WhisperConfig(),
            ollama=OllamaConfig(),
            tts=TTSConfig(),
            audio=AudioConfig(),
            websocket=WebSocketConfig(),
            vad=VADConfig(),
        )


config = Config.from_env()
