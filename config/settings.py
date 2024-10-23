from dataclasses import dataclass
from pathlib import Path
import os

@dataclass
class TTSConfig:
    sample_rate: int = 48000
    base_path: Path = Path(__file__).parent.parent / "models" / "tts" / "variants" / "Silero"
    model_file: str = "silero_tts.pt"
    model_url: str = "https://models.silero.ai/models/tts/ru/v3_1_ru.pt"
    default_speaker: str = "xenia"
    speakers: list = ('aidar', 'baya', 'kseniya', 'xenia', 'eugene', 'random')
    device: str = "cpu"
    num_threads: int = 4

@dataclass
class Live2DConfig:
    api_name: str = "VTubeStudioPublicAPI"
    api_version: str = "1.0"
    websocket_url: str = "ws://127.0.0.1:8001"
    plugin_name: str = "TTS Integration"
    plugin_developer: str = "test"
    request_id: str = "test"
    token_file: str = "token.json"

@dataclass
class AppConfig:
    tts: TTSConfig = TTSConfig()
    live2d: Live2DConfig = Live2DConfig()
