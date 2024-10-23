from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
import os

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    model_paths: Dict[str, Dict[str, str]] = field(default_factory=dict)
    base_path: Path = Path(__file__).parent.parent
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_data_type: torch.dtype = torch.bfloat16
    max_model_memory: int = 18
    autocast_enabled: bool = True
    
    def __post_init__(self):
        if not self.model_paths:
            self.model_paths = {
                'instruct': {
                    'id': 'SiberiaSoft/SiberianFredT5-instructor',
                    'localPath': '/variants/SiberianInstructor'
                },
                'dialog': {
                    'id': 'SiberiaSoft/SiberianPersonaFred-2',
                    'localPath': '/variants/SiberianPersonaFred'
                }
            }

@dataclass
class TTSConfig:
    sample_rate: int = 48000
    base_path: Path = Path(__file__).parent.parent
    model_dir: Path = field(init=False)
    model_path: Path = field(init=False)
    model_file: str = "silero_tts.pt"
    model_url: str = "https://models.silero.ai/models/tts/ru/v3_1_ru.pt"
    default_speaker: str = "xenia"
    speakers: Tuple[str, ...] = ('aidar', 'baya', 'kseniya', 'xenia', 'eugene', 'random')
    device: str = "cpu"
    num_threads: int = 4

    def __post_init__(self):
        self.model_dir = self.base_path / "models" / "tts" / "variants" / "Silero"
        self.model_path = self.model_dir / self.model_file

@dataclass
class Live2DConfig:
    api_name: str = "VTubeStudioPublicAPI"
    api_version: str = "1.0"
    websocket_url: str = "ws://127.0.0.1:8001"
    plugin_name: str = "TTS Integration"
    plugin_developer: str = "test"
    request_id: str = "test"
    token_file: Path = Path("token.json")

@dataclass
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    live2d: Live2DConfig = field(default_factory=Live2DConfig)
