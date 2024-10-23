import torch
from torch import package
import os
import logging
from typing import Optional, Tuple
import numpy as np
from pathlib import Path
from datetime import datetime

from .base import BaseTTS
from ...config.settings import TTSConfig
from ...config.constants import TTSError

logger = logging.getLogger(__name__)

def format_text_for_tts(text: str, rate: str = "medium", pitch: str = "medium") -> str:
    """Format text with proper SSML tags"""
    return f'<speak><prosody rate="{rate}" pitch="{pitch}">{text}</prosody></speak>'

class SileroTTS(BaseTTS):
    def __init__(self, config: TTSConfig):
        self.config = config
        self.model = None
        self.device = torch.device(config.device)
        self.initialized = False
   
        torch.set_num_threads(config.num_threads)

    def initialize(self) -> bool:
        """Initialize the Silero TTS model"""
        if not self.initialized:
            logger.info('Started loading TTS model...')
            start_time = datetime.now()
            
            model_path = self.config.base_path / self.config.model_file
            os.makedirs(model_path.parent, exist_ok=True)
            
            if not model_path.is_file():
                logger.info('Downloading model...')
                torch.hub.download_url_to_file(
                    self.config.model_url,
                    str(model_path)
                )
            
            self.model = torch.package.PackageImporter(str(model_path)).load_pickle("tts_models", "model")
            self.model.to(self.device)
            
            self.initialized = True
            logger.info(f'Model loaded in {(datetime.now() - start_time).total_seconds():.2f}s')
            
        return self.initialized

    def generate_audio(self, text: str, speaker: Optional[str] = None) -> np.ndarray:
        """Generate audio from text using Silero TTS"""
        if not self.initialized:
            raise TTSError("TTS model not initialized")
            
        speaker = speaker or self.config.default_speaker
            
        try:
            ssml_text = format_text_for_tts(text)
            logger.debug(f"Using SSML: {ssml_text}")
            
            try:
                audio = self.model.apply_tts(
                    ssml_text=ssml_text,
                    speaker=speaker,
                    sample_rate=self.config.sample_rate,
                    put_accent=True,
                    put_yo=True
                )
            except Exception as err:
                logger.warning(f"SSML generation failed: {err}, trying without SSML...")
                audio = self.model.apply_tts(
                    text=text,
                    speaker=speaker,
                    sample_rate=self.config.sample_rate,
                    put_accent=True,
                    put_yo=True
                )
            
            if audio is not None:
                logger.info(f"Generated audio of length {len(audio)}")
            return audio
            
        except Exception as err:
            logger.error(f"Audio generation failed: {err}")
            logger.error(f"Text was: {text}")
            raise TTSError(f"Failed to generate audio: {err}")
