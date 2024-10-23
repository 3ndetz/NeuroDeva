import torch
import os
from pathlib import Path
from typing import Optional, Union
from config.settings import TTSConfig
from utils.exceptions import TTSInitializationError, TTSGenerationError

class SileroTTS:
    def __init__(self, config: TTSConfig = None):
        self.config = config or TTSConfig()
        self.model = None
        self.initialized = False
        
        self.device = torch.device(self.config.device)
        torch.set_num_threads(self.config.num_threads)

    def initialize(self) -> bool:
        if self.initialized:
            return True

        try:
            print('[TTS INIT] Started load TTS model...')
            
            os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)
            
            if not os.path.isfile(self.config.model_path):
                print('[TTS INIT] Downloading model...')
                torch.hub.download_url_to_file(
                    'https://models.silero.ai/models/tts/ru/v3_1_ru.pt',
                    self.config.model_path
                )
            
            self.model = torch.package.PackageImporter(
                self.config.model_path
            ).load_pickle("tts_models", "model")
            self.model.to(self.device)
            
            self.initialized = True
            return True
            
        except Exception as e:
            raise TTSInitializationError(f"Failed to initialize TTS: {str(e)}")

    def generate_audio(
        self,
        text: str,
        speaker: Optional[str] = None
    ) -> torch.Tensor:
        try:
            if not self.initialized:
                self.initialize()
                
            speaker = speaker or self.config.default_speaker
            
            try:
                audio = self.model.apply_tts(
                    ssml_text=text,
                    speaker=speaker,
                    sample_rate=self.config.sample_rate,
                    put_accent=True,
                    put_yo=True
                )
            except Exception as err:
                print(f"[TTS WARN] SSML generation failed: {err}, trying without SSML...")
                audio = self.model.apply_tts(
                    text=text,
                    speaker=speaker,
                    sample_rate=self.config.sample_rate,
                    put_accent=True,
                    put_yo=True
                )
            
            if audio is not None:
                print(f"[TTS SUCCESS] Generated audio of length {len(audio)}")
            return audio
            
        except Exception as e:
            raise TTSGenerationError(f"Failed to generate audio: {str(e)}")
