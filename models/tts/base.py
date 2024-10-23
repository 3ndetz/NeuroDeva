from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

class BaseTTS(ABC):
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the TTS model"""
        pass
    
    @abstractmethod
    def generate_audio(self, text: str, speaker: Optional[str] = None) -> np.ndarray:
        """Generate audio from text"""
        pass
