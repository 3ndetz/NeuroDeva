from typing import Optional, Dict, List
import asyncio
from .llm.fred_t5 import FredT5
from .tts.silero_tts import SileroTTS
from .live2d.vtube_studio import VTubeStudioIntegration
from .utils.audio import AudioProcessor
from .config.settings import LLMConfig, TTSConfig, VTubeStudioConfig

class AIIntegration:
    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        tts_config: Optional[TTSConfig] = None,
        vtube_config: Optional[VTubeStudioConfig] = None
    ):
        self.llm = FredT5(llm_config)
        self.tts = SileroTTS(tts_config)
        self.live2d = VTubeStudioIntegration(vtube_config)
        self.audio_processor = AudioProcessor()

    async def initialize(self) -> None:
        """Initialize all components."""
        self.llm.initialize()
        self.tts.initialize()
        await self.live2d.connect()

    async def process_input(
        self,
        text: str,
        context: Optional[List[Dict]] = None,
        llm_params: Optional[Dict] = None
    ) -> None:
        response = await self.llm.generate_response(text, llm_params, context)
        
        audio = self.tts.generate_audio(response["reply"])
        await self.audio_processor.play_with_lipsync(
            audio,
            self.live2d,
            self.tts.config.sample_rate
        )
        
        return response

    async def cleanup(self) -> None:
        if self.live2d.websocket:
            await self.live2d.websocket.close()
