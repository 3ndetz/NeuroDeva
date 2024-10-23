from typing import Optional, Dict, List
import asyncio
from llm.fred_t5 import FredT5
from tts.silero_tts import SileroTTS
from live2d.vtube_studio import VTubeStudioIntegration
from utils.audio import AudioProcessor
from config.settings import LLMConfig, TTSConfig, Live2DConfig

class AIIntegration:
    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        tts_config: Optional[TTSConfig] = None,
        vtube_config: Optional[Live2DConfig] = None
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
    ) -> Dict:
        context = context or []
        response = await self.llm.generate_response(
            text=text,
            params=llm_params,
            repeat_danger_part=context[-1]["content"] if context else ""
        )
        
        audio = self.tts.generate_audio(response["reply"])
        await self.audio_processor.play_with_lipsync(
            audio,
            self.live2d,
            self.tts.config.sample_rate
        )
        
        return response

    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self.live2d.websocket:
            await self.live2d.websocket.close()

async def main():
    print("\nInitializing AI Integration...")
    
    ai = AIIntegration()
    await ai.initialize()
    
    print("[MAIN] All Systems Ready!")
    context = [] 
    
    try:
        while True:
            text = input("\nYou: ")
            if text.lower() == 'quit':
                break
                
          
            response = await ai.process_input(text, context)
            print(f"\nEva: {response['reply']}")
                
           
            context.extend([
                {"role": "user", "content": text},
                {"role": "assistant", "content": response["reply"]}
            ])
    finally:
      
        await ai.cleanup()
        print("\nSystem shutdown complete")

if __name__ == "__main__":
    asyncio.run(main())
