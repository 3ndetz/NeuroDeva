from typing import Optional, Dict, List
import asyncio
from llm.fred_t5 import FredT5
from tts.silero_tts import SileroTTS
from live2d.vtube_studio import VTubeStudioIntegration
from utils.audio import AudioProcessor
from config.settings import LLMConfig, TTSConfig, Live2DConfig
import time
from datetime import datetime

def format_time(seconds: float) -> str:
    return f"{seconds:.2f}s"

class AIIntegration:
    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        tts_config: Optional[TTSConfig] = None,
        vtube_config: Optional[Live2DConfig] = None
    ):
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Initializing components...")
        start_time = time.time()
        
        self.llm = FredT5(llm_config)
        self.tts = SileroTTS(tts_config)
        self.live2d = VTubeStudioIntegration(vtube_config)
        self.audio_processor = AudioProcessor()
        
        init_time = time.time() - start_time
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Components initialized in {format_time(init_time)}")

    async def initialize(self) -> None:
        """Initialize all components with timing measurements."""
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting full initialization...")
        total_start = time.time()
        
        llm_start = time.time()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Initializing LLM model...")
        self.llm.initialize()
        llm_time = time.time() - llm_start
        print(f"[{datetime.now().strftime('%H:%M:%S')}] LLM initialized in {format_time(llm_time)}")
        
        tts_start = time.time()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Initializing TTS model...")
        self.tts.initialize()
        tts_time = time.time() - tts_start
        print(f"[{datetime.now().strftime('%H:%M:%S')}] TTS initialized in {format_time(tts_time)}")
        
        live2d_start = time.time()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Connecting to VTube Studio...")
        await self.live2d.connect()
        live2d_time = time.time() - live2d_start
        print(f"[{datetime.now().strftime('%H:%M:%S')}] VTube Studio connected in {format_time(live2d_time)}")
        
        total_time = time.time() - total_start
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Full initialization completed in {format_time(total_time)}")
        print(f"├── LLM: {format_time(llm_time)}")
        print(f"├── TTS: {format_time(tts_time)}")
        print(f"└── Live2D: {format_time(live2d_time)}")

    async def process_input(
        self,
        text: str,
        context: Optional[List[Dict]] = None,
        llm_params: Optional[Dict] = None
    ) -> Dict:
        process_start = time.time()
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Processing input: '{text}'")
        
        context = context or []
        
        # Generate LLM response
        llm_start = time.time()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Generating LLM response...")
        response = await self.llm.generate_response(
            text=text,
            params=llm_params,
            repeat_danger_part=context[-1]["content"] if context else ""
        )
        llm_time = time.time() - llm_start
        print(f"[{datetime.now().strftime('%H:%M:%S')}] LLM response generated in {format_time(llm_time)}")
        
        tts_start = time.time()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Generating audio...")
        audio = self.tts.generate_audio(response["reply"])
        tts_time = time.time() - tts_start
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Audio generated in {format_time(tts_time)}")
        
        play_start = time.time()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Playing audio with lip sync...")
        await self.audio_processor.play_with_lipsync(
            audio,
            self.live2d,
            self.tts.config.sample_rate
        )
        play_time = time.time() - play_start
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Audio playback completed in {format_time(play_time)}")
        
        total_time = time.time() - process_start
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Total processing completed in {format_time(total_time)}")
        print(f"├── LLM Generation: {format_time(llm_time)}")
        print(f"├── Audio Generation: {format_time(tts_time)}")
        print(f"└── Audio Playback: {format_time(play_time)}")
        
        return response

    async def cleanup(self) -> None:
        """Cleanup resources."""
        cleanup_start = time.time()
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting cleanup...")
        
        if self.live2d.websocket:
            await self.live2d.websocket.close()
            
        cleanup_time = time.time() - cleanup_start
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Cleanup completed in {format_time(cleanup_time)}")

async def main():
    total_start = time.time()
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting AI Integration...")
    
    ai = AIIntegration()
    await ai.initialize()
    
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] All Systems Ready!")
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
        total_time = time.time() - total_start
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Total session time: {format_time(total_time)}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] System shutdown complete")

if __name__ == "__main__":
    asyncio.run(main())
