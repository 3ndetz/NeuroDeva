import asyncio
import logging
from pathlib import Path
import signal
from typing import Optional

from config.settings import AppConfig
from models.tts.silero import SileroTTS
from models.live2d.integration import Live2DIntegration
from utils.audio import AudioProcessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class VTuberApplication:
    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or AppConfig()
        self.tts = SileroTTS(self.config.tts)
        self.live2d = Live2DIntegration(self.config.live2d)
        self.audio_processor = AudioProcessor()
        self.running = False

    async def initialize(self):
        """Initialize all components"""
        logger.info("Initializing VTuber application...")
        self.tts.initialize()
        await self.live2d.connect()
        self.running = True
        logger.info("Initialization complete")

    async def process_message(self, text: str) -> None:
        """Process a text message through TTS and Live2D"""
        try:
            audio = self.tts.generate_audio(text)
            await self.audio_processor.play_with_lipsync(
                audio, 
                self.live2d,
                self.config.tts.sample_rate
            )
        except Exception as e:
            logger.error(f"Failed to process message: {e}")
            raise

    async def cleanup(self):
        """Cleanup resources"""
        self.running = False
        logger.info("Application shutdown complete")

async def main():
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(shutdown(loop)))

    app = VTuberApplication()
    await app.initialize()

    try:
        while app.running:
            text = input("\nEnter text (or 'quit' to exit): ")
            if text.lower() == 'quit':
                break
            await app.process_message(text)
    finally:
        await app.cleanup()

async def shutdown(loop):
    """Cleanup and shutdown the application"""
    logger.info('Shutting down...')
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    [task.cancel() for task in tasks]
    await asyncio.gather(*tasks, return_exceptions=True)
    loop.stop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
