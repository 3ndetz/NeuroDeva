import numpy as np
import sounddevice as sd
from typing import Optional
import asyncio

class AudioProcessor:
    @staticmethod
    def analyze_amplitude(audio_chunk: np.ndarray) -> float:
        """Get audio amplitude for lip sync."""
        if len(audio_chunk) == 0:
            return 0
        return float(np.abs(audio_chunk).mean())


    
    @staticmethod
    async def play_with_lipsync(
        audio: np.ndarray,
        live2d_integration,
        sample_rate: int = 48000,
        chunk_size: int = 2048
    ) -> None:
        """Play audio while synchronizing lip movements."""
        if audio is None:
            return

        try:
            print("[PLAY] Starting audio with lip sync...")
            sd.play(audio, sample_rate)
            
            await live2d_integration._ensure_connection()
            
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i + chunk_size]
                amplitude = AudioProcessor.analyze_amplitude(chunk)
                mouth_value = min(1.0, amplitude * 5.0)
                
                try:
                    await live2d_integration.set_parameter("MouthOpen", mouth_value)
                except Exception as e:
                    print(f"[PLAY WARN] Lip sync error: {e}")
                    
                await asyncio.sleep(chunk_size / sample_rate)
            
            sd.stop()
            await live2d_integration.set_parameter("MouthOpen", 0.0)
            
        except Exception as e:
            print(f"[PLAY ERR] Failed to play audio with lip sync: {e}")
            sd.stop()
            try:
                await live2d_integration.set_parameter("MouthOpen", 0.0)
            except:
                pass
            
