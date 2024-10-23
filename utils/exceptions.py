class TTSError(Exception):
    """Base exception for TTS-related errors."""
    pass

class TTSInitializationError(TTSError):
    """Raised when TTS initialization fails."""
    pass

class TTSGenerationError(TTSError):
    """Raised when audio generation fails."""
    pass

class Live2DError(Exception):
    """Base exception for Live2D-related errors."""
    pass

class Live2DConnectionError(Live2DError):
    """Raised when Live2D connection fails."""
    pass
