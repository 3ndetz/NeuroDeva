from enum import Enum, auto

class WebSocketMessageType(Enum):
    AUTH_TOKEN_REQUEST = "AuthenticationTokenRequest"
    AUTH_REQUEST = "AuthenticationRequest"
    PARAMETER_DATA = "InjectParameterDataRequest"

class TTSError(Exception):
    """Base exception for TTS-related errors"""
    pass

class Live2DError(Exception):
    """Base exception for Live2D-related errors"""
    pass
