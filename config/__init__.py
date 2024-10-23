from .settings import AppConfig, TTSConfig, LLMConfig, Live2DConfig
__all__ = [
    'AppConfig',
    'TTSConfig', 
    'LLMConfig',
    'Live2DConfig',
    'get_default_config'
]

def get_default_config() -> AppConfig:
    return AppConfig()
