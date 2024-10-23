def format_text_for_tts(text: str, rate: str = "medium", pitch: str = "medium") -> str:
    return f'<speak><prosody rate="{rate}" pitch="{pitch}">{text}</prosody></speak>'
