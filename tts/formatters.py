import html

def format_text_for_tts(text: str, rate: str = "medium", pitch: str = "medium") -> str:
    escaped_text = html.escape(text)
    return f'<speak><prosody rate="{rate}" pitch="{pitch}">{escaped_text}</prosody></speak>'
