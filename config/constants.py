from enum import Enum

class ModelType(Enum):
    INSTRUCT = "instruct"
    DIALOG = "dialog"

class StopReason(Enum):
    SYMBOL = "symbol"
    WORD = "word"
    REPEAT = "repeat"
