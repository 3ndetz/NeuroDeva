from transformers import AutoProcessor, BarkModel
import time
from datetime import datetime
#import torch
print('starting program')
t = datetime.now()


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def calcTime(time):
    return bcolors.OKGREEN + str((datetime.now() - time).total_seconds()) + bcolors.ENDC

device = "cuda:0"
processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark").to(device)

model = model.to_bettertransformer() #pip install optimum
#model.enable_cpu_offload()

sample_rate = model.generation_config.sample_rate

print('loaded models', calcTime(t))
t = datetime.now()
voice_preset = "v2/ru_speaker_6"


# device = "cuda"
def generate_bark_audio(text: str, voice_preset: str):
    t = datetime.now()
    inputs = processor(text, voice_preset=voice_preset).to(device)
    audio = model.generate(**inputs)
    print('generated', calcTime(t))
    audio = audio.cpu().numpy().squeeze()
    return audio


import sounddevice as sd

sd.default.samplerate = sample_rate
sd.default.channels = 2


def output_audio(audio, sample_rate=48000):
    print('started output sr =', sample_rate)
    # sd.default.device = ''
    # sd.default.device = 'CABLE-A Input (VB-Audio Cable A), Windows DirectSound'

    sd.play(audio, sample_rate * 1.05)
    audio_timelength = (len(audio) / sample_rate) + 0.5
    time.sleep(audio_timelength)
    sd.stop()
    print('ended output (length of audio =',audio_timelength,'s)')


def generate_audio(text):
    global sample_rate
    audio = generate_bark_audio(text, voice_preset="v2/ru_speaker_6")
    output_audio(audio, sample_rate)

def split_text_to_sentences(inp: str) -> list[str]:
    punkt = '!?.'
    out = ''
    cnt = 0
    #inp = CutSpaces(ninp).lower()
    for char in inp:
        cnt += 1
        if char == '\n' and cnt<30:
            out += ' '
        elif char == '\n':
            out += char
            cnt = 0
        elif cnt > 100 or (cnt>32 and char in punkt):
            out += char
            out += '\n'
            cnt = 0
        else:
            out += char
    out = out.split('\n')
    output = []
    for line in out:
        output.append(line.strip())
    return output
# t = datetime.now()

while True:
    input_text = input("Audio Bark TTS\n:>")
    texts = split_text_to_sentences(input_text)
    for text in texts:
        generate_audio(text)
    print('audio generated!')
