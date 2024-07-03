# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script can be used to simulate cache-aware streaming for ASR models. The ASR model to be used with this script need to get trained in streaming mode. Currently only Conformer models supports this streaming mode.
You may find examples of streaming models under 'NeMo/example/asr/conf/conformer/streaming/'.

It works both on a manifest of audio files or a single audio file. It can perform streaming for a single stream (audio) or perform the evalution in multi-stream model (batch_size>1).
The manifest file must conform to standard ASR definition - containing `audio_filepath` and `text` as the ground truth.

# Usage

## To evaluate a model in cache-aware streaming mode on a single audio file:

python speech_to_text_streaming_infer.py \
    --asr_model=asr_model.nemo \
    --audio_file=audio_file.wav \
    --compare_vs_offline \
    --use_amp \
    --debug_mode

## To evaluate a model in cache-aware streaming mode on a manifest file:

python speech_to_text_streaming_infer.py \
    --asr_model=asr_model.nemo \
    --manifest_file=manifest_file.json \
    --batch_size=16 \
    --compare_vs_offline \
    --use_amp \
    --debug_mode

You may drop the '--debug_mode' and '--compare_vs_offline' to speedup the streaming evaluation.
If compare_vs_offline is not used, then significantly larger batch_size can be used.
Setting `--pad_and_drop_preencoded` would perform the caching for all steps including the first step.
It may result in slightly different outputs from the sub-sampling module compared to offline mode for some techniques like striding and sw_striding.
Enabling it would make it easier to export the model to ONNX.

# Hybrid ASR models
For Hybrid ASR models which have two decoders, you may select the decoder by --set_decoder DECODER_TYPE, where DECODER_TYPE can be "ctc" or "rnnt".
If decoder is not set, then the default decoder would be used which is the RNNT decoder for Hybrid ASR models.


## Evaluate a model trained with full context for offline mode

You may try the cache-aware streaming with a model trained with full context in offline mode.
But the accuracy would not be very good with small chunks as there is inconsistency between how the model is trained and how the streaming inference is done.
The accuracy of the model on the borders of chunks would not be very good.

To use a model trained with full context, you need to pass the chunk_size and shift_size arguments.
If shift_size is not passed, chunk_size would be use as the shift_size too.
Also argument online_normalization should be enabled to simulate a realistic streaming.
The following command would simulate cache-aware streaming on a pretrained model from NGC with chunk_size of 100, shift_size of 50 and 2 left chunks as left context.
The chunk_size of 100 would be 100*4*10=4000ms for a model with 4x downsampling and 10ms shift in feature extraction.

python speech_to_text_streaming_infer.py \
    --asr_model=stt_en_conformer_ctc_large \
    --chunk_size=100 \
    --shift_size=50 \
    --left_chunks=2 \
    --online_normalization \
    --manifest_file=manifest_file.json \
    --batch_size=16 \
    --compare_vs_offline \
    --use_amp \
    --debug_mode

"""


import contextlib
import io
import json
import os
import time
from argparse import ArgumentParser
from dataclasses import dataclass
import numpy as np
import soundfile
import torch
from omegaconf import open_dict

from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
from nemo.collections.asr.parts.utils.transcribe_utils import setup_model
from nemo.utils import logging


def extract_transcriptions(hyps):
    """
        The transcribed_texts returned by CTC and RNNT models are different.
        This method would extract and return the text section of the hypothesis.
    """
    if isinstance(hyps[0], Hypothesis):
        transcriptions = []
        for hyp in hyps:
            transcriptions.append(hyp.text)
    else:
        transcriptions = hyps
    return transcriptions


def calc_drop_extra_pre_encoded(asr_model, step_num, pad_and_drop_preencoded):
    # for the first step there is no need to drop any tokens after the downsampling as no caching is being used
    if step_num == 0 and not pad_and_drop_preencoded:
        return 0
    else:
        return asr_model.encoder.streaming_cfg.drop_extra_pre_encoded


def perform_streaming(
    asr_model, streaming_buffer, compare_vs_offline=False, debug_mode=False, pad_and_drop_preencoded=False, autocast_enabled=True
):
    batch_size = len(streaming_buffer.streams_length)
    if (autocast_enabled):
        logging.info("AMP (autocast) enabled!\n")
        autocast = torch.cuda.amp.autocast
    else:
        @contextlib.contextmanager
        def autocast():
            yield
    if compare_vs_offline:
        # would pass the whole audio at once through the model like offline mode in order to compare the results with the stremaing mode
        # the output of the model in the offline and streaming mode should be exactly the same

        with torch.inference_mode():
            with autocast():
                processed_signal, processed_signal_length = streaming_buffer.get_all_audios()
                with torch.no_grad():
                    (
                        pred_out_offline,
                        transcribed_texts,
                        cache_last_channel_next,
                        cache_last_time_next,
                        cache_last_channel_len,
                        best_hyp,
                    ) = asr_model.conformer_stream_step(
                        processed_signal=processed_signal,
                        processed_signal_length=processed_signal_length,
                        return_transcription=True,
                    )
        final_offline_tran = extract_transcriptions(transcribed_texts)
        logging.info(f" Final offline transcriptions:   {final_offline_tran}")
    else:
        final_offline_tran = None

    cache_last_channel, cache_last_time, cache_last_channel_len = asr_model.encoder.get_initial_cache_state(
        batch_size=batch_size
    )

    previous_hypotheses = None
    streaming_buffer_iter = iter(streaming_buffer)
    pred_out_stream = None
    for step_num, (chunk_audio, chunk_lengths) in enumerate(streaming_buffer_iter):
        with torch.inference_mode():
            with autocast():
                # keep_all_outputs needs to be True for the last step of streaming when model is trained with att_context_style=regular
                # otherwise the last outputs would get dropped

                with torch.no_grad():
                    (
                        pred_out_stream,
                        transcribed_texts,
                        cache_last_channel,
                        cache_last_time,
                        cache_last_channel_len,
                        previous_hypotheses,
                    ) = asr_model.conformer_stream_step(
                        processed_signal=chunk_audio,
                        processed_signal_length=chunk_lengths,
                        cache_last_channel=cache_last_channel,
                        cache_last_time=cache_last_time,
                        cache_last_channel_len=cache_last_channel_len,
                        keep_all_outputs=streaming_buffer.is_buffer_empty(),
                        previous_hypotheses=previous_hypotheses,
                        previous_pred_out=pred_out_stream,
                        drop_extra_pre_encoded=calc_drop_extra_pre_encoded(
                            asr_model, step_num, pad_and_drop_preencoded
                        ),
                        return_transcription=True,
                    )

        if debug_mode:
            logging.info(f"Streaming transcriptions: {extract_transcriptions(transcribed_texts)}")

    final_streaming_tran = extract_transcriptions(transcribed_texts)
    logging.info(f"Final streaming transcriptions: {final_streaming_tran}")

    if compare_vs_offline:
        # calculates and report the differences between the predictions of the model in offline mode vs streaming mode
        # Normally they should be exactly the same predictions for streaming models
        pred_out_stream_cat = torch.cat(pred_out_stream)
        pred_out_offline_cat = torch.cat(pred_out_offline)
        if pred_out_stream_cat.size() == pred_out_offline_cat.size():
            diff_num = torch.sum(pred_out_stream_cat != pred_out_offline_cat).cpu().numpy()
            logging.info(
                f"Found {diff_num} differences in the outputs of the model in streaming mode vs offline mode."
            )
        else:
            logging.info(
                f"The shape of the outputs of the model in streaming mode ({pred_out_stream_cat.size()}) is different from offline mode ({pred_out_offline_cat.size()})."
            )

    return final_streaming_tran, final_offline_tran


from struct import pack, unpack
import librosa
import soundfile as sf
def recover_wav(f):  # на ВХОД (Union object io.BytesIO) ИЛИ (BinaryIO (это open(file,'rb+') ) выдаёт то же что и на ВХОД
    wav_header = "4si4s4sihhiihh4si"
    data = list(unpack(wav_header, f.read(44)))
    assert data[0] == b'RIFF'
    assert data[2] == b'WAVE'
    assert data[3] == b'fmt '
    assert data[4] == 16
    assert data[-2] == b'data'
    assert data[1] == data[-1] + 36
    f.seek(0, 2)
    filesize = f.tell()
    datasize = filesize - 44
    data[-1] = datasize
    data[1] = datasize + 36
    f.seek(0)
    f.write(pack(wav_header, *data))
    return f
def prepare_input_audio(f,orig_sr=48000,orig_channels=2):
    recover_wav(f)
    y, sr = sf.read(f, format='RAW', samplerate=orig_sr, channels=orig_channels, subtype='PCM_16',
                    dtype='float32')  # ,subtype='FLOAT' ,dtype='float32',dtype='int16'
    f.close()
    y = y.transpose()
    if orig_channels>1:
        y = librosa.core.to_mono(y).T
    y = librosa.core.resample(y, orig_sr=sr, target_sr=16000).T
    return y
def prepare_input_audiofile(audio_file_path,orig_sr=48000,orig_channels=2):
    fileOpen = open(audio_file_path, 'rb+')
    filee = fileOpen.read()
    f = io.BytesIO(filee)
    fileOpen.close()
    return prepare_input_audio(f,orig_sr=orig_sr,orig_channels=orig_channels)

def save_outstream_to_file(audio,out_audio_path='my_24bit_file.wav'):
    sf.write(out_audio_path, audio, 16000)
# SAVING FILE
#sf.write('my_24bit_file.wav', y, 16000)

voice_recog_model_name = "stt_ru_fastconformer_hybrid_large_pc"
@dataclass
class StreamingRecogArgsConfig:
    pad_and_drop_preencoded = False  # would perform the caching for all steps including the first step.
    compare_vs_offline = False #You may drop the '--debug_mode' and '--compare_vs_offline' to speedup the streaming evaluation.
    # If compare_vs_offline is not used, then significantly larger batch_size can be used.
    use_amp = True
    device = "cuda" #cuda or cpu
    chunk_size = 100  # The chunk_size of 100 would be 100*4*10=4000ms for a model with 4x downsampling and 10ms shift in feature extraction.
    batch_size = 32
    shift_size = -1  # The shift_size to be used for models trained with full context and offline models
    left_chunks = 2  # The number of left chunks to be used as left context via caching for offline models
    debug_mode = False
    autocast_enabled = True
    
    
thisfolder = os.path.dirname(os.path.realpath(__file__))
    
    
@dataclass
class TranscriptionConfig:
    model_path = f"{thisfolder}/{voice_recog_model_name}/{voice_recog_model_name}.nemo"  # Path to a .nemo file
    pretrained_name = voice_recog_model_name  # Name of a pretrained model
    cuda = -1

def model_init(args,cfg):
    logging.info(f"Using local ASR model from {cfg.model_path}")
    asr_model, model_name = setup_model(cfg, map_location=torch.device(args.device))

    #logging.info(asr_model.encoder.streaming_cfg)
    if (
        args.use_amp
        and torch.cuda.is_available()
        and hasattr(torch.cuda, 'amp')
        and hasattr(torch.cuda.amp, 'autocast')
    ):
        logging.info(f"AMP (AUTOCAST) set to {str(args.autocast_enabled)} (in config)!\n")
    else:
        args.autocast_enabled = False
    # configure the decoding config
    decoding_cfg = asr_model.cfg.decoding
    with open_dict(decoding_cfg):
        decoding_cfg.strategy = "greedy"
        decoding_cfg.preserve_alignments = False
        if hasattr(asr_model, 'joint'):  # if an RNNT model
            decoding_cfg.greedy.max_symbols = 10
            decoding_cfg.fused_batch_size = -1
        asr_model.change_decoding_strategy(decoding_cfg)

    asr_model = asr_model.to(args.device)
    asr_model.eval()

    # chunk_size is set automatically for models trained for streaming. For models trained for offline mode with full context, we need to pass the chunk_size explicitly.
    if args.chunk_size > 0:
        if args.shift_size < 0:
            shift_size = args.chunk_size
        else:
            shift_size = args.shift_size
        asr_model.encoder.setup_streaming_params(
            chunk_size=args.chunk_size, left_chunks=args.left_chunks, shift_size=shift_size
        )

    streaming_buffer = CacheAwareStreamingAudioBuffer(
        model=asr_model,
        online_normalization=False,
        pad_and_drop_preencoded=args.pad_and_drop_preencoded,
    )
    return asr_model, streaming_buffer
def model_transcribe( asr_model, streaming_buffer,args, audio_samples=None, audio_file=None, audio_settings = None):
    start_time = time.time()
    logging.info('PERFORMING TRANSCRIBE STREAMING STARTED!')
    if audio_settings is None:
        audio_settings = {"sr":48000,"channels":2}
    if audio_samples is None and audio_file is None:
        print('ВНИМАНИЕ!!!!! АРГУМЕНТОВ НЕТ!!! ЗАКРЫТИЕ ТРАНСКРАЙБА')
        return "ВЫ ДАЛИ МНЕ ПУСТОТУ!"
    if audio_file is not None:
        audio_samples = prepare_input_audiofile(audio_file, audio_settings["sr"], audio_settings["channels"])
    else:
        audio_samples = prepare_input_audio(audio_samples, audio_settings["sr"], audio_settings["channels"])
    final_streaming_tran = "НИЧЕГО НЕ РАСПОЗНАНО"
    if audio_samples is not None:
        streaming_buffer.reset_buffer()
        processed_signal, processed_signal_length, stream_id = streaming_buffer.append_audio(audio_samples,
                                                                                             stream_id=-1)
        final_streaming_tran_mas, _ = perform_streaming(
            asr_model=asr_model,
            streaming_buffer=streaming_buffer,
            compare_vs_offline=args.compare_vs_offline,
            pad_and_drop_preencoded=args.pad_and_drop_preencoded,
        )
        if len(final_streaming_tran_mas)>0:
            final_streaming_tran = "".join(final_streaming_tran_mas)

    end_time = time.time()
    logging.info(f"The whole streaming process took: {round(end_time - start_time, 2)}s")
    return final_streaming_tran

class NemoSpeechTranscriber():
    """TRANSCRIBER MAIN CLASS"""

    def __init__(self):
        self.args = StreamingRecogArgsConfig()
        self.cfg = TranscriptionConfig()
        self.asr_model = None
        self.streaming_buffer = None
        self.initialized = False

    def check_initialization(self):
        if not self.initialized:
            logging.info('TRANSCRIBER init...')
            start_time = time.time()
            self.asr_model, self.streaming_buffer = model_init(self.args, self.cfg)
            logging.info(f'TRANSCRIBER init ENDED! Time:{round(time.time() - start_time, 2)}s')
            self.initialized = True
        return self.initialized

    def audio_transcribe(self,audio_samples=None,audio_file=None,audio_settings=None):

        result = model_transcribe(audio_samples=audio_samples, audio_file=audio_file, audio_settings=audio_settings,
                                  asr_model=self.asr_model, streaming_buffer=self.streaming_buffer, args=self.args)
        return result
def main_debug():
    args = StreamingRecogArgsConfig()
    cfg = TranscriptionConfig()

    asr_model, streaming_buffer = model_init(args, cfg)
    start_time = time.time()
    def debugTestAudiofiles():
        audiofile_list = ["test.wav"]
        settings_dict = {"sr": 48000, "channels": 1}
        #audiofile_list = ["test.wav","test1.wav","test2vloger.wav","test3.wav","test4old.wav"]
        for audiofile in audiofile_list:

            result = model_transcribe(audio_samples=None, audio_file=audiofile, audio_settings=settings_dict,
                                      asr_model=asr_model, streaming_buffer=streaming_buffer, args=args)
            print('РЕЗУЛЬТАТ 1 ОКОНЧЕН! ТРАНСКРИПЦИЯ:',result)

    debugTestAudiofiles()
    end_time = time.time()
    print('ВСЁ ЗАВЕРШЕНО! РЕЗУЛЬТАТ:',{round(end_time - start_time, 2)},'s')

def file_to_bytes_io(filename):
    fileOpen = open(filename, 'rb+')
    filee = fileOpen.read()
    samples_file = io.BytesIO(filee)
    fileOpen.close()
    return samples_file
if __name__ == '__main__':
    transcriber = NemoSpeechTranscriber()
    transcriber.check_initialization()
    #print('==RESULT 1',transcriber.audio_transcribe(audio_file="test.wav",audio_settings={"sr":48000,"channels":1}))
    #samples = file_to_bytes_io("test1.wav")
    #print('==RESULT 2', transcriber.audio_transcribe(audio_samples=samples))
    #print('==RESULT 3', transcriber.audio_transcribe(audio_file="test2vloger.wav"))
    #print('==RESULT 4', transcriber.audio_transcribe(audio_file="test3.wav", audio_settings={"sr":44100,"channels":2}))
    #main_debug()
