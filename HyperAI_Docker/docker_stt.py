from multiprocessing import Process
import time
from datetime import datetime


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


import multiprocessing
from multiprocessing.managers import SyncManager


class MyManager(SyncManager):
    pass


# control dict
syncdict = {}

# llm
llm_inputQueue = multiprocessing.Queue()
llm_outputQueue = multiprocessing.Queue()
llm_loading_flag = multiprocessing.Event()

# sttCtx = multiprocessing.N

def get_llm_loading_flag():
    return llm_loading_flag

def get_llm_input_q():
    return llm_inputQueue


def get_llm_output_q():
    return llm_outputQueue


# tts
tts_inputQueue = multiprocessing.Queue()
tts_outputQueue = multiprocessing.Queue()


def get_tts_input_q():
    return tts_inputQueue


def get_tts_output_q():
    return tts_outputQueue


def get_dict():
    return syncdict


def nemo_tts_process(manager):
    debug_iter = 0
    from STT.stream_stt_inf import NemoSpeechTranscriber
    transcriber = NemoSpeechTranscriber()
    transcriber.check_initialization()
    while True:
        inp = manager.tts_input_q().get()
        print("INPUT GOT!", debug_iter)  # ,inp,debug_iter)
        debug_iter += 1
        # inp = inp+" "+str(debug_iter)
        output = transcriber.audio_transcribe(audio_samples=inp["bytes_io"], audio_settings=inp["audio_settings"])
        print("PROCESSING READY, SENDING OUT! ", inp)
        manager.tts_output_q().put(output)
        # print('waiting for action, syncdict %s' % (syncdict))
        # time.sleep(5)

if __name__ == "__main__":

    MyManager.register("syncdict", get_dict)
    MyManager.register("tts_input_q", get_tts_input_q)
    MyManager.register("tts_output_q", get_tts_output_q)

    MyManager.register("llm_input_q", get_llm_input_q)
    MyManager.register("llm_output_q", get_llm_output_q)
    MyManager.register("llm_loading_flag",get_llm_loading_flag)
    from other.HyperAI_DockerSecrets import DockerAuthKey
    manager = MyManager(("0.0.0.0", 6006), authkey=DockerAuthKey)

    print("Started listener manager 0.0.0.0 : 6006")
    manager.start()
    STT_Process = Process(
        target=nemo_tts_process,
        args=(manager,))  # Thread(target = a, kwargs={'c':True}).start()
    STT_Process.start()
    print('Wait for loading LLM...')
    manager.llm_loading_flag().wait()
    print('WAITING COMPLETED!')
    while True:
        time.sleep(1)
        if manager.syncdict().get("stop", False) == True:
            print('TERMINATING DOCKER RECIEVER')
            break
        # ii = input("чонадо")
        # if ii=="":
        #    print('TERMINATING DOCKER RECIEVER')
        #    break
    # transcriber.audio_transcribe(audio_file="test2vloger.wav")

    # raw_input("Press any key to kill server".center(50, "-"))
    STT_Process.terminate()
    STT_Process.join()
    manager.shutdown()
