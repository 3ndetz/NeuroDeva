# -*- coding: utf-8 -*-
import importlib, sys, time
import datetime, random, os
import traceback
import multiprocessing, queue
import contextlib


# torch.set_num_threads(4)# если cuda, отрубаем это НИЧЕГО НЕ ДАЕТ ПОЧТИ. Грузит проц, но **** не дает =( прирост менее 20%
# def FindRepeats(inp):
# pip install transformers sentencepiece accelerate
def calcTime(time):
    return bcolors.OKGREEN + str((datetime.datetime.now() - time).total_seconds()) + bcolors.ENDC


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


def TTS_PROCESS(loading_flag, tts_ctx_InputQueue, tts_ctx_OutputQueue):
    t = datetime.datetime.now()

    thisfolder = os.path.dirname(os.path.realpath(__file__))

    import torch
    autocast_enabled = True

    model_data_type = torch.bfloat16

    if (autocast_enabled):
        print("[LLM FREDT5 PRE-INIT] AMP (autocast) enabled!\n")
        # logging.info("AMP (autocast) enabled!\n")
        autocast = torch.cuda.amp.autocast
    else:
        @contextlib.contextmanager
        def autocast(device=None, dtype=None):
            yield

    def generate(model, input_ids, generation_config):
        with torch.inference_mode():
            with autocast(enabled=True, dtype=model_data_type):
                # with torch.no_grad():
                return model.generate(
                    input_ids,
                    generation_config=generation_config,
                )

    class InferenceTTS:

        thisfolder = os.path.dirname(os.path.realpath(__file__))

        def __init__(self):
            self.tts_model = None
            self.streaming_buffer = None
            self.initialized = False

        def check_initialization(self):
            if not self.initialized:
                # logging.info('TRANSCRIBER init...')
                start_time = time.time()
                self.tts_model, self.streaming_buffer = model_init(self.args, self.cfg)
                # logging.info(f'TRANSCRIBER init ENDED! Time:{round(time.time() - start_time, 2)}s')
                self.initialized = True
            return self.initialized

        def audio_generate(self, data):
            result = model_transcribe()
            return result

    tts_class = InferenceTTS()
    tts_class.check_initialization()
    loading_flag.set()
    print('время запуска ' + calcTime(t))
    while True:
        try:
            inp = tts_ctx_InputQueue.get()
            print('[FREDT5 QUEUE] получена очередь', inp)
            answer = tts_class.audio_generate(data=inp)
            tts_ctx_OutputQueue.put(answer)
        except BaseException as err:
            print('[LM FRED T5 ERR] ОШИБКА ПРОЦЕССА ВО FRED T5: ', err)
            print('[LM FRED T5 ERR] ТЕКСТ ОБ***Й ОШИБКИ', traceback.format_exc())
            print("\n[LM FRED T5 ERR] === КОНЕЦ ОШИБКИ ====")
            time.sleep(1)


if __name__ == "__main__":  # DEBUG NOT WORK
    print('ЗАПУСК ЧО')
    manager = multiprocessing.Manager()
    t = datetime.datetime.now()
    fredCtxQueue = manager.Queue()
    fredCtxQueueOutput = manager.Queue()
    loading_flag = manager.Event()
    LargeFREDProc = multiprocessing.Process(
        target=FRED_PROCESS,
        args=(loading_flag, fredCtxQueue, fredCtxQueueOutput,))  # Thread(target = a, kwargs={'c':True}).start()
    LargeFREDProc.start()
    # FRED_PROCESS(fredCtx)
    print('ЗАПУСК ЧО2')


    def FredT5ChatbotQueue(ninp, context, paramsOverride, environment, lmUsername):
        keywords = {"context": context, "paramsOverride": paramsOverride, "environment": environment,
                    "lmUsername": lmUsername}
        fredCtxQueue.put((ninp, keywords))
        return fredCtxQueueOutput.get()


    loading_flag.wait()
    # print(e.getResult()+'mda')
    print('время запуска ТЕСТА ' + calcTime(t))
    while True:
        inp = input("хехе__бой")  # 'чобабке>>'
        if inp != "" or inp != "ext":
            if inp[0] != "!":
                print('ANS', FredT5ChatbotQueue("Как дела шщзшщз", "", None, {"env": "yt", "diags_count": 0}, inp[1:]))
            else:
                print('ANS', FredT5ChatbotQueue(inp, "", None, {"env": "yt"}, "lexeCho"))
