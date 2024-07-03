# -*- coding: utf-8 -*-
import importlib, sys, time
import datetime, random, os
import traceback
import multiprocessing, queue
import contextlib


# torch.set_num_threads(4)# если cuda, отрубаем это НИЧЕГО НЕ ДАЕТ ПОЧТИ. Грузит проц, но ***** не дает =( прирост менее 20%
# def FindRepeats(inp):
# pip install transformers sentencepiece accelerate
def calcTime(time):
    return bcolors.OKGREEN + str((datetime.datetime.now() - time).total_seconds()) + bcolors.ENDC


def GetCmd(ninp, tip="emo"):
    inp = ninp
    result = ""
    brackets = ['[', ']']
    if tip == "emo":
        cmdlist = "агрессия, скука, усталость, интерес, смущение, счастье, веселье, страх".split(', ')
        brackets = ['[', ']']
    elif tip == "cmd":
        cmdlist = "бан, издевайся, попрыгай, смейся, кричи, убегай".split(', ')
        brackets = ['<', '>']
    lbracketIdx = inp.find(brackets[0]) + 1
    rbracketIdx = inp.rfind(brackets[1]) + 1
    emotionContainer = inp[lbracketIdx:rbracketIdx]

    if (lbracketIdx != 0) and (
            rbracketIdx != 0):  # проверка нашли ли мы обе скобки. 0 т.к. мы выше мы прибавили к индексам скобок по 1
        for command in cmdlist:
            if (emotionContainer.find(command) != -1):
                result = command
                break
        inp = inp[:lbracketIdx - 1] + inp[rbracketIdx:]
    return {"cmd": result, "cut": inp}


def CutSpaces(inp):
    result = ""
    cnt = 0
    for letter in inp:
        if (letter == ' '):
            cnt += 1
            if (cnt > 1):
                pass
                # cnt=0
            else:
                result += letter
        else:
            result += letter
            cnt = 0

    # print('!!! БЕЗ ПРОБЕЛА !!!',result)
    return result.strip()


def findRepeatingTokens(sample: list, check: list):
    while True:
        if len(check) > 10 and len(sample) > 10:
            for k, token in enumerate(check):
                if k >= 9:
                    checkWord = [check[k - 9], check[k - 8], check[k - 7], check[k - 6], check[k - 5], check[k - 4],
                                 check[k - 3], check[k - 2], check[k - 1], check[k]]
                    for i, sampleToken in enumerate(sample):
                        if i >= 9:
                            sampleWord = [sample[i - 9], sample[i - 8], sample[i - 7], sample[i - 6], sample[i - 5],
                                          sample[i - 4], sample[i - 3], sample[i - 2], sample[i - 1], sample[i]]
                            if checkWord == sampleWord:
                                return True
            return False
        else:
            return False


# sample = [1,3,5,2,3,7,2,5]
# gen = [3,1,3,5,4,3,5,3,2,0,1,3,5,4,3,3,5,2,3,7,3,5]
# print(sample,gen,foundRepeatingTokens(sample,gen))
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


def FRED_PROCESS(loading_flag, fredCtxQueue, fredCtxQueueOutput, repeatingDict=None):
    if repeatingDict is None:
        repeatingDict = {}
    t = datetime.datetime.now()
    thisfolder = os.path.dirname(os.path.realpath(__file__))
    sys.path.insert(0, thisfolder)

    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig, StoppingCriteria, \
        StoppingCriteriaList, AutoConfig  # from transformers import GPT2Tokenizer, T5ForConditionalGeneration
    # from auto_gptq import AutoGPTQForCausalLM
    # import psutil
    # os_used = sys.platform
    # process = psutil.Process(os.getpid())  # Set highest priority for the python script for the CPU
    # if os_used == "win32":  # Windows (either 32-bit or 64-bit)
    #    process.nice(psutil.HIGH_PRIORITY_CLASS)#REALTIME_PRIORITY_CLASS)
    #    print('[FT5] УСТАНОВЛЕН ВЫСОКИЙ ПРИОРИТЕТ ПРОЦЕССА PID =',os.getpid())
    # elif os_used == "linux":  # linux
    #    process.nice(psutil.IOPRIO_HIGH)
    # else:  # MAC OS X or other
    #    process.nice(20)

    import torch
    import gc
    autocast_enabled = True

    # model_data_type = torch.bfloat16
    cuda_enabled = torch.cuda.is_available()
    if cuda_enabled:
        max_model_memory = int(torch.cuda.mem_get_info()[0] / 1024 ** 3) - 2  # вся память - 2, измеряется в gb
    else:
        max_model_memory = 18  # 18

    model_data_type = torch.bfloat16  # torch.bfloat16 для Fred t5
    torch_device = torch.device("cuda" if cuda_enabled else "cpu")
    print(
        f'[TORCH INIT] DEVICE={torch_device}; MODEL DTYPE={str(model_data_type)}; cuda bf16 support={str(torch.cuda.is_bf16_supported())}')
    # torch.set_default_dtype(model_data_type)
    # torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
    # torch.set_default_tensor_type(torch.cuda.HalfTensor) #быстрее в 3 раза загрузка (30 сек), но медленнее в 1.5 раза инференс. (4.8 сек против 3). Также для включения надо убрать torch dtype при загрузке (pretrained)
    # torch.set_default_tensor_type(torch.cuda.HalfTensor)
    # torch.set_default_device(torch_device)

    if (autocast_enabled):
        print("[LLM FREDT5 PRE-INIT] AMP (autocast) enabled!\n")
        # logging.info("AMP (autocast) enabled!\n")
        autocast = torch.cuda.amp.autocast
    else:
        @contextlib.contextmanager
        def autocast(device=None, dtype=None):
            yield

    def generate(model, input_ids, generation_config, stop_criteria):
        print('[LLM DEBUG MEM] model BEFORE GENERATION, cuda MEM ALLOC =', torch.cuda.memory_allocated())
        # 3490177024 5.2 GB
        if torch.cuda.memory_allocated() > 4000000000:
            torch.cuda.empty_cache()
            print(
                '[LLM DEBUG MEM] MAX MEMORY EXCEEED! CLEASRING CUDA CACHE... \n[LLM DEBUG MEM] NOW (AFTER CLEAR) cuda MEM ALLOC =',
                torch.cuda.memory_allocated())
            # стандарть 3490177536 3490176000 3490177024
            # gc.collect()

        with torch.inference_mode():
            with autocast(enabled=True, dtype=model_data_type):
                # with torch.no_grad():
                with torch.no_grad():
                    result = model.generate(
                        input_ids,
                        generation_config=generation_config,
                        stopping_criteria=StoppingCriteriaList([stop_criteria])
                    )
                    print('[LLM DEBUG MEM] model AFTER GENERATION, cuda MEM ALLOC =', torch.cuda.memory_allocated())
                    return result

    class T5:
        thisfolder = os.path.dirname(os.path.realpath(__file__))
        tokenizer = []
        model = []
        nick = "obama421"
        username = "Пользователь"
        e = []
        ModelLoaded = False
        lastTokensUsed = 0
        device = "cuda"
        context = []

        ModelLocalPaths = {
            'instruct': {'id': 'SiberiaSoft/SiberianFredT5-instructor', 'localPath': '/variants/SiberianInstructor'},
            'dialog': {'id': 'SiberiaSoft/SiberianPersonaFred-2', 'localPath': '/variants/SiberianPersonaFred'},

            }

        # ModelLocalPath = '/variants/FP16Siberian_FRED'
        # ModelID = 'SiberiaSoft/SiberianFRED-T5-XL'

        # '/variants/FP16ruGPT35_8BIT' 'Gaivoronsky/ruGPT-3.5-13B-8bit' ruGPT 3.5. Тупая + необученная + **г знает как её токенами нормально заставить выводить
        # '/variants/FP16Siberian_FRED' 'SiberiaSoft/SiberianFRED-T5-XL' ТОПЧИК V2! Но токсичновата и тупа
        # '/variants/FP16Trained1den4ik' 'Den4ikAI/FRED-T5-XL_instructor_chitchat' ТОПЧИК! Но токсичновата и тупа
        # '/variants/SiberianPersonaFred' '/variants/SiberianPersonaFred' ПЛОХО ГЕНЕРИТ НИКИ. Не токсична но в диалоге лучше.

        def TokenizerDebugPrint(self, inp, debugPrefix='Debug Input >> '):
            tokens = inp
            debugOutputs = []
            for t in tokens:
                debugOutputs.append(t)
                debugOutputs.append(96)  # token '|' = 96, [=65, .=18
            print(debugPrefix, '\n<|||>\n', self.tokenizer.decode(debugOutputs), '\n<|||>')

        def CheckModel(self, forceLoad=False):
            if (not self.ModelLoaded) or forceLoad:
                t = datetime.datetime.now()
                if forceLoad:
                    print('[FT5 DEBUG ЗАГРУЗКА FORCE LOAD!!!!!]')
                print(f'\n=== Загрузка БОЛЬШОЙ модели FT5 на {str(torch_device)} ** ===\n')

                ###original model###
                # self.tokenizer = GPT2Tokenizer.from_pretrained(thisfolder+'/variants/original',eos_token='</s>')
                # self.model = T5ForConditionalGeneration.from_pretrained(self.thisfolder+'/variants/original')
                """ #ВТОРАЯ МОДЕЛЬ (НЕОБЯЗАТЕЛЬНАЯ!)
                self.dialog_tokenizer = AutoTokenizer.from_pretrained(self.ModelLocalPaths["dialog"]["id"],
                                                               cache_dir=self.thisfolder + self.ModelLocalPaths["dialog"]["localPath"])

                self.dialog_model = AutoModelForSeq2SeqLM.from_pretrained(self.ModelLocalPaths["dialog"]["id"],
                                                                   cache_dir=self.thisfolder + self.ModelLocalPaths["dialog"]["localPath"],
                                                                   max_memory={0: f'{max_model_memory//2}GB'},
                                                                   torch_dtype=model_data_type,
                                                                          device_map={'': 0}
                                                                   # torch.float16 или bfloat16
                                                                   )
                self.dialog_model.eval()
                """
                print(f'\n=== Загрузка DIALOG LLM в GPU+eval УСПЕШНО ЗАВЕРШЕНА ({calcTime(t)}c) ===\n')
                # self.instruct_tokenizer =
                self.tokenizer = AutoTokenizer.from_pretrained(self.ModelLocalPaths["instruct"]["id"],
                                                               cache_dir=self.thisfolder +
                                                                         self.ModelLocalPaths["instruct"][
                                                                             "localPath"])
                # self.instruct_model =
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.ModelLocalPaths["instruct"]["id"],
                                                                   cache_dir=self.thisfolder +
                                                                             self.ModelLocalPaths["instruct"][
                                                                                 "localPath"],
                                                                   max_memory={0: f'{max_model_memory // 2}GB'},
                                                                   torch_dtype=model_data_type,
                                                                   device_map={'': 0}
                                                                   # torch.float16 или bfloat16
                                                                   )  # .to(torch_device)
                self.model.eval()
                # debug todo
                # self.dialog_model = self.instruct_model
                # self.dialog_tokenizer = self.instruct_tokenizer
                print(f'\n=== Загрузка INSTRUCT LLM в GPU+eval УСПЕШНО ЗАВЕРШЕНА ({calcTime(t)}c) ===\n')
                # self.model = AutoGPTQForCausalLM.from_quantized(self.ModelID,
                #                                                cache_dir=self.thisfolder + self.ModelLocalPath,
                #                                                max_memory={0: f'{max_model_memory}GB'},
                #                                                torch_dtype=model_data_type,
                #                                                use_triton=False,
                #                                                device=torch_device
                #                                                # torch.float16 или bfloat16
                #                                                ).to(torch_device)  # .cuda().to(torch.bfloat16)#
                # print(f'\n=== Загрузка в CPU FT5** УСПЕШНО ЗАВЕРШЕНА ({calcTime(t)}c) ===\n')
                # self.model.eval()
                # self.model = self.instruct_model
                # self.tokenizer = self.instruct_tokenizer
                self.ModelLoaded = True
                print(f'\n=== Загрузка ПОЛНАЯ LLM в GPU+eval УСПЕШНО ЗАВЕРШЕНА ({calcTime(t)}c) ===\n')

        def FredT5(self, ninp, p=None, repeatDangerPart='',
                   returnStopReason=False):  # [2.0,2.0,50,100]
            if p is None:
                p = {
                    "do_sample": True,
                    "top_p": 0.9,
                    "top_k": 50,
                    "temperature": 0.15,
                    "repetition_penalty": 1.2,
                    "min_length": 15,
                    "max_length": 150,
                    "no_repeat_ngram_size": 5,
                    "num_beams": 1,
                    "tokens_offset": 0,
                    "max_time": 12
                }
            self.CheckModel()
            t = datetime.datetime.now()
            # DEBUG TODO
            # '<extra_id_0>' ДЛЯ T5
            # '<pad>' ДЛЯ RUGPT?

            # if p.get("model_type", "instruct") == "dialog":
            #    print('[DEBUG LLM] MAIN LLM MODEL SET TO DIAG** !!!')
            #    self.model = self.dialog_model
            #    self.tokenizer = self.dialog_tokenizer
            # else:
            #    print('[DEBUG LLM] MAIN LLM MODEL SET TO **INSTRUCT !!!')
            #    self.model = self.instruct_model
            #    self.tokenizer = self.instruct_tokenizer

            inp = ninp + '<extra_id_0>'  # '<SC6>'+ninp+' <extra_id_0>'#'<LM>'+ninp

            # print('DEBUG ИНПУТ МОДЕЛИ === \n',inp)
            # print('БЕЗ СПЕЦТОКЕНОВ: ',[tokenizer.encode(inp,add_special_tokens=False)])
            # print('СО: ',[tokenizer.encode(inp,add_special_tokens=True)])
            input_tokens = self.tokenizer.encode(inp, add_special_tokens=False)
            samplePart = []
            ##DEBUG
            # repeatDangerPart = 'Mame4o спрашивает как дела у пользователя. | У нас всё отлично, давайте продолжим нашу беседу! Кстати, вы видели мои посты про пингвина? А какую книгу читали последнюю? И сколько уже выпили пива вместе со мной? Это было потрясающе! Я до сих пор вспоминаю это с улыбкой на лице.'
            if repeatDangerPart != '':
                samplePart = self.tokenizer.encode(repeatDangerPart, add_special_tokens=False)

            input_cnt = len(input_tokens)
            # print('DEBUG Параметры: ',str(p))
            # self.TokenizerDebugPrint(input_tokens,'DEBUG ИНПУТ П0>')
            input_ids = torch.tensor([input_tokens]).to(torch_device)

            ### МОДУЛЬ ОСТАНОВКИ ГЕНЕРАЦИИ ###

            class KeywordsStoppingCriteria(StoppingCriteria):
                i = 0

                def __init__(self, keywords_ids: list, keywords_words_ids: list,
                             sample: list, controlOut: list):
                    self.controlOut = controlOut
                    self.keywords = keywords_ids
                    self.words = keywords_words_ids
                    self.sample = sample

                def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                    self.i += 1
                    if input_ids[0][-1].item() in self.keywords:
                        self.controlOut.append('symbol')
                        print('[STOP CRITERIA] Early stopping сработал! (по символу)')
                        return True
                    if len(input_ids[0]) > 1:
                        # print('TENSORS',[input_ids[0][-1].item(),input_ids[0][-2].item()],'EXAMPLES',self.words[0])
                        if [input_ids[0][-1].item(), input_ids[0][-2].item()] in self.words:
                            self.controlOut.append('word')
                            print('[STOP CRITERIA]  Early stopping сработал!')
                            return True
                    if (self.i > 5 and self.i % 5 == 0):
                        if findRepeatingTokens(self.sample, input_ids[0].tolist()):
                            self.controlOut.append('repeat')
                            print('[STOP CRITERIA] Early stopping REPEAT FOUND')
                            return True
                    return False

            stop_symbols = ['}', '*', ']']  # ,':']
            stop_words = [['\n', '*'], ['\n', 'Q'], ['Q', ':']]
            stop_ids = [self.tokenizer.encode(w, add_special_tokens=False)[0] for w in stop_symbols]
            stop_ids_words = []
            stoppingCallback = []
            for word in stop_words:
                stop_ids_words.append([self.tokenizer.encode(w, add_special_tokens=False)[0] for w in word])
            stop_criteria = KeywordsStoppingCriteria(stop_ids, stop_ids_words, samplePart, stoppingCallback)

            ### МОДУЛЬ ОСТАНОВКИ ГЕНЕРАЦИИ ###
            try:
                generation_config = GenerationConfig.from_pretrained(self.ModelLocalPaths["instruct"]["id"],
                                                                     cache_dir=self.thisfolder +
                                                                               self.ModelLocalPaths["instruct"][
                                                                                   "localPath"])
            except BaseException as err:
                print('GenConfig не нашелся потому что', err)
                generation_config = GenerationConfig.from_dict({"bos_token_id": 50256, "eos_token_id": 50256,
                                                                "transformers_version": "4.27.1"})  # взято из ruGPT3.5 config

            generation_config.do_sample = p.get("do_sample")
            # generation_config.top_p = p["top_p"]

            # generation_config.repetition_penalty = p["repetition_penalty"]
            # generation_config.top_k = p["top_k"]
            # generation_config.no_repeat_ngram_size = p["no_repeat_ngram_size"]
            generation_config.no_repeat_ngram_size = 2

            # top_p": 0.95, "top_k": 5, "repetition_penalty": 1.03,
            generation_config.top_p = 0.95
            generation_config.top_k = 5
            generation_config.repetition_penalty = 1.03
            generation_config.temperature = p.get("temperature", 0.2)
            generation_config.min_length = p["min_length"]
            generation_config.max_length = p["max_length"]
            generation_config.max_new_tokens = p["max_length"]
            generation_config.max_time = p.get("max_time", 12.0)
            # generation_config.num_beams = 2#p.get("num_beams", 3)  # DEBUG !!!! DEBUG TODO BEAMS
            generation_config.eos_token_id = self.tokenizer.eos_token_id  # self.tokenizer.encode(']',add_special_tokens=False)[0]#tokenizer.eos_token_id
            generation_config.early_stopping = True
            print('DEBUG GEN CONFIG = ', generation_config)
            # torch.manual_seed(random.randint(0, 1000)) #ниче не дает
            restart_generation = True
            attempt = 0
            wasEarlyStopped = False
            result = ""
            while restart_generation and attempt <= 3:
                attempt += 1
                print(f'({calcTime(t)}|{attempt}) [FT5 FT5 DEBUG!!!] DEBUG PRINT ПЕРЕД ГЕНАЦИЕЙ')
                stoppingCallback.clear()
                outputs = generate(self.model, input_ids, generation_config, stop_criteria)
                print(f'({calcTime(t)}|{attempt}) [FT5 FT5 DEBUG!!!] DEBUG PRINT ПОСЛЕ ГЕНАЦИИ')
                # https://huggingface.co/docs/transformers/v4.18.0/en/main_classes/text_generation
                output = None
                if (len(outputs) > 0):
                    self.lastTokensUsed = len(outputs[0])
                    output = outputs[0][1 + p["tokens_offset"]:]
                wasEarlyStopped = len(stoppingCallback) > 0

                # print('DEBUG TOKEN OUTS',outputs)
                result = self.tokenizer.decode(output, skip_special_tokens=True)
                result = CutSpaces(result.replace('<extra_id_0>', '').replace('A:', '').strip())
                print(calcTime(t) + ' - время просчета, токенов [I/O] -',
                      '[' + str(input_cnt) + '/' + str(self.lastTokensUsed) + ']',
                      'earlyStop =', wasEarlyStopped, 'фрагмент генерации =', result[0:7], '\n')
                if len(result) >= 4 or wasEarlyStopped:
                    restart_generation = False
                else:
                    print(
                        '{calcTime(t)} | [FT5 WARNING] МЕНЕЕ 8 СИМВОЛОВ! ЗАПУЩЕНА ПЕРЕЗАГРУЗКА МОДЕЛИ И РЕСТАРТ ГЕНЕРАЦИИ')
                    # self.CheckModel(forceLoad=True)
                    print(f'{calcTime(t)} | FT5 >>>> CPU')
                    self.model.to("cpu")
                    torch.cuda.empty_cache()
                    # https://bytemeta.vip/repo/ultralytics/ultralytics/issues/4057
                    gc.collect()
                    print(f'{calcTime(t)} | FT5 >>>> EMPTED CACHE!')
                    print('{calcTime(t)} | FT5 >>>> CUDA')
                    self.model.to(torch_device)
                    print('{calcTime(t)} | [FT5 WARNING] ПОВТОРНАЯ ЗАГРУЗКА ЗАВЕРШЕНА')
            # self.TokenizerDebugPrint(output,'DEBUG РЕЗУЛЬТ П1>')
            # print('DEBUG РЕЗУЛЬТ П1>'+self.tokenizer.decode(debugOutputs))

            if returnStopReason:

                stoppingReason = ''
                if wasEarlyStopped:
                    stoppingReason = stoppingCallback[0]
                return {"generated": result, "stopped": stoppingReason}  # БЫЛО outputs[0][1:]
            else:
                return result

        def debug(self):
            from LLMExamples import LLMExamples
            examples = importlib.reload(sys.modules['LLMExamples']).examples
            self.e = examples()
            self.e.debug()
            inp = self.e.getResult()
            p = self.e.getParams()
            print("Загрузка текста из подключаемого модуля")
            # print('Параметры: ',str(p))
            # print(inp)
            return self.FredT5(inp, p)

        def debug2(self):
            self.CheckModel()
            generation_config = GenerationConfig.from_pretrained(self.thisfolder + self.ModelLocalPath)
            prompt = '<SC1>Тебя зовут Анфиса. Тебе интересно машинное обучение.' + '\nТы ответил: <extra_id_0>'
            input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids
            out_ids = self.model.generate(input_ids=input_ids.to(torch_device), generation_config=generation_config)
            t5_output = self.tokenizer.decode(out_ids[0][1:])
            if '</s>' in t5_output:
                t5_output = t5_output[:t5_output.find('</s>')].strip()

            t5_output = t5_output.replace('<extra_id_0>', '').strip()
            t5_output = t5_output.split('Собеседник')[0].strip()
            print('B:> {}'.format(t5_output))
            return t5_output

        def chatbot(self, ninp, params, repeat_danger_context=""):

            inp = ninp  # self.e.getResult()
            p = params  # self.e.getParams()
            print("Загрузка текста из подключаемого модуля")
            # print('Параметры: ',str(p))
            # print(inp)
            reply = self.FredT5(inp, p, repeatDangerPart=repeat_danger_context, returnStopReason=True)
            stopReason = reply["stopped"]
            reply = reply["generated"]
            # reply = "Пользователь просит меня не заебывать его. | Я думаю, что это связано с тем фактом,что он очень сильно хочет бана и боится этого больше всего на свете <команда=и****й> [эмоция=интерес]"
            emotion = 'нет'
            command = 'нет'
            # print('R1',reply)
            cmd = GetCmd(reply, tip="emo")
            emotion = cmd["cmd"]
            reply = cmd["cut"]
            # print('R2',cmd)
            cmd = GetCmd(reply, tip="cmd")
            command = cmd["cmd"]
            reply = cmd["cut"]
            reply = CutSpaces(reply)
            result = {
                "stopped": stopReason,
                "reply": reply,
                "emotion": emotion,
                "command": command,
                "tokens": self.lastTokensUsed}

            return result

        def __init__(self, nick='obama726', syspath=''):
            if syspath != '':
                self.thisfolder = syspath  # os.path.dirname(os.path.realpath(__file__))
            print('Загрузка класса FredT5 по пути', self.thisfolder)
            self.nick = nick

    lm = T5()
    lm.CheckModel()
    loading_flag.set()
    print('время запуска ' + calcTime(t))
    while True:
        try:
            llm_input = fredCtxQueue.get()
            print('[FREDT5 QUEUE] получена очередь', llm_input)
            # answer = lm.chatbot(inp[0], **inp[1])
            # llm_input, params, danger_context
            answer = lm.chatbot(llm_input[0], llm_input[1], llm_input[2])
            fredCtxQueueOutput.put(answer)
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
    repeating_dict = manager.dict()

    DOCKER_SENDER_ENABLED = False
    if DOCKER_SENDER_ENABLED:
        from HyperAI_Docker import DockerSender

        # ЧЕКНУТЬ АУТПУТ ПРОЦЕССА strace -ewrite -p $PID
        docker_sender = DockerSender()
    else:

        LargeFREDProc = multiprocessing.Process(
            target=FRED_PROCESS,
            args=(loading_flag, fredCtxQueue, fredCtxQueueOutput,
                  repeating_dict,))  # Thread(target = a, kwargs={'c':True}).start()
        LargeFREDProc.start()
        # FRED_PROCESS(fredCtx)
    print('ЗАПУСК ЧО2')

    from LLMExamples import LLMExamples, get_llm_formed_inputs


    def FredT5ChatbotQueue(ninp, context, paramsOverride, environment, lmUsername):
        if environment.get("own_prompt", False):
            llm_input, params, danger_context = ninp, \
                {"do_sample": True,
                 "top_p": 0.98, "temperature": 0.65, "repetition_penalty": 1.3, "min_length": 10,
                 "max_length": 150, "tokens_offset": 0, "top_k": 50,
                 "no_repeat_ngram_size": 5, "num_beams": 3, "max_time": 12, }, \
                "ну че как дела"
        else:
            llm_input, params, danger_context = get_llm_formed_inputs(inp=ninp, username=lmUsername,
                                                                      params_override=paramsOverride,
                                                                      environment=environment, dialog_context=context,
                                                                      repeating_dict=repeating_dict)
        fredCtxQueue.put((llm_input, params, danger_context))
        out = fredCtxQueueOutput.get()
        return out

        # return docker_sender.chatbot(llm_input, params, danger_context)


    loading_flag.wait()
    # print(e.getResult()+'mda')
    print('время запуска ТЕСТА ' + calcTime(t))
    while True:
        inp = input(
            "1-yt,2-mine,3-БЕЗ ПРОМПТА,!-ник,4-DIALOG mc,5-INSTRUCT mc,6-welcome,без-о системе\n:>")  # 'чобабке>>'
        if inp != "" or inp != "ext":
            if inp[0] == "!":
                print('ANS',
                      FredT5ChatbotQueue("Как дела зшщз", "", None, {"env": "youtube", "diags_count": 0}, inp[1:]))
            elif inp[0] == "1":
                print('ANS',
                      FredT5ChatbotQueue(inp[1:], "", None, {"env": "youtube", "sentence_type": "dialog"}, "lexeCho"))

            elif inp[0] == "2":
                print('ANS',
                      FredT5ChatbotQueue(inp[1:], "", None, {"env": "minecraft", "sentence_type": "dialog"}, "lexeCho"))
            elif inp[0] == "3":
                print('ANS',
                      FredT5ChatbotQueue(inp[1:], "", None,
                                         {"env": "minecraft", "own_prompt": True, "sentence_type": "dialog"},
                                         "lexeCho"))
            elif inp[0] == "4":
                print('ANS',
                      FredT5ChatbotQueue(inp[1:], "", {"model_type": "dialog"},
                                         {"env": "minecraft", "sentence_type": "dialog"}, "lexeCho"))
            elif inp[0] == "5":
                print('ANS',
                      FredT5ChatbotQueue(inp[1:], "", {"model_type": "instruct"},
                                         {"env": "minecraft", "sentence_type": "dialog"}, "lexeCho"))
            elif inp[0] == "6":
                print('ANS',
                      FredT5ChatbotQueue(inp[1:], "", {"model_type": "instruct"},
                                         {"env": "broadcast", "broadcast_type": "stream_ad"}, "lexeCho"))
            elif inp[0] == "7":
                print('ANS',
                      FredT5ChatbotQueue(inp[1:], "", {"model_type": "instruct"},
                                         {"env": "broadcast", "broadcast_type": "status_report"}, "lexeCho"))
            else:
                print('ANS',
                      FredT5ChatbotQueue(inp, "", None, {"env": "youtube", "sentence_type": "about_system"}, "lexeCho"))

        # if inp == "1":
        #
        #    inp = input("Введите сообщение для получения ответа\n>>")
        #    print("Запуск модели")
        #    print("Ответ:")
        #    print('|!|\n'+lm.FredT5(inp)+'\n|!|')
        # if inp == "2":
        #
        #    inp = input("Введите сообщение для получения ответа чатбота\n>>")
        #    print("Запуск модели")
        #    print("Ответ:")
        #    print('|!|\n',lm.chatbot(inp),'\n|!|')
        # if inp == "3":
        #    inp = input("Введите сообщение для АНАЛИЗА НИКА\n>>")
        #    print("Запуск модели")
        #    print("Ответ:")
        #    print('|!|\n',lm.nickAnalyze(inp),'\n|!|')
        # if inp == "":
        #
        #    print("Ответ:")
        #    print('|!|\n'+lm.debug()+'\n|!|')
        if inp == "ext":
            print("выход")
            exit()
    #####Prefix <LM>
    ####lm_text='<LM>Принялся Кутузов рассказывать свою историю как он сюда попал. Началось'
    ####input_ids=torch.tensor([tokenizer.encode(lm_text)]).to(device)
    ####outputs=model.generate(input_ids,eos_token_id=tokenizer.eos_token_id,early_stopping=True)
    ####print(tokenizer.decode(outputs[0][1:]))
    ####
    ##### print result: с того, что он был в армии, служил в артиллерии</s>.
    ####
    #####Prefix <SC1>
    ####lm_text='<SC1>Принялся Кутузов рассказывать свою историю <extra_id_0>. Началось с того, что он был в армии, служил в артиллерии.'
    ####input_ids=torch.tensor([tokenizer.encode(lm_text)]).to(device)
    ####outputs=model.generate(input_ids,eos_token_id=tokenizer.eos_token_id,early_stopping=True)
    ####print(tokenizer.decode(outputs[0][1:]))
    ####
    #####print result: '<extra_id_0>, как он воевал</s>'
    ####
    ##### Prefix <SC5> 
    ####lm_text='<SC5>Принялся Кутузов рассказывать свою историю <extra_id_0>. Началось с того, что он был в армии, служил в артиллерии.'
    ####input_ids=torch.tensor([tokenizer.encode(lm_text)]).to(device)
    ####outputs=model.generate(input_ids,eos_token_id=tokenizer.eos_token_id,early_stopping=True)
    ####print(tokenizer.decode(outputs[0][1:]))

    # print result: '<extra_id_0>, как он стал генералом</s>'
'''  ### ORIGINAL MODEL COPY FRED T5 ####
class T5:
    thisfolder = os.path.dirname(os.path.realpath(__file__))
    tokenizer = []
    model = []
    nick = "obama421"
    username = "Пользователь"
    device = []
    e = []
    ModelLoaded = False
    lastTokensUsed=0
    device = "cuda"
    context = []
    def TokenizerDebugPrint(self,inp,debugPrefix='Debug Input >> '):
        tokens = inp
        debugOutputs = []
        for t in tokens:
            debugOutputs.append(t)
            debugOutputs.append(96) # token '|' = 96, [=65, .=18
        print(debugPrefix,'\n<|||>\n',self.tokenizer.decode(debugOutputs),'\n<|||>')
    def CheckModel(self):
        if(not self.ModelLoaded):
            t = datetime.datetime.now()
            print('Загрузка модели...')
            
            ###original model###
            #self.tokenizer = GPT2Tokenizer.from_pretrained(thisfolder+'/variants/original',eos_token='</s>')
            #self.model = T5ForConditionalGeneration.from_pretrained(self.thisfolder+'/variants/original')
            ###den4ik model###
            
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.thisfolder+'/variants/FP16Trained1den4ik',eos_token='</s>')
            self.model = T5ForConditionalGeneration.from_pretrained(self.thisfolder+'/variants/FP16Trained1den4ik')
            
            #torch_device=torch_device #cpu или cuda (то же что и gpu)
            self.model.to(torch_device)
            self.ModelLoaded = True
            print('Модель загружена! Время:'+calcTime(t))
            
    def FredT5(self,ninp,p={
        "do_sample":True,
        "top_p":0.9,
        "top_k": 50,
        "temperature":0.15,
        "repetition_penalty": 2.0,
        "min_length": 15,
        "max_length": 200,
        "no_repeat_ngram_size": 5,
        "tokens_offset":0
        }):  #[2.0,2.0,50,100]
        self.CheckModel()
        t = datetime.datetime.now()
        inp = '<LM>'+ninp#'<LM>'+ninp
        #print('ИНПУТ МОДЕЛИ === \n\n',inp)
        #print('БЕЗ СПЕЦТОКЕНОВ: ',[tokenizer.encode(inp,add_special_tokens=False)])
        #print('СО: ',[tokenizer.encode(inp,add_special_tokens=True)])
        input_tokens = self.tokenizer.encode(inp,add_special_tokens=False)
        input_cnt = len(input_tokens)
        print('DEBUG Параметры: ',str(p))
        self.TokenizerDebugPrint(input_tokens,'DEBUG ИНПУТ П0>')
        input_ids=torch.tensor([input_tokens]).to(self.device)
        
        
        
        #torch.manual_seed(random.randint(0, 1000)) #ниче не дает
        outputs=self.model.generate(
            input_ids,
            do_sample = p["do_sample"],
            #top_p = p["top_p"],
            
            temperature = p["temperature"],
            repetition_penalty = p["repetition_penalty"],
            min_length = p["min_length"],
            max_length = p["max_length"],
            top_k = p["top_k"],
            no_repeat_ngram_size = p["no_repeat_ngram_size"],
            #num_beams = 2,
            eos_token_id=self.tokenizer.encode(']',add_special_tokens=False)[0],#tokenizer.eos_token_id,
            early_stopping=True) #https://huggingface.co/docs/transformers/v4.18.0/en/main_classes/text_generation
        output = None
        if(len(outputs)>0):
            self.lastTokensUsed = len(outputs[0])
            output = outputs[0][1+p["tokens_offset"]:]
        print(calcTime(t)+' - время просчета, токенов [I/O] -','['+str(input_cnt)+'/'+str(self.lastTokensUsed)+']','\n')
        #print('DEBUG TOKEN OUTS',outputs)
        result = self.tokenizer.decode(output, skip_special_tokens=True)
        
        self.TokenizerDebugPrint(output,'DEBUG РЕЗУЛЬТ П1>')
        #print('DEBUG РЕЗУЛЬТ П1>'+self.tokenizer.decode(debugOutputs))
        result = CutSpaces(result)
       
        return result# БЫЛО outputs[0][1:]
        
    def debug(self):
        examples = importlib.reload(sys.modules['FredExamples']).examples
        self.e = examples()
        self.e.debug()
        inp = self.e.getResult()
        p = self.e.getParams()
        print("Загрузка текста из подключаемого модуля")
        #print('Параметры: ',str(p))
        #print(inp)
        return self.FredT5(inp,p)
    def debug2(self):
        self.CheckModel()
        generation_config = GenerationConfig.from_pretrained(self.thisfolder+'/variants/FP16Trained1den4ik')
        prompt = '<SC1>Тебя зовут Анфиса. Тебе интересно машинное обучение.' + '\nТы ответил: <extra_id_0>'
        input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids
        out_ids = self.model.generate(input_ids=input_ids.to(self.device), generation_config=generation_config)
        t5_output = self.tokenizer.decode(out_ids[0][1:])
        if '</s>' in t5_output:
            t5_output = t5_output[:t5_output.find('</s>')].strip()

        t5_output = t5_output.replace('<extra_id_0>', '').strip()
        t5_output = t5_output.split('Собеседник')[0].strip()
        print('B:> {}'.format(t5_output))
        return t5_output
    def nickAnalyze(self,nick):
        examples = importlib.reload(sys.modules['FredExamples']).examples
        self.e = examples()
        inp = self.e.nickAnalyze(nick)
        p = self.e.getParams()
        print("Загрузка текста из подключаемого модуля")
        #print('Параметры: ',str(p))
        #print(inp)
        return self.FredT5(inp,p)
    def getContext(self):
        result = ""
        if len(self.context)>1:
            for i,record in enumerate(self.context):
                #if i-1==len(self.context) and record["role"] == "assistant":
                #    result+=""
                result+=record["content"]
        return result
    def chatbot(self,ninp,rank=5,context=""):
        if context != "":
            self.context = context
        examples = importlib.reload(sys.modules['FredExamples']).examples
        self.e = examples()
        self.e.username = self.username
        self.e.chatbot(ninp,context=self.getContext())
        inp = self.e.getResult()
        p = self.e.getParams()
        print("Загрузка текста из подключаемого модуля")
        #print('Параметры: ',str(p))
        #print(inp)
        reply = self.FredT5(inp,p)
        #reply = "Пользователь просит меня не заебывать его. | Я думаю, что это связано с тем фактом,что он очень сильно хочет бана и боится этого больше всего на свете <команда=*******> [эмоция=интерес]"
        emotion = 'нет'
        command = 'нет'
        #print('R1',reply)
        cmd = GetCmd(reply,tip="emo")
        emotion = cmd["cmd"]
        reply = cmd["cut"]
        #print('R2',cmd)
        cmd = GetCmd(reply,tip="cmd")
        command = cmd["cmd"]
        reply = cmd["cut"]
        reply = CutSpaces(reply)
        #print('R3',cmd)
        #print('DEBUG REPLY = ',reply)
        #lbracketIdx = reply.find('[')+1
        #rbracketIdx = reply.rfind(']')+1
        #emotionContainer = reply[lbracketIdx:rbracketIdx]
        #if(lbracketIdx != -1) and (rbracketIdx != -1) and emotionContainer.find('=') != -1:
        #    
        #    emotion = emotionContainer[emotionContainer.rfind('=')+1:-1]
        #    reply = reply[:lbracketIdx-1]+reply[rbracketIdx:]
        #else:
        #    print('эмоция не найдена!')
        #    
        #
        #lbracketIdx = reply.find('<')+1
        #rbracketIdx = reply.rfind('>')+1
        #emotionContainer = reply[lbracketIdx:rbracketIdx]
        #if(lbracketIdx != -1) and (rbracketIdx != -1) and emotionContainer.find('=') != -1:
        #    
        #    command = emotionContainer[emotionContainer.find('=')+1:-1]
        #    reply = reply[:lbracketIdx-1]+reply[rbracketIdx:]
        #else:
        #    print('команда не найдена!')
            
            
            
        #emotion=reply.split('\n')[0].replace("Emotion=","")
        #reply = reply[reply.find('\n')+1:]
        result = {
            "reply": reply,
            "emotion": emotion,
            "command": command,
            "tokens": self.lastTokensUsed}
        
        
        return result
    def __init__(self,nick='obama726'):
        self.nick = nick
        self.e = examples()
'''
