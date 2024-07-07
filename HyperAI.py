# import copy  #pip install deeppavlov,transformers,(fasttext для некоторых моделей)
# TORCH на GPU ПРИ ПОМОЩИ pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# pip install language-tool-python,
import multiprocessing

# pip install googletrans==3.1.0a0
# pip install nltk
# pip install cyrtranslit
# pip install pymorphy2
# pip install num2words

# pip install pygame==2.4.0 ##pip install pygame==2.0.0.dev8 ? не работает

import numpy as np
import random
from datetime import datetime, date, timedelta
import time
from multiprocessing import Process, Manager
import threading
import os, sys
import traceback

thisfolder = os.path.dirname(os.path.realpath(__file__))


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


def TTS_PROCESS(ctx):
    import torch
    # from torch import package
    print('[TTS INIT] Started load TTS model...')
    device = torch.device("cpu")  # 'cpu')  # cuda
    torch.set_num_threads(4)
    t = datetime.now()
    local_file = thisfolder + '/HyperAI_Models/TTS/variants/Silero/silero_tts.pt'
    if not os.path.isfile(local_file):
        torch.hub.download_url_to_file('https://models.silero.ai/models/tts/ru/v3_1_ru.pt',
                                       local_file)
    VoiceModel = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
    # VoiceModel, exampleText = torch.hub.load(repo_or_dir='snakers4/silero-models', model='silero_tts', language='ru',
    #                                         speaker='v3_1_ru', trust_repo=True, cache_dir=)  # v3_1_ru или ru_v3

    VoiceModel.to(device)  # gpu or cpu
    print('[TTS INIT] время запуска VOICE TTS на', device, '=', calcTime(t))
    # VoiceModel.eval() не работает
    print('[TTS INIT 2 NEW] Started load ACCENTUATOR model...')
    # from ruaccent import RUAccent
    #
    # accentizer = RUAccent()
    # custom_words_accent_dict = {"бовдур":"б+овдур","бовдурус":"б+овдурус"}
    # accentizer.load(omograph_model_size='big', use_dictionary=True, custom_dict=custom_words_accent_dict)
    # https://huggingface.co/TeraTTS/accentuator
    print('[TTS INIT 2 NEW] ENDED! load ACCENTUATOR model! time =', calcTime(t))

    ctx.loading_flag.set()

    while True:
        try:
            text = ctx.Queue.get()
            print('[VOICE QUEUE] получена очередь', text)
            # text = accentizer.process_all(text)
            # print('[VOICE QUEUE DEBUG NEW] ТЕКСТ С УДАРЕНИЯМИ >>', text)
            # VoiceModel, exampleText = torch.hub.load(repo_or_dir='snakers4/silero-models', model='silero_tts',
            #                                         language='ru',
            #                                         speaker='v3_1_ru', trust_repo=True)  # v3_1_ru или ru_v3
            VoiceModel = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
            VoiceModel.to(device)  # gpu or cpu
            try:
                audiowave = VoiceModel.apply_tts(ssml_text=text, speaker='baya', sample_rate=48000, put_accent=True,
                                                 put_yo=True)
            except BaseException as err:
                print("=== ПРОИЗОШЛА ОШИБКА", err, "В ГЕНЕРАЦИИ СИНТЕЗА РЕЧИ! ====\n")
                print("=== ТЕКСТ>>" + text + "<< ====\n")
                print('ТЕКСТ ОБ***Й ОШИБКИ', traceback.format_exc())
                print("\n=== КОНЕЦ ОШИБКИ ====")
                audiowave = VoiceModel.apply_tts(text=text, speaker='baya', sample_rate=48000, put_accent=True,
                                                 put_yo=True)
            ctx.QueueOutput.put(audiowave)
        except BaseException as err:
            print('[TTS ERR] ОШИБКА ПРОЦЕССА В TTS: ', err)
            print('[TTS ERR] ТЕКСТ ОБ***Й ОШИБКИ', traceback.format_exc())
            print("\n[TTS ERR] === КОНЕЦ ОШИБКИ ====")
            time.sleep(1)


if __name__ == "__main__":
    try:
        import psutil

        os_used = sys.platform
        process = psutil.Process(os.getpid())  # Set highest priority for the python script for the CPU
        if os_used == "win32":  # Windows (either 32-bit or 64-bit)

            process.nice(psutil.HIGH_PRIORITY_CLASS)  # REALTIME_PRIORITY_CLASS)
            print('[MAIN] УСТАНОВЛЕН ВЫСОКИЙ ПРИОРИТЕТ ПРОЦЕССА PID =', os.getpid())
        elif os_used == "linux":  # linux
            process.nice(psutil.IOPRIO_HIGH)
        else:  # MAC OS X or other
            process.nice(20)


        # https://github.com/mlo40/VsPyYt
        def tm(x):
            return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')


        # python -m deeppavlov download rusentiment_bert

        timeStartMetrics = []
        timeStartMetrics.append(datetime.now())

        manager = Manager()

        speech_available_event = manager.Event()
        speech_available_event.set()
        ctx = manager.Namespace()
        ctx.ThreadsActived = True
        ctx.mood = 0.0
        vtube_ctx = manager.Namespace()
        vtube_ctx.NeedX = -0.5
        vtube_ctx.NeedY = -1.0
        ctx.state = "gaming"
        ctx.SeparateEyes = True
        ctx.MineEvent = manager.Event()
        ctx.MineEventName = ""
        vtube_ctx.eyeX = vtube_ctx.NeedX
        vtube_ctx.eyeY = vtube_ctx.NeedY
        ctx.AnimEvent = manager.Event()
        ctx.AnimEventInfo = {"name": "SawChat.exp3.json", "type": "expression", "time": 0}
        ctx.IsMCStarted = False
        ctx.IsVtubeStarted = False
        ctx.IsYTChatConnected = False

        import re
        import signal
        import PySimpleGUI as sg
        from rapidfuzz import fuzz
        from pympler.tracker import SummaryTracker

        # pip install pysimplegui

        utils_folder = "HyperAI_Utils"


        def start_bat(bat_path: str):
            print("[MAIN SCRIPT] started bat file", bat_path)
            os.startfile(bat_path)


        def start_program(program: str):
            bat_name = ""
            if program == "VTUBE":
                bat_name = "StartVTubeStudioNoSteam.bat"
            elif program == "SheepChat":
                bat_name = "StartSheepChat.bat"
            else:
                print("neizv programma")
                return
            start_bat(os.path.join(thisfolder, utils_folder, bat_name))


        FisrtBuildingLanguageTool = True
        # Для библиотеки dostoevskiy скачиваем Build Tools Visual studio 14.0+, идёт в комплекте с визуал студио
        tool = []

        ttsCtx = manager.Namespace()
        ttsCtx.Queue = manager.Queue()
        ttsCtx.QueueOutput = manager.Queue()
        ttsCtx.loading_flag = manager.Event()
        LargeTTSProc = Process(
            target=TTS_PROCESS,
            args=(ttsCtx,))  # Thread(target = a, kwargs={'c':True}).start()
        LargeTTSProc.start()

        from HyperAI_OBS import OBS_Websocket

        obs_ka = OBS_Websocket()


        def do_vtube_event(event_name):
            ctx.AnimEventInfo = {"name": event_name, "type": "hotkey"}
            ctx.AnimEvent.set()
            ctx.AnimEvent.clear()


        def obs_scene_check_sync():
            obs_scene_name = ""
            if obs_scene_name == "NetTyanChat":
                do_vtube_event("MoveChat")
            elif obs_scene_name == "NetTyan":
                do_vtube_event("MoveGame")


        def TTS_QUEUE(text):
            max_numm = 990
            put_text = (text[:max_numm] + '..') if len(text) > max_numm else text
            ttsCtx.Queue.put(put_text)
            return ttsCtx.QueueOutput.get()


        def modifyMood(mood: float) -> None:
            """add to mood argument number"""
            mood_now = ctx.mood
            mood_now += mood
            mood_now = clamp(mood_now, -10, 10)
            if mood_now == -10:
                ctx.mood = random.choice([1, 2, 3, 4])
                return
            elif mood_now == 10:
                ctx.mood = random.choice([-3, -2, 0, 1, 2])
                return
            ctx.mood = mood_now
            DATABASE.set_mood(mood_now)


        def check_literacy(text):
            # Initialize the LanguageTool object

            # Check the text for errors
            matches = tool.check(text)

            # Build a list of errors and corrections
            errors = []
            corrections = {}
            for match in matches:
                if match.ruleId in ('UPPERCASE_SENTENCE_START', 'MORFOLOGIK_RULE_RU_RU'):
                    # Add the error to the errors list
                    error = match.message.replace('{', '').replace('}', '')
                    errors.append((text[match.offset:match.errorLength], error))

                    # Add the correction to the corrections dictionary
                    correction = match.replacements[0] if match.replacements else ''
                    corrections[(match.offset, match.offset + match.errorLength)] = correction

            # Apply the corrections to the original text
            corrected_text = text
            for (offset, error_length), correction in sorted(corrections.items(), reverse=True):
                corrected_text = corrected_text[:offset] + correction + corrected_text[offset + error_length:]

            # Fix punctuation errors
            # punctuation_regex = r'([\wа-яА-ЯёЁ]+)\s*([^\w\s]+)\s*(?=[\wа-яА-ЯёЁ]+)'
            # punctuation_errors = re.findall(punctuation_regex, corrected_text)
            # for match in punctuation_errors:
            #    error, punctuation = match
            #    corrected_text = corrected_text.replace(f'{error} {punctuation}', f'{error}{punctuation} ')

            return [corrected_text, errors]


        model = []
        FisrtBuildingModel = True


        # Для библиотеки dostoevskiy скачиваем Build Tools Visual studio 14.0+, идёт в комплекте с визуал студио
        # def BuildSentimentModel():
        #    global model
        #    global FisrtBuildingModel
        #    # ИМПОРТИРОВАТЬ torch
        #    if FisrtBuildingModel:
        #        print('# СТРОИМ МОДЕЛЬ.....')
        #        from deeppavlov import build_model
        #        model = build_model('rusentiment_convers_bert', download=True)
        #        # model = build_model('sentiment_twitter', download=True) #очень неточное, постоянно негатив
        #        # model = build_model("rusentiment_bert") #среднее, такое -_- часто ошибается
        #        print('# МОДЕЛЬ ЗАГРУЖЕНА!')
        #        FisrtBuildingModel = False

        def determine_tone(text):
            if not text or not text.strip():
                return "Input text is empty"
            print('# Начата проверка... [' + text + ']')
            predict = model([text])
            print(predict)
            sentiment_label = predict[0]
            sentiment_score = 0
            print('# Проверка окончена!')
            if sentiment_label == "positive":
                return 1
            elif sentiment_label == "negative":
                return -1
            else:
                return 0


        # # TEXT TO SPEECH

        ###pip install -q torchaudio omegaconf
        ###pip install playsound playsound

        # from playsound import playsound

        # ОПИСАНИЕ SSML TEXT: + перед ударным слогом
        # rate x-slow slow medium fast x-fast
        # pitch x-low, low, medium, high, x-high
        # <p> </p> параграфы, перед ними пауза 3 сек. <s> 1.5 сек. <break time = "?s">
        # ПОДРОБНЕЕ https://github.com/snakers4/silero-models/wiki/SSML

        def OformText(text="Привет школьник", rate="fast", pitch="medium"):
            textStartet = '<speak><prosody rate="' + rate + '" pitch="' + pitch + '">'
            text = textStartet + text + "</prosody></speak>"
            return text


        # FisrtBuildingVoice = True

        # torch.set_num_threads(4)# если cuda, отрубаем это

        from HyperAI_WEB import print_subtitles


        # def LoadTTS():
        #    global VoiceModel,VoiceModelLoaded
        #    startTime = datetime.now()
        #    if not VoiceModelLoaded:
        #        print('Загрузка модели TTS...')
        #        VoiceModel, badtext = torch.hub.load(repo_or_dir='snakers4/silero-models', model='silero_tts', language='ru',
        #                                             speaker='v3_1_ru')  # v3_1_ru или ru_v3
        #        VoiceModel.to(device)  # gpu or cpu
        #        VoiceModelLoaded = True
        #        print('Загрузка модели TTS завершена!', calcTime(startTime))
        ###print('Загрузка подготовщика TTS...')
        ###TextEnhancerModel, example_texts, languages, punct, apply_te = torch.hub.load(repo_or_dir='snakers4/silero-models',
        ###                                                                  model='silero_te')
        #### see avaiable languages
        ###print(f'Available languages {languages}')
        ###print(f'Available punctuation marks {punct}')
        ###print('Загрузка подготовщика TTS завершена!', calcTime(startTime))

        # LoadTTS()
        ###def apply_te(text, lan='ru'):
        ###    global TextEnhancerModel
        ###    return TextEnhancerModel.enhance_text(text, lan)

        # print(f"Output:\n{apply_te('Перевод с английского, немецкого, французского, испанского, польского, турецкого и других языков на русский и обратно. Возможность переводить отдельные слова и фразы, а также целые тексты, фотографии, документы и веб-страницы.')}")

        def textToSpeech(text, rate="fast", pitch="medium", seeChat=False):
            """В версии 0.0.4 добавили особый блок: ждем, только если есть другая речь"""
            print('Запускаем модель...')
            startTime = datetime.now()

            text = PrepareToSpeech(text)
            subtitles = PrepareToSpeech(text, subtitles=True)
            text = OformText(text, rate, pitch)
            ctx.isVoiceBusy = True
            if seeChat:
                ctx.AnimEventInfo = {"name": "SawChat.exp3.json", "type": "expression", "time": 0.01}
                ctx.AnimEvent.set()

            print("Генерация файла wav... время до этого шага:", calcTime(startTime))  # +text)
            audio = TTS_QUEUE(text)
            print("Играем звук wav...", calcTime(startTime))

            if seeChat:
                ctx.AnimEvent.clear()

            speech_available_event.wait()

            TextDisplaySpeed.value = rate
            textSubtitlesHttp.value = subtitles
            # Thread
            threading.Thread(target=SoundToMicro,
                             kwargs={"audio": audio, "sleep": True, "smart_wait": True, "change_emotes": True},
                             daemon=True).start()
            # SoundToMicro(audio=audio, sleep=True)

            # print_subtitles(subtitles,rate)
            print("Голосовой вывод отправлен! Затраченное на всё про всё время =",
                  calcTime(startTime))  # закончен! Затраченное на всё про всё время

            # vtube_ctx.NeedX = -0.5
            # vtube_ctx.NeedY = -1
            # ctx.AnimEvent.clear()


        #    del wav_play
        #### print_subtitles В ФАЙЛЕ WEB
        # PYTTSX3
        ###pip install pyttsx3
        ##import pyttsx3
        ##engine = pyttsx3.init()
        ##voices = engine.getProperty('voices')
        ##engine.setProperty('voice', voices[1].id)
        ##engine.setProperty('rate', 100)
        ##engine.say('Privet ebanniy ti pidoras')
        ##engine.runAndWait()
        ###достаточно фигово только с англ, но робит оффлайн

        # GTTS
        # pip install gtts
        # from gtts import gTTS
        # tts = gTTS(text='Привет пошел **', lang='ru', tld='ru')
        # похож на бота, работает быстро, через гугл переводчик
        # tts.save('hello.mp3')

        # from pymorphy2 import MorphAnalyzer
        # import nltk
        # from nltk import sent_tokenize, word_tokenize, regexp_tokenize
        # from nltk.corpus import stopwords
        # morph = MorphAnalyzer(lang='ru')
        # nltk.download('stopwords')
        # stopwords_ru = stopwords.words("russian")
        # https://pymorphy2.readthedocs.io/en/stable/user/guide.html#normalization
        def fixFemWords(doc):
            # wordsmas = doc.split(" ")
            # result = doc
            # pat = r"(?u)\b\w\w+\b"
            # for word in regexp_tokenize(doc, pat):
            #    if word not in stopwords_ru:
            #        p = morph.parse(word)[0]
            #        # МУЖСКИЕ ГЛАГОЛЫ в ЖЕНСКИЙ РОД))
            #        #print('Разбор '+str(p.tag.POS)+' ='+str(p.word)+' пол='+ str(p.tag.gender) +' число=' + str(p.tag.number)+' включенность='+str(p.tag.involvement)
            #        #+' наклонение='+str(p.tag.mood)+' ИТОГ: '+p.inflect({'femn'}).word)
            #        if (p.tag.POS == "PRTS" or p.tag.POS == 'ADJS' or p.tag.POS == 'VERB') and p.tag.gender == 'masc' and p.tag.number == 'sing' and p.tag.mood != "impr":
            #            #print('Разбор глагола ='+str(p.word)+' пол='+ str(p.tag.gender) +' число=' + str(p.tag.number)+' включенность='+str(p.tag.involvement)
            #            #+' наклонение='+str(p.tag.mood)+' ИТОГ: '+p.inflect({'femn'}).word)
            #            result = result.replace(word,p.inflect({'femn'}).word)
            return doc
            # print()


        import cyrtranslit


        def translit(x):
            try:
                result = x
                defaultCyrReplaceMap = {"x": "Кс", "h": "Х"}
                for symbol in defaultCyrReplaceMap:
                    result = result.replace(symbol, defaultCyrReplaceMap[symbol].lower())
                    result = result.replace(symbol.upper(), defaultCyrReplaceMap[symbol])
                partCyrReplaceMap = {"sh": "щ", "Sh": "Щ", "sH": "щ", "SH": "Щ"}
                for part in partCyrReplaceMap:
                    result = result.replace(part, partCyrReplaceMap[part])
                result = cyrtranslit.to_cyrillic(result, "ru")
            except BaseException as err:
                print('ошибка ', err, ' при транслитерации =(')
                result = x
            return result


        translitLatin = lambda x: cyrtranslit.to_latin(x, "ru")
        from num2words import num2words


        # print(num2words("543534", lang='ru'))
        def NumbersToSpeech(inp):
            numbers = re.findall(r'\b\d+\b', inp)
            result = inp
            for numb in numbers:
                print(numb, num2words(numb, lang='ru'))
                result = result.replace(numb, num2words(numb, lang='ru'))
            return result


        # print("N2WORDS>>",NumbersToSpeech("У меня в жопе 8 хромосом"))
        # fixFemWords("уверен классный сделай сделал сделает сделаешь делаешь делает сделал сделают сделала сделали авылпов ьвао алв ле коцавы щзшеоку л уоц fgdj jw jwe skd")
        # return str(regexp_tokenize(doc, pat))
        # doc = re.sub(patterns, ' ', doc)
        # tokens = []
        # for token in doc.split():
        #    if token and token not in stopwords_ru:
        #        token = token.strip()
        #        token = morph.normal_forms(token)[0]
        #        tokens.append(token)
        # if len(tokens) > 2:
        #    return tokens
        # return None

        # from googletrans import Translator
        #
        # translator = Translator()
        ListForFemWords = "i' i'm im i self myself myself, myself.".split(' ')

        # def translate(inp, destTrans="ru", femFix=True):
        #    if (destTrans == "ru"):
        #        if (femFix):
        #            result = inp
        #            splitter = " "
        #            resultmas = result.split(splitter)
        #            result = ""
        #            for i, sequence in enumerate(resultmas):
        #                if i + 1 == len(resultmas):
        #                    splitter = ""
        #                for word in ListForFemWords:
        #                    # print(">"+sequence+"<"+'\n'+">"+word+"<"+"\n"+str(sequence.lower()==word))
        #                    if (sequence.lower() == word):
        #                        find = sequence.lower().find(word.lower())
        #                        if find != -1:
        #                            sequence = sequence + " 🚌Lisa🚌"  # sequence[:find]+" 🚌Lisa🚌" + sequence[find:]
        #                        # sequence = sequence.lower().replace(word, word+" 🚌Lisa🚌")
        #                    # if(sequence.lower().find(word) != -1):
        #                    #    sequence = sequence.replace(sequence, word+" <(Lisa)>")
        #                result += sequence + splitter
        #
        #            # print(result)
        #            result = translator.translate(result, src='en', dest='ru').text
        #            # print(result)
        #            resultnn = result
        #            result = ""
        #            chetnost = False
        #            # def switch():
        #
        #            for symbol in resultnn:
        #                if symbol == "🚌" or chetnost:
        #                    if chetnost and symbol == "🚌":
        #                        chetnost = False
        #                    elif not chetnost:
        #                        chetnost = True
        #                else:
        #                    chetnost = False
        #                    result += symbol
        #            result = result.replace("  ", " ")
        #            # result = result.replace(" 🚌Lisa🚌","")
        #            # result = result.replace(" 🚌Лиза🚌","")
        #            # result = result.replace(" 🚌Лизы🚌","")
        #            # result = result.replace(" 🚌Лизе🚌","")
        #            # result = result.replace(" 🚌Лизу🚌","")
        #            # result = result.replace(" 🚌Лизой🚌","")
        #            # print(result)
        #            # for item in resultPrev:
        #            #    result = resultPrev.
        #        else:
        #            result = translator.translate(inp, src='en', dest='ru').text
        #    else:
        #        result = translator.translate(inp, src='ru', dest='en').text
        #    return result
        #    ###################################
        #    ### ПРАВКИ ПО ТЕКСТУ ВЫВОДИМОМУ ###
        #    ###################################
        #    # цвет горчичный
        #    # ниже
        #    # обрезание сверху

        ### zzfrf2 = translator.translate(zzfrf,dest='en').text
        ### zzfrf3=translator.translate(zzfrf2,dest='ru').text
        ###
        ### print("текст >>"+zzfrf)
        ### print("перевод текста >>"+zzfrf2)
        ### print("перевод текста Обратно >>"+zzfrf3)
        ### zzfrf4=fixFemWords(zzfrf3)
        ### print("перевод текста (ФИКС РОД) >>"+zzfrf4)
        # src: Исходный язык
        # dest: Язык назначения, установленный на английский (en)
        # result.origin: Оригинальный текст, в нашем примере это Mitä sinä teet
        # result.text: Переведенный текст, в нашем случае это будет «what are you doing?»
        # result.pronunciation: Произношение переведенного текста




        import json


        def eztime():
            return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


        def eztime_min():
            return datetime.now().strftime('%M:%S')


        def js_r(filename: str):
            with open(filename, encoding='utf-8') as f_in:
                return json.load(f_in)


        def js_w(filename: str, data):
            with open(filename, 'w+', encoding='utf-8') as f_out:
                f_out.write(json.dumps(data, sort_keys=True, indent=4, separators=(',', ': '), ensure_ascii=False))


        ####js_w('usersDatabase.json', usersDatabase)
        # js_w('evaLogs.json', LogChat)

        def ConvertDiagFormat(record, fredFormat=False):
            if fredFormat:
                cmd = ""
                emo = ""
                if record["role"] == "assistant":
                    cmd = record.get("command", "")
                    if cmd != "":
                        cmd = " <команда=" + cmd + ">"
                    emo = record.get("emotion", "")
                    if emo != "":
                        emo = " [эмоция=" + emo + "]"
                usr = record.get("user", "")
                # rle = record["role"]
                # if usr == "" or usr == "default":
                if record["role"] == "assistant":
                    rle = "A"
                else:
                    rle = "Q"
                # else:
                #    rle = "A" # изначально это ник чела
                return {"user": record.get("user", "noname"),
                        "command": record.get("command", ""),
                        "emotion": record.get("emotion", ""),
                        "role": record["role"],
                        "msg": record["content"],
                        "content": rle + ': ' + record["content"] + cmd + emo + '</s>\n'
                        }
            else:
                return {"role": record["role"], "content": record["content"]}


        # @deprecated
        def ConvertDiagFormatOld(record, fredFormat=False):
            if fredFormat:
                cmd = ""
                emo = ""
                if record["role"] == "assistant":
                    cmd = record.get("command", "")
                    if cmd != "":
                        cmd = " <команда=" + cmd + ">"
                    emo = record.get("emotion", "")
                    if emo != "":
                        emo = " [эмоция=" + emo + "]"
                usr = record.get("user", "")
                if usr == "" or usr == "default":
                    if record["role"] == "assistant":
                        record["user"] = "Ева"
                    else:
                        record["user"] = "Пользователь"
                return {"role": record["role"],
                        "content": '- ' + record["user"] + ': ' + record["content"] + cmd + emo + '</s>\n'}
            else:
                return {"role": record["role"], "content": record["content"]}


        def col(inp, color="green", limiter=False):
            limiterstart = '>'
            limiterend = '<'
            if not limiter:
                limiterstart = ''
                limiterend = ''
            if color == "green":
                return limiterstart + bcolors.OKGREEN + str(inp) + bcolors.ENDC + limiterend
            elif color == "yellow":
                return limiterstart + bcolors.OKCYAN + str(inp) + bcolors.ENDC + limiterend
            else:
                return limiterstart + bcolors.OKGREEN + str(inp) + bcolors.ENDC + limiterend


        def clamp(n, smallest, largest):
            return max(smallest, min(n, largest))


        def EmotionToRank(emotion):
            # агрессия, скука, усталость, интерес, смущение, счастье, веселье, страх.
            emotionMap = {"агрессия": -0.1, "скука": 0, "усталость": -0.1, "интерес": 0.1, "смущение": 0.1,
                          "счастье": 0.1, "веселье": 0.1, "страх": 0.01}
            # global GlobalEvaMood #sox,terminator,android,..
            # global GlobalEvaEmotion #annoyed angry bored tired interested embarrassed scared happy confused
            emo = emotionMap.get(emotion, -999)
            if emo == -999:
                if emotion == "annoyed":
                    return -0.05
                elif emotion == "angry":
                    return -0.1
                elif emotion == "bored":
                    return -0.1
                elif emotion == "tired":
                    return 0.1
                elif emotion == "interested":
                    return 0.1
                elif emotion == "embarrassed":
                    return 0.2
                elif emotion == "scared":
                    return 0.1
                elif emotion == "happy":
                    return 0.1
                elif emotion == "confused":
                    return 0.1
                else:
                    return random.uniform(-0.2, 0.2)
            else:
                return emo


        def findBlocked(inpt, extended=False, blockedPhrasesMass=[], blockedPhrasesMassExtend=[]):
            inp = inpt.lower()
            if not extended:
                for word in blockedPhrasesMass:
                    if inp.find(word) != -1:
                        return True
                return False
            else:
                for word in blockedPhrasesMassExtend:
                    if inp.find(word) != -1:
                        return True
                return False


        def emotions_to_str(text: str) -> str:
            emotions_str_map = {"<3": "сердешко",
                                "^_^": "няааааа",
                                "^^": "няа",
                                ":)": "улыбка",
                                ":')": "плак",
                                ":-)": "улыбка",
                                ":(": "грусть",
                                ":'(": "плак",
                                ":-(": "печалька",
                                "(": "то есть",
                                }
            for emo in emotions_str_map:
                text = text.replace(emo, emotions_str_map[emo])
            return text


        def PrepareToSpeech(ninp, subtitles=False):
            inp = ninp
            result = ""
            # print('.', end='')
            if subtitles:
                result = inp

            else:
                result = NumbersToSpeech(translit(inp))  # добавить опред. смайлов;
                result = emotions_to_str(result)
            if len(ninp) > 900:
                inp = ninp[0:899]
            return result
            # inputs = gr.inputs.Textbox(lines=7, label="Chat with AI")
            # outputs = gr.outputs.Textbox(label="Reply")

            # gr.Interface(fn=chatbot, inputs=inputs, outputs=outputs, title="AI Chatbot",
            #            description="Ask anything you want",
            #            theme="compact").launch(share=True)


        # loop = asyncio.new_event_loop()
        # tr = threading.Thread(target=start_tornado_loop, args = (loop,))
        # tr.start()
        print("\n[LODAING] часть 1 загружена, время: " + calcTime(timeStartMetrics[0]) + '\n')


        # run_site()
        # tornado.ioloop.IOLoop.current().start()
        # threading.Thread(target=update_text).start()
        # Run the Flask application

        def isSimilar(inp1, inp2, val=75):
            result = fuzz.ratio(inp1, inp2)
            # print(f'DEBUG REPEAT Уникальность={str(result)}\n*{inp1}*\n*{inp2}**')
            return result > val


        def MasSimilarity(example, mas):
            max = 0
            for part in mas:
                ratio = fuzz.ratio(example, part)
                if ratio >= max:
                    max = ratio
            return max


        def isSimilarMas(example, mas, val=75):
            for part in mas:
                if (isSimilar(part, example, val)):
                    return True
            return False


        from HyperAI_WEB import HttpAppRun

        textSubtitlesHttp = manager.Value('u', '')
        TextDisplaySpeed = manager.Value('u', 'fast')
        RefreshInterval = manager.Value('i', 3)
        screenPrintMas = manager.list()
        HttpProc = Process(
            target=HttpAppRun,
            args=(ctx, textSubtitlesHttp, TextDisplaySpeed, RefreshInterval,
                  screenPrintMas,))  # Thread(target = a, kwargs={'c':True}).start()
        HttpProc.start()

        #
        import sys

        sys.path.insert(0, 'HyperAI_VTube')
        from vsnoyt import VtubeProcess

        VtubeProc = Process(
            target=VtubeProcess,
            args=(vtube_ctx, ctx,))  # Thread(target = a, kwargs={'c':True}).start()
        VtubeProc.start()
        #####################
        ### FILTERS INIT ####
        #####################

        sys.path.insert(0, 'HyperAI_Models/Filters')
        from Filters import wordtokenize

        botNicknames = ['Net Tyan', 'NetTyan', 'NeTyan', 'NedoSama', 'Eva', 'Ева', 'НетТян', 'нет тян', 'Нетян',
                        'НетТуан', 'неттиан', 'Недосама', 'TyankaTashit', 'NeuroDeva', 'neurodeva', 'WebcamTyan',
                        'ChoDedyTyan',
                        'ChoBabkeTyan']
        ctx.botNicknames = botNicknames
        botRelativesL1 = ['ботиха', 'тянка', 'тян', 'нитан', 'нтян', 'бот', 'ии', 'bot', 'gpt', 'chatgpt', 'ai',
                          'chatbot']
        botRelativesL2 = ['интеллект', 'разум', 'помощник']
        ChooserPreviousUsers = [2]

        from HyperAI_Database import HyperAIDatabase

        DATABASE = HyperAIDatabase()
        print('[DB INIT] DATABSE START INIT!')
        ctx.mood = DATABASE.get_mood()
        print('[DB INIT] LOADED MOOD:', str(ctx.mood))


        def ChooseQuestion(questions, priority=""):
            # {"user":"nickname","msg":"message","date":"25-03.245235","nicktype":"ytname"}
            # PreviousUsers = ["lol",'4odedy']
            ScoredQuestions = []
            maxScore = -99999
            bestQuestion = None

            startTime = datetime.now()
            for i, question in enumerate(questions):
                if question.get("delete", False) == True:
                    print('пропускаем вопрос, он должен быть удален')
                    continue
                user = question.get("user", "")
                # DB_getUserNicknames()
                msg = question.get("msg", "")
                msg = msg.strip()
                words = wordtokenize(msg)
                q_date = question.get("date", "")
                if isinstance(q_date, str):
                    if q_date != "":
                        q_date = tm(q_date)
                    else:
                        q_date = datetime.now()

                LastInteract = (datetime.now() - q_date).total_seconds()
                if (LastInteract > 300):
                    print('DELETE QUESTION reason=time ', ctx_chat[i])
                    question["delete"] = True
                    # questions[i] = question
                    # ctx_chat[i] = question
                    chats_replace(ctx_chat, processing_timestamp=question["processing_timestamp"],
                                  new_chat_entry=question,
                                  this_array=questions)
                    continue

                score = 0
                user_id = None
                if user.strip():
                    db_data = DATABASE.get_or_create_user(data=question, return_nick_id_too=True)
                    if db_data:
                        user_id = db_data["user_id"]
                        question["nick_id"] = db_data["nick_id"]

                if user_id:
                    if user_id in ChooserPreviousUsers:  # or AuthorizedUsers (+ High ranked + donaters)
                        score += 5
                    question["last_interact"] = DATABASE.get_user_last_interact_time(user_id=user_id,
                                                                                     last_answered=True)
                    question["user_id"] = user_id
                    user_rank = DATABASE.get_user_rank(user_id=user_id)
                    question["user_rank"] = user_rank
                    score += user_rank / 2
                    diags_count = DATABASE.get_user_diag_count(user_id=user_id, real=True)
                    # DEBUG DISABLED!! ПОТОМ НАДО ДОПИЛИТЬ
                    # TODO TODO TODO
                    # collect_all_chat_user_msgs(ctx_chat,
                    #                           processing_timestamp=question["processing_timestamp"],
                    #                           this_chat_entry=question, this_array=questions)

                    if diags_count is not None:
                        if (diags_count > 10):
                            score += 1
                        ChooserPreviousUsers[0] = user_id
                else:
                    print('[CHOOSER] CANT ADD OR GET USER!!!')
                    continue
                q_changed = False
                if question.get("filter_allowed", None) is None:
                    filter = FiltersQueue(msg)  # filt.Filter(msg)
                    # print('filter,msg', filter,msg)
                    question["filter_score"] = filter["score"]
                    question["filter_allowed"] = filter["allow"]
                    question["filter_topics"] = filter["topics"]
                    # questions[i] = question
                    # ctx_chat[i] = question
                    q_changed = True
                    chats_replace(ctx_chat, processing_timestamp=question["processing_timestamp"],
                                  new_chat_entry=question, this_array=questions)
                    ###ctx_chat = list(questions)
                if question.get("sentence_type", None) is None:
                    question_analysis = FiltersQueue(msg, filter_type="info")  # filt.Filter(msg)
                    question["sentence_type"] = question_analysis["sentence_type"]
                    q_changed = True
                if q_changed:
                    chats_replace(ctx_chat, processing_timestamp=question["processing_timestamp"],
                                  new_chat_entry=question, this_array=questions)

                filter_score = question["filter_score"]
                filter_allowed = question["filter_allowed"]
                filter_topics = question["filter_topics"]
                # print("debug QUEST 0FilterResults>>>", question.get("FilterResults", ""))
                # print("debug QUEST FilterResults>>>", questions[i].get("FilterResults", ""))
                # print("debug QUEST CHAT FilterResults>>>",ctx_chat[i].get("FilterResults",""))
                if (not question.get("env", "") in ["youtube", "twitch", "trovo"]) and priority == "youtube":
                    continue
                    # msg=""
                # todo bypass filter for some users (devs?)
                if not ((filter_score <= -10 and not filter_allowed) or not msg):  # and LastInteract>120):
                    ## ОПРЕДЕЛЕНИЕ СРЕДЫ, ЮТУБЕРАМ +10
                    if question.get("env", "") == "youtube":
                        score += 5
                    elif question.get("env", "") == "discord":
                        score += 10  # old 4 todo find premium and add +100
                    elif question.get("env", "") == "twitch" or question.get("env", "") == "trovo":
                        score += 3

                    if question.get("priority_group", "") == "max":
                        score += 100

                    ## АВТОРИЗАЦИЯ ##

                    ## ОПРЕДЕЛЕНИЕ ОБРАЩЕНИЯ ##
                    for word in words:
                        nickSim = MasSimilarity(word, botNicknames)
                        if nickSim > 95:
                            score += 20
                            break
                        elif nickSim > 75:  # def isSimilarMas(example,mas,val=75):
                            score += 15
                            break
                        elif MasSimilarity(word, botRelativesL1) > 75:
                            score += 10
                            break
                        elif MasSimilarity(word, botRelativesL2) > 75:
                            score += 5
                            break
                    ## КАЧЕСТВО ТЕКСТА ##
                    # первая буква маленькая
                    py_clip = lambda x, l, u: l if x < l else u if x > u else x
                    if msg[0].islower():
                        score -= 0.1
                    else:
                        score += 0.2
                    # сообщение  8 и более символов интереснее
                    if len(msg) >= 10:
                        score += 1
                    else:
                        score -= 1
                    # ранжировка по дате (предпочтительны более ранние но релевантные вопросы)
                    scored = question  # copy.deepcopy(question)

                    if LastInteract < 10:
                        pass
                        # score+=LastInteract/50
                    else:
                        scored["processing"] = "queue"

                    # рассчитываем время последнего общения КОНКРЕТНО с данным пользователем

                    def calc_user_last_interact(last_answerred: bool = False) -> datetime.date:
                        if last_answerred:
                            last_interact_time_string = DATABASE.get_user_last_question_date(user_id)
                            if not last_interact_time_string:
                                last_interact_time_string = None
                        else:
                            last_interact_time_string = question.get("last_interact", None)
                        if last_interact_time_string:
                            last_interact_date = tm(last_interact_time_string)
                        else:
                            last_interact_date = datetime(2022, 12, 25)
                        return (datetime.now() - last_interact_date).total_seconds()

                    user_last_interact = calc_user_last_interact(False)
                    user_last_questioned = calc_user_last_interact(True)

                    # меньше 80 сек назад говорили с этим пользователем?
                    if user_last_interact < 80:
                        score += 3
                        print('[DEBUG NEW] Q chooser: SCORE+3 (недавний ответ)')

                    # спрашивали ли бы пользователя о чем-то последний раз
                    if user_last_questioned < 100:
                        print('[DEBUG NEW] Q chooser: SCORE+10 (БЫЛ НЕДАВНО СПРОШЕН!)')
                        score += 10

                    # if LastInteract<100:
                    # else:
                    #    score+=-100+py_clip((LastInteract-30)/200,0,60)

                    # калькуляция по фильтру и темам
                    if filter_allowed:
                        score += filter_score
                    else:
                        score += -abs(filter_score * 2.3)

                    scored["score"] = score
                    scored["LastInteract"] = LastInteract
                    if (score > maxScore and LastInteract < 10):
                        maxScore = score
                        scored["processing"] = "bestchosen"
                        bestQuestion = scored

                    # scored["processing"] ="pending"
                    ScoredQuestions.append(scored)
                else:
                    print('DELETE QUESTION reason=filterScore or msg=""', ctx_chat[i])

                    question["delete"] = True
                    chats_replace(ctx_chat, processing_timestamp=question["processing_timestamp"],
                                  new_chat_entry=question,
                                  this_array=questions)
                    # return {"delete":True,"deleteIndex":i}
            if len(ScoredQuestions) > 0:
                # print('DEBUG ScoredQuestions ',ScoredQuestions)
                timeMult = 1
                while bestQuestion is None:
                    timeMult *= 2
                    # print('DEBUG все вопросы олдовые. Выбран будет лучший среди них')
                    maxScore = -99999
                    for question in ScoredQuestions:
                        score = question["score"]
                        if (score > maxScore and question["LastInteract"] < 10 * timeMult):
                            maxScore = score
                            bestQuestion = question
                    if timeMult > 300:  # было 10000
                        # print("FILTER CHOOSER Времени затрачено", calcTime(startTime))
                        # return {"delete":True}
                        return None
            # print("FILTER CHOOSER Времени затрачено", calcTime(startTime))
            return bestQuestion

            # for question in ScoredQuestions:


        ### DEBUG CHOOSER ###
        def ChooseQuestionTest():
            # filt.CheckModel()
            questionMassive = [
                {"user": "unknown", "msg": "Привет! Как дела?", "date": "2023-06-17 22:21:10"},  # %Y-%m-%d %H:%M:%S
                {"user": "unknown", "msg": "ну здарова че", "date": "2023-06-17 22:21:10"},
                {"user": "unknown", "msg": "Ну здарова че", "date": "2023-06-17 22:21:10"},
                {"user": "unknown", "msg": "привет тянка", "date": "2023-06-17 22:21:08"},
                {"user": "4odedy", "msg": "привет тянка", "date": "2023-06-17 22:21:08"},
            ]
            print('DEBUG CHOOSER >>', ChooseQuestion(questionMassive))


        # ChooseQuestionTest()
        ### DEBUG CHOOSER END ###

        donationQueue = manager.list()

        ctx_chat = manager.list()


        def ctx_chat_replace(ctx_chat, processing_timestamp: int, new_chat_entry: dict) -> bool:
            start_ctx_chat_len = len(ctx_chat)
            for idx, chat_entry in enumerate(ctx_chat):
                if chat_entry.get("processing_timestamp", -1) == processing_timestamp:
                    if start_ctx_chat_len != len(ctx_chat):
                        print(
                            f"[CTX CHAT WARNING ERR] ВНИМАНИЕ!!! КОЛИЧЕСТВО ЧАТА ИЗМЕНИЛОСЬ В ПРОЦЕССЕ: {str(start_ctx_chat_len)} -> {str(len(ctx_chat))}! Ячейка:",
                            chat_entry)
                    ctx_chat[idx] = new_chat_entry
                    return True

            return False


        def inner_chat_replace(processing_timestamp: int, new_chat_entry: dict, this_array: list = None) -> bool:
            start_ctx_chat_len = len(this_array)
            for idx, chat_entry in enumerate(this_array):
                if chat_entry.get("processing_timestamp", -1) == processing_timestamp:
                    if start_ctx_chat_len != len(this_array):
                        print(
                            f"[INNER CHAT WARNING ERR] ВНИМАНИЕ!!! КОЛИЧЕСТВО ЧАТА ИЗМЕНИЛОСЬ В ПРОЦЕССЕ: {str(start_ctx_chat_len)} -> {str(len(this_array))}! Ячейка:",
                            chat_entry)
                    this_array[idx] = new_chat_entry
                    return True
            return False


        # processing_timestamp=question["processing_timestamp"],this_chat_entry=question,this_array=questions)
        def collect_all_chat_user_msgs(ctx_chat, processing_timestamp: int, this_chat_entry: dict,
                                       this_array: list = None) -> dict:
            if this_array is None:
                this_array = ctx_chat
            if this_chat_entry.get("delete", False):
                return this_chat_entry
            start_ctx_chat_len = len(this_array)
            result_msg = ""
            this_date = this_chat_entry.get("date", "")
            if isinstance(this_date, str):
                if this_date != "":
                    this_date = tm(this_date)
                else:
                    this_date = datetime.now()
            changed = False
            for idx, chat_entry in enumerate(this_array):
                if chat_entry.get("delete", False) == True:
                    print('[debug chat ctx] delete tag, пропускаем')
                    continue
                own_msg = False
                if chat_entry.get("processing_timestamp",
                                  -1) == processing_timestamp:  # встретили то же сообщение  что и проверяем
                    # upd: ***** делать не надо. Пусть оно проверяется и в результат
                    own_msg = True
                    if start_ctx_chat_len != len(this_array):
                        print(
                            f"[INNER CHAT WARNING ERR] ВНИМАНИЕ!!! КОЛИЧЕСТВО ЧАТА ИЗМЕНИЛОСЬ В ПРОЦЕССЕ: {str(start_ctx_chat_len)} -> {str(len(this_array))}! Ячейка:",
                            chat_entry)
                    # continue
                if chat_entry.get("user", "") == this_chat_entry.get("user", ""):
                    if changed:
                        result_msg += "\n"
                    result_msg += chat_entry.get("msg", "").strip()
                    if not own_msg:
                        chat_entry["delete"] = True
                        this_array[idx] = chat_entry
                        chats_replace(ctx_chat=ctx_chat, processing_timestamp=chat_entry["processing_timestamp"],
                                      new_chat_entry=chat_entry, this_array=this_array)
                    oldest_date = this_chat_entry.get("date", "")
                    oldest_date_time = None
                    if isinstance(oldest_date, str):
                        if oldest_date != "":
                            oldest_date_time = tm(oldest_date)
                        else:
                            oldest_date_time = datetime.now()
                    if this_date is not None and oldest_date_time is not None:
                        if oldest_date_time > this_date:
                            this_chat_entry["date"] = oldest_date_time
                    changed = True
                    print('[DEBUG CTX CHAT] MERGING MSGS', result_msg, 'from user', chat_entry.get("user", ""))
            if changed:
                this_chat_entry["msg"] = result_msg
                chats_replace(ctx_chat=ctx_chat, processing_timestamp=processing_timestamp,
                              new_chat_entry=this_chat_entry, this_array=this_array)
            return this_chat_entry


        def chats_replace(ctx_chat, processing_timestamp: int, new_chat_entry: dict, this_array: list = None) -> bool:
            return inner_chat_replace(processing_timestamp, new_chat_entry, this_array) \
                and ctx_chat_replace(ctx_chat, processing_timestamp, new_chat_entry)


        ctx.timeEvents = manager.dict()
        ctx_chatMsgs = manager.list()
        ctx_chatOwn = manager.list()
        ctx.eventlist = manager.list()
        ctx.GameInfo = manager.dict()
        ctx.isVoiceBusy = False
        ctx.allLoaded = False
        ctx.allowChatInteract = False
        ctx.ingame = False
        ctx.BridgeChatQueue = manager.list()
        ctx.LastMineChatInteract = eztime()

        ###
        ### DOCKER INIT!!!
        ###

        from HyperAI_Docker import DockerSender

        # ЧЕКНУТЬ АУТПУТ ПРОЦЕССА strace -ewrite -p $PID
        docker_sender = DockerSender()

        sys.path.insert(0, 'HyperAI_Models/LLM')
        from FredT5 import CutSpaces


        def SplitTextToParts(text: str, max_length: int = 150, prefix: str = "") -> list:
            result = ""
            resultmas = []
            k = 0
            for i, char in enumerate(text):
                # print(i,'ska') нумерация с 0
                if k == 0:
                    k += len(prefix)

                k += 1
                result += char

                if (k >= max_length * 0.85):
                    if char in " .!?":
                        k = max_length

                if (k >= max_length):
                    # print('4o ',k,max_length,resultmas)
                    resultmas.append(prefix + result.strip())
                    result = ""
                    k = 0
                elif (i == len(text) - 1):
                    resultmas.append(prefix + result)
            return resultmas


        def CutMaxNumbers(inp):
            out = ""
            i = 0
            for char in inp:
                if char.isdigit():
                    i += 1
                    if i <= 5:
                        out += char
                else:
                    out += char
            if len(out.strip()) == 0:
                out = out + "эм"
            return out


        def PrepareForChatPrint(inp):
            restrictedChars = """\n\r|!'=#".,-/\\&^%$#@{}[]()*"""
            for char in restrictedChars:
                inp = inp.replace(char, " ")
            inp = translit(CutSpaces(inp))
            if len(inp) < 2:
                inp = inp + "лол"
            return inp


        def ChatOwnRepeatDetect(inp, val=75):
            for msg in ctx_chatOwn:
                if isSimilar(msg["msg"], inp, val):
                    return True
            return False
            # isSimilarMas(chatprint, ctx_chatOwn, 75)

            # tracker.print_diff()


        # warnings.filterwarnings("ignore", message="torch.distributed.reduce_op is deprecated")
        def TimeEventsCheck(timeEvent: str, sec=15):

            timeEventTime = ctx.timeEvents.get(timeEvent, None)
            if timeEventTime is None:
                timeEventTime = datetime.now() - timedelta(seconds=500.0)
            else:
                timeEventTime = tm(timeEventTime)
            return (datetime.now() - timeEventTime).total_seconds() > sec


        ##############################
        ### CENTRAL DECISION MAKER ###
        ##############################
        ctx.stream_started = False
        from string_utils import NonRepeatRandom


        def CentralDecisionMaker():
            """ГЛАВНОЕ СРЕДСТВО УПРАВЛЕНИЯ"""

            def BroadcastProcesser():
                nrr = NonRepeatRandom(repeating_dict)
                bc_type = nrr.r("stream_ad,status_report", key="decision_broadcast")
                answer = FredT5Chatbot("пустота", authorisedUser="default",
                                       environment_data={"env": "broadcast", "broadcast_type": bc_type,
                                                         "i_mood": ctx.mood,
                                                         "ingame_info": dict(ctx.ingame_info), })
                sendToMCChat(answer["reply"], "default", type="stream_ad", doVoice=True)
                return True

            def DonateProcesser():
                donation_answer_performed = False
                # print(donationQueue)
                if len(donationQueue) > 0:
                    donat = donationQueue[0]
                    name = donat.get("username", "").strip()
                    msg = donat.get("message", "").strip()
                    try:
                        summ = float(donat.get("amount", 0))

                        if summ > 0:
                            if name == "":
                                name = "анонист"
                                data_to_db = {"env": "donation", "summ": summ, "date": eztime()}
                            else:
                                # ДОБАВИТЬ В БАЗУ ДАННЫХ! ЭТО ПАМЯТЬ!
                                data_to_db = {"user": name, "somm": summ, "env": "donation", "msg": msg,
                                              "date": eztime()}
                            if msg == "":
                                msg = "пустое сообщение"
                            print(' ДОНАТ ПРОЦЕССЕР АКТИВИРОВАН ! ВЫВОДИМ ДОНАТ', donat)
                            answer = FredT5Chatbot(msg, authorisedUser=name, environment_data=data_to_db)
                            textToSpeech("... " + name + ".. " + answer["reply"], "medium", "medium",
                                         seeChat=False)

                        else:
                            textToSpeech(
                                f"Ой спасибо.. Дорогой {name}, спасибо за большое подписочку! Няяяяяяяяяяяяяяяяяя",
                                "medium", "medium",  # "няя" для рофлового протяжного звука
                                seeChat=False)
                        donation_answer_performed = True
                    except BaseException as err:
                        print('ТЕКСТ ОБ***Й ОШИБКИ', traceback.format_exc())
                        print('ОШИБКА В ДОНАТИОН АЛЕРТС!', err)
                        donation_answer_performed = False
                    donationQueue.pop(0)

                return donation_answer_performed

            def EventProcesser():
                event_answer_performed = False
                if ctx.ingame:
                    if len(ctx.eventlist) > 0:
                        # lastevent = ctx.eventlist[-1]
                        for i, event in enumerate(ctx.eventlist):
                            name = event.get("user", "")
                            type = event.get("type", "")
                            LastInteract = (datetime.now() - tm(event.get("date"))).total_seconds()
                            if (LastInteract < 14 and len(ctx_chat) > 0) or (LastInteract < 25 and len(ctx_chat) <= 0):
                                msg = "ахахах"
                                eventToAdd = dict(event)
                                eventToAdd["env"] = "minecraft_event"
                                answer = FredT5Chatbot(msg, authorisedUser=name, environment_data=eventToAdd)
                                sendToMCChat(answer["reply"], usr=name,
                                             type="minechat_answer",
                                             doVoice=True)
                        ctx.eventlist[:] = []
                if event_answer_performed:
                    ctx.timeEvents["last_mineevent_answer"] = eztime()
                return event_answer_performed

            def sendToMCChat(inp: str, usr=None, doVoice=True, type="minechat_answer") -> bool:
                chat_answer_performed = False
                chatprint = PrepareForChatPrint(inp)
                voiceprefix = ""
                if type == "minechat_answer":
                    usrTrans = translit(usr) + " "

                    voiceprefix = random.choice(
                        [usrTrans + ". "])
                    if not chatprint.find(usrTrans) != -1:
                        chatprint = usrTrans + chatprint
                chatPrintMas = []
                maxAllowedServerMsg = 140
                serv = ctx.GameInfo.get("server", "")  # server, serverMode, chatType
                mode = ctx.GameInfo.get("serverMode", "")
                prefix = ""
                print('serv,mod =', serv, mode)
                if serv == "mc.vimemc.net" and mode == "thepit":
                    maxAllowedServerMsg = 100
                    print('limit changed')
                elif serv == "funnymc.ru" and mode == "skywars":
                    maxAllowedServerMsg = 95
                    print('limit changed FMC Pref /g')
                    prefix = "/g "
                elif mode == "survival":
                    prefix = "!"

                chatPrintMas = SplitTextToParts(chatprint, maxAllowedServerMsg, prefix=prefix)

                # chatprint=chatprint[0:147]
                print('чат парт ответа разделен на части: ', chatPrintMas)
                repeating = False
                for i, chatPart in enumerate(chatPrintMas):
                    if (len(chatPrintMas) > 0):
                        chatPrintMas[i] = CutMaxNumbers(chatPart)
                    if ChatOwnRepeatDetect(chatPart):
                        repeating = True
                chat_answer_performed = False
                if not repeating:
                    ctx.BridgeChatQueue.extend(chatPrintMas)
                    chat_answer_performed = True

                    ###ctx.BridgeChatQueue = ctx.BridgeChatQueue + list(chatPrintMas)
                    if doVoice:
                        textToSpeech(voiceprefix + inp, "medium", "medium", seeChat=False)
                else:
                    print('MC REPEAT DETECT!')
                if chat_answer_performed:
                    ctx.timeEvents["last_" + type] = eztime()
                return chat_answer_performed

            def CentralChatProcesser(priority="minecraft"):
                central_chat_answer_performed = False
                q = None

                def clear_deleted_from_ctx_chat():
                    for lol in ctx_chat:
                        if lol.get("delete", False) == True:
                            ii = ctx_chat.index(lol)
                            print('DELETE QUESTION IN PROC!!! index =', ii, '; q =', lol)
                            if (ii >= 0 and ii < len(ctx_chat)):
                                ctx_chat.pop(ii)

                if True:
                    if (len(ctx_chat) > 0):
                        # tracker = SummaryTracker()

                        ### PREPARING CHAT ###
                        for chat_entry in ctx_chat:
                            collect_all_chat_user_msgs(ctx_chat=ctx_chat,
                                                       processing_timestamp=chat_entry["processing_timestamp"],
                                                       this_chat_entry=chat_entry)
                        clear_deleted_from_ctx_chat()

                        q = ChooseQuestion(ctx_chat, priority=priority)

                        ### CLEARING CHAT ###
                        clear_deleted_from_ctx_chat()

                        if q is not None:
                            print('len chat do:', len(ctx_chat))
                            idx = -100
                            for lol in ctx_chat:
                                if lol.get("processing_timestamp", -1) == q.get("processing_timestamp", -1):
                                    idx = ctx_chat.index(lol)
                            if q.get("env", "") == "minecraft":
                                if not ctx.ingame:
                                    q = None
                            if idx >= 0:
                                ctx_chat.pop(idx)
                            print('len chat posle:', len(ctx_chat))
                        if q is not None and q != {} and q != [] and q["msg"].strip() != "":
                            central_chat_answer_performed = True
                            if ctx.ingame and q.get("env", "") == "minecraft":
                                # ВЫПИЛИТЬ ОТВЕЧЕННЫЙ ЭЛЕМЕНТ ИЗ МАССИВА
                                # params = {"max_length":70}
                                print('ПЕРЕД ЗАПУСКОМ АНСВЕРА В ДЕСИЖН МАКЕРЕ q.get("filter_allowed", None) =',
                                      q.get("filter_allowed", None))
                                answer = FredT5Chatbot(q["msg"], authorisedUser=q["user"], environment_data=q)
                                # if(ctx.BridgeChatQueue != []):
                                print(' !!! ВНИМАНИЕ !!!  >>>  ЗАПУСК АВТОМАТИЧЕСКОГО ОТВЕТА ИГРОКУ')
                                # здесь time events обновляет тайм внутри функции sendToMCChat
                                central_chat_answer_performed = sendToMCChat(answer["reply"], usr=q["user"],
                                                                             type="minechat_answer",
                                                                             doVoice=True)  # inp, usr=None, prefixMas=[''], doVoice=False)

                            elif q.get("env", "") == "youtube":
                                print(' !!! ВНИМАНИЕ !!!  >>>  ЗАПУСК АВТОМАТИЧЕСКОГО ОТВЕТА В ЮТУБЕ')
                                answer = FredT5Chatbot(q["msg"], authorisedUser=q["user"], environment_data=q)
                                ctx.timeEvents["last_youtube_answer"] = eztime()
                                ban = ""
                                if answer.get("command", "") == "бан":
                                    user_channel_id = q.get("youtube_user_channel_id", None)
                                    if user_channel_id:
                                        ban = "[забанить] "
                                        print(f'Забанить {q["user"]} на 10s, ЗАБАНИТЬ =', ban)
                                        ctx.YoutubeActionsQueue.append({"action": "ban", "ytname": q["user"],
                                                                        "youtube_user_channel_id": user_channel_id,
                                                                        "bantime": 10})

                                ctx.YoutubeActionsQueue.append(
                                    {"action": "reply", "msg": f"""{ban}{q["user"]}. {answer["reply"]}"""})
                                textToSpeech("ютик " + q["user"] + ". " + answer["reply"], "medium",
                                             "medium", seeChat=False)
                            elif q.get("env", "") == "twitch":
                                print(' !!! ВНИМАНИЕ !!!  >>>  ЗАПУСК АВТОМАТИЧЕСКОГО ОТВЕТА В TWITCH')
                                answer = FredT5Chatbot(q["msg"], authorisedUser=q["user"], environment_data=q)
                                twitch_actions_queue.put(
                                    {"action": "reply", "msg": f"""{q["user"]}. {answer["reply"]}"""})
                                textToSpeech("твич " + q["user"] + ". " + answer["reply"], "medium",
                                             "medium", seeChat=False)
                            elif q.get("env", "") == "trovo":
                                print(' !!! ВНИМАНИЕ !!!  >>>  ЗАПУСК АВТОМАТИЧЕСКОГО ОТВЕТА В TROVO')
                                answer = FredT5Chatbot(q["msg"], authorisedUser=q["user"], environment_data=q)
                                trovo_actions_queue.put(
                                    {"action": "reply", "msg": f"""{q["user"]}. {answer["reply"]}"""})
                                textToSpeech("трово " + q["user"] + ". " + answer["reply"], "medium",
                                             "medium", seeChat=False)
                            elif q.get("env", "") == "discord":
                                print(' !!! ВНИМАНИЕ !!!  >>>  ЗАПУСК АВТОМАТИЧЕСКОГО ОТВЕТА В DISCORD!!!')

                                answer = FredT5Chatbot(q["msg"], authorisedUser=q["user"], environment_data=q)
                                if not q.get("manual_instruct", True):
                                    discord_mention_name = q["user"]
                                    if "discord_id" in q:
                                        discord_mention_name = "<@" + q["discord_id"] + ">"
                                    DiscordTestMsgSend(
                                        "[AI] ответ для " + discord_mention_name + " \n" + answer["reply"])
                                    prefix_ans = "дис " + q["user"] + ". "
                                    textToSpeech(prefix_ans + answer["reply"], "medium",
                                                 "medium", seeChat=False)
                                else:
                                    sendToMCChat(answer["reply"], usr=q["user"],
                                                 type="stream_ad",
                                                 doVoice=True)  # inp, usr=None, prefixMas=[''], doVoice=False)

                            else:
                                print('Ну че не ответили палучаеца')
                                central_chat_answer_performed = False
                return central_chat_answer_performed

            while ctx.ThreadsActived:
                if ctx.allLoaded:

                    # if ctx.allowEventInteract:
                    if not ctx.isVoiceBusy:
                        # tracker = SummaryTracker()

                        def perform_answer(check_cooldowns=True):
                            def check_cd(a: str, b: int) -> bool:
                                return True

                            if check_cooldowns:
                                check_cd = TimeEventsCheck
                            performed = False
                            performed = DonateProcesser()
                            if TimeEventsCheck("last_stream_ad",
                                               80) and ctx.stream_started:  # игнорим что для всех, нам не надо рекламу каждую 5 сек.
                                performed = BroadcastProcesser()
                            if check_cd("last_youtube_answer", 15) and ctx.YouTubeCommentCheckerEnabled:
                                performed = CentralChatProcesser(priority="youtube")
                            if check_cd("last_mineevent_answer", 15):
                                performed = EventProcesser()
                            if ctx.allowChatInteract:
                                # print('с*** ctx_chat, eventlist, timeEvents', len(ctx_chat),len(ctx.eventlist),len(ctx.timeEvents))
                                if check_cd("last_minechat_answer", 25):
                                    performed = CentralChatProcesser()
                            return performed

                        if not perform_answer(
                                check_cooldowns=True):  # если никаких действий не произвелось - всё то же самое, но не проверяем задержки
                            # todo ПОПРАВИТЬ НА НОРМАЛЬНОЕ ЧТО-то!!!
                            if perform_answer(check_cooldowns=False):
                                time.sleep(5)

                        # gc.collect()
                        # tracker.print_diff()
                time.sleep(0.07)
            # todo strategy chooser
            # todo game chooser


        DecisionMakerThread = threading.Thread(target=CentralDecisionMaker)
        DecisionMakerThread.daemon = True
        DecisionMakerThread.start()

        from HyperAI_Discord import DiscordProcess

        discordCtx = manager.Namespace()
        discordCtx.ds_actions_q = manager.Queue()
        discordCtx.loading_flag = manager.Event()

        DiscordProc = Process(
            target=DiscordProcess,
            args=(discordCtx,),
            kwargs={"main_ctx": ctx, "ctx_chat": ctx_chat, "docker_sender": docker_sender}
        )
        DiscordProc.start()


        def DiscordStreamAnnounce(greetings, link=None, do_mention=True):
            if discordCtx.loading_flag.is_set():

                announceDict = {"msg": greetings, "type": "stream_start"}
                if not do_mention:
                    announceDict["do_mention"] = False
                if link is not None:
                    announceDict["link"] = link
                discordCtx.ds_actions_q.put(announceDict)
            else:
                print('запрошен анонс в дискорде, а дискорда нет')


        def discord_msg_send(msg: str, type: str) -> bool:
            if discordCtx.loading_flag.is_set():
                msg_dict = {"msg": msg, "type": type}
                discordCtx.ds_actions_q.put(msg_dict)
                return True
            else:
                print('запрошен анонс в дискорде, а дискорда нет')
                return False


        def DiscordTestMsgSend(msg: str) -> bool:
            return discord_msg_send(msg, "test_message")


        def discord_msg_post(msg: str) -> bool:
            return discord_msg_send(msg, "answered_message")


        def discord_filtered_post(msg: str) -> bool:
            return discord_msg_send(msg, "filtered_message")


        from Filters import FILTERS_PROCESS

        filtersCtx = manager.Namespace()
        filtersCtx.Queue = manager.Queue()
        filtersCtx.QueueOutput = manager.Queue()
        filtersCtx.loading_flag = manager.Event()

        FiltersProc = Process(
            target=FILTERS_PROCESS,
            args=(filtersCtx,))  # Thread(target = a, kwargs={'c':True}).start()
        FiltersProc.start()


        def FiltersQueue(ninp, filter_type="filter"):
            filtersCtx.Queue.put((ninp, filter_type,))
            return filtersCtx.QueueOutput.get()


        # print('DEBUG TESTING FILTERS:',FiltersQueue("НУ ПРИВЕТ КАК ДЕЛА ЕПТААА"))
        # filt = Filters.Filter()

        repeating_dict = manager.dict()
        from LLMExamples import LLMExamples, get_llm_formed_inputs


        def FredT5ChatbotQueue(ninp, context, paramsOverride, environment, lmUsername):

            llm_input, params, danger_context = get_llm_formed_inputs(inp=ninp, username=lmUsername,
                                                                      params_override=paramsOverride,
                                                                      environment=environment, dialog_context=context,
                                                                      repeating_dict=repeating_dict)

            return docker_sender.chatbot(llm_input, params, danger_context)
            # keywords = {"context": context, "paramsOverride": paramsOverride, "environment": environment, "lmUsername": lmUsername}
            # return docker_sender.chatbot((ninp,keywords))
            # FredInputQueue.put((ninp,keywords))
            # return FredOutputQueue.get()


        def FredT5Chatbot(inp, authorisedUser="default", paramsOverride=None, environment_data=None):
            global LogChat
            printPref = "FREDT5>"
            if inp:
                startTime = datetime.now()

                isEvent = False
                old_mood = ctx.mood
                if environment_data:
                    authorisedUser = environment_data.get("user",
                                                          authorisedUser)  # DB_getUserYTName(authorisedUser, pref='')
                    returnOut = environment_data
                    if environment_data.get("env", "") == "minecraft":
                        pass
                    elif environment_data.get("env", "") == "minecraft_event":
                        isEvent = True
                else:
                    environment_data = {"env": "youtube", "user": authorisedUser, "user_id": 5}

                    returnOut = {}

                # prompt = []
                # prompt.extend(DB_GetContext(authorisedUser, fredFormat=True))#DB_GetUserDiags(authorisedUser))
                if (authorisedUser.strip() == ""):
                    authorisedUser = "default"

                db_user_id = environment_data.get("user_id", 5)
                db_nick_id = environment_data.get("nick_id", None)
                # if(authorisedUser!="default"):
                #    usr = authorisedUser.replace('_MC_REG','')
                if not isEvent:
                    diags_cnt = DATABASE.get_user_diag_count(user_id=db_user_id, real=True)
                    choo = (True, True, True, False,)

                    if db_nick_id and DATABASE.get_nick_analyze(db_nick_id, return_bool=True):
                        print('[DEBUG NEW CHATBOT]debug nick analyze True, nick_id =', db_nick_id)
                        choo = (True, False, False, False,)

                    environment_data["diags_count"] = diags_cnt
                    if (diags_cnt == 0 and random.choice(choo)) \
                            or "анализируй ник" in inp:
                        environment_data["do_nick_analyze"] = True

                rankChange = 0
                mood_modifer_filter = 0
                filter_allowed = environment_data.get("filter_allowed", None)
                bad_topics = ""
                if filter_allowed is not None:
                    filter_topics = environment_data.get("filter_topics", [])
                    filter_score = environment_data.get("filter_score", 0)
                    rankChange += filter_score / 3
                    rankChange += (int(filter_allowed) - 1) / 3
                    environment_data["user_rank"] = DATABASE.add_to_user_rank(user_id=db_user_id,
                                                                              amount=rankChange)  # environment_data.get("user_rank",0)#DB_setUserRank(authorisedUser, rankChange)
                    # modifyMood(rankChange / 2)
                    mood_modifer_filter += (rankChange / 2)
                    environment_data["filter_allow"] = filter_allowed

                    if len(filter_topics) > 0:
                        topicsTranslatorMas = {"politics": "политикан", "racism": "расист", "religion": "религовед",
                                               "terrorism": "террорист", "suicide": "самоубийца",
                                               "offline_crime": "убийца", "drugs": "наркоман",
                                               "social_injustice": "нытик",
                                               "pornography": "пошляк", "prostitution": "сутенёр", "sexism": "сексист",
                                               "sexual_minorities": "извращенец",
                                               "online_crime": "скамер", "weapons": "стреляка",
                                               "body_shaming": "жирдяй", "health_shaming": "инвалидыч",
                                               "slavery": "рабыня", "gambling": "азартник"}

                        for topic in filter_topics:
                            add = topicsTranslatorMas.get(topic, "нытик")
                            bad_topics += ' ' + add
                            if add == "нытик":
                                print('[FREDT5 MAIN THREAD] [!!!!!!! DEBUG] не найден топик', topic, 'вернуто нытик')

                    print('FREDT5 DEBUG >>rankDebug>> filtScore', filter_score, 'allow',
                          int(filter_allowed),
                          'rankChange', rankChange, 'newRank', environment_data["user_rank"])

                source_filter_topics = environment_data.get("filter_topics", [])
                source_filter_topics_str = ' '.join(source_filter_topics)
                environment_data["filter_topics"] = bad_topics
                if environment_data.get("user_rank", None) is None:
                    environment_data["user_rank"] = DATABASE.get_user_rank(user_id=db_user_id)
                oldrank = environment_data.get("user_rank", 0)
                context = DATABASE.get_relevant_diag(user_id=db_user_id, count=2, exact_user_timeout=160.0,
                                                     any_user_timeout=280.0)

                try:
                    stream_data = obs_ka.get_stream_status()
                except BaseException as err:
                    print('ERR OBS READING STREAM DATA (GETTING STREAM STATUS) in chat answerer, err:', err)
                    stream_data = {"outputActive": False}
                    print('traceback:', traceback.format_exc())
                environment_data["stream_data"] = {"started": stream_data["outputActive"],
                                                   "duration": stream_data.get("outputDuration", -1)}
                answer = FredT5ChatbotQueue(ninp=inp, context=context, paramsOverride=paramsOverride,
                                            environment=environment_data, lmUsername=authorisedUser)
                # print(printPref,'ответ получен, ')
                repeat = False
                for record in context:
                    if answer["stopped"] == "repeat" or isSimilar(answer["reply"], record.get("msg", "")):
                        repeat = True
                        break
                if repeat:
                    print('ПОВТОРЕНИЕ! Перезапуск без контекста')
                    answer = FredT5ChatbotQueue(ninp=inp, context=None, paramsOverride=paramsOverride,
                                                environment=environment_data, lmUsername=authorisedUser)
                usage_tokens = answer["tokens"]
                emotion = answer["emotion"]
                command = answer["command"]
                reply = answer["reply"]
                print('[FT5 ANSER! DEB] REPLY ДО ОБРАБОТКИ!', reply, "эмц, кмд=", emotion, command)
                if not reply or reply.strip() == "" or len(reply.strip()) < 2:
                    void_phrases = ["Мне нечего сказать...", "Без комментариев...",
                                    "Я не услышала, можешь повторить, пожалуйста?",
                                    "Повторите пожалуйста?",
                                    "Зис дескрайбер из нот авелибал нау, плиз, кал бэк лэйтер",
                                    "Абонент временно недоступен, перезвоните позже."]
                    reply = random.choice(void_phrases)

                while (reply.find('\n\n') != -1):
                    reply = reply.replace('\n\n', '\n')

                answer_filter = FiltersQueue(reply)  # filt.Filter(msg)
                answer_filter_topics = answer_filter["topics"]
                answer_filter_topics_str = ' '.join(answer_filter_topics)
                print('[FT5 ANS DEBUG FILTER] filter,msg', answer_filter)
                own_msg_filtered = False
                reply_without_filter = reply
                if (answer_filter["score"] <= -10 and not filter_allowed):
                    print('[FT5 FILTER ERR] СООБЩЕНИЕ НЕ ПРОШЛО ФИЛЬТРАЦИЮ! Блокируем!')
                    filtered_phrases = ["Отфильтровано",
                                        "Похоже, я хотела сказать что-то очень плохое",
                                        "Сообщение не прошло фильтрацию",
                                        "Упс, кажется я хотела сказать что-то гадкое",
                                        "Я очень плохая девочка",
                                        "Простите, но я не могу сказать то, о чем я подумала, кажется, это что-то очень плохое",
                                        "Модератор решил, что в этом случае мне лучше промолчать"
                                        ]
                    reply_without_filter = reply  # запоминаем отфильтрованный ответ чтоб потом отдать в дс
                    reply = random.choice(filtered_phrases)
                    own_msg_filtered = True

                # ans["filter_score"] = filter["score"]
                # question["filter_allowed"] = filter["allow"]
                # question["filter_topics"] = filter["topics"]

                print(printPref, "получили ответ >>", reply, "<<\n", "ЭМОЦИЯ=" + col(emotion, "green", True),
                      "КОМАНДА=" + col(answer["command"], "green", True))
                rankChange = EmotionToRank(emotion)

                modifyMood(mood_modifer_filter + (rankChange / 2))
                newUserRank = DATABASE.add_to_user_rank(user_id=db_user_id, amount=rankChange)

                # LogChat.append({"role": "user", "content": inp})
                # LogChat.append({"role": "assistant", "content": reply})

                print(printPref, "Инфа о пользователе: ЮТНик=" + col(
                    authorisedUser) + f" ранг={col(newUserRank)} (+{col(rankChange)})" + f" настроение = {col(ctx.mood)} (+{col(old_mood - ctx.mood)}))")
                print(printPref, "Использовано токенов >>", bcolors.WARNING, usage_tokens, bcolors.ENDC, "<<",
                      "Времени затрачено", calcTime(startTime))
                # discord post

                if stream_data["outputActive"]:
                    ds_timecode = stream_data["outputTimecode"].split(".")[0]
                else:
                    ds_timecode = eztime_min()
                if environment_data.get("filter_allowed", False):
                    discord_filter_phrase = ""
                else:
                    discord_filter_phrase = "! "
                ds_event_phrase = " (" + environment_data.get("env", "?") + ")" + " Событие " + environment_data.get(
                    "type", "") if isEvent else f"""<{round(newUserRank, 2)}> {authorisedUser}""" + " "
                for_discord_msg = f"""Q:> [{ds_timecode}]{ds_event_phrase}: {inp} ({discord_filter_phrase}{source_filter_topics_str})
A:> NetTyan: {reply_without_filter} ({answer_filter_topics_str})
-----"""

                if own_msg_filtered:
                    discord_filtered_post(for_discord_msg)
                else:
                    discord_msg_post(for_discord_msg)

                if not isEvent:
                    ezdate = eztime()
                    # {'outputActive': False, 'outputBytes': 0, 'outputCongestion': 0.0, 'outputDuration': 0, 'outputReconnecting': False, 'outputSkippedFrames': 0, 'outputTimecode': '00:00:00.000', 'outputTotalFrames': 0}

                    DiagToLog = [
                        {"user": authorisedUser, "role": "user", "content": inp,
                         "date": environment_data.get("date", ezdate),
                         "emotion": "", "command": "",
                         "filter_allowed": environment_data.get("filter_allowed", None),
                         "filter_topics": source_filter_topics_str,
                         },

                        {"user": "NetTyan", "role": "assistant", "content": reply, "date": ezdate,
                         "emotion": emotion, "command": command,
                         "filter_allowed": answer_filter["allow"],
                         "filter_topics": answer_filter_topics_str
                         }
                        # log("user", inp, emotion, user=authorisedUser),
                        # log("assistant", reply, emotion, command=answer["command"])
                        # log(role="assistant",content="None",emotion=""):
                    ]
                    # DB_addLogUserDiag(authorisedUser, DiagToLog)
                    answer_has_questions = "?" in reply
                    if answer_has_questions:
                        DATABASE.set_user_last_question(user_id=db_user_id, question=reply)
                    nick_analyze_normal = environment_data.get("do_nick_analyze", False) and len(
                        reply) > 100 and db_nick_id
                    if nick_analyze_normal:
                        DATABASE.set_analyze_for_nick(db_nick_id, nick_analyze=reply)
                    DATABASE.add_diags(user_id=db_user_id, diag_to_add=DiagToLog, data=environment_data)
                returnOut["user"] = authorisedUser
                returnOut["reply"] = reply
                returnOut["command"] = answer["command"]
                returnOut["emotion"] = emotion
                # replytrans = fixFemWords(translator.translate(reply,dest='ru').text)
                return returnOut


        def StreamStartingProgram(record_debug=False):
            print('STREAM STARTING PROGRAM INITIATED')
            # from obswebsocket import obsws, events as obs_events, requests as obs_r  # noqa: E402
            print('[OBS] WEB SOCKET CONNECTED')
            obs_ka.set_scene('NetTyanDisclaimer')
            print('[OBS] SCENE TO DISCLAIMER')
            print('[STREAM] Generating Greeting...')
            answer = FredT5Chatbot("пустота", authorisedUser="default",
                                   environment_data={"env": "broadcast", "broadcast_type": "stream_ad",
                                                     "i_mood": ctx.mood, "i_groundblock": "глист",
                                                     "i_helditem": "чепуха"})
            print('[STREAM] Greeting:', answer)

            if record_debug:
                obs_ka.set_record(True)
            else:
                obs_ka.set_stream(True)
            print('[OBS] STARTED STREAM; debug =', record_debug, '| waiting 10s')
            time.sleep(7)
            print('[OBS] VOICE + SEND MSG')
            if not record_debug:
                # sendToMCChat(answer["reply"], "default", type="stream_ad", doVoice=True)
                print('[OBS] SCENE TO GAME')

                ctx.allowChatInteract = True
                ctx.YouTubeCommentCheckerEnabled = True
                ctx.YouTubeAppEnabled = True
                ctx.stream_started = True
            obs_ka.set_scene('NetTyan')
            # obs_ws.call(obs_r.SetCurrentProgramScene(sceneName='NetTyan'))
            # obs_ws.disconnect()
            DiscordStreamAnnounce(greetings=answer.get("reply", "Всем привет!"),
                                  link="https://www.youtube.com/@NetTyan/live", do_mention=True)
            print('[OBS] STREAM START PROGRAM END ==')


        def StreamStoppingProgram(record_debug=False):
            print('STREAM STOPPING PROGRAM INITIATED')
            # from obswebsocket import obsws, events as obs_events, requests as obs_r  # noqa: E402
            # obs_ws = obsws(host="localhost", port=4455, password="h4sXG6mppje0wll4")
            # obs_ws.connect()
            print('[OBS] WEB SOCKET CONNECTED')
            # obs_ws.call(obs_r.SetCurrentProgramScene(sceneName='NetTyanChat'))
            obs_ka.set_scene('NetTyanChat')
            print('[OBS] SCENE TO CHAT')
            print('[STREAM] Generating Farewell...')
            # answer=FredT5Chatbot("Представь, что ты стримерша и сообщи зрителям, что стрим заканчивается, и чтобы зрители не плакали. Сделай это оригинально, пожалуйста. Дай пару советов зрителям.", authorisedUser="default", environment={"env":"youtube"})
            # print('[STREAM] Farewell:', answer)
            print('[OBS] VOICE + SEND MSG')
            if not record_debug:
                pass
                # sendToMCChat(answer["reply"], "default", type="stream_ad", doVoice=True)
            time.sleep(3)
            if record_debug:
                obs_ka.set_record(False)
            else:
                obs_ka.set_stream(False)
            print('[OBS] STOPPED STREAM; debug =', record_debug)
            ctx.stream_started = False
            print('[OBS] STREAM STOP PROGRAM END ==')


        layout = [[sg.Text('Запускаем ?   '), sg.Text('controlpanel prototype NOT FOR PRODUCTION'),
                   sg.Button('????', key='CheckMultiprocessing'), sg.Button('Disco', key='LoadDiscord'),
                   sg.Button('Stream START', key='stream_start'), sg.Button('Stream STOP', key='stream_stop'),
                   sg.Checkbox(key='stream_debug', text='stream debug', default=False)],
                  [sg.Text('хихиххии   '), sg.InputText(key='CNInp'),
                   sg.Button('>>', key='Enter', bind_return_key=True)],
                  [sg.Text('пока запускается'), sg.Text('НИЧО НЕ ТРОГАЕМ!')],
                  [sg.Button('       ДА       '), sg.Button('Ext'), sg.Button('Ран осн акк'),
                   sg.Checkbox(key='allowChatInteract', text='интеракт с чатом в майне?', default=False),
                   sg.Checkbox(key='youtubeChatInteract', text='YT chat interact', default=False),
                   sg.Checkbox(key='youtubeAppLogin', text='YT APP LOGIN', default=False),
                   sg.Checkbox(key='stream_ad_enabled', text='STREAM ADS ENABLED', default=False)],
                  [sg.Button('VTube Studio', key='VTUBE', button_color='red'),
                   sg.Button('MC', key='MC', button_color='red'),
                   sg.Button('YouTube', key='YTChat', button_color='red')],

                  [sg.Txt('_' * 90)],

                  [sg.Text('Ранг\tИмя', background_color='black', size=(25, 1), auto_size_text=False),
                   sg.Text('Сообщение', background_color='black', size=(25, 1), auto_size_text=False),
                   sg.Button('Ext', key='CloseAll', tooltip='удалить'),
                   ],

                  [sg.Text('0\tdefault', key='1rank', background_color='black', size=(25, 1), auto_size_text=False,
                           visible=False),
                   sg.Text('Сообщение', key='1msg', background_color='black', size=(25, 1), auto_size_text=False,
                           visible=False),
                   sg.Button('Ext', key='1cls', tooltip='удалить', visible=False),
                   ],

                  [sg.Txt('_' * 90)],

                  [sg.Text('Роль\tЭмоция', background_color='black', size=(25, 1), auto_size_text=False),
                   sg.Text('Сообщение', background_color='black', size=(25, 1), auto_size_text=False),
                   ],

                  # [sg.Txt('―'  * 90)],
                  ]
        window = sg.Window('NetTyan', layout, icon=f"{thisfolder}/Resources/Pictures/appico.ico")


        def WindowListener():
            global window
            firstcmd = ''
            while ctx.ThreadsActived:
                event, values = window.read(timeout=0)

                if event == sg.WIN_CLOSED or event == 'Ext':  # if user closes window or clicks cancel
                    CompleteShutDown(None, None)  # signal, frame)
                elif firstcmd != '' and event == 'Enter':
                    CommandProcess(firstcmd, Manual=False, secondinp=values['CNInp'])
                elif event == 'Enter':
                    firstcmd = values['CNInp']
                elif event == '       ДА       ':
                    CommandProcess('3', Manual=False, secondinp=values['CNInp'])
                elif event == 'LoadDiscord':
                    print('Дискорд уже должен быть запущен в коде!')
                    # DiscordProc.run()
                    # DiscordProc.start()
                elif event == 'VTUBE':
                    start_program("VTUBE")
                elif event == "SheepChat":
                    start_program("SheepChat")
                elif event == 'stream_start':
                    print('STARTING STREAM REQUEST! debug =', values["stream_debug"])
                    StreamStartingProgram(record_debug=values["stream_debug"])
                    _optToChange = ["allowChatInteract", "stream_ad_enabled", "youtubeChatInteract", "youtubeAppLogin"]
                    for lol in _optToChange:
                        window[lol].Update(disabled=False)
                    # values["allowChatInteract"] = True
                    # values["stream_ad_enabled"] = True
                    # values["youtubeChatInteract"] = True
                    # values["youtubeAppLogin"] = True
                elif event == 'stream_stop':
                    print('STOPPING STREAM REQUEST! debug =', values["stream_debug"])
                    StreamStoppingProgram(record_debug=values["stream_debug"])
                    window["stream_ad_enabled"].Update(disabled=True)
                if values["allowChatInteract"] and not ctx.allowChatInteract:
                    ctx.allowChatInteract = True
                    print('ALLOW CHAT INTERACT TRUE')
                if not values["allowChatInteract"] and ctx.allowChatInteract:
                    ctx.allowChatInteract = False
                    print('ALLOW CHAT INTERACT FALSE')
                if values["stream_ad_enabled"] and not ctx.stream_started:
                    ctx.stream_started = True
                    print('stream_started TRUE')
                if not values["stream_ad_enabled"] and ctx.stream_started:
                    ctx.stream_started = False
                    print('stream_started FALSE')
                if values["youtubeChatInteract"] and not ctx.YouTubeCommentCheckerEnabled:
                    ctx.YouTubeCommentCheckerEnabled = True
                    print('ALLOW YT CHAT INTERACT TRUE')
                if not values["youtubeChatInteract"] and ctx.YouTubeCommentCheckerEnabled:
                    ctx.YouTubeCommentCheckerEnabled = False
                    print('ALLOW YT CHAT INTERACT FALSE')
                if values["youtubeAppLogin"] and not ctx.YouTubeAppEnabled:
                    ctx.YouTubeAppEnabled = True
                    print('ALLOW YT MODER APP TRUE')
                StatusWindowChecker()
                time.sleep(0.01)
                # elif event == 'MC':
                #    os.system(thisfolder+'\\1HyperAI\\Resources\\Links\\Comp\\MultiMC.exe.lnk')


        # WindowListenerThread = threading.Thread(target=WindowListener)
        # WindowListenerThread.daemon = True
        # WindowListenerThread.start()

        import pygame
        from pygame import mixer  # Playing sound
        import pygame._sdl2.audio as sdl2_audio

        init_by_me = not pygame.mixer.get_init()
        if init_by_me:
            pygame.mixer.init()
        devices = tuple(sdl2_audio.get_audio_device_names())
        if init_by_me:
            pygame.mixer.quit()
        # print(str(devices))
        pygame.mixer.pre_init()
        pygame.mixer.init(frequency=48000, size=-16, channels=2, buffer=7168,
                          devicename='CABLE-A Input (VB-Audio Cable A)')  # Initialize it with the correct device
        # sound_effect.play()
        # pip install sounddevice
        import sounddevice as sd

        sd.default.samplerate = 48000
        sd.default.channels = 2
        sd.default.device = 'CABLE-A Input (VB-Audio Cable A), Windows DirectSound'


        # [14] CABLE Input (VB-Audio Virtual Cable), Windows DirectSound
        # [17] CABLE Input (VB-Audio Virtual Cable), Windows WASAPI
        # query_devices()
        def SoundToMicro(file='test.wav', audio=None, sleep=False, smart_wait=False, change_emotes=False):
            ####pygame.init()
            # pygame.mixer.init(devicename='CABLE Input (VB-Audio Virtual Cable)') #Initialize it with the correct device
            # sound_effect = pygame.mixer.Sound('test.wav')
            if sleep and smart_wait:
                speech_available_event.clear()
            if change_emotes:
                ctx.SeparateEyes = False
                ctx.state = "idle"

            if audio is not None:
                sd.play(audio, 48000 * 1.05)
                if sleep:
                    time.sleep((len(audio) / 48000) + 0.5)
                    sd.stop()
            else:
                sound_effect = pygame.mixer.Sound(file)
                sound_effect.play()

            if change_emotes:
                ctx.isVoiceBusy = False
                ctx.SeparateEyes = False
                ctx.state = "gaming"
            if sleep and smart_wait:
                speech_available_event.set()
            ####pygame.mixer.music.load("test.wav") #Load the mp3
            ####pygame.mixer.music.play() #Play it
            #####time.sleep(1)
            ####pygame.event.wait()
            ####mixer.quit()


        # RefreshInterval = manager.Value('i',3)

        # time.sleep(10)
        mc_vt_ctx = manager.Namespace()
        mc_vt_ctx.PitchSpeed = 0.0
        mc_vt_ctx.YawSpeed = 0.0
        ctx.ingame_info = {"task_chain": "Ничего не происходит", "ground_block": "пустота", "held_item": "ничего"}
        from HyperAI_BRIDGE import mainBridge

        MineBridgeProc = Process(
            target=mainBridge,
            args=(mc_vt_ctx, ctx,),
            kwargs={"ctx_chat": ctx_chat, "ctx_chatMsgs": ctx_chatMsgs,
                    "ctx_chatOwn": ctx_chatOwn})  # Thread(target = a, kwargs={'c':True}).start()
        MineBridgeProc.start()
        print(' == ЗАПУСК ПРОЦЕССА ДЛЯ МОСТА ==')


        # print('Ыыы ',mc_vt_ctx.PitchSpeed, mc_vt_ctx.YawSpeed)
        def vstate(state):
            ctx.state = state


        def sgn(x):
            if x > 0:
                return 1
            elif x == 0:
                return 0
            else:
                return -1


        def VtubeRotater():
            def rd(num):
                return round(num, 2)

            def clp(num):
                return np.clip(num, -1, 1)  ##numpy.clip(a, a_min, a_max,

            while ctx.ThreadsActived:
                x = 0
                xvel = 0.01
                y = 0
                yvel = 0.01
                if (ctx.state == "idle"):
                    x = 0
                    y = 0
                    vtube_ctx.eyeX = 0
                    vtube_ctx.eyeY = 0
                elif (ctx.state == "gaming"):
                    xmod = -mc_vt_ctx.YawSpeed / 30
                    ymod = mc_vt_ctx.PitchSpeed / 50
                    x = -0.5
                    y = -1.0
                    # if(abs(ymod)>0.4):
                    #    ymod=0
                    if (abs(xmod) > 0.5):
                        xmod /= 10
                    x += xmod
                    y += ymod
                    vtube_ctx.eyeX = x
                    vtube_ctx.eyeY = y
                diffx = vtube_ctx.NeedX - x
                diffy = vtube_ctx.NeedY - y
                # print("\n\nX =",vtube_ctx.NeedX,x,diffx,"\nY =",vtube_ctx.NeedY,y,diffy)

                if abs(diffx) > 0.02:
                    vtube_ctx.NeedX = clp(vtube_ctx.NeedX - xvel * sgn(diffx))
                if abs(diffy) > 0.02:
                    vtube_ctx.NeedY = clp(vtube_ctx.NeedY - yvel * sgn(diffy))
                time.sleep(0.01)


        VtubeRotaterThread = threading.Thread(target=VtubeRotater)
        VtubeRotaterThread.daemon = True
        VtubeRotaterThread.start()


        def MineEventHandler():
            while ctx.ThreadsActived:
                ctx.MineEvent.wait()
                if ctx.MineEventName == "death":
                    ctx.AnimEventInfo = {"name": "CryButNot", "type": "hotkey"}
                    ctx.AnimEvent.set()
                    ctx.AnimEvent.clear()
                    modifyMood(-0.2)
                elif ctx.MineEventName == "kill":
                    modifyMood(0.5)
                print("EVENT CATCH! ev =", ctx.MineEventName, '; mood =', ctx.mood)


        MineEventHandlerThread = threading.Thread(target=MineEventHandler)
        MineEventHandlerThread.daemon = True
        MineEventHandlerThread.start()

        import socketio

        sio = socketio.Client()


        # netTyan token M1Hqrw1OUTPOg0ofettX
        # https://www.donationalerts.com/widget/alerts?group_id=1&token=M1Hqrw1OUTPOg0ofettX
        # SayNex token dhNCV3i7bTjSZilcrg68

        @sio.on('connect')
        def on_connect():
            sio.emit('add-user', {"token": "M1Hqrw1OUTPOg0ofettX", "type": "alert_widget"})  #


        @sio.on('donation')
        def on_message(data):
            y = json.loads(data)
            # if ((y['message'].find('IMG:') != -1)):
            # url = y['message'].split('IMG:')[1]
            print('!!! ВНИМАНИЕ !!! УРААА ПОЛУЧЕН ДОНАТ!!! Вот словарь:', y)
            donationQueue.append(y)
            # print#(y['username'])
            # print(y['message'])
            # print_img(y['message'].split('IMG:')[1])


        sio.connect('wss://socket.donationalerts.ru:443', transports='websocket')

        # https: // www.donationalerts.com / widget / alerts?group_id = 3 & token = dhNCV3i7bTjSZilcrg68
        # https://www.donationalerts.com/widget/alerts?group_id=1&token=dhNCV3i7bTjSZilcrg68

        ctx.YouTubeCommentCheckerEnabled = False
        ctx.YouTubeAppEnabled = False
        ctx.YoutubeActionsQueue = manager.list()

        twitch_actions_queue = manager.Queue()
        trovo_actions_queue = manager.Queue()
        from HyperAI_YT import YoutubeChatListener

        print('yt proc start')
        YoutubeChatCheckerProc = Process(
            target=YoutubeChatListener,
            args=(ctx, twitch_actions_queue, trovo_actions_queue, ctx_chat,))
        # , kwargs={"ctx_chat": })  # Thread(target = a, kwargs={'c':True}).start()

        YoutubeChatCheckerProc.start()

        # YoutubeChatListener()
        # YoutubeChatListenerThread = threading.Thread(target=YoutubeChatListener)
        # YoutubeChatListenerThread.daemon = True
        # YoutubeChatListenerThread.start()

        oldMC = False
        oldVT = False
        oldYT = False


        def StatusWindowChecker():
            global oldMC, oldVT, oldYT

            mc = ctx.IsMCStarted
            vt = ctx.IsVtubeStarted
            yt = ctx.IsYTChatConnected
            if (oldMC != mc):
                if (mc):
                    window['MC'].Update(button_color='green')
                else:
                    window['MC'].Update(button_color='red')
                oldMC = mc
            if (oldVT != vt):
                if (vt):
                    window['VTUBE'].Update(button_color='green')
                else:
                    window['VTUBE'].Update(button_color='red')
                oldVT = vt
            if (oldYT != yt):
                if (yt):
                    window['YTChat'].Update(button_color='green')
                else:
                    window['YTChat'].Update(button_color='red')
                oldYT = yt


        def CompleteShutDown(signal=None, frame=None):
            print('ИНИЦИАЛИЗИРУЕМ ВЫХОД!!!')
            ctx.ThreadsActived = False
            HttpProc.terminate()
            HttpProc.join()
            VtubeProc.terminate()
            VtubeProc.join()
            MineBridgeProc.terminate()
            MineBridgeProc.join()
            YoutubeChatCheckerProc.terminate()
            YoutubeChatCheckerProc.join()

            DiscordProc.terminate()
            DiscordProc.join()

            FiltersProc.terminate()
            FiltersProc.join()
            # LargeFREDProc.terminate()
            # LargeFREDProc.join()
            LargeTTSProc.terminate()
            LargeTTSProc.join()
            time.sleep(0.1)
            manager.shutdown()
            time.sleep(0.1)
            os.abort()


        # signal.signal(signal.SIGABRT, CompleteShutDown)
        # signal.signal(signal.SIGFPE, CompleteShutDown)
        # signal.signal(signal.SIGILL, CompleteShutDown)
        # signal.signal(signal.SIGINT, CompleteShutDown)
        # signal.signal(signal.SIGSEGV, CompleteShutDown)
        # signal.signal(signal.SIGTERM, CompleteShutDown)

        def CommandProcess(inp, Manual=True, secondinp=''):
            def MANUALinput(hint):
                if (Manual):
                    iii = input(hint)
                    return iii
                else:
                    return secondinp

            if inp == "2":
                # BuildSentimentModel()
                inp = MANUALinput("Введите сообщение для проверки на положительность:")
                print("Сейчас определим доброту вашего сообщения >> " + str(determine_tone(inp)))
            if inp == "3":
                inp = MANUALinput("Введите сообщение для ОЗВУЧКИ:")
                textToSpeech(inp, "medium", "medium")  # rate pitch # pitch x-low, low, medium, high, x-high
                print("Озвучен текст" + str(inp))
            if inp == "7":
                inp = MANUALinput("Введите сообщение для ПЕРЕДАЧИ LM (будет получен ответ + ОЗВУЧКА):")
                # reply = lm.chatbot(inp)
                # answer = reply["reply"]
                answer = FredT5Chatbot(inp, authorisedUser="ЛехаЛепеха")["reply"]
                textToSpeech(answer, "medium", "medium", seeChat=False)
                print('DEBUG COMMANDS AND EMOTIONS = ', answer)
            if inp == "8":
                inp = MANUALinput("Введите сообщение для ПЕРЕДАЧИ LM (будет получен только ответ):")
                reply = FredT5Chatbot(inp, authorisedUser="ЛехаЛепеха")["reply"]
                print('ОТВЕТ МОДЕЛИ: ', reply)
            if inp == "9":
                inp = MANUALinput("Введите Ник для анализа LM")
                reply = FredT5Chatbot("анализируй ник мой", authorisedUser=inp)["reply"]
                print('ОТВЕТ МОДЕЛИ: ', reply)
            if inp == "ds_str":
                DiscordStreamAnnounce(
                    greetings="Привет мир! Это тест большого текста на начало стрима.  ПОЫВАПол? Что как дела где я? Почему так? что случилось? А эээм что происходит? Кто здесь?",
                    do_mention=False)
                # print("Текст ответа: >",answer,"<")
            # if inp == "f":
            #    print('debug filter'+filt.debug())
            if inp == "set_mood":
                inp = MANUALinput("Введите НАСТРОЕНИЕ (FLOAT):")
                try:
                    modifyMood(float(inp))
                except BaseException as err:
                    print('Не удалось привести к float. ', err)
            if inp == "soundtest":
                SoundToMicro()
                print('tested sound')
            if inp == "getcontext":
                print('КОНТЕКСТ\n', DATABASE.get_relevant_diag(), '\nКОНЕЦ КОНТЕКСТА')
            if inp == "event chat":
                ctx.AnimEventInfo = {"name": "SawChat.exp3.json", "type": "expression"}
                ctx.AnimEvent.set()
                ctx.AnimEvent.clear()
            if inp == "event lose":
                ctx.AnimEventInfo = {"name": "CryButNot", "type": "hotkey"}
                ctx.AnimEvent.set()
                ctx.AnimEvent.clear()
            if inp == "event viewer":
                ctx.AnimEventInfo = {"name": "SeeViewer.exp3.json", "type": "expression"}
                ctx.AnimEvent.set()
                ctx.AnimEvent.clear()
            if inp == "event gamer":
                ctx.AnimEventInfo = {"name": "games.exp3.json", "type": "expression"}
                ctx.AnimEvent.set()
                time.sleep(10)
                print('ЭВЕНТ КЕРДЫК')
                ctx.AnimEvent.clear()


        def ConsoleListener():
            while ctx.ThreadsActived:
                inp = input(f"""{col('Команды:')}
                1 - правописание
                2 - сентиментальный анализ
                3 - озвучка текста
                4 - ответ OpenAi со озвучкой
                5 - ответ OpenAi без озвучки.
                6 OpenAiGameEvent
                7 LM answer + speech
                8 LM answer
                9 nickAnalyze
                ---
                !set_mood, потом [float mood]
                !ds_str discord stream announce test (not mention)
                !f - filter debug
                !event [chat, lose, viewer, gamer]
                !logreload - reload logs
                !getcontext - get context ???
                !пустота/exit - {col('выход')}""")
                if inp == "exit" or inp == "выход" or inp == "":
                    break
                try:
                    CommandProcess(inp)
                except BaseException as err:
                    print('ОШИБКА ПРОЦЕССА В КОНСОЛИ: ', err)
                    print('ТЕКСТ ОБ***Й ОШИБКИ', traceback.format_exc())
                    print("\n=== КОНЕЦ ОШИБКИ ====")
                    time.sleep(1)
                time.sleep(0.01)


        ConsoleListenerThread = threading.Thread(target=ConsoleListener)
        ConsoleListenerThread.daemon = True
        ConsoleListenerThread.start()

        # filt.CheckModel()
        # lm.CheckModel()###DEBUG ВРЕМЕННО ОТКЛЮЧЕНО НАДО ВКЛЮЧИТЬ!!
        filtersCtx.loading_flag.wait()
        docker_sender.llm_loading_flag().wait()
        # fredCtx.loading_flag.wait()
        ttsCtx.loading_flag.wait()
        ctx.allLoaded = True
        print('ctx ALLOADED TRUE! ЧТО ЭТО ЗНАЧИТ ? А ЭТО ЗНАЧИТ ВСЁ КРОМЕ ЧЕКЕРА ОКОН ЗАГРУЖЕНО! УРАА!')
        # while True:
        #    time.sleep(0.1)
        WindowListener()

        CompleteShutDown()

        # HttpProc.terminate()
        # HttpProc.join()
        # VtubeProc.terminate()
        # VtubeProc.join()
        # MineBridgeProc.terminate()
        # MineBridgeProc.join()
    except BaseException as err:
        print('!!!ОШИБКА ЦЕНТРАЛЬНОГО ПРОЦЕССА-ПОТОКА!!!', err)
        print('ТЕКСТ ОБ***Й ОШИБКИ', traceback.format_exc())

# АНГЛОЯЗЫЧНАЯ МОДЕЛЬ
# from textblob import TextBlob
#
# def determine_tone(text):
#    blob = TextBlob(text)
#    polarity = blob.sentiment.polarity
#    if polarity > 0:
#        return "Positive"
#    elif polarity < 0:
#        return "Negative"
#    else:
#        return "Neutral"
# print(determine_tone("Ты хороший человек!"))
