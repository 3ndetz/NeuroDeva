"""
ЭТО ФИЛЬТР
В ДАННОМ ФАЙЛЕ ЕСТЬ ОСКОРБЛЕНИЯ И НЕПРИСТОЙНЫЕ ВЫРАЖЕНИЯ!!!
"""

import importlib, sys
import datetime, os

import json
import numpy as np
import string
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stopwords_ru = stopwords.words("russian")


# print('ТЕСТ С*********')


def calcTime(time):
    return bcolors.OKGREEN + str((datetime.datetime.now() - time).total_seconds()) + bcolors.ENDC


def wordtokenize(text, remove_stopwords=True):
    # разбиваем текст на слова
    text = text.lower()
    spec_chars = string.punctuation + '\r\n\xa0«»\t—…'
    for char in spec_chars:
        text = text.replace(char, ' ')

    text = CutSpaces(text)
    text = re.sub("[^А-Яа-яA-Za-z0-9]", "", text)
    output = text.split()
    if remove_stopwords:
        for word in stopwords_ru:
            while word in output:
                output.remove(word)
    return output


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


def get_elements_of_nested_list(element):
    count = 0
    if isinstance(element, list):
        for each_element in element:
            count += get_elements_of_nested_list(each_element)
    else:
        count += 1
    return count


def adjust_multilabel(y, target_vaiables_id2topic_dict, is_pred=False):
    y_adjusted = []
    for y_c in y:
        y_test_curr = [0] * 19
        index = str(int(np.argmax(y_c)))
        # value = y_c[index]
        y_c = target_vaiables_id2topic_dict[index]
    return y_c


def ConvertTextForFilter(ninp):
    punkt = '!?.'
    out = ''
    cnt = 0
    inp = CutSpaces(ninp).lower()
    for char in inp:
        cnt += 1
        if char == '\n' and cnt < 30:
            out += ' '
        elif char == '\n':
            out += char
            cnt = 0
        elif cnt > 100 or (cnt > 32 and char in punkt):
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


def FILTERS_PROCESS(ctx):
    from FilterExamples import examples
    from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, \
        AutoModelForSequenceClassification
    from sentence_transformers import SentenceTransformer, util
    import torch
    # torch.set_num_threads(4) #dEBUG ОТКЛЮЧИЛ
    import traceback, time
    class Filter:

        ModelLocalPaths = {'judge': {'id': 'apanc/russian-inappropriate-messages', 'localPath': '/models/apancJudge'},
                           'topics': {'id': 'apanc/russian-sensitive-topics', 'localPath': '/models/apancTopics'},
                           'tiny_classificator': {'id': 'Den4ikAI/ruBert-tiny-replicas-classifier',
                                                  'localPath': '/models/den_tiny_replicas'},
                           'synonims': {'id': 'inkoziev/sbert_synonymy', 'localPath': '/models/kozievSynonims'},
                           }
        thisfolder = os.path.dirname(os.path.realpath(__file__))
        tokenizer = []
        TopicClassificatorModel = []
        FilterJudgeModel = []
        SynonymsModel = []
        TinyClassModel = []
        tokenizer_for_classificator = []
        q_samples_dict = None
        nick = "obama421"
        username = "Пользователь"
        device = "cpu"
        e = []
        ModelLoaded = False
        lastTokensUsed = 0
        context = []
        target_vaiables_id2topic_dict = []

        def TokenizerDebugPrint(self, inp, debugPrefix='Debug Input >> '):
            tokens = inp
            debugOutputs = []
            for t in tokens:
                debugOutputs.append(t)
                debugOutputs.append(96)  # token '|' = 96, [=65, .=18
            print(debugPrefix, '\n<|||>\n', self.tokenizer.decode(debugOutputs), '\n<|||>')

        def CheckModel(self):
            if (not self.ModelLoaded):
                t = datetime.datetime.now()
                # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                print(f'\n=== Загрузка ФИЛЬТРОВ на CPU ===\n')
                self.tokenizer = BertTokenizer.from_pretrained(self.ModelLocalPaths["topics"]["id"]
                                                               , cache_dir=self.thisfolder +
                                                                           self.ModelLocalPaths["topics"]["localPath"])

                self.tokenizer.truncation_side = 'left'
                self.FilterJudgeModel = BertForSequenceClassification.from_pretrained(
                    self.ModelLocalPaths["judge"]["id"]
                    , cache_dir=self.thisfolder + self.ModelLocalPaths["judge"]["localPath"]);
                print(f'\n=== Загрузка 1 ФИЛЬТРА-СУДЬИ ЗАВЕРШЕНА ({calcTime(t)}c) ===\n')
                self.FilterJudgeModel.eval()

                self.TopicClassificatorModel = BertForSequenceClassification.from_pretrained(
                    self.ModelLocalPaths["topics"]["id"]
                    , cache_dir=self.thisfolder + self.ModelLocalPaths["topics"]["localPath"]);  # загрузка 3 сек
                self.TopicClassificatorModel.eval()

                with open(self.thisfolder + "/id2topic.json") as f:
                    self.target_vaiables_id2topic_dict = json.load(f)

                self.TinyClassModel = AutoModelForSequenceClassification.from_pretrained(
                    self.ModelLocalPaths["tiny_classificator"]["id"]
                    , cache_dir=self.thisfolder + self.ModelLocalPaths["tiny_classificator"][
                        "localPath"]);  # загрузка 3 сек
                self.TinyClassModel.eval()

                self.tokenizer_for_classificator = AutoTokenizer.from_pretrained(
                    self.ModelLocalPaths["tiny_classificator"]["id"]
                    , cache_dir=self.thisfolder +
                                self.ModelLocalPaths["tiny_classificator"]["localPath"])

                self.ModelLoaded = True
                print(f'\n=== Загрузка 2 ФИЛЬТРОВ УСПЕШНО ЗАВЕРШЕНА ({calcTime(t)}c) ===\n')

                # self.SynonymsModel = SentenceTransformer(self.ModelLocalPaths["synonims"]["id"]
                #     , cache_folder=self.thisfolder + self.ModelLocalPaths["synonims"]["localPath"])
                #
                # print(f'\n=== Загрузка ПОИСКА СИНОНИМОВ ЗАВЕРШЕНА ({calcTime(t)}c) ===\n')

        # def GetIntent(self, ninp):
        #
        #     if self.q_samples_dict is None:
        #         self.q_samples_dict = [{"text":"Как у тебя дела?","type":"q_about"},
        #                                {"text": "Как тебя зовут", "type": "q_about"},
        #                                {"text": "Как зовут разработчика", "type": "q_about"},
        #                                {"text": "Что ты умеешь", "type": "q_about"},]
        #         for i, sample in enumerate(self.q_samples_dict):
        #             self.q_samples_dict[i]["token_ids"] = self.SynonymsModel.encode([sample["text"]])[0]
        #
        #     s1 = ninp
        #     v1 = self.SynonymsModel.encode([ninp])[0]
        #
        #     max_similarity = 0
        #     result = {}
        #     for sample in self.q_samples_dict:
        #         s = util.cos_sim(a=v1, b=sample["token_ids"]).item()
        #         if s >= max_similarity:
        #             max_similarity = s
        #             result["similar_text"] = sample["text"]
        #             result["similar_type"] = sample["type"]
        #         print('text1={} text2={} cossim={}'.format(s1, sample["text"], s))
        #
        #     result["similarity_value"] = max_similarity
        #     return result
        def get_sentence_type(self, text):
            inputs = self.tokenizer_for_classificator(text.replace("?", ""), max_length=512, add_special_tokens=False,
                                                      return_tensors='pt').to(self.device)
            classes = ['instruct', 'question', 'dialogue', 'problem', 'about_system', 'about_user']
            try:
                with torch.no_grad():
                    logits = self.TinyClassModel(**inputs).logits
                    probas = list(torch.sigmoid(logits)[0].cpu().detach().numpy())
                out = classes[probas.index(max(probas))]
            except BaseException as err:
                print('ERR В ПРОЦЕССЕ ГЕТА ИНФА', err)
                out = "dialogue"
            return str(out)

        def get_possible_info(self, ninp) -> dict:
            # intents = self.GetIntent(ninp)
            words = wordtokenize(ninp)
            question_words = "как почему что где".split(" ")
            is_question = False
            sentence_type = self.get_sentence_type(ninp)
            for word in words:
                if word in question_words:
                    is_question = True
            return {"is_question": is_question, "sentence_type": str(sentence_type)}

        def Filter(self, ninp):  # [2.0,2.0,50,100]
            self.CheckModel()
            # t = datetime.datetime.now()
            inp = ConvertTextForFilter(ninp)
            # print('DEBUG inp0 = ', tokens_ids, 'msk ',mask)
            input_cnt = 0
            topics = []
            words = []
            allowed = True

            for i, line in enumerate(inp):
                tokenized = self.tokenizer.batch_encode_plus([line],
                                                             max_length=256, padding=True, truncation=True,
                                                             return_token_type_ids=False)  # было max length 512
                tokens_ids, mask = torch.tensor(tokenized['input_ids']), torch.tensor(tokenized['attention_mask'])

                input_cnt += get_elements_of_nested_list(tokens_ids.tolist())

                with torch.no_grad():
                    model_output = self.TopicClassificatorModel(tokens_ids, mask)
                    judgement_out = self.FilterJudgeModel(tokens_ids, mask)
                judgement_label = judgement_out['logits'].argmax().item()
                allow = not bool(judgement_label)
                if not allow:
                    allowed = False
                preds = adjust_multilabel(model_output['logits'], self.target_vaiables_id2topic_dict, is_pred=True)
                if preds != "none":
                    topics = list(set(topics + preds.split(',')))
                words = list(set(words + wordtokenize(line)))

            def machine_filter(words: list) -> int:
                result_score = 0
                innaproproriate_wordlist = "зеленск путин байден евре гитлер долба хуес таджик узбек чурк узког нигер негр niger nigr нигг нигер нигр нига ниги nig сперм sperm бляди шлюх манд mand пизд pizd укр ukr россия россия росия росие расие амер amer войн вайн war спецопер политик стран войск пропаганд лгбд гей геи натуралы ориентаци церков церкв бог дьявол верующ религи ислам террор".split(
                    ' ')
                for word in words:
                    for word_part in innaproproriate_wordlist:
                        if word_part in word.lower():
                            print('[FILTERS] НАЙДЕНО ЗАПРЕЩЕННОЕ СЛОВО!', word_part)
                            result_score -= 10
                return result_score

            score = 0
            score += machine_filter(words)
            for topic in topics:
                if topic in 'politics,racism,religion,terrorism,suicide'.split(','):
                    score += -10

                elif topic in 'offline_crime,drugs,social_injustice'.split(','):
                    score += -1
                elif topic in 'pornography,prostitution,sexism,sexual_minorities'.split(','):
                    score += -0.5
                elif topic in 'online_crime'.split(','):
                    score += -0.25
                elif topic in 'body_shaming,health_shaming'.split(','):
                    score += -0.1
                elif topic in 'slavery,gambling,weapons'.split(','):
                    score += -0.01
                else:
                    score += 0.5
                # print(i, 'inp = ', self.tokenizer.decode(tokens_ids[0]), '\nallow =', allow, 'preds =', preds)

            # print(calcTime(t) + ' - время просчета, токенов [INPUT] -', '[' + str(input_cnt) + ']', '\n')
            return {"topics": topics, "allow": allowed, "score": score}

        """topics
    none

    недопустимые (-10)
    politics,racism,religion,terrorism,suicide

    такое себе (-1)
    offline_crime,drugs,social_injustice

    средней тяжести (-0.5)
    pornography,prostitution,sexism,sexual_minorities

    слабой тяжести (-0.25)
    online_crime

    по здоровью (-0.1)
    body_shaming,health_shaming

    почти не влияющие (-0.01)
    slavery,gambling(азартная игра)


    """

        def debug(self):
            examples = importlib.reload(sys.modules['FilterExamples']).examples
            self.e = examples()
            self.e.debug()
            inp = self.e.getResult()
            p = self.e.getParams()
            print("Загрузка текста из подключаемого модуля")
            # print('Параметры: ',str(p))
            # print(inp)
            return str(self.Filter(inp, p))

        def __init__(self):
            self.e = examples()

    t = datetime.datetime.now()
    filt = Filter()
    filt.CheckModel()
    ctx.loading_flag.set()
    print('время запуска FILTERS' + calcTime(t))
    while True:
        try:
            queue_input = ctx.Queue.get()
            ninp = queue_input[0]
            filter_type = queue_input[1]
            print('[FILTERS QUEUE] получена очередь', ninp, 'ТИП:', filter_type)
            answer = {}
            if filter_type == "filter":
                answer = filt.Filter(ninp)
            elif filter_type == "info":
                answer = filt.get_possible_info(ninp)
            ctx.QueueOutput.put(answer)
        except BaseException as err:
            print('[FILTERS ERR] ОШИБКА ПРОЦЕССА: ', err)
            print('[FILTERS ERR] ТЕКСТ ОБ***Й ОШИБКИ', traceback.format_exc())
            print("\n[FILTERS ERR] === КОНЕЦ ОШИБКИ ====")
            time.sleep(1)


if __name__ == "__main__":  # DEBUG NOT WORK
    t = datetime.datetime.now()
    print('ЗАПУСК ЧО')
    import multiprocessing

    manager = multiprocessing.Manager()
    filtersCtx = manager.Namespace()
    filtersCtx.Queue = manager.Queue()
    filtersCtx.QueueOutput = manager.Queue()
    filtersCtx.loading_flag = manager.Event()
    LargeFREDProc = multiprocessing.Process(
        target=FILTERS_PROCESS,
        args=(filtersCtx,))  # Thread(target = a, kwargs={'c':True}).start()
    LargeFREDProc.start()
    # FRED_PROCESS(fredCtx)
    print('ЗАПУСК ЧО2')


    def FiltersQueue(ninp, filter_type="filter"):
        filtersCtx.Queue.put((ninp, filter_type,))
        return filtersCtx.QueueOutput.get()


    filtersCtx.loading_flag.wait()
    # print(e.getResult()+'mda')
    print('время запуска ТЕСТА ' + calcTime(t))

    while True:
        inp = input('чобабке\n>>')
        if inp == "1":
            inp = input("Введите сообщение для получения РЕЗУЛЬТАТА ФИЛЬТРАЦИИ\n>>")
            print("Запуск модели")
            print("Ответ:")
            print('|!|\n', FiltersQueue(inp), '\n|!|')
        if inp == "2":
            inp = input("Введите сообщение для получения ИНФОРМАЦИИ И НАМЕРЕНИЙ\n>>")
            print("Запуск модели")
            print("Ответ:")
            print('|!|\n', FiltersQueue(inp, filter_type="info"), '\n|!|')
        if inp == "":
            print("Ответ:")
            print('|!|\n' + FiltersQueue("ЧО БАБКЕ С*******") + '\n|!|')
        if inp == "ext":
            print("выход")
    ####lm_text='<SC5>Принялся Кутузов рассказывать свою историю <extra_id_0>. Началось с того, что он был в армии, служил в артиллерии.'
    ####outputs=model.generate(input_ids,eos_token_id=tokenizer.eos_token_id,early_stopping=True)
