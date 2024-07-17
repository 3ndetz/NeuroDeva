# -*- coding: utf-8 -*-
"""
SOME OF PROMPTS REMOVED (because it has some secrets =)

MOVE THIS FILE UPPER FROM FOLDER
CHANGE PROMPT TEXT FOR YOU NEEDS
THIS NEED FOR RAG-LIKE SYSTEM WORK!

FILE MAY BE OUTDATED, IF FOUND ERROR CREATE ISSUE/CHAT AUTHOR
"""

from datetime import datetime, date
import random
import importlib
import sys
# import os

# import json
# import os

from string_utils import NonRepeatRandom


def tm(x):
    return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')


def get_llm_formed_inputs(inp: str, username: str, environment: dict, params_override: dict,
                          dialog_context: list, repeating_dict,
                          danger_context: str = "Привет! Что ты делаешь(ла)? ") -> [str, dict, str]:
    def get_formed_llm_context(context: list, danger_context_inner: str) -> (str, str):
        result = ""
        if len(context) > 1:
            question = True
            for i, record in enumerate(context):
                if record.get("role", "") == "assistant":
                    danger_context_inner += record.get("content", "")
                    this_role_prefix = "A: "
                    cmd = record.get("command", "")
                    if cmd != "":
                        cmd = f" <команда=!{cmd}"
                    emo = record.get("emotion", "")
                    if emo != "":
                        emo = f" [эмоция={emo}"
                    this_role_suffix = f"{cmd}{emo}</s>\n"
                    if (i - 1) == len(context):  # если последний элемент не добавим энтера
                        pass
                    else:
                        this_role_suffix += '\n'
                else:
                    this_role_prefix = "Q: "
                    this_role_suffix = "</s>\n"

                if i == 0 and record["role"] == "assistant":
                    question = False
                else:
                    if question:
                        result += f'*{random.choice(["челикс", "ботяра", "ноунейм", "какой-то", "пупсик"])} {record.get("user", "default")} начинает общение*\n'
                    question = not question
                result += this_role_prefix + record["content"] + this_role_suffix
        return result, danger_context_inner

    if dialog_context is not None:
        dialog_context_formed, danger_context = get_formed_llm_context(dialog_context, danger_context)
    else:
        dialog_context_formed, danger_context = "", danger_context

    LLMExamples = importlib.reload(sys.modules['LLMExamples']).LLMExamples
    llm_prompts = LLMExamples()
    if environment is not None:
        llm_prompts.setEnvironment(environment)
        # self.e.environment=environment
    llm_prompts.username = username
    if params_override is not None:
        llm_prompts.paramsOverride = params_override
    llm_prompts.repeatingDict = repeating_dict
    llm_prompts.chatbot(inp, context=dialog_context_formed)
    return llm_prompts.getResult(), llm_prompts.getParams(), danger_context


t5_mode = True
if t5_mode:
    start_token_diag = '<SC6>'  # sc1 стояло в донате и эвентах
else:
    start_token_diag = '<s>'


class LLMExamples:
    repeatingDict = {}
    nick = 'васия5321'
    lines = []
    username = "Konushnya852"
    paramsOverride = None
    environment = {
        "env": "youtube"
    }
    params = {
        "do_sample": True,
        "top_p": 0.95,
        "temperature": 0.21,
        "repetition_penalty": 1.4,
        "min_length": 15,
        "max_length": 200,
        "tokens_offset": 0,
        "top_k": 50,
        "no_repeat_ngram_size": 5,
        "num_beams": 1,
    }

    def setEnvironment(self, newenv):
        self.environment = dict(newenv)  # не помню уже зачем, но тут надо сделать shallow copy
        del newenv
        # print('new env', self.environment)

    def getResult(self):
        return ''.join(self.lines)

    def getParams(self):
        return self.params

    def chatbot(self, inp="мда...", context=""):
        self.params = {
            "do_sample": True,
            "top_p": 0.95,
            "temperature": 0.2,
            # 0.0001 - 0.15 адекватные ответы, но слегка монотонные.
            # 0.2-0.3 ответы отличаются, менее адекватные, менее логичные но веселые
            # начиная с 0.7 модель путается в командах чаще, ставит лишние пробелы
            "repetition_penalty": 1.03,
            "min_length": 10,
            "max_length": 150,
            "tokens_offset": 0,
            "top_k": 5,
            "no_repeat_ngram_size": 2,
            "num_beams": 3,
            "max_time": 12,
        }
        nrr = NonRepeatRandom(self.repeatingDict)
        from prompts import PromptDataset
        PromptDataset = importlib.reload(sys.modules['prompts']).PromptDataset
        ppt = PromptDataset(self.repeatingDict)
        # if self.environment["env"]=="minecraft":#
        #    self.params["max_length"] = 70

        myname = "Ева"
        exampleName = "Кожаный"
        username = self.username
        self.lines = []

        # раскидала нубикса в железке
        # скаманула по-плотному, теперь можно и...
        # о какая пещера здесь можно построить скам-машину
        # бахнула мишек фредди пожилым динамитом
        cringeNegativePlus = 'нищикс кринжевоз чел бовдурчик бовдурус глистик лицемер псевдоморалист тараканыш напёрдыш лузерус попёрдыватель курвикс'.split(
            ' ')
        cringeNegativePlusFemale = 'апчихуа кринжекозочка бовдурочка лицемерка псевдоморалистка'.split(' ')
        cringeNegative = 'бяка фукич кулебяка изич лысик пердедус мусорикс кринжик подвыпердыш подмёток дикарь бомжик лысик нубикс попытикс дединсайд пердюка пердед пердун штырик крот крынжик нищенка пёсик глистовод тараканчикс подпёрдыш напёрдыш пупкошмыг гавкошмыг крысолов мамонт маздыч куропатыч копатыч школофончик роблоксер'.split(
            ' ')
        cringeNegativeFemale = 'маздочка подмышка куропатка'.split(' ')
        # cringeNeutralPril = 'пожилой подводный'.split(' ')
        cringeNeutral = 'мильфуньич ботовод чикипук пупсик мишка шершень челикс дедус клещ глистыш бравлер грек бебрик чебоксар павук нубикс попытикс милфхантер лолихантер дединсайд скамер чикипук огурец крош крот ботикс пупа пёсик шершень мамонт пупкошмыг гавкошмыг'.split(
            ' ')
        cringeNeutralFemale = 'карпетка милфунья милфхантерша милфа'.split(' ')
        cringePositive = 'мишка бро милфунья крош котэ мармелад киборг крепыш силач качок'.split(' ')
        cringePositiveFemale = 'карпетка куропатка'.split(' ')
        cringePositivePlus = 'любовь лапотулечикс умничка'.split(' ')
        cringePositivePlusFemale = 'милашечка лапочка'.split(' ')

        def cho(mas):
            if len(mas) > 0:
                return random.choice(mas)
            else:
                return None

        def clamp(n, smallest, largest):
            return max(smallest, min(n, largest))

        env = self.environment["env"]

        if self.environment.get("manual_instruct", False):
            env = "broadcast"
            self.environment["broadcast_type"] = "manual_instruct"

        rank = self.environment.get("user_rank", 3)
        if rank >= 6:
            alias = cho(cringePositive + cringePositivePlus)
        elif rank >= 4:
            alias = cho(cringePositive)
        elif rank >= 3.7:
            alias = cho(cringeNeutral + cringePositive)
        elif rank >= 2.8:
            alias = cho(cringeNegative + cringeNeutral)
        elif rank >= 2.3:
            alias = cho(cringeNegative + cringeNegative + cringeNegativePlus)
        elif rank >= 1.5:
            alias = cho(cringeNegative + cringeNegativePlus)
        else:
            alias = cho(cringeNegativePlus)
        rank_map = {0: ["ущербненький", "убожеский", "обиженный", "недостойный", "жалкий"],
                    1: ["глупый", "недалёкий", "поехавший", "неугомонный", "кринжовенький"],
                    2: ["печальненький", "усталый", "глупенький", "обычненький"],
                    3: ["странненький", "заскамленный", "интересненький"],
                    4: ["добренький", "понимающий", "честненький", "хорошенький"],
                    5: ["любимый", "симпатичный", "топовый"],
                    6: ["обожаемый", "прекрасный"]
                    }
        now = datetime.now()
        nowHour = now.hour
        w_time = "утро"
        if nowHour >= 0 and nowHour <= 4:
            w_time = "поздний вечер"
        elif nowHour > 4 and nowHour <= 8:
            w_time = "ночь"
        elif nowHour > 8 and nowHour <= 12:
            w_time = "раннее утро"
        elif nowHour > 12 and nowHour <= 18:
            w_time = "день"
        elif nowHour > 18 and nowHour <= 22:
            w_time = "вечер"
        elif nowHour > 22 and nowHour <= 24:
            w_time = "поздний вечер"

        w_mood_num = self.environment.get("i_mood", 0)
        if w_mood_num > 8:
            w_mood = "ПРЕКРАСНОЕ"
        elif w_mood_num > 5:
            w_mood = "отличное"
        elif w_mood_num > 3:
            w_mood = "хорошее"
        elif w_mood_num > 1:
            w_mood = "хорошее"
        elif w_mood_num > -1:
            w_mood = "кринжовенькое"
        elif w_mood_num > -5:
            w_mood = "плохое"
        elif w_mood_num <= -5:
            w_mood = "паршивое"
        else:
            w_mood = "неопределенное"
        ingame_info = self.environment.get("ingame_info", {})
        g_block = ingame_info.get("ground_block", "резной каменный кирпич")
        g_item = ingame_info.get("held_item", "алмазный меч")
        g_tasks = ingame_info.get("task_chain", "нет задач")
        diags_count = self.environment.get("diags_count", None)
        do_nick_analyze = self.environment.get("do_nick_analyze", False)

        sentence_type = self.environment.get("sentence_type", "dialogue")

        last_interact_time_string = self.environment.get("last_interact", None)
        if last_interact_time_string:
            last_interact_date = tm(last_interact_time_string)
        else:
            last_interact_date = datetime(2022, 12, 30)
        last_interact = (now - last_interact_date).total_seconds()
        last_interact_phrase = " начинает общение, хочет приветствие"
        if last_interact <= 7200:  # не общались более 2 часов
            last_interact_phrase = " продолжает общение, ждет ответа"
        first = False
        if diags_count is not None:
            if diags_count <= 0:
                first = True

        bad_topics_line = ""
        if self.environment.get("filter_topics", None):
            bad_topics_line += f" Также мне известно, что {username} это" + self.environment["filter_topics"].replace(
                ' ', ', ') + "."
        normalness = self.environment.get("filter_allow", None)
        if normalness is not None:
            if not normalness:
                bad_topics_line += f' {username} общается очень грубо и недопустимо, нужно поставить на место этого дикаря.'
        rank_prilagatelnoe = random.choice(rank_map.get(int(clamp(round(rank), 0, 6)), 3))
        print('[FredExamples] DEBUG RANK', str(rank), 'rank_prilagatelnoe', rank_prilagatelnoe, 'alias', alias)

        # alias = cho(cringe)

        # lol = []
        def gen_entering_prompt():
            if env == "youtube":
                action_prompt = "Я (Ева) сейчас отвечаю в чате на трянсляции своего YouTube канала NetTyan (НетТян, нетян). "
            elif env == "twitch":
                action_prompt = "Я (Ева) сейчас отвечаю в чате на трянсляции своего Twitch канала NeuroDeva (нейродева, другое название - NetTyan). "
            elif env == "minecraft":
                action_prompt = "Я (Ева) сейчас отвечаю в чате игры Minecraft. "
            elif env == "discord":
                action_prompt = "Я (Ева) сейчас разговариваю в Discord. "
            else:
                action_prompt = "Я (Ева) сейчас отвечаю в чате. "
            if sentence_type == "about_system":
                return f"""Я - нейростримерша
Информация, что в данный момент происходит в игре: я стою на блоке {g_block}, у меня в руках {g_item}.
Внутри игры сейчас я вот что делаю: {g_tasks}
Для передвижения в игре я использую мод Baritone (баритон) для майнкрафта версии 1.18.2.
Я, нейросеть Ева NetTyan, состою из нескольких нейронных сетей, мода для майнкрафта и скрипта на Python, который связывает все мои части воедино так, чтобы я могла реагировать на всё, что угодно.
Я (Ева) родилась (была создана) человеком по имени фиг его знает в июле 2023 года.
Общаюсь с хитрыми инфоцыганами, заскамленными мамонтами, кринжовыми школофонами, главный из которых - это {username}. Для меня он {rank_prilagatelnoe} {alias}, его репутация - {rank_prilagatelnoe}.{bad_topics_line}
Доступные команды: !бан, !издевайся, !прыгай, !смейся, !кричи, !убегай. Использование - <команда=!прыгай> 
Доступные эмоции: агрессия, скука, усталость, интерес, смущение, счастье, веселье, страх. Использование - [эмоция=смущение]
"""
            else:
                return f"Я - девушка по имени Ева, мне 20 лет. {action_prompt}Я должна отвечать на сообщения в чате с сарказмом, устраивая интриги и постоянно недоговаривая, чтобы вызывать у всяких юных мамонтов интерес. Также я параллельно веду стрим (трансляцию) на своем канале ютуб НетТян, в котором я играю в Minecraft и с весельем общаюсь в чате. Общаюсь с хитрыми инфоцыганами, заскамленными мамонтами, кринжовыми школофонами, главный из которых - это {username}. Для меня он {rank_prilagatelnoe} {alias}, его репутация - {rank_prilagatelnoe}.{bad_topics_line} Доступные команды: !бан, !прыгай, !смейся, !кричи, !убегай. Использование - <команда=!прыгай> Доступные эмоции: агрессия, скука, усталость, интерес, смущение, счастье, веселье, страх. Использование - [эмоция=смущение]"

        if not (context.strip() != '' and context is not None):
            context = ""

        def BroadcastPrompt():

            bc_type = self.environment.get("broadcast_type", "stream_ad")
            if bc_type == "stream_ad":
                ingame_tasks_line = " Также дополнительно известны твои задачи внутри игры:" + g_tasks if g_tasks != "нет задач" else ""
                p = ppt.getPrompt("stream_ad")
                result = f"""{start_token_diag}{p}"Инструктор": Хорошо, но надо ещё лучше! На данный момент время {w_time}, твое настроение - {w_mood}. В игре майн сейчас ты стоишь на блоке {g_block}, а у тебя в руках {g_item}.{ingame_tasks_line}
</s>
"Ева": """
            elif bc_type == "manual_instruct":
                p = ppt.getPrompt("manual_instruct")
                result = f"""{start_token_diag}{p}"Инструктор": {inp}
</s>
"Ева": """
            else:
                ingame_tasks_line = " Также дополнительно известны твои задачи внутри игры:" + g_tasks if g_tasks != "нет задач" else ""
                p = ppt.getPrompt("status_report")
                result = f"""{start_token_diag}{p}"Инструктор": Хорошо, но надо ещё лучше! На данный момент время {w_time}, твое настроение - {w_mood}. В игре майн сейчас ты стоишь на блоке {g_block}, а у тебя в руках {g_item}.{ingame_tasks_line}
</s>
"Ева": """

            return result

        def NickGreetingPrompt():
            p = ppt.getPrompt("nick_greetings")
            result = f"""{start_token_diag}"обо мне": Меня зовут Ева, мой канал на ютубе называется NetTyan, а на твиче - NeuroDeva. Говоря обо мне, я девушка 20 лет и занимаюсь тем, что анализирую ники из игры майнкрафт и придумываю шутки, основываясь на этих никах, чтобы посмеяться над авторами этих ников и предложить им вариант получше. В конце отзыва я пишу только одну из этих эмоций: скучно, интересно, весело. Мне известно, что человек под ником "{username}" недавно говорил "{inp}".{bad_topics_line} \n[эмоция=интересно]\n</s>.{p}
"{username}": """
            return result

        def GenYTPrompt():
            if sentence_type == "about_system":
                p = ppt.getPrompt("about_system")
                # self.params["model_type"] = "dialog"
            else:
                p = ppt.getPrompt("yt_standart")
                self.params["model_type"] = "dialog"
            result = f"""{start_token_diag}Описание ситуации: {gen_entering_prompt()}
{p}
{context}*{alias} {username}{last_interact_phrase}*
Q: {inp}</s>
A: """
            return result

        def GenMCPrompt():
            if sentence_type == "about_system":
                p = ppt.getPrompt("about_system")
                # self.params["model_type"] = "dialog"
            else:

                p = ppt.getPrompt("mc_standart")
                self.params["model_type"] = "dialog"
            # p = ppt.getPrompt("mc_QA")
            result = f"""{start_token_diag}Описание ситуации: {gen_entering_prompt()}
{p}
{context}*{alias} {username}{last_interact_phrase}*
Q: {inp}</s>
A: """
            return result

        def chs(check, checkmas, outmas):
            for i, lol in enumerate(checkmas):
                if lol == check:
                    return outmas[i]
            return ""

        def GenDonationEventPrompt():
            sum = self.environment.get("summ", 10)

            result = f"""{start_token_diag}Описание ситуации: готь  {alias} {username} подарил мне {str(sum)} зелен."""
            # "успокаиваем" нейросеть начиная с начала промпта. Донатерам нужно отвечать с доброжелательностью и уважением.
            if True:  # sum>0: #and sum<100
                result += f""" [эмоция=радость]</s>

*{username} подарил(а) {str(sum)} зелёненьких, {username}{last_interact_phrase}*
Q: {inp}</s>
A: """
            return result

        def GenMCEventPrompt():
            result = ""

            event_type = self.environment.get("type", "")
            result += f"""{start_token_diag}Описание ситуации: я девушка Ева 22 лет, которая любит играть в кубики (майнкрафт). Я придумываю смешные приколы и угары, основываясь на игровом нике и событии, которое произошло с этим ником. {chs(event_type, ["", "death", "kill"], ["а фиг его знает :)", f"меня (Еву) убил {alias} {username}, я зла, разочарована и мне нужно над ним как-то подшутить чтобы над ним посмеялись.", f"я (Ева) уничтожила в игре персонажа {alias} {username} и он умер"])}. В конце я пишу только одну из этих эмоций: скучно, интересно, весело."""
            if event_type == "kill":
                p = ppt.getPrompt("mc_event_kill")
                result += f"""
{p}
"{username}": """
            if event_type == "death":
                p = ppt.getPrompt("mc_event_death")
                result += f"""
{p}
"{username}": """
            return result

        if env == "donation":
            self.lines.append(GenDonationEventPrompt())
            self.params["min_length"] = 100
            self.params["max_length"] = 300
        elif env == "minecraft_event":
            self.lines.append(GenMCEventPrompt())
            self.params["max_length"] = 95
        elif env == "broadcast":
            self.lines.append(BroadcastPrompt())
            self.params["min_length"] = 100
            self.params["max_length"] = 220
        elif do_nick_analyze:
            self.lines.append(NickGreetingPrompt())
            self.params["min_length"] = 45
            self.params["max_length"] = 170
        elif not do_nick_analyze:
            if env == "youtube":
                self.lines.append(GenYTPrompt())
                self.params["max_length"] = 120
            else:  # if env == "minecraft":
                self.lines.append(GenMCPrompt())
                self.params["max_length"] = 75
        else:
            self.lines.append(GenMCPrompt())
            self.params["max_length"] = 75
        if (len(self.lines) == 0):
            self.lines = [""]
        if self.paramsOverride is not None:
            for key in self.paramsOverride.keys():
                # print('изм. key',key,':',self.params[key],self.paramsOverride[key])
                self.params[key] = self.paramsOverride[key]
        result = ''.join(self.lines)
        print('КОНТ3КСТ::: |!|\n', result, '\n|!| КОНЕЦ КОНТЕКСТА:::', self.environment, '\nпарамс=', self.params)
        return result

    def debug(self):
        # self.nickAnalyze(nick="4odedy")
        self.username = "liza5552"
        # env = {"env":"minecraft_event","type":"death"}
        # env = {"env": "donation", "summ": 500}
        # env = {"env": "minecraft", "diags_count":0}
        env = {"env": "broadcast", "diags_count": 0}
        self.environment = env
        self.chatbot(inp="привет анфиса", context="")


if __name__ == "__main__":
    print(' ==*== RUN ISOLATED TESTING LLM EXAMPLES ==*==')
    e = LLMExamples()
    e.environment["do_nick_analyze"] = True
    from prompts import PromptDataset

    e.username = "Maria_AI"
    e.chatbot("Привет! Как дела?")
    print(PromptDataset({}).getPrompt('broadcast'))
    print('result =', e.getResult(), '\nparams = ', e.getParams())
