import os
import time
import asyncio
import threading
import traceback
import datetime
import random

from discord import ChannelType
from HyperAI_Models.STT.stt_discord_pycord import pycord_voice_client
# from HyperAI_Models.STT.docker_sender import DockerSTTSender
from HyperAI_Docker import DockerSender


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


def eztime():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def tm(x):
    return datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')


def calcTime(time):
    return bcolors.OKGREEN + str((datetime.datetime.now() - time).total_seconds()) + bcolors.ENDC


###USAGE
# import MainControlDiscord
#                from MainControlDiscord import MainControlDiscord as MCD
#                DiscordClass = MCD(CNNMainInput,EnterInputEv,ScriptedProcesses,VisoredName,VisoredEv)
class MainControlDiscord:
    ####https://discord.com/developers/applications/919901202177736714/bot
    ####https://habr.com/ru/post/494600/
    VisoredName = [["4o"]]

    def __init__(self, main_ctx, ctxx, ctx_chat, docker_sender=None):
        self.docker_sender = docker_sender
        self.guild_id = 1122595942986694658
        self.recognition_client = None
        timeStartMetrics = []
        timeStartMetrics.append(datetime.datetime.now())

        def GetTimeDelta(timemas):
            if len(timemas) > 0:
                timemas.append(datetime.datetime.now())
                return round((timemas[-1] - timemas[0]).total_seconds(), 1)
            else:
                return 0

        def admincmd(cmds):
            if not isinstance(cmds, str):
                for lol in cmds:
                    print(lol)
                    os.system(lol)
            else:
                os.system(cmds)

        def installLibraries():
            CmdInstallLibraries = [
                """pip install pycord['VOICE']""",
            ]
            admincmd(CmdInstallLibraries)
            print('ДИСКОРД: Вроде как закончили с установкой! загружаемся!!!')

        try:
            import discord
        except ImportError as imperr:
            print(f'\n\nДИСКОРД: ОШИБКА ИМПОРТА БИБЛИОТЕК!!!\n{imperr}\n ==== Запускаем установщик!!!!!! ====\n\n ')
            installLibraries()
            import discord
        from discord.ext import commands, tasks
        from discord.ext.commands import has_permissions

        def tbool(bol, type='text'):
            if type == 'text':
                if bol:
                    return 'ода'
                else:
                    return 'неа'
            if type == 'color':
                if bol:
                    return 'green'
                else:
                    return 'red'

        def cutIndex(sstr, shablon, caseIgnore=True):
            if caseIgnore:
                sstr = sstr.lower()
                shablon = shablon.lower()
            mm = sstr.find(shablon)
            if mm != -1:
                if mm + len(shablon) == len(sstr):
                    return ''
                else:
                    return sstr[sstr.find(shablon) + len(shablon):]
            else:
                return False

        # print('НАШЛИ ИЛИ НЕТ >'+cutIndex('алисаа','алиса')+'<')
        print('[ДИСКОРД МОДУЛЬ] ИНИЦИАЛИЗИРУЕМСЯ...', GetTimeDelta(timeStartMetrics))
        from HyperAI_Secrets import DiscordToken
        self.TOKEN = DiscordToken
        self.CPREF = '.'

        # class MyBot(commands.Bot):
        #    async def setup_hook(self):
        #        #pass
        #        self.loop.create_task(checker_loop())
        intents = discord.Intents().all()

        # intents = discord.Intents(messages = True, guilds = True, reactions = True, members = True, presences = True)

        def generate_nickname_variants(nicks: set) -> list:
            """ТОЛЬКО ДЛЯ ДИСКОРДА!!!! ДЛЯ ДРУГИХ ИСПОЛЬЗУЙТЕ ДРУГОЕ!!! ТУТ ЗАПЯТЫЕ ДОБАВЛЯЮТСЯ И ПРОБЕЛЫ"""
            out = []
            for nick in nicks:
                if len(nick) <= 2:
                    print('мелкий ник')
                    continue
                out.append(nick)  # "ева"
                out.append(nick[0].upper() + nick[1:])  # Ева
            out2 = []
            for nick in out:
                out2.append(nick + ",")
            out.extend(out2)
            for i, nickVar in enumerate(out):
                out[i] = nickVar + " "

            # nicks.update(out)
            return out  # nicks

        bot_command_prefixes = generate_nickname_variants(
            {"ева", "eva", "неттян", "нетян", "тян", "натян", "нетТян", "nettyan", "netTyan"})
        bot_command_prefixes.extend(['!', self.CPREF])
        print('Bot command prefixes (nicks) =', bot_command_prefixes)
        activity = discord.Game(name="симулятор скамера кринжовых школофонов")
        self.bot = commands.Bot(command_prefix=bot_command_prefixes, intents=intents, activity=activity)
        # self.alice = commands.Bot(command_prefix="")
        # bot = client
        self.LastMyMsgs = []
        self.MsgsQueue = []
        # asyncio.run(self.LastMsgsChecker())
        ####
        ####loop = asyncio.new_event_loop()
        ####def f(loop):
        ####    asyncio.set_event_loop(loop)
        ####    loop.run_forever()
        #####async def csglobal():
        #####    #await self.LastMsgsCheckerLoop()
        #####    asyncio.ensure_future(self.LastMsgsCheckerLoop())
        ####
        ####thread2 = threading.Thread(target=f, args =(loop,),daemon = True)
        ####thread2.start()
        ####
        ####@asyncio.coroutine
        ####def g():
        ####    yield from asyncio.sleep(1)
        ####    print('Hello, world!')
        ####def mda(lol):
        ####    while True:
        ####        if len(self.LastMyMsgs)>0:
        ####            print('мда удаляем',self.LastMyMsgs[0])
        ####            a1 = loop.create_task(self.DeleteMsg())
        ####            #loop.run_until_complete(a1)
        ####            #self.LastMyMsgs[0].delete()
        ####
        ####loop.call_soon_threadsafe(mda, g())
        # self.client = discord.Client()
        firstCheck = True
        CheckCtx = None
        CheckStartMsg = None

        def UpdateEvents():
            # nonlocal MsgsQueue
            nonlocal firstCheck
            lastViv = ''
            while True:
                time.sleep(0.01)

                if firstCheck:
                    firstCheck = False
                    if len(self.VisoredName[0]) > 2:
                        emb = discord.Embed(title='ПРОСЛУШКА')
                        emb.add_field(name='==== ' + self.VisoredName[0] + ' ====',
                                      value='-    начинаем прослушку    -', inline=True)
                        emb.add_field(name='==== последние 5 логов ====',
                                      value='-    выводим последние 5 сообщений    -', inline=False)
                        self.MsgsQueue.append([emb, 15])
                        outmas(self.VisoredName[1])
                        for lol in self.VisoredName[1]:
                            if lol != '':
                                self.MsgsQueue.append([lol, 10])
                        emb2 = discord.Embed(title='НАЧИНАЕМ ВЫВОДИТЬ ЧАТ')
                        emb2.add_field(name='! ==== ' + self.VisoredName[0] + ' ==== !', value='-    ты кто НН    -',
                                       inline=True)
                        self.MsgsQueue.append([emb2, 10])
                        time.sleep(1)
                else:
                    try:
                        if (len(self.VisoredName) > 1):
                            if len(self.VisoredName[1]) > 4:
                                vivod = self.VisoredName[1][4]
                                if vivod != '' and lastViv != vivod:
                                    self.MsgsQueue.append([vivod, 8])
                                    lastViv = vivod
                    except BaseException as err:
                        print('[DISCORD] ОШИБКА ПРИ ПЕЧАТАНИИИ ВЫВОДА!!!!')
                        print('[DISCORD] ТЕКСТ ОБ***Й ОШИБКИ:\n', err, '\nТРЕЙСБЕК:\n', traceback.format_exc())
                        print('[DISCORD] КОНЕЦ ТРЕЙСБЕКА')

                # MsgsQueue.append()
                # VisoredEvAS.set()
                # print('Обновляем visored AS!',VisoredEvAS.is_set())
                # VisoredEvAS.clear()
                # time.sleep(0.01)

        thread2 = threading.Thread(target=UpdateEvents, daemon=True)
        thread2.start()

        def CheckPerms(id):
            DevIds = ['464798128021700619', '', '']
            if id in DevIds:
                return True
            else:
                return False

        def outmas(maaas):
            maaasn = "из " + str(len(maaas)) + " элем."
            print('\n ==== ВЫВОД МАССИВА ' + maaasn + ' ===== \n')
            for i, lol in enumerate(maaas):
                print(f'{i} >{str(lol)}<')
            print('\n ==== КОНЕЦ ' + maaasn + ' ===== \n')

        async def SendInChat(ctx, whatto):
            msg = await ctx.send(whatto)
            self.LastMyMsgs.append(msg)

        async def SendInChatQ(whatto, RemoveTime):
            nonlocal CheckCtx
            RemoveTime = 0
            if not CheckCtx is None and whatto != '':
                msg = None
                # print('ДАУН')
                # print(isinstance(whatto, discord.embeds.Embed))
                if not isinstance(whatto, discord.embeds.Embed):  # cutIndex(str(type(whatto)),'embed'):
                    if len(whatto) > 0 and len(whatto) < 1000:
                        if RemoveTime != 0:
                            msg = await CheckCtx.send(whatto, delete_after=RemoveTime)
                        else:
                            msg = await CheckCtx.send(whatto)
                    else:
                        print('[Discord] Сообщение не соответствует требованиям. Слишком много или нет знаков.')
                else:
                    if RemoveTime != 0:
                        msg = await CheckCtx.send(embed=whatto, delete_after=RemoveTime)
                    else:
                        msg = await CheckCtx.send(embed=whatto)
                if not msg is None:
                    self.LastMyMsgs.append(msg)

        @tasks.loop(seconds=0.1)
        async def checker_loop():
            youtube_chat_connected = False
            self.stream_link = 'https://www.youtube.com/@NetTyan'
            twitch_stream_link = "https://www.twitch.tv/neurodeva"
            while True:
                try:
                    # do something
                    # tsm = [datetime.datetime.now()]
                    # print('running loop',len(self.MsgsQueue))
                    await asyncio.sleep(0.1)  # 0.01 DEBUG
                    if len(self.MsgsQueue) > 0:
                        nsg = self.MsgsQueue[0][0]
                        ttm = self.MsgsQueue[0][1]
                        if len(self.MsgsQueue) > 2:
                            IsEmbed = False
                            for lol in self.MsgsQueue:
                                if isinstance(lol[0], discord.embeds.Embed):
                                    IsEmbed = True
                                    nsg = lol[0]
                                    ttm = lol[1]
                                    break
                                nsg += lol[0] + '\n'
                                self.MsgsQueue.remove(lol)
                            if IsEmbed:
                                # await SendInChatQ(nsg,ttm)
                                self.MsgsQueue.remove([nsg, ttm])
                        else:
                            self.MsgsQueue.remove([nsg, ttm])
                        await SendInChatQ(nsg, ttm)
                    if main_ctx.IsYTChatConnected:
                        if not youtube_chat_connected:
                            await self.bot.change_presence(activity=discord.Streaming(name="Стрим NetTyan!",
                                                                                      platform="Twitch",
                                                                                      game="Minecraft",
                                                                                      url=twitch_stream_link))  # self.stream_link))
                            youtube_chat_connected = True
                    elif youtube_chat_connected:
                        youtube_chat_connected = False
                    if ctxx.ds_actions_q.qsize() > 0:

                        act = ctxx.ds_actions_q.get()
                        act_type = act.get("type", "")
                        if act_type == "stream_start":
                            print('[DISCORD ACTION WORKER] ПОЛУЧЕН ЗАПРОС НА АНОНС СТРИМА!')
                            self.stream_link = act.get("link", self.stream_link)
                            channel = self.bot.get_channel(1128963785571434506)
                            timeString = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
                            mention_text = ""
                            if act.get("do_mention", True):
                                mention_text = "@everyone \n"
                            hello_text = mention_text + "НАЧАЛСЯ СТРИМ :fire: :sunglasses: "
                            emb = discord.Embed(title='ЗАЛЕТАЙ НА СТРИМ =) Ссылочки на Twitch и Ютуб ниже!',
                                                url=self.stream_link,
                                                description=act["msg"],
                                                color=discord.Color.brand_green()
                                                )
                            emb.set_author(name="NetTyan", url=self.stream_link,
                                           icon_url="https://static-cdn.jtvnw.net/jtv_user_pictures/ed3cdd8e-8de2-4acd-a2f8-5b30b99eed3e-profile_image-70x70.png")
                            emb.add_field(name='YouTube', value=self.stream_link, inline=True)
                            emb.add_field(name='Twitch', value=twitch_stream_link, inline=True)
                            allowed_mentions = discord.AllowedMentions(everyone=True)
                            stream_message = await channel.send(content=hello_text, embed=emb,
                                                                allowed_mentions=allowed_mentions)
                            await stream_message.publish()
                            thread = await channel.create_thread(name="стрим " + timeString,
                                                                 type=ChannelType.public_thread,
                                                                 message=stream_message)
                            await thread.send("Сюда можете присылать угарные таймкоды со стрима и комментировать)")
                        elif act_type == "test_message":
                            print('[DISCORD ACTION WORKER] ПОЛУЧЕН ЗАПРОС НА ВЫПИСЫВАНИЕ СООБЩЕНИЯ!')
                            channel = self.bot.get_channel(1132660411477544962)
                            await channel.send(act["msg"])
                        elif act_type == "answered_message":
                            print('[DISCORD ACTION WORKER] ПОЛУЧЕН ЗАПРОС НА ВЫПИСЫВАНИЕ СООБЩЕНИЯ В ОТВЕТЫ НА СТРИМЕ!')
                            channel = self.bot.get_channel(1160988908172099685) #ОТВЕТЫ НА СТРИМЕ
                            await channel.send(act["msg"])
                        elif act_type == "filtered_message":
                            print('[DISCORD ACTION WORKER] ОТФИЛЬТРОВАННОЕ СООБЩЕНИЕ!')
                            channel = self.bot.get_channel(1160988972730810399)  # ОТФИЛЬТРОВАНО
                            await channel.send(act["msg"])

                    while len(self.LastMyMsgs) > 10:
                        # print('мда удаляем',self.LastMyMsgs[0])
                        if not self.LastMyMsgs[0] is None:
                            await self.LastMyMsgs[0].delete()
                            # if len(self.LastMyMsgs)>11:
                            #    await asyncio.sleep(0.05)
                        self.LastMyMsgs.pop(0)
                    # print('ИТЕРАЦИЯ >',GetTimeDelta(tsm))
                except BaseException as err:
                    print('[DISCORD] ОШИБКА ПРИ ПЕЧАТАНИИИ ВЫВОДА В ЧАТ ДСа!!!!')
                    print('[DISCORD] ТЕКСТ ОБ***Й ОШИБКИ:\n', err, '\nТРЕЙСБЕК:\n', traceback.format_exc())
                    print('[DISCORD] КОНЕЦ ТРЕЙСБЕКА')

        # self.bot.loop.create_task(checker_loop())
        # async def main():
        # #   async with self.bot:
        #        self.bot.loop.create_task(my_task())
        # self.bot.loop.create_task(my_task())
        # loop.run_until_complete(asyncio.gather(my_taskTwo()))

        @self.bot.event
        async def on_voice_state_update(member, prev, cur):

            channel = self.bot.get_channel(1122595944798625925)
            user = f"{member.name}#{member.discriminator}"
            # print('loll',member, prev, cur)
            if cur.channel == channel:  # в канале для траснляций?
                if not (cur.afk or cur.deaf or cur.mute):
                    if cur.self_mute and not prev.self_mute:  # Would work in a push to talk channel
                        print(f"[DISCORD VOICE CHANNEL] {user} stopped talking! (micro off)")
                    elif (prev.self_mute and not cur.self_mute):  # As would this one
                        print(f"[DISCORD VOICE CHANNEL] {user} started talking! (micro on)")
                    elif not cur.self_mute:
                        print(f"[DISCORD VOICE CHANNEL] {user} started talking! (other, join or smth)")

            elif prev.channel == channel:
                print(f"[DISCORD VOICE CHANNEL] {user} stopped talking! (leaved)")

            if cur.afk and not prev.afk:
                print(f"[DISCORD VOICE CHANNEL] {user} went AFK!")
            elif prev.afk and not cur.afk:
                print(f"[DISCORD VOICE CHANNEL] {user} is no longer AFK!")

        @self.bot.event
        async def on_message(msg):
            answer = ''

            if msg.author == self.bot.user:
                return
            # print('DEBUG GET CTX')
            # ctx = await self.bot.get_context(msg)
            # print(ctx)
            # print(msg.channel.id)
            # СТРИМЫ 1128963785571434506
            # ОБЩЕНИЕ 1123869287552127007
            # МЕМЫ 1122595944521797788
            # БАГИ 1122595944521797789
            # ЗНАКОМСТВО 1122595944521797787
            # ОБЪЯВЛЕНИЯ 1122595944521797784
            if msg.channel.id == 1130078872256389120:
                text = msg.content
                print('КОМАНДА В МАЙНКРАФТ >>'+text)
                if text:
                    if len(text)>1:
                        if text[0] == "!":
                            ctx_chat.append(
                                {"date": eztime(), "user": msg.author.name,
                                 "priority_group": "max",
                                 "manual_instruct": True,
                                 "msg": text,
                                 "env": "discord",
                                 "server": "1122595942986694658", "discord_id": str(msg.author.id),
                                 "processing_timestamp": time.time_ns()})
                        else:
                            main_ctx.BridgeChatQueue.append(msg.content)
            # print('END OF CTX')
            if msg.content.startswith('хихихи'):
                ####msg.author.mention - выделение автора через @
                ####msg.author - полное имя ДС автора вместе с #2321
                ####msg.author.name - просто имя автора
                ####
                # answer = 'Может, мне тоже посмеяться, {0.author.mention} ?'.format(msg) #С ВЫДЕЛЕНИЕМ ЧЕРЕЗ @
                answer = 'Может, мне тоже посмеяться, {0} ?'.format(msg.author.name[0:5] + 'что-то там')
                # await msg.reply(answer, delete_after = 15, mention_author=False)#msg.channel.send(answer)
            if msg.content.startswith('ЦифруМне'):
                answer = '{0}, твой DISCORD ID = >{1}<'.format(msg.author.name[0:5] + 'что-то там', msg.author.id)
            for bot_nick in bot_command_prefixes:
                if msg.content.lower().find(bot_nick) != -1:

                    if msg.content.find('300') != -1:
                        answer = 'Я ТВОЮ МАТЬ Е'
                    elif msg.content.find('ответь ') != -1:
                        # answer = 'услышала вопрос, анализирую ответ...'
                        await msg.reply(f"<@{str(msg.author.id)}> услышала вопрос, думаю над ответом...",
                                        delete_after=6,
                                        mention_author=True)
                        ctx_chat.append(
                            {"date": eztime(), "user": msg.author.name, "msg": str(cutIndex(msg.content, 'ответь ')),
                             "env": "discord",
                             "server": "1122595942986694658", "discord_id": str(msg.author.id),
                             "processing_timestamp": time.time_ns()})

                    elif cutIndex(msg.content, 'скажи ') == '':
                        answer = 'нет не ' + cutIndex(msg.content, 'скажи ')
                    elif cutIndex(msg.content, 'алиса') == '':
                        answer = 'а?'
            if answer != '':
                await msg.reply(answer, mention_author=False)
            else:
                if CheckPerms(str(msg.author.id)):
                    await self.bot.process_commands(msg)
                else:
                    pass
            #        await msg.reply(f"п**** ***** тип на {msg.author.name[0:5]}что-то там! Вот кста цифры твоей мамы: >{msg.author.id}<", delete_after = 6, mention_author=False)
            #        await msg.delete()

        @self.bot.event
        async def on_ready():
            print('[DISCORD] МОДУЛЬ Подключен + готов к работе! >', GetTimeDelta(timeStartMetrics))
            print('[DISCORD] LOGIN {} - ID {}'.format(self.bot.user.name, self.bot.user.id))
            ctxx.loading_flag.set()
            # await self.bot.change_presence(activity=discord.Game(name="симулятор скамера кринжовых школофонов <3"))
            # подключаемся к VC
            guild = self.bot.get_guild(1122595942986694658)  # id сервера
            self.guild = guild
            checker_loop.start()
            if self.recognition_client is not None:
                self.recognition_client.start_recognition_functions()
            else:
                print("[DISCORD STT] RECOGNITION CLIENT NOT READY! [DISABLED]")
            # guild_ids = [g.id for g in self.bot.guilds]
            # print(guild_ids)
            # print(guild)

            await voice_start(guild=guild)

        @self.bot.event
        async def on_command_error(ctx, error):
            fun_osks = ["лоШара", "ты бот", "кривожоп", "зачитай заклинание че папе че маме",
                        "где девственность оставил"]
            await ctx.message.reply(f"{random.choice(fun_osks)}: {str(error)}", delete_after=6, mention_author=False)
            await ctx.message.delete()  # чистим чат от твоих команд

        # @self.bot.event
        # async def on_message_error(ctx, error):#не робит!
        #    if isinstance(error, discord.ext.commands.errors.CommandNotFound):
        #        await ctx.send("п*****")
        #        await ctx.message.delete() #чистим чат от твоих команд
        #        @self.bot.command(pass_context=True)
        #        async def delcommes(ctx):
        #            await ctx.send("This is the response to the users command!")
        #            await ctx.message.delete()
        # pip install pyaudio
        # pi install PyNaCl
        # python3 -m pip install -U discord.py[voice]
        import pyaudio

        # PYAUDIO DEVICE LIST
        p = pyaudio.PyAudio()
        input_device_count = p.get_device_count()

        device_to_out_name = "CABLE-A Output (VB-Audio Cable A)"
        audio_input_device_index = 32
        for i in range(input_device_count):
            device_info = p.get_device_info_by_index(i)
            device_name = device_info['name']

            device_index = device_info['index']
            if device_name == device_to_out_name:
                audio_input_device_index = int(device_index)
                print(f"Input Device {i}: {device_name} ({device_index}) <<< CHOSEN!")
            else:
                print(f"Input Device {i}: {device_name} ({device_index})")
        #audio_input_device_index = 20

        # input_stream = p.open(format=pyaudio.paInt16, channels=2, rate=48000, input=True,
        #                      input_device_index=audio_input_device_index, frames_per_buffer=960)
        # Input Device 20: CABLE Output (VB-Audio Virtual Cable) (20) ВСЁ НОРМ (было 21)

        # from discord import FFmpegPCMAudio, PCMVolumeTransformer

        # source = FFmpegPCMAudio("test.wav", **FFMPEG_OPTIONS)
        class PyAudioPCM(discord.AudioSource):
            # ПЛОХОЕ КАЧЕСТВО + ЖОПА РАЗРЫВНЫЙ ЗВУК ПРИ ВОСПР. 'index': 2, 'structVersion': 2, 'name': 'CABLE Output (VB-Audio Virtual '
            # Invalid number of channels 'index': 6, 'structVersion': 2, 'name': 'CABLE Input (VB-Audio Virtual C'
            # ПРОСТО ЖОПУ ПОРВАЛО 'index': 10, 'structVersion': 2, 'name': 'CABLE Output (VB-Audio Virtual Cable)'
            # Invalid number of channels 'index': 14, 'structVersion': 2, 'name': 'CABLE Input (VB-Audio Virtual Cable)'
            # то же самое 17, но там 'defaultSampleRate': 48000.0
            # то же самое что в 1 но лучше тк ср внорме {'index': 21, 'structVersion': 2, 'name': 'CABLE Output (VB-Audio Virtual Cable)', 'hostApi': 2, 'maxInputChannels': 2, 'maxOutputChannels': 0, 'defaultLowInputLatency': 0.003, 'defaultLowOutputLatency': 0.0, 'defaultHighInputLatency': 0.01, 'defaultHighOutputLatency': 0.0, 'defaultSampleRate': 48000.0}
            def __init__(self, channels=2, rate=48000, chunk=960, input_device=audio_input_device_index) -> None:
                p = pyaudio.PyAudio()
                print('[DISCORD VOICE] Коннект:', p.get_device_info_by_index(input_device))
                self.chunks = chunk
                self.input_stream = p.open(format=pyaudio.paInt16, channels=channels, rate=rate, input=True,
                                           input_device_index=input_device, frames_per_buffer=chunk)

            def read(self) -> bytes:
                return self.input_stream.read(self.chunks, exception_on_overflow=False)
            # USING vc.play(PyAudioPCM(), after=lambda e: print(f'Ошибка в проигрывании error: {e}') if e else None)

        # pip install ffmpeg
        import ffmpeg

        @self.bot.command(aliases=['звук'], pass_context=True)  # разрешаем передавать агрументы
        async def voice_test(ctx):  # создаем асинхронную фунцию бота
            await voice_test(ctx)

        async def voice_start(ctx=None, guild=None):  # создаем асинхронную фунцию бота

            # 1122595944798625925
            print('Voice starting initiated')
            channel = self.bot.get_channel(1122595944798625925)  # голосовой Комната для траснляций

            contextless = False

            if ctx is None:
                contextless = True
            if ctx is None and guild is None:
                print('Дебил?')
                return
            if contextless:
                vc = discord.utils.get(self.bot.voice_clients, guild=guild)
            else:
                vc = ctx.voice_client
            # vc.start_recording
            # vc = discord.utils.get(self.bot.voice_clients, guild=ctx.guild)
            if vc is not None:
                await vc.move_to(channel)
                print('Moving to channel..')
            else:
                print('Connecting..')
                vc = await channel.connect()
            if vc is None:
                print("FFFF НЕ ПОДКЛЮЧИЛИСЬ")

            if self.recognition_client is not None:
                self.recognition_client.start_listening()

                # vc = await ctx.voice_client.move_to(channel)
            # ИГРАТЬ ЗВУК! #
            # vc.play(discord.FFmpegPCMAudio(executable="C:/ffmpeg/bin/ffmpeg.exe", source="F:\\Onix\\Pictures\\1Pet\\NetTyan\\стрим\playlist\\2 fonk+doki\\0КУКИЕС.mp3"))
            # 0КУКИЕС.mp3# 2surprise.ogg # #F:/Onix/Downloads/minebot/1HyperAI/GTTS (гугл транслейтор).mp3

            # ИГРАТЬ УСТРОЙСТВО (МИКРО И Т Д) #
            vc.play(PyAudioPCM(), after=lambda e: print(f'Ошибка в проигрывании error: {e}') if e else None)
            if not contextless:
                await ctx.message.reply('Тестирование звука...', delete_after=6, mention_author=False)
                await ctx.message.delete()  # чистим чат от твоих команд
            # vc.send_audio_packet(input_stream.read(960, exception_on_overflow = False), encode=False)
            # print('done playing')

        @self.bot.command(aliases=['clear', 'чистка', 'посудомойка'])  # разрешаем передавать агрументы
        @has_permissions(manage_messages=True)
        async def clearCommand(ctx, amount: int):  # создаем асинхронную фунцию бота
            amount += 1
            print('[DISCORD] очищаем чат на', amount)
            await ctx.channel.purge(limit=abs(int(amount)))

        @clearCommand.error
        async def on_errr(ctx, err):
            if isinstance(err, commands.MissingPermissions):
                await ctx.message.reply('У тя нет прав бездарь тупой пзхвапзва на ' + str(ctx.message.author),
                                        delete_after=6, mention_author=False)

        @self.bot.command()  # разрешаем передавать агрументы
        async def ahelp(ctx):  # создаем асинхронную фунцию бота
            print('[DISCORD] запрос на команду помощи!')
            emb = discord.Embed(title='Навигация по командам')
            emb.add_field(name=self.CPREF + 'clear', value='Очистить чат', inline=False)
            emb.add_field(name=self.CPREF + 'stat', value='статус', inline=False)
            emb.add_field(name=self.CPREF + 'gs', value='глобальное смс ботярам', inline=False)
            emb.add_field(name=self.CPREF + 'ch', value='чек ник_ботяры', inline=False)

            await ctx.message.reply(embed=emb, mention_author=False)

        @self.bot.command(aliases=['stat', 'status', 'стат', 'статус'])  # разрешаем передавать агрументы
        async def statusbots(ctx):  # создаем асинхронную фунцию бота
            print('[DISCORD] запрос на вывод СТАТУСА')
            emb = discord.Embed(title=f'Самочувствие NetTyan',
                                description="Вывод важных показателей статуса")
            emb.add_field(name=str(main_ctx.mood), value="настроение", inline=True)
            emb.add_field(name="[YT ON]" if main_ctx.IsYTChatConnected else "[YT OFF]", value="youtube", inline=True)
            ingame = main_ctx.ingame
            emb.add_field(name="[MC ON]" if ingame else "[MC OFF]", value="minecraft",
                          inline=True)
            if ingame:
                emb.add_field(name=str(main_ctx.ingame_info.get("ground_block","пустота"))+" | "+str(main_ctx.ingame_info.get("held_item","ничего")), value="блок | вещь",
                              inline=True)
                emb.add_field(name=str(main_ctx.ingame_info.get("task_chain", "нет задач")), value="игровые задачи",
                              inline=False)



            await ctx.send(content="Докладываю статус:",embed=emb)

            # emb.add_field(name = '-статус-', value = '-вывод статуса-', inline=False)
            for k, q in enumerate(ctx_chat):

                emb = discord.Embed(title=f'#{str(k)} {q.get("user", "??? (UNDEFINED)")}',
                                    description=str(q.get("date")))
                emb.add_field(name='Сказал:', value=q.get("msg","(ничего)"), inline=False)
                filt_allowed = q.get("filter_allowed", None)


                if filt_allowed is not None:
                    filt_result = "Допуск: "
                    filt_result += "Да" if filt_allowed else "Нет"

                    filt_topics = str(q.get("filter_topics", ""))
                    if filt_topics:
                        filt_result+= "; Темы: " + filt_topics
                    sentence_type = str(q.get("sentence_type",""))
                    if sentence_type:
                        emb.add_field(name='Тип сообщения:', value=sentence_type, inline=True)

                    emb.add_field(name='Фильтрация:', value=filt_result, inline=True)
                emb.add_field(name='Откуда?', value=q.get("env","???"), inline=True)
                if k == 0:
                    await ctx.send(content="Вывод состояния чата:",embed=emb)



                # chat_entry
            # , delete_after = 20)
            # await ctx.message.delete() #чистим чат от твоих команд

        ####await ctx.send(embed = discord.Embed(title = 'Текст', description = f'Текст', colour = 0x09F2C8), delete_after = 10)
        ####Выше пример кода, тебе нужно использовать delete_after и число через которое бот удалит своё сообщение, пример delete_after = 10
        # @self.bot.command(aliases=['','?'])
        # async def voidcmmd(ctx):
        #    await ctx.send( 'а?' , delete_after = 3)
        #    await ctx.message.delete() #чистим чат от твоих команд
        ####@self.bot.command(aliases=[' ', 'скажи'],pass_context=True)  # разрешаем передавать агрументы
        ########ДЛЯ РАЗРЕШЕНИЙ ТОЛЬКО АДМИНАМ:
        ########commands.has_permissins (administrator = True)
        ####async def trollll(ctx, *args):  # создаем асинхронную фунцию бота
        ####    arg = " ".join(args)
        ####    print('получили',arg)
        ####    if arg.find('300') != -1:
        ####        answer = 'Я ТВОЮ МАТЬ Е'
        ####    else:
        ####        answer = 'нет не '+arg
        ####    await SendInChat(ctx,answer)
        ####    await ctx.message.delete() #чистим чат от твоих команд
        ####    #await ctx.send(answer)  # отправляем обратно аргумент
        ####    print('отправили',answer)
        ####    #print('мас ',self.LastMyMsgs)

        @self.bot.command(aliases=['гс', 'gs'], pass_context=True)  # разрешаем передавать агрументы
        ####ДЛЯ РАЗРЕШЕНИЙ ТОЛЬКО АДМИНАМ:
        ####commands.has_permissins (administrator = True)
        async def GlobalSendMsg(ctx, *args):  # создаем асинхронную фунцию бота
            arg = " ".join(args)
            # print('получили',arg)
            # answer = 'нет не '+arg
            # await SendInChat(ctx,answer)
            # ctxx.CNNMainInput.value = arg
            # ctxx.EnterInputEv.set()
            # print('[DISCORD GLOBAL]',ctxx.CNNMainInput.value)
            # ctxx.EnterInputEv.clear()
            await ctx.message.reply('отправила:\n' + arg, delete_after=6, mention_author=False)
            await ctx.message.delete()  # чистим чат от твоих команд
            # if not CheckStartMsg is None and len(arg)<3:
            #    CheckStartMsg.delete()
            # await ctx.send(answer)  # отправляем обратно аргумент
            # print('отправили',answer)
            # print('мас ',self.LastMyMsgs)

        @self.bot.command(aliases=['пр', 'ch', 'check', 'чек'], pass_context=True)  # разрешаем передавать агрументы
        ####ДЛЯ РАЗРЕШЕНИЙ ТОЛЬКО АДМИНАМ:
        ####commands.has_permissins (administrator = True)
        async def CheckMsgs(ctx, arg):  # создаем асинхронную фунцию бота
            if True:
                nonlocal firstCheck
                nonlocal CheckCtx
                nonlocal CheckStartMsg
                if not CheckStartMsg is None:
                    await CheckStartMsg.delete()
                    CheckStartMsg = None
                for lol in range(10):
                    self.LastMyMsgs.append(None)
                await asyncio.sleep(0.5)
                CheckCtx = ctx
                firstCheck = True
                # VisoredEv.set()
                # VisoredEvAS.set()
                # VisoredEvAS.clear()
                # VisoredEv.clear()

                # EnterInputEv.clear()
                if len(self.VisoredName[0]) > 2:
                    print('[DISCORD GLOBAL] ПРОСЛУШИВАЕМ БОТЯРУ НА', arg)
                    CheckStartMsg = await ctx.message.reply('Чекаем ботяру:\n' + arg + '\n___________',
                                                            mention_author=False)
                else:
                    await ctx.message.reply('закончила прослушивать: указан ' + arg, delete_after=5,
                                            mention_author=False)
                    print('[DISCORD GLOBAL] Заканчиваем прослушивать: указан - ', arg)

                # await ctx.send(answer)  # отправляем обратно аргумент
                # print('отправили',answer)
                # print('мас ',self.LastMyMsgs)
            else:
                await ctx.message.reply('нет такого: ' + arg, delete_after=5, mention_author=False)
            await ctx.message.delete()  # чистим чат от твоих команд

        @self.bot.command(pass_context=False)
        async def удалиласт(ctx):
            print('получили удалиласт')
            await self.DeleteMsg()
            print('тип удалили')

        print('[DISCORD] модуль загружен ФАЗА 1', GetTimeDelta(timeStartMetrics))

        # @asyncio.coroutine
        # async def shutdown():
        #    print('ЗАКРЫВАЕМСЯ')
        #    await self.bot.close()
        #    #exit()
        #    await self.bot.login(self.TOKEN, bot = True)
        #    print('ЗАКРЫЛИСЬ')
        def RunBot():
            SuccesfullRun = False
            FirstRun = True
            # self.bot.run(self.TOKEN)
            try:
                if FirstRun:
                    print('[DISCORD] начинаем запуск.....', GetTimeDelta(timeStartMetrics))
                    #if self.docker_sender is None:
                    #    print(
                    #        '[DISCORD VC] ДОКЕР СЕНДЕР DOCKER SENDER НЕ ПЕРЕДАН КАК АРГУМЕНТ! ИНИЦИАЛИЗИРУЕМ САМОСТОЯТЕЛЬНО!!!!')
                    #    # for use in MAIN надо добавить add path для локального докер сендера
                    #    self.docker_sender = DockerSender()
                    if self.docker_sender is not None:
                        self.recognition_client = pycord_voice_client(self.bot, autonomus=False, ctx_chat=ctx_chat,
                                                                      guild_id=self.guild_id, docker_sender=docker_sender)
                    else:
                        print('[DISCORD STT PRE-INIT] RUN WITHOUT STT!!!')
                    # pycord_voice_client(self.bot,autonomus=False)
                    self.bot.run(self.TOKEN)  # , log_handler=None) # log_handler=None отключает Логи дискорда
                    FirstRun = False
            except BaseException as err:
                # print('>'+str(err)+'<')
                while not SuccesfullRun and str(
                        err) == 'Cannot connect to host discord.com:443 ssl:default [getaddrinfo failed]':
                    try:
                        print('ОШИБКА ПОДКЛЮЧЕНИЯ К СЕРВЕРУ! ПЕРЕЗАПУСК ЧЕРЕЗ 3 СЕК...')
                        # asyncio.run(shutdown())
                        # self.bot.loop.stop()
                        time.sleep(3)
                        # asyncio.run(shutdown())
                    except BaseException as errn:
                        if str(errn) == 'Cannot connect to host discord.com:443 ssl:default [getaddrinfo failed]':
                            pass
                        else:
                            SuccesfullRun = True
                else:
                    SuccesfullRun = True
                    print('[DISCORD] ОШИБКА ПРИ ЗАПУСКЕ!!!!')
                    print('[DISCORD] ТЕКСТ ОБ***Й ОШИБКИ:\n', err, '\nТРЕЙСБЕК:\n', traceback.format_exc())
                    print('[DISCORD] КОНЕЦ ТРЕЙСБЕКА')
                    time.sleep(3)
                # print('Бегу по тропинке в голове ляллляля')
                # self.client.run(self.TOKEN)

        RunBot()
        # threadMain = threading.Thread(target=RunBot,daemon = True)
        # threadMain.start()
        # time.sleep(10)
        # print('[DISCORD] модуль загружен ФАЗА 2',GetTimeDelta(timeStartMetrics))
        # ctxx.loading_flag.set()

    async def DeleteMsg(self):
        await self.LastMyMsgs[0].delete()
        self.LastMyMsgs.pop(0)

    async def LastMsgsCheckerLoop(self):
        while True:
            await asyncio.sleep(0.01)
            if len(self.LastMyMsgs) > 2:
                print('мда удаляем', self.LastMyMsgs[0])
                await self.LastMyMsgs[0].delete()
                # await self.DeleteMsg()


def DiscordProcess(ctx, main_ctx, ctx_chat, docker_sender=None):
    print('[DISCORD MODULE] Инициирован запуск ДИСКОРД ПРОЦЕССА')
    t = datetime.datetime.now()
    start = True
    while start:
        try:
            MainControlDiscord(main_ctx, ctx, ctx_chat,
                               docker_sender=docker_sender)  # DiscordClass = MainControlDiscord(ctx)
        except BaseException as err:
            print('[DISCORD ERROR STOP]')
            print('[DISCORD] ОШИБКА В ГЛАВНОМ ПОТОКЕ!!!!')
            print('[DISCORD] ТЕКСТ ОБ***Й ОШИБКИ:\n', err, '\nТРЕЙСБЕК:\n', traceback.format_exc())
            print('[DISCORD] КОНЕЦ ТРЕЙСБЕКА')
            time.sleep(5)
            print('[DISCORD RESTART INITIATED]')
    print('Дискорд процесс завершен')
    # print('[DISCORD MODULE] дискорд запущен! Время:', calcTime(t))
    # while True:
    #    time.sleep(1)


if __name__ == "__main__":
    #                from MainControlDiscord import MainControlDiscord as MCD
    #            DiscordClass = MCD(CNNMainInput,EnterInputEv,ScriptedProcesses,VisoredName,VisoredEv)
    import multiprocessing

    manager = multiprocessing.Manager()
    discordCtx = manager.Namespace()
    ctx = manager.Namespace()
    ctx.IsYTChatConnected = False
    ctx.ingame = True
    ctx.mood = 0.1
    ctx.ingame_info={}
    ctx_chat = manager.list()
    discordCtx.ds_actions_q = manager.Queue()
    discordCtx.loading_flag = manager.Event()
    docker__ssender = DockerSender()

    DiscordProcess(main_ctx=ctx, ctx=discordCtx, ctx_chat=ctx_chat, docker_sender=docker__ssender)

    # DiscordProc = multiprocessing.Process(
    #    target=DiscordProcess,
    #    args=(discordCtx,),
    #    kwargs={"docker_sender": docker__ssender}
    # )  # Thread(target = a, kwargs={'c':True}).start()
    # DiscordProc.start()

    # DiscordClass = MainControlDiscord(ctx)
    # print(e.getResult()+'mda')
    inp = input("4odedy")
    # print('время запуска '+calcTime(t))
    # async def LastMsgsChecker(self):
    #    import threading
    #    import asyncio
    #    thread2 = threading.Thread(target=asyncio.run, args =(self.LastMsgsCheckerLoop(),),daemon = True)
    #    thread2.start()
    #    return thread2
