import asyncio
import datetime
import io
import time

import discord
from discord.ext import commands, tasks

from datetime import datetime
def eztime():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
def tm(x):
    return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

class pycord_voice_client():
    def __init__(self, bot, autonomus=False, ctx_chat=None, guild_id=None, docker_sender=None):
        if ctx_chat is None:
            self.ctx_chat = []
        else:
            self.ctx_chat = ctx_chat
        self.stt_class = docker_sender
        self.bot = bot
        if guild_id is None:
            self.guild_id = 1122595942986694658
        else:
            self.guild_id = guild_id
        self.guild = None

        self.autonomus = autonomus
        self.listening_loop_started = False
        self.sink = None
        self.vc = None
        self.listening_loop_delay = 1.0
        self.debug_text_channel = None


        #init recogloop  variables
        self.voice_streak_found = 0
        self.silence_streak_found = 0
        self.voice_confirmed_iters = 0
        self.voice_recog_stared = False
        self.prev_voice_recog_stared = False
        self.collected_voice_data = {}
    def start_listening(self):
        self.listening_loop_started = True
    def stop_listening(self):
        self.listening_loop_started = False
    def start_recognition_functions(self):

        stt_class = self.stt_class
        bot = self.bot
        def speech_recognition(bytes_io_audio):
            if stt_class is not None:
                return stt_class.transcribe(audio_bytes_io=bytes_io_audio)
            else:
                return "Not implemented =( надо stt class передать, а он None почему-то"
        connections = {}
        #class Sinks(Enum):
        #    mp3 = discord.sinks.MP3Sink()
        #    wav = discord.sinks.WaveSink()
        #    pcm = discord.sinks.PCMSink()
        #    ogg = discord.sinks.OGGSink()
        #    mka = discord.sinks.MKASink()
        #    mkv = discord.sinks.MKVSink()
        #    mp4 = discord.sinks.MP4Sink()
        #    m4a = discord.sinks.M4ASink()
        #self.listening_loop_started = False

        async def finished_callback_sendfiles(sink, channel: discord.TextChannel, *args):

            #await sink.vc.disconnect()
            #audio wav Audio: PCM 48000Hz stereo 1536kbps [A: pcm_s16le, 48000 Hz, 2 channels, s16, 1536 kb/s]
            #Audio: PCM 48000Hz stereo 1536kbps [A: pcm_s16le, 48000 Hz, 2 channels, s16, 1536 kb/s]
            #Audio: 0x0000 48000Hz stereo 1536kbps [A: adpcm_dtk, 48000 Hz, stereo, s16]
            ###audiolist = []
            ###for user_id, audio in sink.audio_data.items():
            ###    f = wave.open(audio.file, "rb")
            ###    audiolist.append([f.getparams(), f.getnframes()])
            ###    print('params =',f.getparams())
            ###    f.close()
            ###    #f.seek(0)
            #wave merge https://stackoverflow.com/questions/2890703/how-to-join-two-wav-files-using-python
                #f.seek(0) #0 - установить курсор в начало файла, 1 - на файл в последнюю измененную видимо позицию, 2 - в конец файла
            #print('AUDIOLIST >>',audiolist,'<<')
            recorded_users = [f"<@{user_id}>" for user_id, audio in sink.audio_data.items()]
            if len(recorded_users)>0:

                files = []
                for user_id, audio in sink.audio_data.items():

                    send_filename = bot.get_user(user_id).display_name +"-" + datetime.datetime.now().strftime('%Mm-%Ss') + f".{sink.encoding}"
                    files.append(discord.File(audio.file, send_filename))
                await channel.send(
                    f"Finished! Recorded audio for {', '.join(recorded_users)}.", files=files
                )
        async def finished_callback_recognize(sink, channel: discord.TextChannel, *args):
            #recorded_users = [f"<@{user_id}>" for user_id, audio in sink.audio_data.items()]
            sink_audio_data = sink.audio_data.copy()
            sink_audio_data_items = sink_audio_data.items()
            if len(sink_audio_data_items) > 0:
                speechmas = []
                for user_id, audio in sink_audio_data_items:
                    #files.append(audio.file)
                    recognized = speech_recognition(audio.file)
                    if recognized.strip() != "" and len(recognized)>2:
                        username = bot.get_user(user_id).name
                        speechmas.append([username,recognized,user_id])
                        self.ctx_chat.append({"date":eztime(), "user":username, "msg":recognized,"env":"discord",
                        "server":"1122595942986694658", "discord_id":str(user_id), "processing_timestamp":time.time_ns()})
                        #await channel.send(f"""<@{user_id}>: {recognized}""")
                if len(speechmas)>0:
                    nw = datetime.now().strftime('%M:%S')
                    out = []
                    for record in speechmas:
                        line = "["+nw+"] [РАСПОЗНАНА РЕЧЬ] "+record[0]+": "+record[1]
                        print(f"[DISCORD VC] {line}")
                        out.append(line)
                    await channel.send("\n".join(out))

        def loop_start():
            print('[DISCORD VC]DEBUG LOOP START')
            self.guild = bot.get_guild(self.guild_id)
            if self.debug_text_channel is None:
                self.debug_text_channel = bot.get_channel(1132660411477544962)
            print(self.debug_text_channel,self.guild)

            voice_recognition_loop.start()


        if self.autonomus:
            @bot.event
            async def on_ready():
                # print('\n'.join(map(repr, self.get_all_channels())))
                #loop_start()

                print('[PYCORD] Logged in to Discord as {} - ID {}'.format(bot.user.name, bot.user.id))
                print('all systems ready')
                #loop_start()
                #voice_recognition_loop.start()

            @bot.command(aliases=['слушай'], pass_context=True)
            async def vc_listen(ctx):
                """Record your voice!"""
                voice = ctx.author.voice

                if not voice:
                    return await ctx.send("You're not in a vc right now")
                self.debug_text_channel = bot.get_channel(
                    1132660411477544962)  # <- open test closed test 1130078872256389120)
                self.vc = await voice.channel.connect()
                connections.update({ctx.guild.id: self.vc})
                #print('БРЕД')
                #sink = discord.sinks.WaveSink()
                #vc.start_recording(sink, finished_callback_recognize, self.debug_text_channel, )
                await ctx.send("The LISTENING has started!")

                self.listening_loop_started = True



        async def check_vc_initialize():
            if self.vc is not None:

                if self.sink is None:
                    self.sink = discord.sinks.WaveSink()
                #if self.sink.finished:
                #    self.sink = discord.sinks.WaveSink()
                if not self.vc.recording:
                    self.vc.start_recording(self.sink, finished_callback_recognize, self.debug_text_channel, )
                    #self.vc.start_recording(self.sink, finished_callback_recognize, self.debug_text_channel, )
                return True
            else:
                if self.guild is not None:
                    self.vc = discord.utils.get(self.bot.voice_clients, guild=self.guild)
                else:
                    self.guild = self.bot.get_guild(self.guild_id)
                    self.vc = discord.utils.get(self.bot.voice_clients, guild=self.guild)
                return False

        @tasks.loop(seconds=self.listening_loop_delay)
        async def voice_recognition_loop():
            #print('cho')
            if self.listening_loop_started:
                #print('cho2')
                initialized = await check_vc_initialize()
                if not initialized:
                    print('not init')
                    return
                vc = self.vc
                listening_loop_delay = self.listening_loop_delay
                voice_streak_found = self.voice_streak_found
                silence_streak_found = self.silence_streak_found
                voice_confirmed_iters = self.voice_confirmed_iters
                voice_recog_stared = self.voice_recog_stared
                prev_voice_recog_stared = self.prev_voice_recog_stared

                collected_voice_data = self.collected_voice_data


                #listening_loop_delay = 1
                #await asyncio.sleep(listening_loop_delay)

                audios = self.sink.get_all_audio()
                sound_found = False
                if audios is not None:
                    for lol in audios:
                        sound_found = True
                        break
                # print('[DISCORD VOICE DEUBG] soundfound,len_collected',sound_found,len(collected_voice_data))
                if sound_found:
                    silence_streak_found = 0
                    voice_streak_found += 1
                    # print('Речь обнаружена!')
                    for user, audio in self.sink.audio_data.items():
                        if user not in collected_voice_data:
                            file = io.BytesIO()
                            collected_voice_data.update({user: discord.sinks.AudioData(file)})
                        collected_voice_data[user].write(audio.file.getvalue())
                        # collected_voice_data = collected_voice_data | sink.audio_data
                    if voice_streak_found > 1:
                        # print('НАЧАТО РАСПОЗНАВАНИЕ РЕЧИ!')
                        voice_confirmed_iters += 1
                        voice_recog_stared = True
                    if voice_confirmed_iters > 0:
                        pass
                        # print('sound record iter',voice_confirmed_iters,'time =',voice_confirmed_iters*listening_loop_delay)
                    if voice_confirmed_iters * listening_loop_delay >= 10:  # 10 сек
                        # print('sound record time reached')
                        # print('Завершение записи речи')
                        voice_recog_stared = False
                        voice_confirmed_iters = 0
                else:
                    silence_streak_found += 1
                    voice_streak_found = 0
                    if voice_confirmed_iters > 0:
                        if voice_recog_stared == True:
                            # print('ВАННА ОЧИЩЕНА!!!!')
                            # sink.cleanup()
                            voice_recog_stared = False
                    elif voice_confirmed_iters > 1:  # более 2 пустот подряд
                        # print('Завершение записи речи')
                        voice_recog_stared = False
                        voice_confirmed_iters = 0
                    if silence_streak_found > 1:
                        collected_voice_data.clear()
                # if sound_found_attempts
                if prev_voice_recog_stared != voice_recog_stared:
                    if voice_recog_stared:
                        pass
                        # print('ЗАПИСЬ РЕЧИ CONFIRMED! len =',len(collected_voice_data))
                    else:
                        # print('СТОП 100% ЗАПИСЬ РЕЧИ!')
                        voice_confirmed_iters = 0
                        await asyncio.sleep(1)
                        if vc.recording and not vc.paused:
                            vc.toggle_pause()
                        # sink.audio_data = sink.audio_data | collected_voice_data
                        self.sink.audio_data = collected_voice_data
                        if vc.recording:
                            vc.stop_recording()
                        await asyncio.sleep(1)
                        # print('RECORD RESTARTING!')
                        self.sink = discord.sinks.WaveSink()
                        vc.start_recording(self.sink, finished_callback_recognize, self.debug_text_channel, )
                        # merge collected data!
                        collected_voice_data.clear()
                    prev_voice_recog_stared = voice_recog_stared
                # sink.cleanup()
                self.sink.audio_data = {}
                self.collected_voice_data = collected_voice_data
                self.listening_loop_delay = listening_loop_delay
                self.voice_streak_found = voice_streak_found
                self.silence_streak_found = silence_streak_found
                self.voice_confirmed_iters = voice_confirmed_iters
                self.voice_recog_stared = voice_recog_stared
                self.prev_voice_recog_stared = prev_voice_recog_stared
        @bot.command(aliases=['запись'], pass_context=True)
        async def record(ctx):
            """Record your voice!"""
            voice = ctx.author.voice

            if not voice:
                return await ctx.send("You're not in a vc right now")

            vc = await voice.channel.connect()
            connections.update({ctx.guild.id: vc})

            vc.start_recording(
                discord.sinks.WaveSink(),
                finished_callback_recognize,
                ctx.channel,
            )
            await ctx.send("The recording has started!")


        @bot.command(aliases=['стоп'], pass_context=True)
        async def stop(ctx: discord.ApplicationContext):
            """Stop recording."""
            self.listening_loop_started = False
            if ctx.guild.id in connections:
                vc = connections[ctx.guild.id]
                vc.stop_recording()
                del connections[ctx.guild.id]
                await ctx.message.delete()
            else:
                await ctx.send("Not recording in this guild.")

        #loop_start()
        loop_start()
        #voice_recognition_loop.start()
if __name__ == '__main__':
    intents = discord.Intents().all()

    bot = commands.Bot(command_prefix=('!', 'eva ', 'Eva ', 'ева ', 'ева, ', 'Ева ', 'Ева, ', 'неттян, '),
                       intents=intents)
    from docker_sender import DockerSTTSender
    from HyperAI_Secrets import DiscordToken
    stt_class_client_sender = DockerSTTSender()

    bot_voice_class = pycord_voice_client(bot, autonomus=True, stt_class=stt_class_client_sender)
    bot_voice_class.start_recognition_functions()
    #bot_voice_class.
    bot.run(DiscordToken)