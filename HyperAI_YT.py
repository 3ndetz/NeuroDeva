import time
from datetime import datetime
import pytchat
import requests
import json
import threading
import os
# pip install python-dotenv
from dotenv import load_dotenv
# pip install google-auth-oauthlib
# pip install google-api-python-client
from google_auth_oauthlib.flow import InstalledAppFlow
import random
from googleapiclient.discovery import build
import traceback

print('imported AI YT0')
def eztime():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def tm(x):
    return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')


print('imported AI YT01')


def YoutubeChatListener(ctx, twitch_actions_queue, trovo_actions_queue, ctx_chat):
    print('0[PRE PRE PRE INIT YT + TWITCH] yt listener start....')
    load_dotenv()
    print('[PRE PRE PRE INIT YT + TWITCH] yt listener start....')

    def CheckApp(youtube):
        if youtube is None and ctx.YouTubeAppEnabled:
            youtube = AuthorizeApp()
        return youtube

    def AuthorizeApp():
        file = "HyperAI_Social/youtube/client_secret.json"
        flow = InstalledAppFlow.from_client_secrets_file(file, scopes={
            'openid',
            'https://www.googleapis.com/auth/userinfo.email',
            'https://www.googleapis.com/auth/userinfo.profile',
            'https://www.googleapis.com/auth/youtube',
            'https://www.googleapis.com/auth/youtube.force-ssl',
            'https://www.googleapis.com/auth/youtube.readonly',
        })
        flow.run_local_server(
            host='localhost',
            port=5500,
            authorization_prompt_message="")
        credentials = flow.credentials
        # Building the youtube object:
        youtube = build('youtube', 'v3', credentials=credentials)

        # Settings
        _delay = 1

        # https://github.com/shieldnet/Youtube-livestream-api-bot/blob/master/youtubechat/ytchat.py
        # delete ban

        # https://github.com/nategentile/ban_youtube_bots/blob/main/main.py
        return youtube

    youtube = None
    liveChatId = None

    def getLiveChatId(yt_liveChatId, LIVE_STREAM_ID):
        nonlocal youtube
        """
        It takes a live stream ID as input, and returns the live chat ID associated with that live stream

        LIVE_STREAM_ID: The ID of the live stream
        return: The live chat ID of the live stream.
        """
        if yt_liveChatId is None and ctx.YouTubeAppEnabled:
            stream = youtube.videos().list(
                part="liveStreamingDetails",
                id=LIVE_STREAM_ID,  # Live stream ID
            )

            yt_response = stream.execute()
            # print("\nLive Stream Details:  ", json.dumps(response, indent=2))

            yt_liveChatId = yt_response['items'][0]['liveStreamingDetails']['activeLiveChatId']
            print("\nLive Chat ID: ", yt_liveChatId)
        return yt_liveChatId

    # Access user's channel Name:
    def getUserName(userId):
        """
        It takes a userId and returns the userName.

        userId: The user's YouTube channel ID
        return: User's Channel Name
        """
        channelDetails = youtube.channels().list(
            part="snippet",
            id=userId,
        )
        yt_response = channelDetails.execute()
        # print(json.dumps(response, indent=2))
        userName = yt_response['items'][0]['snippet']['title']
        return userName

    def yt_execute(yt_snippet):
        try:
            response = yt_snippet.execute()
            return response
        except BaseException as err:
            print('[YT ERR] err while executing:', err)
            return False

    def yt_exec(yt_snippet):
        response = yt_execute(yt_snippet)
        if response is False:
            print('[YT ERR EXEC] провалена 1 попытка выполнить запрос, пробуем снова')
            response = yt_execute(yt_snippet)
        return response

    # print(getUserName("UC0YXSy_J8uTDEr7YX_-d-sg"))
    def tempban(yt_liveChatId, channel_id, timee=10):
        nonlocal youtube
        print('до попытки бана')
        ban = youtube.liveChatBans().insert(
            part="snippet",
            body={
                "snippet": {
                    "liveChatId": yt_liveChatId,
                    "type": "temporary",
                    "banDurationSeconds": timee,
                    "bannedUserDetails": {
                        "channelId": str(channel_id)
                    }
                }
            }
        )
        print("[YT LIVECHAT] BAN TO 4ell sent!", yt_exec(ban))

    def sendReplyToLiveChat(yt_liveChatId, message):
        nonlocal youtube
        """
        It takes a liveChatId and a message, and sends the message to the live chat.

        liveChatId: The ID of the live chat to which the message should be sent
        message: The message you want to send to the chat
        """
        if not isinstance(message, str):
            message = "[AI] Сообщение ответа не прошло фильтрацию [ЭТАП4]."
        if len(message) >= 200:
            print('[YOUTUBE LIVECHAT] ДЛИНА СООБЩЕНИЯ ПЕРВЫСИЛО МАКСИМУМ!', len(message))
            message = message[0:150]
        reply = youtube.liveChatMessages().insert(
            part="snippet",
            body={
                "snippet": {
                    "liveChatId": yt_liveChatId,
                    "type": "textMessageEvent",
                    "textMessageDetails": {
                        "messageText": message,
                    }
                }
            }
        )
        print("[YT CHAT BOT] Send message response:", yt_exec(reply))

    def getYoutubeUserId(YOUTUBE_STREAM_API_KEY, YouTubeName):
        channel_ids = requests.get(
            f'https://www.googleapis.com/youtube/v3/search?part=id&q={YouTubeName}&type=channel&key={YOUTUBE_STREAM_API_KEY}').json()[
            'items']
        if len(channel_ids) > 0:
            channel_id = channel_ids[0]['id']['channelId']
            return channel_id
        return None

    # import time

    # pip install pytchat
    # Set API key and YouTube video ID
    # добывается в гугл клауде https://console.cloud.google.com/apis/
    from HyperAI_Social.SocialConfigs import YOUTUBE_STREAM_API_KEY

    # Set YouTube channel ID КАНАЛ ОТКУДА БЕРЕМ СТРИМ. ID канала узнать можно через код элемента поиск channel id
    # [TEST] The Good Life Radio x Sensual Musique https://www.youtube.com/channel/UChs0pSaEoNLV4mevBFGaoKA
    from HyperAI_Social.SocialConfigs import CHANNEL_ID


    # Get channel information
    # чисто url самого канала и его описания иконки и т д
    # url = f"https://www.googleapis.com/youtube/v3/channels?part=snippet%2CcontentDetails%2Cstatistics&id={CHANNEL_ID}&key={API_KEY}"

    # юрл стрима текущего

    YoutubeStreamURL = f"https://www.googleapis.com/youtube/v3/search?part=snippet&channelId={CHANNEL_ID}&eventType=live&type=video&key={YOUTUBE_STREAM_API_KEY}"


    from twitchAPI import Twitch
    from twitchAPI.oauth import UserAuthenticator
    from twitchAPI.types import AuthScope, ChatEvent
    from twitchAPI.chat import Chat, EventData, ChatMessage, ChatSub, ChatCommand
    import asyncio

    from HyperAI_Social.SocialConfigs import TWITCH_APP_ID, TWITCH_APP_SECRET, TWITCH_TARGET_CHANNEL

    TWITCH_USER_SCOPE = [AuthScope.CHAT_READ, AuthScope.CHAT_EDIT]

    twitch_chat = None

    async def twitch_chat_reply(ninp: str):
        nonlocal twitch_chat
        try:
            await twitch_chat.send_message(TWITCH_TARGET_CHANNEL, ninp)
        except BaseException as err:
            print('[TWITCH LIVECHAT ERR] не удалось отправить сообщение', ninp, 'в twitch chat потому что', err)

    async def on_ready(ready_event: EventData):
        print('[TWTICH BOT LOAD] Bot is ready for work, joining channels')

        await ready_event.chat.join_room(TWITCH_TARGET_CHANNEL)
        await twitch_chat_reply(
            f"[AI] [CONNECTED->{datetime.now().strftime('%M:%S')}] Подключен twitch! Всем привет, система работает =)")

    async def on_message(msg: ChatMessage):
        twitch_username = msg.user.name
        # print(f'[TWITCH CHAT {msg.room.name}] {msg.user.name}: {msg.text}')
        msg = {"env": "twitch", "msg": msg.text, "user": twitch_username,
               "processing_timestamp": time.time_ns(), "date": eztime()}
        # pre, rank, user, msg, clan, team, server, serverMode, chat_type, precision
        if twitch_username in ctx.botNicknames:
            print('[YT] Встречено собственное сообщение', msg["user"], 'вносим в базу', msg["msg"])
            # ctx_chatOwn.append(msg)
            ###ctx_chatOwn = ctx_chatOwn + [msg]
            # ctx.LastMineChatInteract = datetime.now()
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [TWITCH CHAT]", msg["user"], '>',
                  msg["msg"])
            ctx_chat.append(msg)

    # this will be called whenever someone subscribes to a channel ПЛАТНАЯ ПОДПИСКА
    async def on_sub(sub: ChatSub):
        print(f'[TWITCH +SUB] New subscription in {sub.room.name}, type: {sub.sub_plan}, msg: {sub.sub_message}')

    # this will be called whenever the !reply command is issued
    async def test_command(cmd: ChatCommand):
        if len(cmd.parameter) == 0:
            await cmd.reply('you did not tell me what to reply with')
        else:
            await cmd.reply(f'{cmd.user.name}: {cmd.parameter}')

    async def run_twitch_bot():
        nonlocal twitch_chat
        twitch = await Twitch(TWITCH_APP_ID, TWITCH_APP_SECRET)
        auth = UserAuthenticator(twitch, TWITCH_USER_SCOPE)
        token, refresh_token = await auth.authenticate()
        await twitch.set_user_authentication(token, TWITCH_USER_SCOPE, refresh_token)
        # await twitch.set_user_authentication('vi4veb8whrz6uacio4ilj9pmkrimk3', TWITCH_USER_SCOPE, ) #access token после ручного запроса
        twitch_chat = await Chat(twitch)
        twitch_chat.register_event(ChatEvent.READY, on_ready)
        twitch_chat.register_event(ChatEvent.MESSAGE, on_message)
        twitch_chat.register_event(ChatEvent.SUB, on_sub)
        # you can directly register commands and their handlers, this will register the !reply command
        twitch_chat.register_command('reply', test_command)
        twitch_chat.start()
        # ЗАКРЫТИЕ!
        # chat.stop()
        # await twitch.close()

    # lets run our setup
    # asyncio.run(run_twitch_bot())
    print('НАЧИНАЕМ ЧЕКАТЬ ЮТУБ...')

    def twitch_actions_executor_func():
        while True:
            if twitch_chat is not None:
                t_act_inp = twitch_actions_queue.get()
                t_act = t_act_inp.get("action", "")
                try:
                    if t_act == "reply":
                        asyncio.run(twitch_chat_reply("[AI] " + t_act_inp.get("msg", "пустота")))
                        time.sleep(1)
                    # elif t_act == "ban":
                    #    if channel_id:
                    #        print('user', channel_id)
                    #        tempban(liveChatId, channel_id=channel_id,
                    #                timee=t_act_inp.get("bantime", 20))
                    #        time.sleep(1)
                    #    else:
                    #        print("БАН НЕ ВЫДАН ТАК КАК НЕ СООБЩЕНО ID")
                except BaseException as err:
                    print('TWICH action print queue err, q=', t_act)
                    print('ОШИБКА ВЫВОДА В ЧАТ TWITCH! ', err)
                    print('ТЕКСТ ОБ***Й ОШИБКИ', traceback.format_exc())
            time.sleep(1)

    def run_twitch_bot_func():
        asyncio.run(run_twitch_bot())
        tt = threading.Thread(target=twitch_actions_executor_func, daemon=True)
        tt.start()

    print('LELL')
    t = threading.Thread(target=run_twitch_bot_func, daemon=True)
    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(run_twitch_bot())

    #
    twitch_started = False
    # выше так было
    # а теперь стало тк тест надо же сделать

    # t.start()
    # twitch_started = True
    from HyperAI_Social.TrovoClient import trovo_client_thread
    print('STARTING YT CHECKER! 00')
    trovo_started = False
    trovo_thread = trovo_client_thread(ctx_chat, trovo_actions_queue)
    print('STARTING YT CHECKER! 01')
    while ctx.ThreadsActived:
        if ctx.YouTubeCommentCheckerEnabled:
            print('[PRE INIT YT] включил коммент чекер? вход в ветку ютуба и твича для запуска непосредственно')
            if not trovo_started:
                try:
                    trovo_thread.start()
                except BaseException as err:
                    print('[TROVO]ОШИБКА ПОДКЛЮЧЕНИЯ! TROVO BOT! ', err)
                    print('[TROVO]ТЕКСТ ОБ***Й ОШИБКИ', traceback.format_exc())
                    print("\n[TROVO]=== КОНЕЦ ОШИБКИ ====")
                trovo_started = True

            if not twitch_started:
                try:
                    t.start()

                except BaseException as err:
                    print('[TWITCH]ОШИБКА ПОДКЛЮЧЕНИЯ! TWITCH BOT! ', err)
                    print('[TWITCH]ТЕКСТ ОБ***Й ОШИБКИ', traceback.format_exc())
                    print("\n[TWITCH]=== КОНЕЦ ОШИБКИ ====")
                    twitch_error = True
                twitch_started = True
            try:
                print('[PRE INIT YT] COMMENT CHECHING ENABLED!!! RUNNING TWITCH BOT...')

                print('[PRE INIT YT] TWITCH INIT ENDED! RUNNING YT BOT...')
                response = requests.get(YoutubeStreamURL)
                print(response)
                streams = json.loads(response.text).get('items', [])
                chat = None
                VIDEO_ID = None
                liveChatId = None
                print('YT>> отправлен запрос к каналу ютуб...')
                if (len(streams) > 0):
                    firstStream = streams[0]
                    VIDEO_ID = firstStream['id']['videoId']
                    print('YT>>стрим найден и подключен. ', firstStream)
                    chat = pytchat.LiveChat(video_id=VIDEO_ID)
                    StreamActived = True
                    youtube = CheckApp(youtube)
                    liveChatId = getLiveChatId(liveChatId, VIDEO_ID)
                    # https://github.com/taizan-hokuto/pytchat/wiki/LiveChat
                else:
                    StreamActived = False
                    time.sleep(10)
                    print('YT>>ERR>> на канале стримов нет в данный момент')
                while ctx.YouTubeCommentCheckerEnabled and StreamActived and chat is not None and chat.is_alive():
                    answered = False
                    try:
                        if len(ctx.YoutubeActionsQueue) > 0:
                            if ctx.YouTubeAppEnabled:
                                q = ctx.YoutubeActionsQueue[0]
                                youtube = CheckApp(youtube)
                                if liveChatId is not None and VIDEO_ID is not None:
                                    liveChatId = getLiveChatId(liveChatId, VIDEO_ID)
                                if youtube is not None and liveChatId is not None:
                                    act = q.get("action", "")
                                    try:
                                        if act == "reply":
                                            sendReplyToLiveChat(liveChatId, "[AI] " + q.get("msg", "пустота"))
                                            time.sleep(1)
                                        elif act == "ban":
                                            # if q.get("ytname", None) is not None:
                                            #   channel_id = getYoutubeUserId(YOUTUBE_STREAM_API_KEY,q.get("ytname"))
                                            channel_id = q.get("youtube_user_channel_id", None)

                                            if channel_id:
                                                print('ban channel_id', channel_id)
                                                tempban(liveChatId, channel_id=channel_id,
                                                        timee=q.get("bantime", 20))
                                                time.sleep(1)
                                            else:
                                                print("БАН НЕ ВЫДАН ТАК КАК НЕ СООБЩЕНО ID")
                                    except BaseException as err:
                                        print('youtube action print queue err, q=', q)
                                        print('ОШИБКА ВЫВОДА В ЧАТ ЮТУБА! ', err)
                                        print('ТЕКСТ ОБ***Й ОШИБКИ', traceback.format_exc())
                                time.sleep(0.2)
                                ctx.YoutubeActionsQueue.pop(0)
                                answered = True
                    except BaseException as err:
                        if not answered and len(ctx.YoutubeActionsQueue) > 0:
                            ctx.YoutubeActionsQueue.pop(0)
                        print('ОШИБКА ПОДКЛЮЧЕНИЯ! ПОХОЖЕ СТРИМ ЗАКОНЧИЛСЯ1! ', err)
                        print('ТЕКСТ ОБ***Й ОШИБКИ', traceback.format_exc())
                        print("\n=== КОНЕЦ ОШИБКИ ====")
                        ctx.IsYTChatConnected = False
                        time.sleep(3)
                    try:
                        data = chat.get()

                        items = data.items
                        # print("lol", items,chat)
                        # обработка каждого сообщения в чате
                        for c in items:
                            # getYoutubeUserId(YOUTUBE_STREAM_API_KEY,c.author.name)
                            if c.message == "!hello lol":
                                ctx.YoutubeActionsQueue.append({"action": "reply", "msg": "hello" + c.author.name})
                            ytname = c.author.name
                            # print(f"YT>>{c.datetime} [{col(str(thissrank))}|{col(ytname)}] {col(c.message, 'yellow')}")
                            msg = {"env": "youtube", "msg": c.message, "user": ytname,
                                   "youtube_user_channel_id": c.author.channelId,
                                   "youtube_moderator": c.author.isChatModerator,
                                   "processing_timestamp": time.time_ns(),
                                   "date": eztime()}
                            # pre, rank, user, msg, clan, team, server, serverMode, chat_type, precision
                            if (msg["user"] in ctx.botNicknames):
                                print('[YT] Встречено собственное сообщение', msg["user"], 'вносим в базу', msg["msg"])
                                # ctx_chatOwn.append(msg)
                                ###ctx_chatOwn = ctx_chatOwn + [msg]
                                # ctx.LastMineChatInteract = datetime.now()
                            else:
                                print(f"[{datetime.now().strftime('%H:%M:%S')}] [YOUTUBE CHAT]", msg["user"], '>',
                                      msg["msg"])
                                ctx_chat.append(msg)

                        ctx.IsYTChatConnected = True
                        time.sleep(2)
                    except BaseException as err:
                        print('2ОШИБКА ПОДКЛЮЧЕНИЯ! ПОХОЖЕ СТРИМ ЗАКОНЧИЛСЯ2! ', err)
                        print('ТЕКСТ ОБ***Й ОШИБКИ', traceback.format_exc())
                        print("\n=== КОНЕЦ ОШИБКИ ====")
                        ctx.IsYTChatConnected = False
                        time.sleep(10)
            except BaseException as err:
                print('1ОШИБКА ПОДКЛЮЧЕНИЯ! ПОХОЖЕ СТРИМ ЗАКОНЧИЛСЯ111! ', err)
                print('ТЕКСТ ОБ***Й ОШИБКИ', traceback.format_exc())
                print("\n=== КОНЕЦ ОШИБКИ ====")
                ctx.IsYTChatConnected = False
                time.sleep(10)
        time.sleep(0.1)


if __name__ == "__main__":
    import multiprocessing

    manager = multiprocessing.Manager()
    ctx = manager.Namespace()
    ctx.IsYTChatConnected = False
    ctx_chat = manager.list()
    ctx_twitch_actions_queue = manager.Queue()
    ctx_trovo_actions_queue = manager.Queue()
    ctx.YoutubeActionsQueue = manager.list()
    ctx.YouTubeCommentCheckerEnabled = False  # todo debug true nado (твич тестил)
    ctx.botNicknames = ["Net Tyan", "NetTyan", "neurodeva", "NeuroDeva"]
    ctx.ThreadsActived = True
    ctx.YouTubeAppEnabled = True
    ctx.IsYTChatConnected = False
    twitch_actions_queue = manager.Queue()
    twitch_actions_queue.put({"action": "reply", "msg": f"""Как дела? Тест =)"""})
    YoutubeChatListener(ctx, ctx_twitch_actions_queue, ctx_trovo_actions_queue, ctx_chat)
