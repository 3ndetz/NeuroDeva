import threading

import requests
import websockets
import json
import asyncio
import os
import time
import traceback
from datetime import datetime
from threading import Thread


def eztime():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


this_folder = os.path.dirname(os.path.realpath(__file__))
# 14.11.2023 get token
#https://trovo.live/?access_token=ikpsth1tpooaqywx-vxdya&expires_in=864000&token_type=OAuth&scope=user_details_self%20channel_details_self%20channel_update_self%20channel_subscriptions%20chat_send_self%20send_to_my_channel%20manage_messages%20chat_connect&client_id=ee39d87723e9ea1f3d1c184c1338952d&language=en-US&response_type=token&redirect_uri=https%3A%2F%2Ftrovo.live&state=statedata

#old tok eiy2sof5psozpdr9kh6a1q

from HyperAI_Secrets import TrovoClientID,TrovoAccessToken


class TrovoClient:
    def __init__(self):
        self.CLIENT_ID = TrovoClientID
        self.ACCESS_TOKEN = TrovoAccessToken  # todo get from file
        self.SELF_CHANNEL_ID = 229069544
        self.act_listener_thread = None

    def actions_listener(self, trovo_actions_queue):
        listener_active = True
        while listener_active:
            t_act_inp = trovo_actions_queue.get()
            t_act = t_act_inp.get("action", "")
            try:
                if t_act == "reply":
                    self.send_chat("[AI] " + t_act_inp.get("msg", "пустота"))
                    time.sleep(1)
                if t_act == "terminate":
                    listener_active = False
            except BaseException as err:
                print('TROVO action print queue err, q =', t_act)
                print('ОШИБКА ВЫВОДА В ЧАТ TROVO! ', err)
                print('ТЕКСТ ОБ***Й ОШИБКИ', traceback.format_exc())

    def actions_listener_thread_start(self, _trovo_actions_queue):
        self.act_listener_thread = Thread(target=self.actions_listener, args=(_trovo_actions_queue,), daemon=True)
        self.act_listener_thread.start()

    def __del__(self):
        print('[TROVO REQ CLASS REMOVE] Stopping')

    def _generate_request_headers(self) -> dict:
        return {
            'Accept': 'application/json',
            'Client-ID': self.CLIENT_ID,
            'Authorization': 'OAuth ' + self.ACCESS_TOKEN
        }

    def _generate_link_for_new_token(self) -> str:
        return f"""https://open.trovo.live/page/login.html?client_id={self.CLIENT_ID}
        &response_type=token
        &scope=user_details_self+channel_details_self+channel_update_self+channel_subscriptions+chat_send_self+send_to_my_channel+manage_messages+chat_connect
        &redirect_uri=https%3A%2F%2Ftrovo.live
        &state=statedata"""

    def _req(self, url: str, data=None):
        "get request result. If data is None do GET, if data is not none do POST request"
        try:
            if data:
                response = requests.post(url=url, headers=self._generate_request_headers(), data=data)
            else:
                response = requests.get(url=url, headers=self._generate_request_headers())
            data = response.json()
            if data.get("error", "") != "":
                raise Exception("Response returned error. Response dict: " + str(data))
            return data
        except BaseException as err:
            print('[Trovo REQUEST ERR]', err)
        return None

    def validate_access_token(self) -> bool:
        url = "https://open-api.trovo.live/openplatform/validate"
        resp = self._req(url)
        if resp and len(str(resp.get("expire_ts", ""))) > 1:
            print('[trovo DEBUG validating token] expire_ts=', resp.get("expire_ts", "NO INFO"))
            return True
        return False

    def get_chat_token(self):
        url = 'https://open-api.trovo.live/openplatform/chat/token'
        resp = self._req(url)
        if resp:
            token = resp.get("token", None)
        else:
            token = None
        return token

    def send_chat(self, msg: str) -> bool:
        url = 'https://open-api.trovo.live/openplatform/chat/send'
        resp = self._req(url, json.dumps({"content": msg}))  # '{"content":"'+msg+'"}')
        # Trovo REQUEST ERR] Response returned error. Response dict: {'status': -1000, 'error': 'internalServerError', 'message': 'Internal server error.'}
        print("[TROVO CHAT SEND] sended chat, got resp =", resp)
        if resp is not None and resp == {}:  # if success
            return True
        else:
            return False

    def send_command(self, cmd: str) -> bool:  # todo test
        url = 'https://open-api.trovo.live/openplatform/channels/command'
        resp = self._req(url, json.dumps({"command": cmd,
                                          "channel_id": self.SELF_CHANNEL_ID}))  # '{"command":"' + cmd + '", "channel_id":'+str(self.SELF_CHANNEL_ID)+'}')
        # example commands in trovo live: “settitle xxx”, “mod @xxx”, “ban xxx”
        print("[TROVO COMMAND SEND] sended CMD, got resp =", resp)
        if resp and resp.get("is_success", False):  # if success
            print('[TROVO COMMAND SEND] SUCCESS')
            return True
        else:
            print('[TROVO COMMAND SEND] FAILED COMMAND')
            return False


class TrovoWebSocketClient:

    def __init__(self, connect_token: str, _ctx_chat=None):
        if _ctx_chat is None:
            _ctx_chat = []
        self.ctx_chat = _ctx_chat
        self.connected = False
        self.token = connect_token
        self.connection = None
        self.first_connection = True
        print('[TROVO WS INIT] Starting trovo WS session...')
        self.start_session()

    def start_session(self):
        #loop = asyncio.get_event_loop()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        loop.run_until_complete(self.check_connection())
        tasks = [
            asyncio.ensure_future(self.heartbeat_task()),
            asyncio.ensure_future(self.listener_task()),
        ]

        loop.run_until_complete(asyncio.wait(tasks))

    async def check_connection(self):
        """Connecting to webSocket server"""
        if self.connected:
            return True
        else:
            try:
                self.connection = await websockets.connect('wss://open-chat.trovo.live/chat')
                if self.connection.open:
                    print('[TROVO WS CONN] Connection stablished. Client correcly connected')
                    await self.connection.send(json.dumps(
                        {"type": "AUTH",
                         "nonce": "client-auth",
                         "data": {"token": self.token}}))
                    self.connected = True
                    return True
                else:
                    self.connected = False
            except BaseException as err:
                print("[TROVO WS CONN] err while connecting:", err)
            return False

    async def handle_user_message(self, msg_dict: dict):
        print("[TROVO DEBUG WS CHAT] GOT MSG! ", msg_dict)
        try:
            t = msg_dict["type"]
            if msg_dict.get('send_time', 0) > int(time.time()) - 100:  # drop old chat entries
                if t == 0:
                    # https://developer.trovo.live/docs/Chat%20Service.html#_3-4-chat-message-types-and-samples
                    msg = {"env": "trovo", "msg": msg_dict["content"], "user": msg_dict["nick_name"],
                           "trovo_user_channel_id": msg_dict["sender_id"],
                           "processing_timestamp": time.time_ns(), "date": eztime()}
                    if msg["user"] == "NetTyan":  # todo was in ctx.BotNicknames, solve another?
                        print('[YT] Встречено собственное сообщение', msg["user"], 'вносим в базу', msg["msg"])
                    else:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] [TROVO CHAT]", msg["user"], '>', msg["msg"])
                        self.ctx_chat.append(msg)
                else:  # 5001 - подписался с подвыпердом, 5003 - подписался, 5004 - зашел на канал, 5005 - подарил подписки,
                    # 5006 - кто-то подарил что-то другому, 5007 - приветствие рейдера, 5012 - стрим оф\он, 5013 - зритель отписался
                    print('got msg another that 0 type')
        except BaseException as err:
            print('[TROVO WS CHAT PARSE ERR] msg parse err',err)

    async def listener_task(self):
        '''
            Receiving all server messages and handling them
        '''
        # todo err hander
        while True:
            try:
                if not self.connected:
                    raise Exception("not connected to websocket")
                message = await self.connection.recv()
                msg = json.loads(message)
                if msg["type"] != "PONG":
                    if msg["type"] == "CHAT":
                        msg_list = msg["data"]["chats"]
                        for msg_dict in msg_list:
                            await self.handle_user_message(msg_dict)
                await asyncio.sleep(0.05)
                # print('Received message from server: ' + str(message))
            except BaseException as err:
                print('[TROVO WS RECV ERR] err while recieve message', err)
                await asyncio.sleep(5)

    async def heartbeat_task(self):
        """Ping - pong messages to verify connection is alive"""
        while True:
            try:
                if not (await self.check_connection()):
                    raise Exception("not connected to websocket after checking connecting")
                await self.connection.send(json.dumps(
                    {"type": "PING",
                     "nonce": "client-ping"}))
                await asyncio.sleep(25)
            except BaseException as err:
                print('[TROVO WS HEARTBEAT ERR] err while send or recieve ping', err)
                await asyncio.sleep(5)


def run_trovo_client(ctx_chat, trovo_actions_queue):
    trovo_rq = TrovoClient()
    trovo_rq.actions_listener_thread_start(trovo_actions_queue)
    # trovo_rq.send_chat("Упс")
    # trovo_rq.send_command("lol")

    trovo_connect_msg = f"[CONNECTED->{datetime.now().strftime('%M:%S')}] Подключен trovo! Всем привет, система работает =)"  # [AI]
    trovo_actions_queue.put({"action": "reply", "msg": trovo_connect_msg})

    trovo_chat_token = trovo_rq.get_chat_token()
    trovo_ws = TrovoWebSocketClient(connect_token=trovo_chat_token, _ctx_chat=ctx_chat)


def trovo_client_thread(ctx_chat, trovo_actions_queue):
    trovo_thread = threading.Thread(target=run_trovo_client, args=(ctx_chat, trovo_actions_queue,), daemon=True)
    return trovo_thread

if __name__ == "__main__":
    import multiprocessing

    manager = multiprocessing.Manager()

    ctx__chat = manager.list()
    trovo_actions_queuee = manager.Queue()
    # print(json.dumps({"content": "ыыы","id":5})) # выводит {"content": "\u044b\u044b\u044b", "id": 5}
    print('runned client')
    trovo_client_thread(ctx__chat, trovo_actions_queuee).start()
    print('continue code')
    time.sleep(10)
    print(ctx__chat)

    time.sleep(2)
    print(ctx__chat)
    time.sleep(15)
    print('exiting..')