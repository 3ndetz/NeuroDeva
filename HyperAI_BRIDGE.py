import datetime
import multiprocessing
import sys
import traceback
from py4j.java_gateway import JavaGateway, CallbackServerParameters
from HyperAI_Secrets import Razrabs
import numpy as np
import threading
# Connect to the Java gateway server
import copy
# print(str(gateway))
# print(gateway.entry_point)
import time
import queue
from datetime import datetime
import numpy


# НУЖЕН ОТДЕЛЬНЫЙ ПРОЦЕСС(((
def eztime():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def tm(x):
    return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')


def mainBridge(mc_vt_ctx, ctx, ctx_chat, ctx_chatMsgs, ctx_chatOwn):
    captchaQueueInput = multiprocessing.Queue()
    captchaQueueOutput = multiprocessing.Queue()

    def solve_captcha_worker():
        while True:
            image_bytes = captchaQueueInput.get()
            sys.path.insert(0, f'HyperAI_Helpers/captcha_solver/4_vk_mod/code')
            from onnx_inference import solve_captcha
            result = solve_captcha(image_bytes)
            captchaQueueOutput.put(result)

    def requestCaptchaResolve(image_bytes):
        if captchaQueueInput.qsize() > 0:
            while not captchaQueueInput.empty():
                captchaQueueInput.get()
            # with captchaQueueInput.mutex:
            #    captchaQueueInput.queue.clear()
            #    captchaQueueInput.all_tasks_done.notify_all()
            #    captchaQueueInput.unfinished_tasks = 0
        captchaQueueInput.put(image_bytes)

    gateway = None

    threading.Thread(target=solve_captcha_worker).start()
    # ctx_chatMsgs = []

    while ctx.ThreadsActived:
        try:
            ctx.lastmsg = "hz"
            print('запуск python callback')

            class PythonCallback(object):
                def isStarted(self):
                    return True

                def onCaptchaSolveRequest(self, image_bytes):
                    print('GOT CAPTCHA REQUEST!')
                    requestCaptchaResolve(image_bytes)

                def onUpdateServerInfo(self, infoMas=None):
                    for cho in infoMas:
                        ctx.GameInfo[cho] = infoMas[cho]
                    print('GameInfoUpdated', ctx.GameInfo)

                def onVerifedChat(self, msgMas=None):
                    if msgMas is None:
                        pass
                        # msgMas = {}
                    else:
                        # print("RECIEVED BIG MSG:",msgMas.get("user",""),'>',msgMas.get("msg",""),' CLAN=',msgMas.get("clan","")," NONEXIST=",msgMas.get("gdfjkgdf",""))
                        msgDict = {"user": msgMas.get("user", ""),
                                   "msg": msgMas.get("msg", ""),
                                   }
                        # pre, rank, user, msg, clan, team, server, serverMode, chat_type, precision
                        fields = ["pre", "rank", "clan", "team", "server", "serverMode", "chat_type", "precision"]
                        for field in fields:
                            if msgMas.get(field, "") is not None:
                                if msgMas.get(field, "").strip() != "":
                                    msgDict[field] = msgMas.get(field, "")
                        ctx_chatMsgs.append(msgDict)
                    return msgMas

                def onChatMessage(self, msg=""):
                    # print(string)
                    ctx.lastmsg = msg
                    # print('recieved msg >>',msg)
                    return msg

                def onDeath(self, killer="unknown"):
                    if killer is not None and killer.strip() != "" and killer != "unknown":
                        ctx.eventlist.append({"type": "death", "user": killer, "happiness_score": -3, "date": eztime()})
                    ctx.MineEventName = "death"
                    ctx.MineEvent.set()
                    ctx.MineEvent.clear()
                    # print('MINECRAFT DEATH FROM',killer)

                def onKill(self, killed="unknown"):
                    if killed is not None and killed.strip() != "" and killed != "unknown":
                        ctx.eventlist.append({"type": "kill", "user": killed, "happiness_score": 1, "date": eztime()})
                    ctx.MineEventName = "kill"
                    ctx.MineEvent.set()
                    ctx.MineEvent.clear()

                def onDamage(self, amount=0):
                    pass
                    # print('MINECRAFT DAMAGE =',str(amount))

                class Java:
                    implements = ["adris.altoclef.PythonCallback"]

            cb = PythonCallback()
            # gateway = JavaGateway()
            gateway = JavaGateway(
                callback_server_parameters=CallbackServerParameters(),
                python_server_entry_point=cb,
                start_callback_server=True
            )
            # start_callback_server=True)

            # print('запуск gateway entry point')
            # try:
            #     print(gateway.entry_point.inGame())
            # except BaseException as err:
            #     print('err, ',err)

            e = gateway.entry_point

            # def
            # print("ГЕЙТВЕЦЙ ","")
            def ingame(loop=True):
                if (loop):
                    needLog = False
                    fail = True
                    while fail and ctx.ThreadsActived:
                        try:
                            ez = e.inGame()
                            # ctx.BridgeEntry = e
                            fail = False
                            if (not ctx.IsMCStarted):
                                ctx.IsMCStarted = True
                            if needLog:
                                print('BRIDGE CONNECTION SUCCESSFUL')
                            ctx.ingame = ez
                            return ez
                        except BaseException as err:
                            # print('INGAME ERROR', err)
                            # print('ТЕКСТ ОБ***Й ОШИБКИ', traceback.format_exc())
                            if needLog:
                                print('BRIDGE CONNECTION FAILED', err)
                            ctx.IsMCStarted = False
                            ctx.ingame = False
                            time.sleep(5)
                            # return False
                        finally:
                            needLog = False  # False
                else:
                    try:
                        ctx.BridgeEntry = []
                        ez = e.inGame()
                        if (not ctx.IsMCStarted):
                            ctx.IsMCStarted = True
                        # print('CONNECTION SUCCESSFUL')
                        return ez
                    except BaseException as err:
                        print(
                            'ERROR BRIDGE когда чекал запущенный майн. Его видимо нет в списке процессов или ещё че похуже')
                        ctx.IsMCStarted = False
                        return False

            print('ща чекнем майн')
            time.sleep(1)
            print('Minecraft bridge: В игре? =', ingame())
            mc_vt_ctx.PitchSpeed = 0.0
            ctx.YawSpeed = 0.0
            AngSpeedTableP = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            AngSpeedTableY = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

            def updater():
                ii = 0
                while ctx.ThreadsActived:
                    time.sleep(0.01)
                    # print('DEBUG ACTIVED')
                    if (len(ctx_chatMsgs) > 0):
                        for msg in ctx_chatMsgs:
                            # print('Processing msgsmas', msg)
                            msg["date"] = eztime()
                            msg["processing_timestamp"] = time.time_ns()
                            msg["env"] = "minecraft"
                            # pre, rank, user, msg, clan, team, server, serverMode, chat_type, precision

                            if (msg["user"] in ctx.botNicknames):
                                print('Встречено собственное сообщение', msg["user"], 'вносим в базу', msg["msg"])
                                ctx_chatOwn.append(msg)
                                ###ctx_chatOwn = ctx_chatOwn + [msg]
                                ctx.LastMineChatInteract = eztime()
                            else:
                                print(f"[{datetime.now().strftime('%H:%M:%S')}] [MC CHAT]", msg["user"], '>',
                                      msg["msg"])

                                message_l = msg["msg"].lower()
                                player = msg["user"]

                                if player in Razrabs:
                                    if ingame(loop=False):
                                        try:
                                            if message_l.find("за мной") != -1:
                                                e.RunInnerCommand(f"""@follow {player}""")
                                            elif message_l.find("вперед") != -1:
                                                e.RunInnerCommand("@test killall")
                                            elif message_l.find("мочи") != -1:
                                                e.RunInnerCommand(f"""@punk {msg["msg"].split(' ')[1]}""")
                                            elif message_l.find("стоп") != -1:
                                                e.RunInnerCommand("@stop")
                                        except BaseException as err:
                                            print('ERROR WHILE EXEC RAZRAB COMMAND', err)

                                ctx_chat.append(msg)
                                ###ctx_chat = ctx_chat + [msg]
                        ###ctx_chatMsgs = []
                        # ctx_chatMsgs.clear() не работает для ListProxy manager
                        ctx_chatMsgs[:] = []
                    ####
                    if (ingame()):
                        try:
                            captchaSolved = captchaQueueOutput.get(block=False)
                            if captchaSolved is not None:
                                print("[BRIDGE CAPTCHA] GET CAPTCHA OUTPUT ! Entering in chat >>", captchaSolved, '<<')
                                e.CaptchaSolvedSend(captchaSolved["result"], float(captchaSolved["predict"]))
                        except queue.Empty:
                            pass
                        ii += 1
                        if ii > 30:
                            g_block = e.getGroundBlock()
                            g_item = e.getHeldItem()
                            g_tasks = e.getTaskChainString()
                            ctx.ingame_info = {"task_chain": g_tasks, "ground_block": g_block, "held_item": g_item}
                        if (len(ctx.BridgeChatQueue) > 0) and (
                                datetime.now() - tm(ctx.LastMineChatInteract)).total_seconds() >= 7:
                            ctx.LastMineChatInteract = eztime()
                            chat_msg = ctx.BridgeChatQueue[0]
                            is_command = False
                            try:
                                if chat_msg[0] == "$":
                                    is_command = True
                                    chat_msg = chat_msg[1:]
                            except:
                                is_command = False
                            try:
                                if is_command:
                                    print("[MC] RUN CMD " + chat_msg)
                                    e.RunInnerCommand("@" + chat_msg)
                                else:
                                    print("[MC] RUN CHAT " + chat_msg)
                                    e.ChatMessage(chat_msg)
                            except:
                                pass
                            ctx.BridgeChatQueue.pop(0)
                        if (len(ctx_chatOwn) > 80):
                            ctx_chatOwn.pop(0)
                        goalRotation = e.getGoalRotation()
                        if (goalRotation is None):
                            AngSpeedTableP.append(e.getPitch())
                            AngSpeedTableY.append(e.getYaw())
                            if len(AngSpeedTableP) > 10:
                                AngSpeedTableP.pop(0)
                            if len(AngSpeedTableY) > 10:
                                AngSpeedTableY.pop(0)
                            mc_vt_ctx.PitchSpeed = -AngSpeedTableP[5] + AngSpeedTableP[0]
                            ctx.YawSpeed = -AngSpeedTableY[5] + AngSpeedTableY[0]
                        else:
                            # print('ыы',e.ChatMessage("ПриветМир"))
                            mc_vt_ctx.PitchSpeed = goalRotation.getPitch()
                            ctx.YawSpeed = goalRotation.getYaw()
                    else:
                        # print('НЕ В ИГРЕ!!!')
                        time.sleep(2)

            def listener():
                while ctx.ThreadsActived:
                    time.sleep(0.01)
                    if (ingame()):
                        time.sleep(5)
                        print('@test killall')
                        e.ExecuteCommand("@test killall")
                        time.sleep(3)
                        print('стоп')
                        e.ExecuteCommand("@stop")
                        time.sleep(1)

            print("UPDATER STARTING...")
            updaterThread = threading.Thread(target=updater)
            updaterThread.start()
            updaterThread.join()
            # EventListenerThread = threading.Thread(target=listener)
            # EventListenerThread.start()
            # while True:
            # time.sleep(0.05)
            # mc_vt_ctx.PitchSpeed = PitchSpeed+1+mc_vt_ctx.PitchSpeed
            # ctx.YawSpeed = PitchSpeed
            # print('Скорость в 10 тиков: ',mc_vt_ctx.PitchSpeed,ctx.YawSpeed)
        except BaseException as err:
            print('Mine Bridge Произошла большая ошибка >', err)
            print('ТЕКСТ ОБ***Й ОШИБКИ', traceback.format_exc())
            time.sleep(5)
        finally:
            print('Достигнут конец процесса Mine Bridge, перезапускаем его...')
            try:
                if (gateway is not None):
                    gateway.shutdown_callback_server()
                    gateway.shutdown()
                    time.sleep(1)
                    gateway = None
            except BaseException as err:
                print('MineBridge: не удалось завершить gateway')
                time.sleep(1)


if __name__ == "__main__":
    manager = multiprocessing.Manager()
    ctx = manager.Namespace()
    ctx.ThreadsActived = True
    ctx.NeedX = -0.5
    ctx.NeedY = -1.0
    ctx.state = "gaming"
    ctx.SeparateEyes = True
    ctx.MineEvent = manager.Event()
    ctx.MineEventName = ""
    ctx.eyeX = ctx.NeedX
    ctx.eyeY = ctx.NeedY
    ctx.AnimEvent = manager.Event()
    ctx.AnimEventName = "SawChat.exp3.json"
    ctx.AnimEventTime = 0
    ctx.IsMCStarted = False
    ctx.IsVtubeStarted = False
    ctx.IsYTChatConnected = False
    mc_vt_ctx = manager.Namespace()
    mc_vt_ctx.PitchSpeed = 0.0
    ctx.YawSpeed = 0.0
    ctx.state = ""
    mainBridge(ctx)

# mainBridge()
# Get an instance of the Java class
# hello = gateway.jvm.com.example.Hello()
#
## Call the Java method and print the result
# print(message)
###def add_some_values(dict):
###  # Создание списка имён и их запись в dict
###  names = ["Вася", "Аня", "Лена", "Никита"]
###  for name in names:
###      dict.add(name)
###
###  # Удаление значения по ключу 4
###  deleted = dict.remove(4)
###
###  # Вывод содержимого dict
###  for i in range(dict.length()):
###      print(i, "\t", dict.get(i))
###
###  print("\nУдалённый элемент =", deleted)
###  return 1
###
###def mainBridge():
###  # Инициализация JavaGateway
###  gateway = JavaGateway()
###  print(str(gateway))
###  print(gateway.entry_point)
###  # Получение доступа к объекту класса Dict
###  #dict = gateway.entry_point.getDict()
###  #
###  #add_some_values(dict)
###  ## Очистка Dict
###  #dict.clear()
###  return 0
