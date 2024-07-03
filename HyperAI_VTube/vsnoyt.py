from setup import *

if os.path.exists('custom.py'):
    import customfunc
    from customfunc import *


def VtubeProcess(vtube_ctx, ctx):
    import json
    import time
    import os

    import setup
    from setup import setup
    import threading
    async def wsconnect():
        fail = True
        while fail and ctx.ThreadsActived:
            try:
                ez = await websockets.connect('ws://127.0.0.1:8001')
                fail = False
                if (not ctx.IsVtubeStarted):
                    ctx.IsVtubeStarted = True
                # print('CONNECTION SUCCESSFUL')
                return ez
            except BaseException as err:
                ctx.IsVtubeStarted = False
                # print('!! **VTUBE STUDIO CONNECTION FAILED** !!',err)
                time.sleep(5)
                # return False

    async def DoEvent(websocket, event_name="CryButNot", event_type="hotkey", bool_val=True):
        async def innerFunc():
            if event_type == "hotkey":
                await ExHotkey(websocket, event_name)  # xHotkey(websocket,hid,IID):
            else:
                await ExpresState(websocket, event_name, bool_val)

        try:
            await innerFunc()
        except:
            websocket = await wsconnect()
            commandlist = await setup(websocket)
            await innerFunc()

    async def EventListener():
        websocket = await wsconnect()
        commandlist = await setup(websocket)
        while ctx.ThreadsActived:
            ctx.AnimEvent.wait()
            event_dict = ctx.AnimEventInfo
            event_name = event_dict["name"]
            event_type = event_dict.get("type", "hotkey")
            event_time = event_dict.get("time", 0)
            await DoEvent(websocket, event_name, event_type, True)

            if event_type != "hotkey":
                if event_time>0:
                    time.sleep(event_time)
                    # если ивент идёт в данный момент и он такой же как и был то НЕ НАДО ОТКЛЮЧАТЬ
                if ctx.AnimEvent.is_set() and event_name == ctx.AnimEventInfo["name"]:  #:(AnimEvent.is_set()) and eventname!=ctx.AnimEventName:
                    pass
                else:
                    await DoEvent(websocket, event_name, event_type, False)
            # time.sleep(0.6)

    async def startListeningCycle():
        websocket = await wsconnect()
        commandlist = await setup(websocket)
        oldNeedX = vtube_ctx.NeedX
        oldNeedY = vtube_ctx.NeedY
        await setNeedXY(websocket, vtube_ctx.NeedX, vtube_ctx.NeedY)
        # await createparam(websocket,"NeedEyeX",-1,1,0)#createparam(websocket,name,mn,mx,defolt)
        # await createparam(websocket,"NeedEyeY",-1,1,0)#createparam(websocket,name,mn,mx,defolt)
        while ctx.ThreadsActived:
            # word = input("enter command ")
            time.sleep(0.02)
            # if(oldNeedX != NeedX or oldNeedY != NeedY):
            #    oldNeedX = NeedX
            #    oldNeedY = NeedY
            # print("Changed!",oldNeedX,oldNeedY)
            # print("DEBUG",NeedX,NeedY)
            try:
                await setEyeNeedXY(websocket, vtube_ctx.eyeX, vtube_ctx.eyeY)
                await setNeedXY(websocket, vtube_ctx.NeedX, vtube_ctx.NeedY)
                # await setNeedXY(websocket,NeedX,NeedY)
                # await doteststuff(websocket)
                # print('повернута!')
            except:
                # print('Ошибка! переподключение')
                websocket = await wsconnect()
                commandlist = await setup(websocket)
                await setEyeNeedXY(websocket, vtube_ctx.eyeX, vtube_ctx.eyeY)
                await setNeedXY(websocket, vtube_ctx.NeedX, vtube_ctx.NeedY)
                # await setNeedXY(websocket,NeedX,NeedY)
                # await doteststuff(websocket)
                # print('повернута!')
            # word = input("enter command ")
            # for key in commandlist['COMMANDS']:
            #    if word == key:
            #        print('executing')
            #        mdinf = await getmd(websocket)
            #        s = mdinf["data"]["modelPosition"]["size"]
            #        r = mdinf["data"]["modelPosition"]["rotation"]
            #        x = mdinf["data"]["modelPosition"]["positionX"]
            #        y = mdinf["data"]["modelPosition"]["positionY"]
            #        cm = commandlist['COMMANDS'][key]
            #        await eval(cm)

    def EventChecker():
        asyncio.run(EventListener())

    EvCheckerThread = threading.Thread(target=EventChecker, daemon=True)
    EvCheckerThread.start()
    asyncio.run(startListeningCycle())
