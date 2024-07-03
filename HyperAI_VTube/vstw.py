import json
import os
import setup
from setup import *

if os.path.exists('customfunc.py'):
    import customfunc
    from customfunc import *

bot='bots'

async def main():
    try:
        websocket = await websockets.connect('ws://127.0.0.1:8001')
    except:
        print("Couldn't connect to vtube studio")
        input("press enter to quit program")
        quit()
    twitch = await websockets.connect('ws://irc-ws.chat.twitch.tv:80')
    cmm = await setup(websocket)
    ###############################################
    #           platform selction, setup          #
    ###############################################
    link="https://twitchtokengenerator.com/?scope=bits:read+chat:read+channel:read:subscriptions+channel:read:redemptions+channel:read:hype_train+channel:read:editors+user:read:blocked_users+user:read:follows+channel:read:goals+channel:read:vips&auth=auth_stay"
    with open('token.json') as json_file:
        data = json.load(json_file)
        json_file.close()
   
    if (data['authenticationkeytwitch'] == ''):
        print("click authorize, copy the token from access_token=, till the & seperator")
        webbrowser.open(link)
        twauthtoken = input()
        with open('token.json', "w") as json_file:
            data["authenticationkeytwitch"] = twauthtoken
            json_file.write(json.dumps(data))
            json_file.close()
        data = json.load(open('token.json'))
        json_file.close()
        
    channel=input("input channel name ")
    await twitch.send('PASS oauth:'+data['authenticationkeytwitch'])
    await twitch.send('NICK '+bot)
    await twitch.send('JOIN #'+channel)
    res = await twitch.recv()
    print(res)
    res = await twitch.recv()
    print(res)
    res = await twitch.recv()
    print(res)
    while True:
        res = await twitch.recv()#getting twitch chat is soo fucking easy. Parsing it is hell
        message_list = res.split(':')#thanks to elburz article:https://interactiveimmersive.io/blog/content-inputs/twitch-chat-in-touchdesigner/
        user_message = message_list[-1]
        user_name = message_list[1].split('!')[0]
        print(user_message,user_message[0:len(user_message)-2])
        for key in cmm['COMMANDS']:
            if user_message[0:len(user_message)-2] == key:
                print('executing')
                mdinf = await getmd(websocket)
                s = mdinf["data"]["modelPosition"]["size"]
                r = mdinf["data"]["modelPosition"]["rotation"]
                x = mdinf["data"]["modelPosition"]["positionX"]
                y = mdinf["data"]["modelPosition"]["positionY"]
                cm = cmm['COMMANDS'][key]
                await eval(cm)
asyncio.run(main())
