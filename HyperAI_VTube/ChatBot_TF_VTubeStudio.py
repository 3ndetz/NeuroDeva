import os
import json
import configparser

import asyncio
import websockets

import func
from func import *

import webbrowser
async def setup(websocket):
    if os.path.exists('token.json'):
        print('Loading authtoken From File...')
        with open('token.json', "r") as json_file:
            data = json.load(json_file)
            authtoken = (data['authenticationkey'])
            confirm = await authen(websocket,authtoken)
            if authtoken == "" or confirm["data"]["authenticated"] == False:
                print('Error Token Invalid')
                print('Fetching New Tokens...')
                authtoken = await token(websocket)
                print(authtoken)
                print('Saving authtoken for Future Use...')
                data["authenticationkey"] = authtoken
                json_file.close()
                json_file = open('token.json', "w")
                json_file.write(json.dumps(data))
                json_file.close()
                print("Saving finished")
            else:
                json_file.close()
    else:
            print('Fetching New Tokens...')
            authtoken = await token(websocket)
            print(authtoken)
            print('Saving authtoken for Future Use...')
            with open('token.json', "w") as json_file:
                jsonfilecon = {
                            "chatspeed": 0.1,
                            "authenticationkey": authtoken,
                            "authenticationkeytwitch": ""
                        }
                json_file.write(json.dumps(jsonfilecon))
                json_file.close()
            await authen(websocket,authtoken)
    if os.path.exists('commands.ini'):
        config = configparser.ConfigParser()
        commandlist = config.read('commands.ini')
    else:
        config = configparser.ConfigParser()
        with open('commands.ini', "w") as configfile:
            config['COMMANDS'] = {
                    "!spin": "spin(websocket,x,y,s)",
                    "!reset": "mdmv(websocket,0.2,False,0,0,0,-76)",
                    "!rainbow": "rainbow(websocket)"
                }
            config.write(configfile)
    with open('token.json') as json_file:
        data = json.load(json_file)
        json_file.close()
    ###############################
    #   command auto generation   #
    ###############################
    mdls = await listvtsmodel(websocket)
    runs = mdls["data"]["numberOfModels"]
    i=0
    for key in config['COMMANDS']:
        i+=1
    nmumm = runs - i
    if i < nmumm:
        with open('commands.ini', "w") as configfile:
            for i in range(runs):
                ff = mdls["data"]["availableModels"][i]["modelName"]
                gg = mdls["data"]["availableModels"][i]["modelID"]
                name = "!"+ff
                mdss = mdch.__name__+"("+"websocket"+",'"+str(gg)+"')"
                config['COMMANDS'][name] = mdss
            config.write(configfile)
        ###############################
        # command auto generation end #
        ###############################
    print("Successfully Loaded")
    print("Detected Commands")
    for key in config['COMMANDS']:
        print(key)
    return config




from os import system, name
import json
import time
import asyncio
######################################
#          plugin settings           #
######################################
dev = "test"
reqid = "test"
name = "test"
v = "1.0"
######################################
#             functions              #
######################################


try:
    websocket = websockets.connect('ws://127.0.0.1:8001')
except:
    print("Couldn't connect to vtube studio")
    input("press enter to quit program")
    quit()
#cmm = await setup(websocket)
async def token(websocket):
    payload = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": v,
            "requestID": reqid,
            "messageType": "AuthenticationTokenRequest",
            "data": {
                "pluginName": name,
                "pluginDeveloper": dev,
            }
        }
    await websocket.send(json.dumps(payload))
    json_data = await websocket.recv()
    pack = json.loads(json_data)
    authtoken = pack['data']['authenticationToken']
    return authtoken

async def authen(websocket,authtoken):
    payload={
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": v,
            "requestID": reqid,
            "messageType": "AuthenticationRequest",
            "data": {
                "pluginName": name,
                "pluginDeveloper": dev,
                "authenticationToken": authtoken
            }
        }
    await websocket.send(json.dumps(payload))
    json_data = await websocket.recv()
    pack = json.loads(json_data)
    return pack

async def gettrackparam(websocket):
    payload = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": v,
            "requestID": reqid,
            "messageType": "InputParameterListRequest"
        }
    await websocket.send(json.dumps(payload))
    json_data = await websocket.recv()
    pack = json.loads(json_data)
    return pack

async def getmd(websocket):
    payload = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": v,
            "requestID": reqid,
            "messageType": "CurrentModelRequest"
        }
    await websocket.send(json.dumps(payload))
    json_data = await websocket.recv()
    pack = json.loads(json_data)
    return pack

async def listvtsmodel(websocket):
    payload = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": v,
            "requestID": reqid,
            "messageType": "AvailableModelsRequest"
        }
    await websocket.send(json.dumps(payload))
    json_data = await websocket.recv()
    pack = json.loads(json_data)
    return pack

async def getapi(websocket):
    payload = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": v,
            "requestID": reqid,
            "messageType": "APIStateRequest",
        }
    await websocket.send(json.dumps(payload))
    json_data = await websocket.recv()
    pack = json.loads(json_data)
    return pack

async def getstat(websocket):
    payload = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": v,
            "requestID": reqid,
            "messageType": "StatisticsRequest"
        }
    await websocket.send(json.dumps(payload))
    json_data = await websocket.recv()
    pack = json.loads(json_data)
    return pack

async def getvtsfolder(websocket):
    payload = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": v,
            "requestID": reqid,
            "messageType": "VTSFolderInfoRequest"
        }
    await websocket.send(json.dumps(payload))
    json_data = await websocket.recv()
    pack = json.loads(json_data)
    authres = pack['data']
    return pack

async def gethotkeys(websocket,mdid):
    payload = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": v,
            "requestID": reqid,
            "messageType": "HotkeysInCurrentModelRequest",
            "data": {
                "modelID": mdid,
            }
        }
    await websocket.send(json.dumps(payload))
    json_data = await websocket.recv()
    pack = json.loads(json_data)
    return pack

async def getexstate(websocket,expressionfile,detail):
    payload = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": v,
            "requestID": reqid,
            "messageType": "ExpressionStateRequest",
            "data": {
                "details": detail,
                "expressionFile": expressionfile,
            }
        }
    await websocket.send(json.dumps(payload))
    json_data = await websocket.recv()
    pack = json.loads(json_data)
    return pack

async def facecheck(websocket):
    payload = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": v,
            "requestID": reqid,
            "messageType": "FaceFoundRequest"
        }
    await websocket.send(json.dumps(payload))
    json_data = await websocket.recv()
    pack = json.loads(json_data)
    return pack['data']['found']

async def mdch(websocket,mdid):
    payload = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": v,
            "requestID": reqid,
            "messageType": "ModelLoadRequest",
            "data": {
                "modelID": mdid
                }
        }
    await websocket.send(json.dumps(payload))
    json_data = await websocket.recv()
    pack = json.loads(json_data)
    return pack

async def mdmv(websocket,time,revelance,xp,yp,rot,size):
    payload = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": v,
            "requestID": reqid,
            "messageType": "MoveModelRequest",
            "data": {
                "timeInSeconds": time,
                "valuesAreRelativeToModel": revelance,
                "positionX": xp,
                "positionY": yp,
                "rotation": rot,
                "size": size
            }
        }
    await websocket.send(json.dumps(payload))
    json_data = await websocket.recv()
    pack = json.loads(json_data)
    return pack

async def listArtM(websocket,mdid):
    payload = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": v,
            "requestID": reqid,
            "messageType": "ArtMeshListRequest"
        }
    await websocket.send(json.dumps(payload))
    json_data = await websocket.recv()
    pack = json.loads(json_data)
    return pack

async def ExHotkey(websocket,hid,IID):
    payload = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": v,
            "requestID": reqid,
            "messageType": "HotkeyTriggerRequest",
            "data": {
                "hotkeyID": hid,
                "itemInstanceID": IID
        }
    }
    await websocket.send(json.dumps(payload))
    json_data = await websocket.recv()
    pack = json.loads(json_data)
    return pack
    
async def ExpresState(websocket,exf,state):
    payload = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": v,
            "requestID": reqid,
            "messageType": "ExpressionActivationRequest",
            "data": {
                "expressionFile": exf,
                "active": state
        }
    }
    await websocket.send(json.dumps(payload))
    json_data = await websocket.recv()
    pack = json.loads(json_data)
    return pack
    
async def getloc(websocket):
    payload = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": v,
            "requestID": reqid,
            "messageType": "SceneColorOverlayInfoRequest"
    }
    await websocket.send(json.dumps(payload))
    json_data = await websocket.recv()
    pack = json.loads(json_data)
    return pack
    
async def AskMeshSelect(websocket,to,ho,ram,aama):
    payload = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": v,
            "requestID": reqid,
            "messageType": "ArtMeshSelectionRequest",
            "data": {
                "textOverride": to,
                "helpOverride": ho,
                "requestedArtMeshCount": ram,
                "activeArtMeshes": aama
        }
    }
    await websocket.send(json.dumps(payload))
    json_data = await websocket.recv()
    pack = json.loads(json_data)
    return pack
    
async def getitem(websocket,IAS,IIIS,IIAIF,OIFN,OIIID):
    payload = {
        "apiName": "VTubeStudioPublicAPI",
        "apiVersion": v,
        "requestID": reqid,
        "messageType": "ItemListRequest",
        "data": {
            "includeAvailableSpots": IAS,
            "includeItemInstancesInScene": IIIS,
            "includeAvailableItemFiles": IIAIF,
            "onlyItemsWithFileName": OIFN,
            "onlyItemsWithInstanceID": OIIID
        }
    }
    await websocket.send(json.dumps(payload))
    json_data = await websocket.recv()
    pack = json.loads(json_data)
    return pack
    
async def mvitem(websocket,BOFA):
    payload = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": v,
            "requestID": reqid,
            "messageType": "ItemMoveRequest",
            "data": {
                "itemsToMove": BOFA
        }
    }
    await websocket.send(json.dumps(payload))
    json_data = await websocket.recv()
    pack = json.loads(json_data)
    return pack
    
async def conitem(websocket,IID,FPS,FC,BR,ALPH,FS,ASFA,SAPS,APS):
    payload = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": v,
            "requestID": reqid,
            "messageType": "ItemAnimationControlRequest",
            "data": {
                "itemInstanceID": IID,
                "framerate": FPS,
                "frame": FC,
                "brightness": BR,
                "opacity": ALPH,
                "setAutoStopFrames": FS,
                "autoStopFrames": ASFA,
                "setAnimationPlayState": SAPS,
                "animationPlayState": APS
        }
    }
    await websocket.send(json.dumps(payload))
    json_data = await websocket.recv()
    pack = json.loads(json_data)
    return pack
    
async def rmitem(websocket,LSOAIS,LSOPI,KP,AOIID,AOFN):
    payload = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": v,
            "requestID": reqid,
            "messageType": "ItemUnloadRequest",
                "data": {
                "unloadAllInScene": LSOAIS,
                "unloadAllLoadedByThisPlugin": LSOPI,
                "allowUnloadingItemsLoadedByUserOrOtherPlugins": KP,
                "instanceIDs": AOIID,
                "fileNames": AOFN
        }
    }
    await websocket.send(json.dumps(payload))
    json_data = await websocket.recv()
    pack = json.loads(json_data)
    return pack
    
async def loaditem(websocket,FN,X,Y,S,R,FT,O,SAOT,SS,SOS,SF,SL,SAU):
    payload = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": v,
            "requestID": reqid,
            "messageType": "ItemLoadRequest",
            "data": {
                "fileName": FN,
                "positionX": X,
                "positionY": Y,
                "size": S,
                "rotation": R,
                "fadeTime": FT,
                "order": O,
                "failIfOrderTaken": SAOT,
                "smoothing": SS,
                "censored": SOS,
                "flipped": SF,
                "locked": SL,
                "unloadWhenPluginDisconnects": SAU
        }
    }
    await websocket.send(json.dumps(payload))
    json_data = await websocket.recv()
    pack = json.loads(json_data)
    return pack
    
    

async def TintArtM(websocket,r,g,b,a,tintall,num,exactarray,conarray,tagexactarray,tagconarray):
    payload = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": v,
            "requestID": reqid,
            "messageType": "ColorTintRequest",
            "data": {
                "colorTint": {
                    "colorR": r,
                    "colorG": g,
                    "colorB": b,
                    "colorA": a
                },
                "artMeshMatcher": {
                    "tintAll": tintall,
                    "artMeshNumber": num,
                    "nameExact": exactarray,
                    "nameContains": conarray,
                    "tagExact": tagexactarray,
                    "tagContains": tagconarray
                }
            }
        }
    await websocket.send(json.dumps(payload))
    json_data = await websocket.recv()
    pack = json.loads(json_data)
    return pack
    


async def spin(websocket,x,y,s):
    await mdmv(websocket,0.2,False,x,y,90,s)
    time.sleep(0.1)
    await mdmv(websocket,0.2,False,x,y,180,s)
    time.sleep(0.1)
    await mdmv(websocket,0.2,False,x,y,270,s)
    time.sleep(0.1)
    await mdmv(websocket,0.2,False,x,y,360,s)
    time.sleep(0.1)

async def rainbow(websocket):
    await TintArtM(websocket,255,0,0,255,True,"","","","","")
    time.sleep(0.2)
    await TintArtM(websocket,255,127,0,255,True,"","","","","")
    time.sleep(0.2)
    await TintArtM(websocket,255,255,0,255,True,"","","","","")
    time.sleep(0.2)
    await TintArtM(websocket,0,255,0,255,True,"","","","","")
    time.sleep(0.2)
    await TintArtM(websocket,0,0,255,255,True,"","","","","")
    time.sleep(0.2)
    await TintArtM(websocket,46,43,95,255,True,"","","","","")
    time.sleep(0.2)
    await TintArtM(websocket,139,0,255,255,True,"","","","","")
    time.sleep(0.2)
    await TintArtM(websocket,255,255,255,255,True,"","","","","")