import json
import os
import websockets

import multiprocessing
from typing import Optional, Dict, Any
from config.settings import Live2DConfig
from utils.exceptions import Live2DConnectionError
from llm.fred_t5 import FredT5

class VTubeStudioIntegration:
    def __init__(self, config: Live2DConfig = None):
        self.config = config or Live2DConfig()
        self.websocket = None
        self.auth_token = None
        self.connected = False
        
        self.llm = FredT5()
        self.repeating_dict = {}

    async def connect(self) -> bool:
        try:
            self.websocket = await websockets.connect(self.config.websocket_url)
            print("[VTUBE] Connected to VTube Studio")
            
            if os.path.exists(self.config.token_file):
                with open(self.config.token_file, "r") as json_file:
                    data = json.load(json_file)
                    self.auth_token = data.get('authenticationkey', '')
            
            if not self.auth_token:
                self.auth_token = await self._get_token()
                
            self.connected = await self._authenticate()
            return self.connected
            
        except Exception as e:
            raise Live2DConnectionError(f"Failed to connect: {str(e)}")

    async def _get_token(self) -> str:
        payload = {
            "apiName": self.config.api_name,
            "apiVersion": self.config.api_version,
            "requestID": self.config.request_id,
            "messageType": "AuthenticationTokenRequest",
            "data": {
                "pluginName": self.config.plugin_name,
                "pluginDeveloper": self.config.plugin_developer
            }
        }
        
        await self.websocket.send(json.dumps(payload))
        response = await self.websocket.recv()
        token = json.loads(response)['data']['authenticationToken']

        with open(self.config.token_file, "w") as json_file:
            json.dump({"authenticationkey": token}, json_file)

        return token

    async def _authenticate(self) -> bool:
        payload = {
            "apiName": self.config.api_name,
            "apiVersion": self.config.api_version,
            "requestID": self.config.request_id,
            "messageType": "AuthenticationRequest",
            "data": {
                "pluginName": self.config.plugin_name,
                "pluginDeveloper": self.config.plugin_developer,
                "authenticationToken": self.auth_token
            }
        }
        
        await self.websocket.send(json.dumps(payload))
        response = json.loads(await self.websocket.recv())
        return response['data']['authenticated']

    async def _ensure_connection(self) -> bool:
        if not self.connected or not self.websocket or self.websocket.closed:
            await self.connect()
        return self.connected

    async def set_parameter(self, param_name: str, value: float) -> None:
        try:
            retry_count = 3
            while retry_count > 0:
                if not await self._ensure_connection():
                    retry_count -= 1
                    continue

                payload = {
                    "apiName": self.config.api_name,
                    "apiVersion": self.config.api_version,
                    "requestID": self.config.request_id,
                    "messageType": "InjectParameterDataRequest",
                    "data": {
                        "mode": "set",
                        "parameterValues": [
                            {
                                "id": param_name,
                                "value": value
                            }
                        ]
                    }
                }
                
                await self.websocket.send(json.dumps(payload))
                await self.websocket.recv()
                return
                
        except Exception as e:
            print(f"[VTUBE ERR] Failed to set parameter after retries: {e}")
            self.connected = False

    

    async def get_llm_response(self, text: str, context: list) -> Dict[str, Any]:
        try:
            return await self.llm.generate_response(
                text=text,
                params=None,  
                repeat_danger_part=context[-1]["content"] if context else ""
            )
        except Exception as e:
            print(f"[LLM ERR] Failed to get response: {e}")
            return {
                "reply": "",
                "emotion": "нет",
                "command": "нет",
                "tokens": 0,
                "stopped": ""
            }

    async def cleanup(self) -> None:
        if self.websocket:
            await self.websocket.close()
