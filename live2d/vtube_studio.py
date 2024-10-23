import json
import os
import websockets
import multiprocessing
from typing import Optional, Dict, Any
from ..config.settings import VTubeStudioConfig
from ..utils.exceptions import Live2DConnectionError
from .fred_t5 import FRED_PROCESS, get_llm_formed_inputs

class VTubeStudioIntegration:
    """Integration with VTube Studio for Live2D avatar control."""
    def __init__(self, config: VTubeStudioConfig = None):
        self.config = config or VTubeStudioConfig()
        self.websocket = None
        self.auth_token = None
        self.connected = False
        
        self.fred_process = None
        self.fred_input_queue = None
        self.fred_output_queue = None
        self.repeating_dict = None

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
        """Request new authentication token from VTube Studio."""
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
            if not await self._ensure_connection():
                return

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
            
        except Exception as e:
            print(f"[VTUBE ERR] Failed to set parameter: {e}")
            self.connected = False

    def setup_fred_process(self) -> None:
        manager = multiprocessing.Manager()
        loading_flag = manager.Event()
        self.fred_input_queue = manager.Queue()
        self.fred_output_queue = manager.Queue()
        self.repeating_dict = manager.dict()
        
        self.fred_process = multiprocessing.Process(
            target=FRED_PROCESS,
            args=(loading_flag, self.fred_input_queue, self.fred_output_queue, self.repeating_dict)
        )
        self.fred_process.start()
        loading_flag.wait()

    async def get_llm_response(self, text: str, context: list) -> Dict[str, Any]:
        if not self.fred_input_queue:
            raise RuntimeError("Fred process not initialized. Call setup_fred_process() first.")
            
        llm_input, params, danger_context = get_llm_formed_inputs(
            inp=text,
            username="User",
            environment={"env": "dialogue", "sentence_type": "dialog"},
            dialog_context=context,
            params_override=None,
            repeating_dict=self.repeating_dict
        )
        
        self.fred_input_queue.put((llm_input, params, danger_context))
        return self.fred_output_queue.get()

    async def cleanup(self) -> None:
        if self.fred_process:
            self.fred_process.terminate()
            self.fred_process.join()
            
        if self.websocket:
            await self.websocket.close()
