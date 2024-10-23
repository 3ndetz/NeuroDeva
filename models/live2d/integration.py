import json
import websockets
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import asyncio

from ...config.settings import Live2DConfig
from ...config.constants import Live2DError, WebSocketMessageType

logger = logging.getLogger(__name__)

class Live2DIntegration:
    def __init__(self, config: Live2DConfig):
        self.config = config
        self.websocket = None
        self.authtoken: Optional[str] = None
        self.connected = False

    async def ensure_connection(self) -> bool:
        """Ensure we have a valid connection, reconnecting if necessary"""
        if not self.connected or not self.websocket or self.websocket.closed:
            return await self.connect()
        return self.connected

    async def connect(self) -> bool:
        """Connect to VTube Studio"""
        try:
            self.websocket = await websockets.connect(self.config.websocket_url)
            logger.info("Connected to VTube Studio")
            
            token_path = Path(self.config.token_file)
            if token_path.exists():
                data = json.loads(token_path.read_text())
                self.authtoken = data.get('authenticationkey', '')
            
            if not self.authtoken:
                self.authtoken = await self.get_token()
                
            authenticated = await self.authenticate()
            self.connected = authenticated
            return authenticated
            
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            self.connected = False
            return False

    async def get_token(self) -> str:
        """Get authentication token from VTube Studio"""
        payload = {
            "apiName": self.config.api_name,
            "apiVersion": self.config.api_version,
            "requestID": self.config.request_id,
            "messageType": WebSocketMessageType.AUTH_TOKEN_REQUEST.value,
            "data": {
                "pluginName": self.config.plugin_name,
                "pluginDeveloper": self.config.plugin_developer
            }
        }
        
        await self.websocket.send(json.dumps(payload))
        response = await self.websocket.recv()
        token = json.loads(response)['data']['authenticationToken']

        # Save token to file
        token_path = Path(self.config.token_file)
        token_path.write_text(json.dumps({"authenticationkey": token}))

        return token

    async def authenticate(self) -> bool:
        """Authenticate with VTube Studio using token"""
        payload = {
            "apiName": self.config.api_name,
            "apiVersion": self.config.api_version,
            "requestID": self.config.request_id,
            "messageType": WebSocketMessageType.AUTH_REQUEST.value,
            "data": {
                "pluginName": self.config.plugin_name,
                "pluginDeveloper": self.config.plugin_developer,
                "authenticationToken": self.authtoken
            }
        }
        
        await self.websocket.send(json.dumps(payload))
        response = json.loads(await self.websocket.recv())
        return response['data']['authenticated']

    async def set_talking_parameter(self, value: float) -> None:
        """Set the mouth open parameter for the Live2D model"""
        try:
            if not await self.ensure_connection():
                return

            payload = {
                "apiName": self.config.api_name,
                "apiVersion": self.config.api_version,
                "requestID": self.config.request_id,
                "messageType": WebSocketMessageType.PARAMETER_DATA.value,
                "data": {
                    "mode": "set",
                    "parameterValues": [
                        {
                            "id": "MouthOpen",
                            "value": value
                        }
                    ]
                }
            }
            
            await self.websocket.send(json.dumps(payload))
            await self.websocket.recv()
            
        except Exception as e:
            logger.error(f"Failed to set parameter: {e}")
            self.connected = False
            raise Live2DError(f"Failed to set parameter: {e}")
