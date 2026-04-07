from fastapi import WebSocket
from server_azure import proxy as azure_proxy

class AzureProvider:

    async def handle(self, ws: WebSocket, config: dict):
        # Connection already accepted by main.py router
        await azure_proxy(ws)