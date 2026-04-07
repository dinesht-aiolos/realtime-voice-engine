from fastapi import WebSocket
from server import proxy as deepgram_proxy

class DeepgramProvider:

    async def handle(self, ws: WebSocket, config: dict):
        # Connection already accepted by main.py router
        await deepgram_proxy(ws)

