import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from core.router import ProviderRouter
from core.config import VoiceConfig
from utils.logger import logger

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

router = ProviderRouter()


@app.get("/")
async def root():
    return {"status": "Voice Platform Running 🚀"}

@app.websocket("/ws")
async def unified_ws(ws: WebSocket):
    # ✅ MUST ACCEPT FIRST
    await ws.accept()

    logger.info("Client connected")

    try:
        # Now safe
        init_msg = await ws.receive_text()
        data = json.loads(init_msg)

        if data.get("type") != "INIT":
            await ws.close()
            return

        config = VoiceConfig(**data)

        provider = router.get_provider(config.provider)

        # 👉 Pass control
        await provider.handle(ws, config.dict())

    except WebSocketDisconnect:
        logger.info("Client disconnected")

    except Exception as e:
        logger.error(f"Error: {str(e)}")