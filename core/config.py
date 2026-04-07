from pydantic import BaseModel
from typing import Optional

class VoiceConfig(BaseModel):
    provider: str          # deepgram / azure
    model: Optional[str] = None
    voice: Optional[str] = None
    stt: Optional[str] = None
    tts: Optional[str] = None