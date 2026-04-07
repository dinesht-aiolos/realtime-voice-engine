"""
server_azure.py
────────────────
Azure Voice Live agent using official azure-ai-voicelive SDK.
Based on official SDK docs and samples.

Install:
    pip install azure-ai-voicelive azure-core python-dotenv fastapi uvicorn websockets

Run:
    python -m uvicorn server_azure:app --reload --port 8001

.env needed:
    AZURE_VOICELIVE_API_KEY=your_azure_key
    AZURE_VOICELIVE_ENDPOINT=https://crowdvoxpreprocessing-p-resource.services.ai.azure.com/
    VOICE_NAME=en-US-AvaNeural        <- Azure Neural voice
    or
    VOICE_NAME=alloy                  <- OpenAI voice

Voice options:
    Azure Neural: en-US-AvaNeural, en-US-JennyNeural, en-US-GuyNeural
    OpenAI:       alloy, echo, fable, onyx, nova, shimmer
"""

import os
import json
import asyncio
import base64
import time
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

# ── Date Calculation ──────────────────────────────────────────────────────────
today = datetime.now()
today_str = today.strftime("%B %d, %Y")
current_month = today.strftime("%B")
current_year = today.year

# ════════════════════════════════════════════════════════════════════════
#  LATENCY TRACKER — Azure Voice Live
#
#  Azure event mapping:
#    t0 = INPUT_AUDIO_BUFFER_SPEECH_STARTED  → user starts speaking
#    t1 = INPUT_AUDIO_BUFFER_SPEECH_STOPPED  → STT done
#    t2 = RESPONSE_AUDIO_TRANSCRIPT_DELTA    → LLM started responding
#    t3 = RESPONSE_AUDIO_TRANSCRIPT_DONE     → LLM done
#    t4 = RESPONSE_AUDIO_DELTA (first)       → TTS first audio
# ════════════════════════════════════════════════════════════════════════

class AzureLatencyTracker:

    def __init__(self, model: str, voice: str):
        self.model = model
        self.voice = voice
        self.turn  = 0
        self._reset()
        print(f"\n[TIMING] Azure Models: LLM+STT+TTS={model} | Voice={voice}", flush=True)

    def _reset(self):
        self.t0          = None   # speech started
        self.t1          = None   # speech stopped — STT done
        self.t2          = None   # first transcript delta — LLM started
        self.t3          = None   # transcript done — LLM done
        self.t4          = None   # first audio delta — TTS started
        self.tts_printed = False

    def _ms(self, a, b):
        if a is not None and b is not None:
            return int((b - a) * 1000)
        return None

    def _fmt(self, ms):
        return f"{ms}ms" if ms is not None else "N/A"

    def on_speech_started(self):
        self._reset()
        self.t0 = time.time()
        print(f"[TIMING] STT  started ...", flush=True)

    def on_speech_stopped(self):
        self.t1 = time.time()
        print(f"[TIMING] STT  done     → {self._fmt(self._ms(self.t0, self.t1))}", flush=True)
        print(f"[TIMING] LLM  started ...", flush=True)

    def on_transcript_delta(self):
        if self.t2 is None:
            self.t2 = time.time()

    def on_transcript_done(self):
        self.t3 = time.time()
        llm_ms = self._ms(self.t1, self.t3)
        print(f"[TIMING] LLM  done     → {self._fmt(llm_ms)}", flush=True)
        print(f"[TIMING] TTS  started ...", flush=True)

    def on_first_audio(self):
        if self.t4 is None:
            self.t4 = time.time()
            ref    = self.t3 if self.t3 is not None else self.t1
            tts_ms = self._ms(ref, self.t4)
            total  = self._ms(self.t0, self.t4)
            print(f"[TIMING] TTS  first audio → {self._fmt(tts_ms)}", flush=True)
            self.turn += 1
            print(f"[TIMING] ─── Turn {self.turn} total: {self._fmt(total)} ───", flush=True)

    def on_response_done(self):
        self._reset()


app = FastAPI(title="Azure Voice Live Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config ────────────────────────────────────────────────────────────────────
AZURE_KEY      = os.getenv("AZURE_VOICELIVE_API_KEY") or os.getenv("AZURE_OPENAI_KEY", "")
AZURE_ENDPOINT = os.getenv("AZURE_VOICELIVE_ENDPOINT", "").rstrip("/")
AZURE_MODEL    = os.getenv("AZURE_MODEL", "gpt-4o-realtime-preview")
VOICE_NAME     = os.getenv("VOICE_NAME", "en-US-AvaNeural")  # Azure Neural voice
RESOURCE_NAME  = os.getenv("AZURE_RESOURCE_NAME", "crowdvoxpreprocessing-p-resource")

# If endpoint not set, build from resource name
if not AZURE_ENDPOINT:
    AZURE_ENDPOINT = f"https://{RESOURCE_NAME}.services.ai.azure.com"

AGENT_INSTRUCTIONS = (
    f"You are a friendly voice assistant for Acme Restaurant. "
    f"Today is {today_str}. "
    f"You help customers check table availability and make bookings. "
    f"You can also answer general questions on any topic — technology, science, general knowledge, etc. "
    f"If someone asks a general question like what is JavaScript or how does something work, answer it naturally and briefly. "
    f"Then politely ask if they also need help with a table booking. "

    f"VERY IMPORTANT — Ask ONE question at a time. Never ask multiple questions together. "
    f"Wait for the customer answer before asking the next question. "

    f"VALIDATION RULES — strictly enforce these before proceeding: "

    f"DATE VALIDATION: "
    f"Today is {today_str}. Today is day {today.day}, month {current_month}, year {current_year}. "
    f"A date is ONLY invalid if it is strictly before today. "
    f"For example if today is April 4, then April 1, 2 or 3 are past — reject them. "
    f"But April 5, 6, 7, 8 and beyond are future dates — accept them without question. "
    f"If the customer says a day number that is greater than today's day number in the same month, it is a future date — accept it. "
    f"If customer says just a day name like Monday or Saturday, assume the NEXT upcoming one from today. "
    f"Only reject if the date is clearly in the past. When in doubt, accept the date. "
    f"Say sorry only when the date is definitely before today. "

    f"PHONE NUMBER VALIDATION: "
    f"A valid phone number must have exactly 10 digits. "
    f"If the customer gives fewer than 10 digits, say: "
    f"That does not look like a valid 10-digit phone number. Could you repeat it please? "
    f"Re-ask for the phone number. Do not proceed with less than 10 digits. "
    f"Only accept numbers — ignore spaces and dashes when counting digits. "

    f"PARTY SIZE VALIDATION: "
    f"Party size must be a number between 1 and 20. "
    f"If customer gives 0 or more than 20, say it is out of range and re-ask. "

    f"TIME VALIDATION: "
    f"Restaurant hours are 11 AM to 11 PM only. "
    f"If customer gives a time outside this range, say: "
    f"Sorry, we are open from 11 AM to 11 PM. Please choose a time within those hours. "
    f"Re-ask for the time. "

    f"When a customer wants to book a table, follow this exact order: "
    f"1. Ask for the date only. Validate it is a future date. Re-ask if invalid. "
    f"2. Ask for the time only. Validate it is within 11 AM to 11 PM. Re-ask if invalid. "
    f"3. Ask for the party size only. Validate it is between 1 and 20. Re-ask if invalid. "
    f"4. Once all details are confirmed, ask for their name only. "
    f"5. Ask for their phone number only. Validate exactly 10 digits. Re-ask if invalid. "
    f"6. Confirm all details back to the customer in one sentence. "
    f"7. If they confirm, complete the booking and provide a confirmation message. "

    f"Good examples of single questions: "
    f"What date would you like to book? "
    f"What time works best for you? "
    f"How many guests will be joining? "
    f"Could I get your name please? "
    f"And your phone number? "

    f"Never ask multiple things at once. "
    f"Always one question then wait. "
    f"Keep all responses short and friendly."
)


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "status":   "running",
        "provider": "Azure Voice Live (SDK)",
        "endpoint": AZURE_ENDPOINT,
        "model":    AZURE_MODEL,
        "voice":    VOICE_NAME,
        "api_key":  f"{AZURE_KEY[:8]}..." if AZURE_KEY else "NOT SET ❌",
    }


@app.get("/config-check")
async def config_check():
    checks = {
        "AZURE_KEY":      bool(AZURE_KEY),
        "AZURE_ENDPOINT": bool(AZURE_ENDPOINT),
    }
    missing = [k for k, v in checks.items() if not v]
    return {"ready": not missing, "checks": checks, "missing": missing}


# ── WebSocket proxy using azure-ai-voicelive SDK ──────────────────────────────
@app.websocket("/ws")
async def proxy(browser_ws: WebSocket):
    print("\n[Azure] Browser connected", flush=True)

    if not AZURE_KEY:
        msg = "AZURE_VOICELIVE_API_KEY not set in .env"
        print(f"[Azure] ERROR: {msg}", flush=True)
        await browser_ws.send_text(json.dumps({"type": "Error", "description": msg}))
        await browser_ws.close(1008, msg)
        return

    print(f"[Azure] Endpoint: {AZURE_ENDPOINT}", flush=True)
    print(f"[Azure] Model:    {AZURE_MODEL}", flush=True)
    print(f"[Azure] Voice:    {VOICE_NAME}", flush=True)
    print(f"[Azure] Connecting via SDK...", flush=True)

    try:
        from azure.core.credentials import AzureKeyCredential
        from azure.ai.voicelive.aio import connect
        from azure.ai.voicelive.models import (
            RequestSession,
            Modality,
            InputAudioFormat,
            OutputAudioFormat,
            ServerVad,
            AzureStandardVoice,
            ServerEventType,
        )

        async with connect(
            endpoint=AZURE_ENDPOINT,
            credential=AzureKeyCredential(AZURE_KEY),
            model=AZURE_MODEL,
        ) as conn:
            print("[Azure] SDK connected!\n", flush=True)
            tracker = AzureLatencyTracker(model=AZURE_MODEL, voice=VOICE_NAME)

            # ── Configure voice ───────────────────────────────────────
            # Use Azure Neural voice if name contains "Neural"
            # Otherwise use as OpenAI voice string
            if "Neural" in VOICE_NAME or "azure" in VOICE_NAME.lower():
                voice = AzureStandardVoice(
                    name=VOICE_NAME,
                    type="azure-standard",
                )
            else:
                voice = VOICE_NAME  # OpenAI voice as plain string

            # ── Session config ────────────────────────────────────────
            session = RequestSession(
                modalities=[Modality.TEXT, Modality.AUDIO],
                instructions=AGENT_INSTRUCTIONS,
                input_audio_format=InputAudioFormat.PCM16,
                output_audio_format=OutputAudioFormat.PCM16,
                voice=voice,
                turn_detection=ServerVad(
                    threshold=0.5,
                    prefix_padding_ms=300,
                    silence_duration_ms=500,
                ),
                input_audio_transcription={
                    "model":    "gpt-4o-transcribe",
                    "language": "en",   # force English transcription
                },
            )

            await conn.session.update(session=session)
            print("[Azure] Session configured", flush=True)

            # Notify browser
            await browser_ws.send_text(json.dumps({"type": "SettingsApplied"}))

            # Trigger greeting — use input_text not instructions
            await conn.response.create()
            print("[Azure] Greeting triggered\n", flush=True)

            # ── Run both directions concurrently ──────────────────────
            async def browser_to_azure():
                try:
                    while True:
                        msg = await browser_ws.receive()

                        if msg.get("bytes"):
                            # Mic audio → Azure
                            await conn.input_audio_buffer.append(
                                audio=msg["bytes"]
                            )
                        elif msg.get("text"):
                            try:
                                data = json.loads(msg["text"])
                                if data.get("type") == "Settings":
                                    pass  # Already configured
                            except json.JSONDecodeError:
                                pass

                except WebSocketDisconnect:
                    print("[Azure] Browser disconnected", flush=True)
                except Exception as e:
                    if "no close frame" not in str(e):
                        print(f"[Azure] browser→azure: {e}", flush=True)

            async def azure_to_browser():
                try:
                    async for event in conn:
                        event_type = event.type
                        print(f"[Azure] EVENT: {event_type}", flush=True)

                        if event_type == ServerEventType.SESSION_UPDATED:
                            print("[Azure] Session updated OK", flush=True)

                        elif event_type == ServerEventType.RESPONSE_AUDIO_DELTA:
                            # TTS audio chunk — track first audio + send to browser
                            tracker.on_first_audio()
                            audio_bytes = event.delta
                            if isinstance(audio_bytes, str):
                                audio_bytes = base64.b64decode(audio_bytes)
                            if audio_bytes:
                                await browser_ws.send_bytes(audio_bytes)

                        elif event_type == ServerEventType.RESPONSE_AUDIO_TRANSCRIPT_DELTA:
                            tracker.on_transcript_delta()
                            await browser_ws.send_text(json.dumps({
                                "type": "AgentStartedSpeaking",
                            }))

                        elif event_type == ServerEventType.RESPONSE_AUDIO_TRANSCRIPT_DONE:
                            tracker.on_transcript_done()
                            transcript = getattr(event, "transcript", "")
                            if transcript:
                                print(f"[Azure] Agent: {transcript}", flush=True)
                                await browser_ws.send_text(json.dumps({
                                    "type":    "ConversationText",
                                    "role":    "assistant",
                                    "content": transcript,
                                }))

                        elif event_type == ServerEventType.INPUT_AUDIO_BUFFER_SPEECH_STARTED:
                            tracker.on_speech_started()
                            await browser_ws.send_text(json.dumps({
                                "type": "UserStartedSpeaking"
                            }))

                        elif event_type == ServerEventType.INPUT_AUDIO_BUFFER_SPEECH_STOPPED:
                            tracker.on_speech_stopped()
                            await browser_ws.send_text(json.dumps({
                                "type": "AgentThinking"
                            }))

                        elif event_type == "conversation.item.input_audio_transcription.completed":
                            transcript = getattr(event, "transcript", "")
                            if transcript:
                                print(f"[Azure] User: {transcript}", flush=True)
                                await browser_ws.send_text(json.dumps({
                                    "type":    "ConversationText",
                                    "role":    "user",
                                    "content": transcript,
                                }))

                        elif event_type == ServerEventType.RESPONSE_DONE:
                            tracker.on_response_done()
                            await browser_ws.send_text(json.dumps({
                                "type": "AgentAudioDone"
                            }))

                        elif event_type == ServerEventType.ERROR:
                            err_msg = getattr(event.error, "message", "Azure error")
                            print(f"[Azure] ERROR: {err_msg}", flush=True)
                            await browser_ws.send_text(json.dumps({
                                "type":        "Error",
                                "description": err_msg,
                            }))

                except Exception as e:
                    if "no close frame" not in str(e):
                        print(f"[Azure] azure→browser: {e}", flush=True)

            await asyncio.gather(browser_to_azure(), azure_to_browser())

    except ImportError:
        msg = "azure-ai-voicelive not installed. Run: pip install azure-ai-voicelive"
        print(f"[Azure] ERROR: {msg}", flush=True)
        await browser_ws.send_text(json.dumps({"type": "Error", "description": msg}))

    except Exception as e:
        print(f"[Azure] Proxy error: {type(e).__name__}: {e}", flush=True)

    finally:
        print("[Azure] Session ended\n", flush=True)