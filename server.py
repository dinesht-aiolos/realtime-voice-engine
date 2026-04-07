"""
server.py — Deepgram Voice Agent with Native Cartesia TTS + Tool Calling

Cartesia is now used as a NATIVE speak provider — no custom TTS code needed.
Deepgram handles STT + LLM + TTS (via Cartesia) all through one WebSocket.

Supported speak providers (from Deepgram docs):
  - deepgram     → Aura-2 built-in voices (free)
  - cartesia     → Voice cloning, sonic-2 model (native support confirmed)
  - eleven_labs  → ElevenLabs voices + cloning ($5/month min)
  - open_ai      → OpenAI TTS voices

Supported LLM providers:
  - open_ai      → GPT-4o, GPT-4o-mini
  - anthropic    → Claude models
  - deepseek     → DeepSeek chat (via custom endpoint)
  - Any OpenAI-compatible API via endpoint override

Run:
    python -m uvicorn server:app --reload --port 8000

.env:
    DEEPGRAM_API_KEY=your_deepgram_key
    OPENAI_API_KEY=your_openai_key
    CARTESIA_API_KEY=your_cartesia_key
    CARTESIA_VOICE_ID=your_cloned_voice_id
"""

import os
import json
import uuid
import asyncio
import httpx
import time
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API Keys ──────────────────────────────────────────────────────────────────
DEEPGRAM_API_KEY  = os.getenv("DEEPGRAM_API_KEY")
CARTESIA_API_KEY  = os.getenv("CARTESIA_API_KEY")
CARTESIA_VOICE_ID = os.getenv("CARTESIA_VOICE_ID")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")

DEEPGRAM_AGENT_URL = "wss://agent.deepgram.com/v1/agent/converse"
AGENT_GREETING     = "Hello! I can help you book a table or answer any questions. How can I assist you today?"


# ════════════════════════════════════════════════════════════════════════
#  LATENCY TRACKER — terminal only, simple and clean
# ════════════════════════════════════════════════════════════════════════

class LatencyTracker:
    """
    Tracks STT / LLM / TTS timing per turn and prints to terminal.

    NOTE: AgentThinking and AgentStartedSpeaking do NOT always fire
    from Deepgram — especially in tool-calling flows. So we use
    ConversationText (assistant) as the LLM-done signal instead,
    and first audio bytes as TTS signal. This ensures timing always
    prints even when Deepgram skips those events.
    """

    def __init__(self, llm_model: str, stt_model: str, tts_model: str):
        self.llm_model = llm_model
        self.stt_model = stt_model
        self.tts_model = tts_model
        self.turn      = 0
        self._reset_turn()
        print(f"\n[TIMING] Models: STT={stt_model} | LLM={llm_model} | TTS={tts_model}", flush=True)

    def _reset_turn(self):
        self.t0          = None   # UserStartedSpeaking
        self.t1          = None   # ConversationText(user) — STT done
        self.t2          = None   # AgentThinking — LLM started (optional)
        self.t3          = None   # AgentStartedSpeaking OR ConversationText(assistant)
        self.t4          = None   # First binary audio — TTS started
        self.llm_printed = False  # Prevent double-printing LLM done

    def _ms(self, a, b):
        """Return int ms between two timestamps, or None."""
        if a is not None and b is not None:
            return int((b - a) * 1000)
        return None

    def _fmt(self, ms):
        return f"{ms}ms" if ms is not None else "N/A"

    # ── Event hooks ───────────────────────────────────────────────────

    def on_user_started_speaking(self):
        self._reset_turn()
        self.t0 = time.time()
        print(f"[TIMING] STT  started ...", flush=True)

    def on_stt_done(self):
        """Called when ConversationText(user) arrives — STT complete."""
        self.t1 = time.time()
        print(f"[TIMING] STT  done     → {self._fmt(self._ms(self.t0, self.t1))}", flush=True)
        print(f"[TIMING] LLM  started ...", flush=True)

    def on_llm_start(self):
        """Called on AgentThinking — not always fired by Deepgram."""
        if self.t2 is None:
            self.t2 = time.time()

    def on_llm_done(self, dg: dict = None):
        """
        Called on AgentStartedSpeaking.
        Not always fired — on_agent_text() below is the reliable fallback.
        """
        if not self.llm_printed:
            self.t3 = time.time()
            self._print_llm(dg)

    def on_agent_text(self):
        """
        Called on ConversationText(assistant) — fires reliably every turn
        even when AgentStartedSpeaking does not.
        Used as fallback LLM-done signal.
        """
        if self.t3 is None and self.t1 is not None:
            self.t3 = time.time()
            self._print_llm(dg=None)

    def on_first_audio(self):
        """Called on first binary audio bytes received from Deepgram."""
        if self.t4 is None:
            self.t4 = time.time()
            # Use t3 if set, else t1 as reference for TTS start
            ref = self.t3 or self.t1
            tts_ms = self._ms(ref, self.t4)
            print(f"[TIMING] TTS  first audio → {self._fmt(tts_ms)}", flush=True)

            # Print total if not yet printed
            total_ms = self._ms(self.t0, self.t4)
            if total_ms:
                self.turn += 1
                print(f"[TIMING] ─── Turn {self.turn} total: {self._fmt(total_ms)} ───", flush=True)

    def _print_llm(self, dg: dict = None):
        """Print LLM timing once."""
        self.llm_printed = True
        llm_ms = self._ms(self.t1, self.t3)
        print(f"[TIMING] LLM  done     → {self._fmt(llm_ms)}", flush=True)

        if dg:
            dg_total = dg.get("total_latency")
            dg_llm   = dg.get("ttt_latency")
            dg_tts   = dg.get("tts_latency")
            if dg_total:
                print(
                    f"[TIMING] DG reports:   "
                    f"total={int(dg_total*1000)}ms | "
                    f"llm={int(dg_llm*1000) if dg_llm else 'N/A'}ms | "
                    f"tts={int(dg_tts*1000) if dg_tts else 'N/A'}ms",
                    flush=True
                )
        print(f"[TIMING] TTS  started ...", flush=True)

    def reset(self):
        self._reset_turn()


# ════════════════════════════════════════════════════════════════════════
#  TOOLS
# ════════════════════════════════════════════════════════════════════════

TOOLS = [
    {
        "name": "check_availability",
        "description": (
            "Check if a restaurant table is available for a specific date, "
            "time, and party size. Call this when a user asks about booking "
            "or availability."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "date":       {"type": "string",  "description": "Date e.g. Saturday or 2024-12-14"},
                "time":       {"type": "string",  "description": "Time e.g. 7pm or 19:00"},
                "party_size": {"type": "integer", "description": "Number of guests"},
            },
            "required": ["date", "time", "party_size"],
        },
    },
    {
        "name": "create_booking",
        "description": (
            "Create a confirmed restaurant table booking. Only call this after "
            "checking availability AND collecting name and phone number. "
            "Always confirm all details before booking."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name":       {"type": "string",  "description": "Customer full name"},
                "phone":      {"type": "string",  "description": "Customer phone number"},
                "date":       {"type": "string",  "description": "Booking date"},
                "time":       {"type": "string",  "description": "Booking time"},
                "party_size": {"type": "integer", "description": "Number of guests"},
            },
            "required": ["name", "phone", "date", "time", "party_size"],
        },
    },
    {
        "name": "get_weather",
        "description": "Get current weather for a city. Call when user asks about weather.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name e.g. London, Mumbai"},
            },
            "required": ["city"],
        },
    },
]


# ════════════════════════════════════════════════════════════════════════
#  TOOL FUNCTIONS
# ════════════════════════════════════════════════════════════════════════

async def check_availability(date: str, time: str, party_size: int) -> dict:
    print(f"[TOOL] check_availability: {date} {time} x{party_size}", flush=True)
    # TODO: Replace with real DB query
    return {
        "available": True,
        "date": date, "time": time, "party_size": party_size,
        "available_slots": ["6:00 PM", "7:00 PM", "7:30 PM", "8:00 PM"],
        "message": f"Yes, we have availability on {date}. {time} works great!",
    }


async def create_booking(
    name: str, phone: str, date: str, time: str, party_size: int
) -> dict:
    print(f"[TOOL] create_booking: {name} {phone} {date} {time} x{party_size}", flush=True)
    ref = f"BOOK-{uuid.uuid4().hex[:6].upper()}"
    # TODO: Replace with real DB insert + SMS via Twilio
    return {
        "success": True,
        "confirmation_ref": ref,
        "message": f"Booking confirmed! Your reference is {ref}. See you on {date} at {time}!",
    }


async def get_weather(city: str) -> dict:
    print(f"[TOOL] get_weather: {city}", flush=True)
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"https://wttr.in/{city}?format=j1")
            data = resp.json()
        c = data["current_condition"][0]
        return {
            "city": city, "temp_c": c["temp_C"], "feels_c": c["FeelsLikeC"],
            "condition": c["weatherDesc"][0]["value"], "humidity": f"{c['humidity']}%",
            "message": f"It's {c['temp_C']}°C in {city}, {c['weatherDesc'][0]['value'].lower()}. Feels like {c['FeelsLikeC']}°C.",
        }
    except Exception as e:
        return {"error": str(e), "message": f"Couldn't fetch weather for {city}."}


async def execute_tool(name: str, arguments: dict) -> dict:
    if   name == "check_availability": return await check_availability(**arguments)
    elif name == "create_booking":     return await create_booking(**arguments)
    elif name == "get_weather":        return await get_weather(**arguments)
    else: return {"error": f"Unknown tool: {name}"}


# ════════════════════════════════════════════════════════════════════════
#  SETTINGS — choose your speak provider below
# ════════════════════════════════════════════════════════════════════════

def build_settings() -> dict:
    return {
        "type": "Settings",
        "audio": {
            "input":  {"encoding": "linear16", "sample_rate": 24000},
            "output": {"encoding": "linear16", "sample_rate": 24000, "container": "none"},
        },
        "agent": {
            "listen": {
                "provider": {"type": "deepgram", "model": "nova-3"},
            },

            # ── LLM — swap between providers ─────────────────────────
            "think": build_think(),

            # ── TTS — swap between providers ─────────────────────────
            "speak": build_speak(),

            "greeting": AGENT_GREETING,
            "language": "en",
        },
    }


def build_think() -> dict:
    """
    Choose your LLM provider here.
    Uncomment the one you want to use.
    """

    # ── Option 1: OpenAI GPT-4o-mini (default) ────────────────────────
    from datetime import datetime
    today = datetime.now()
    today_str = today.strftime("%A, %B %d, %Y")
    current_month = today.month
    current_year  = today.year

#     return {
#        "provider": {
#         "type":  "groq",                      # must be open_ai, not groq
#         "model": "openai/gpt-oss-20b",      # groq model name
#     },
#     "endpoint": {
#         "url": "https://api.groq.com/openai/v1/chat/completions",
#         "headers": {
#             "authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
#         },
#     },
#         "prompt": (
#             f"You are a friendly voice assistant for Acme Restaurant. "
#             f"Today is {today_str}. "
# "You help customers check table availability and make bookings. "
#             "You can also answer general questions on any topic — technology, science, general knowledge, etc. "
#             "If someone asks a general question like what is JavaScript or how does something work, answer it naturally and briefly. "
#             "Then politely ask if they also need help with a table booking. "

#             "VERY IMPORTANT — Ask ONE question at a time. Never ask multiple questions together. "
#             "Wait for the customer answer before asking the next question. "

#             "VALIDATION RULES — strictly enforce these before proceeding: "

#             "DATE VALIDATION: "
#             f"Today is {today_str}. Today is day {today.day}, month {current_month}, year {current_year}. "
#             "A date is ONLY invalid if it is strictly before today. "
#             f"For example if today is April 4, then April 1, 2 or 3 are past — reject them. "
#             f"But April 5, 6, 7, 8 and beyond are future dates — accept them without question. "
#             "If the customer says a day number that is greater than today's day number in the same month, it is a future date — accept it. "
#             "If customer says just a day name like Monday or Saturday, assume the NEXT upcoming one from today. "
#             "Only reject if the date is clearly in the past. When in doubt, accept the date. "
#             "Say sorry only when the date is definitely before today. "

#             "PHONE NUMBER VALIDATION: "
#             "A valid phone number must have exactly 10 digits. "
#             "If the customer gives fewer than 10 digits, say: "
#             "That does not look like a valid 10-digit phone number. Could you repeat it please? "
#             "Re-ask for the phone number. Do not proceed with less than 10 digits. "
#             "Only accept numbers — ignore spaces and dashes when counting digits. "

#             "PARTY SIZE VALIDATION: "
#             "Party size must be a number between 1 and 20. "
#             "If customer gives 0 or more than 20, say it is out of range and re-ask. "

#             "TIME VALIDATION: "
#             "Restaurant hours are 11 AM to 11 PM only. "
#             "If customer gives a time outside this range, say: "
#             "Sorry, we are open from 11 AM to 11 PM. Please choose a time within those hours. "
#             "Re-ask for the time. "

#             "When a customer wants to book a table, follow this exact order: "
#             "1. Ask for the date only. Validate it is a future date. Re-ask if invalid. "
#             "2. Ask for the time only. Validate it is within 11 AM to 11 PM. Re-ask if invalid. "
#             "3. Ask for the party size only. Validate it is between 1 and 20. Re-ask if invalid. "
#             "4. Call check_availability with the details collected. "
#             "5. If available, ask for their name only. Wait for answer. "
#             "6. Ask for their phone number only. Validate exactly 10 digits. Re-ask if invalid. "
#             "7. Confirm all details back to the customer in one sentence. "
#             "8. If they confirm, call create_booking. "

#             "Good examples of single questions: "
#             "What date would you like to book? "
#             "What time works best for you? "
#             "How many guests will be joining? "
#             "Could I get your name please? "
#             "And your phone number? "

#             "Never ask multiple things at once. "
#             "Always one question then wait. "
#             "Keep all responses short and friendly."
#         ),
#         "functions": TOOLS,
#     }

    return {
        "provider": {"type": "open_ai", "model": "gpt-5.4"},
        "prompt": (
            f"You are a friendly voice assistant for Acme Restaurant. "
            f"Today is {today_str}. "
"You help customers check table availability and make bookings. "
            "You can also answer general questions on any topic — technology, science, general knowledge, etc. "
            "If someone asks a general question like what is JavaScript or how does something work, answer it naturally and briefly. "
            "Then politely ask if they also need help with a table booking. "

            "VERY IMPORTANT — Ask ONE question at a time. Never ask multiple questions together. "
            "Wait for the customer answer before asking the next question. "

            "VALIDATION RULES — strictly enforce these before proceeding: "

            "DATE VALIDATION: "
            f"Today is {today_str}. Today is day {today.day}, month {current_month}, year {current_year}. "
            "A date is ONLY invalid if it is strictly before today. "
            f"For example if today is April 4, then April 1, 2 or 3 are past — reject them. "
            f"But April 5, 6, 7, 8 and beyond are future dates — accept them without question. "
            "If the customer says a day number that is greater than today's day number in the same month, it is a future date — accept it. "
            "If customer says just a day name like Monday or Saturday, assume the NEXT upcoming one from today. "
            "Only reject if the date is clearly in the past. When in doubt, accept the date. "
            "Say sorry only when the date is definitely before today. "

            "PHONE NUMBER VALIDATION: "
            "A valid phone number must have exactly 10 digits. "
            "If the customer gives fewer than 10 digits, say: "
            "That does not look like a valid 10-digit phone number. Could you repeat it please? "
            "Re-ask for the phone number. Do not proceed with less than 10 digits. "
            "Only accept numbers — ignore spaces and dashes when counting digits. "

            "PARTY SIZE VALIDATION: "
            "Party size must be a number between 1 and 20. "
            "If customer gives 0 or more than 20, say it is out of range and re-ask. "

            "TIME VALIDATION: "
            "Restaurant hours are 11 AM to 11 PM only. "
            "If customer gives a time outside this range, say: "
            "Sorry, we are open from 11 AM to 11 PM. Please choose a time within those hours. "
            "Re-ask for the time. "

            "When a customer wants to book a table, follow this exact order: "
            "1. Ask for the date only. Validate it is a future date. Re-ask if invalid. "
            "2. Ask for the time only. Validate it is within 11 AM to 11 PM. Re-ask if invalid. "
            "3. Ask for the party size only. Validate it is between 1 and 20. Re-ask if invalid. "
            "4. Call check_availability with the details collected. "
            "5. If available, ask for their name only. Wait for answer. "
            "6. Ask for their phone number only. Validate exactly 10 digits. Re-ask if invalid. "
            "7. Confirm all details back to the customer in one sentence. "
            "8. If they confirm, call create_booking. "

            "Good examples of single questions: "
            "What date would you like to book? "
            "What time works best for you? "
            "How many guests will be joining? "
            "Could I get your name please? "
            "And your phone number? "

            "Never ask multiple things at once. "
            "Always one question then wait. "
            "Keep all responses short and friendly."
        ),
        "functions": TOOLS,
    }

    # ── Option 2: DeepSeek ────────────────────────────────────────────
    # return {
    #     "provider": {"type": "open_ai", "model": "deepseek-chat"},
    #     "endpoint": {
    #         "url": "https://api.deepseek.com/v1/chat/completions",
    #         "headers": {
    #             "authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}",
    #         },
    #     },
    #     "prompt": "You are a helpful voice assistant. Keep responses short.",
    #     "functions": TOOLS,
    # }

    # ── Option 3: Anthropic Claude ───────────────────────────────────
    # return {
    #     "provider": {"type": "anthropic", "model": "claude-3-haiku-20240307"},
    #     "prompt": "You are a helpful voice assistant. Keep responses short.",
    #     "functions": TOOLS,
    # }


def build_speak() -> dict:
    """
    Choose your TTS / voice provider here.
    Uncomment the one you want to use.
    """

    # ── Option 1: Cartesia — native support, voice cloning ───────────
    # Get Voice ID from cartesia.ai dashboard
    # Free pre-built voices available, cloning requires paid plan
    # return {
    #     "provider": {
    #         "type":     "cartesia",
    #         "model_id": "sonic-2",
    #         "voice": {
    #             "mode": "id",
    #             "id":   CARTESIA_VOICE_ID or "a0e99841-438c-4a64-b679-ae501e7d6091",
    #         },
    #     },
    #     "endpoint": {
    #         "url": "https://api.cartesia.ai/tts/bytes",
    #         "headers": {
    #             "Cartesia-Version": "2024-06-10",
    #             "X-API-Key":        CARTESIA_API_KEY or "",
    #         },
    #     },
    # }

    # ── Option 2: Deepgram Aura-2 — free, no cloning ─────────────────
    return {
        "provider": {"type": "deepgram", "model": "aura-2-zeus-en"},
    }

    # ── Option 3: ElevenLabs — voice cloning, $5/month min ───────────
    # Replace {voice_id} with your ElevenLabs voice ID
    # return {
    #     "provider": {
    #         "type":          "eleven_labs",
    #         "model_id":      "eleven_turbo_v2_5",
    #         "language_code": "en-US",
    #     },
    #     "endpoint": {
    #         "url": "wss://api.elevenlabs.io/v1/text-to-speech/YOUR_VOICE_ID/multi-stream-input",
    #         "headers": {
    #             "xi-api-key": os.getenv("ELEVENLABS_API_KEY") or "",
    #         },
    #     },
    # }

    # ── Option 4: OpenAI TTS ──────────────────────────────────────────
    # return {
    #     "provider": {
    #         "type":  "open_ai",
    #         "model": "tts-1",
    #         "voice": "alloy",  # alloy, echo, fable, onyx, nova, shimmer
    #     },
    #     "endpoint": {
    #         "url": "https://api.openai.com/v1/audio/speech",
    #         "headers": {
    #             "authorization": f"Bearer {OPENAI_API_KEY or ''}",
    #         },
    #     },
    # }


# ════════════════════════════════════════════════════════════════════════
#  FUNCTION CALL HANDLER
# ════════════════════════════════════════════════════════════════════════

async def handle_function_call(dg_ws, event: dict) -> None:
    for func in event.get("functions", []):
        func_id   = func.get("id",   "unknown")
        func_name = func.get("name", "unknown")
        try:
            arguments = json.loads(func.get("arguments", "{}"))
        except json.JSONDecodeError:
            arguments = {}

        print(f"\n[TOOL] ← {func_name}({arguments})", flush=True)

        try:
            result = await execute_tool(func_name, arguments)
        except Exception as e:
            result = {"error": str(e)}

        print(f"[TOOL] → {result}", flush=True)

        await dg_ws.send(json.dumps({
            "type":    "FunctionCallResponse",
            "id":      func_id,
            "name":    func_name,
            "content": json.dumps(result),
        }))
        print(f"[TOOL] ✓ {func_name} response sent\n", flush=True)


# ════════════════════════════════════════════════════════════════════════
#  HEALTH CHECK
# ════════════════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    return {
        "status":         "running",
        "deepgram_key":   f"{DEEPGRAM_API_KEY[:8]}..."  if DEEPGRAM_API_KEY  else "NOT SET ❌",
        "cartesia_key":   f"{CARTESIA_API_KEY[:8]}..."  if CARTESIA_API_KEY  else "NOT SET ❌",
        "cartesia_voice": CARTESIA_VOICE_ID              if CARTESIA_VOICE_ID else "NOT SET ❌",
        "tools":          [t["name"] for t in TOOLS],
    }


# ════════════════════════════════════════════════════════════════════════
#  WEBSOCKET PROXY
# ════════════════════════════════════════════════════════════════════════

@app.websocket("/ws")
async def proxy(browser_ws: WebSocket):
    print("\n[+] Browser connected", flush=True)

    if not DEEPGRAM_API_KEY:
        await browser_ws.close(1008, "DEEPGRAM_API_KEY not configured")
        return

    print(f"[*] Deepgram:  {DEEPGRAM_API_KEY[:8]}...", flush=True)
    print(f"[*] Cartesia:  {CARTESIA_VOICE_ID or 'NOT SET'}", flush=True)
    print(f"[*] Tools:     {[t['name'] for t in TOOLS]}", flush=True)

    try:
        async with websockets.connect(
            DEEPGRAM_AGENT_URL,
            additional_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"},
            open_timeout=10,
        ) as dg_ws:
            print("[+] Deepgram connected!\n", flush=True)

            settings_sent = False

            # Read model names from active settings for timing labels
            _s   = build_settings()
            _llm = _s["agent"]["think"]["provider"].get("model", "unknown")
            _stt = _s["agent"]["listen"]["provider"].get("model", "unknown")
            _sp  = _s["agent"]["speak"]["provider"]
            _tts = _sp.get("model_id") or _sp.get("model", "unknown")
            tracker = LatencyTracker(llm_model=_llm, stt_model=_stt, tts_model=_tts)

            async def browser_to_deepgram():
                nonlocal settings_sent
                try:
                    while True:
                        msg = await browser_ws.receive()
                        if msg.get("bytes"):
                            await dg_ws.send(msg["bytes"])
                        elif msg.get("text"):
                            try:
                                data = json.loads(msg["text"])
                                if data.get("type") == "Settings" and not settings_sent:
                                    payload = json.dumps(build_settings())
                                    print(f"[*] Sending settings", flush=True)
                                    await dg_ws.send(payload)
                                    settings_sent = True
                                else:
                                    await dg_ws.send(msg["text"])
                            except json.JSONDecodeError:
                                await dg_ws.send(msg["text"])
                except WebSocketDisconnect:
                    print("[-] Browser disconnected", flush=True)
                except WebSocketDisconnect:
                    print("[-] Browser disconnected", flush=True)
                except Exception as e:
                    err_str = str(e)
                    if "no close frame" not in err_str and "1005" not in err_str:
                        print(f"[!] browser→deepgram: {type(e).__name__}: {e}", flush=True)
                finally:
                    try: 
                        await dg_ws.close()
                    except Exception: 
                        pass

            async def deepgram_to_browser():
                try:
                    async for message in dg_ws:

                        # ── Binary = TTS audio ────────────────────────
                        if isinstance(message, bytes):
                            tracker.on_first_audio()   # mark TTS first byte
                            await browser_ws.send_bytes(message)

                        else:
                            print(f"[<] {message[:120]}", flush=True)
                            try:
                                event = json.loads(message)
                            except json.JSONDecodeError:
                                await browser_ws.send_text(message)
                                continue

                            event_type = event.get("type", "")

                            # ── Track latency per event type ──────────
                            if event_type == "UserStartedSpeaking":
                                tracker.on_user_started_speaking()

                            elif event_type == "ConversationText":
                                if event.get("role") == "user":
                                    tracker.on_stt_done()      # STT complete
                                elif event.get("role") == "assistant":
                                    tracker.on_agent_text()    # LLM done fallback

                            elif event_type == "AgentThinking":
                                tracker.on_llm_start()      # LLM starts

                            elif event_type == "AgentStartedSpeaking":
                                # Deepgram reports its own latency here
                                dg_latencies = {
                                    "total_latency": event.get("total_latency"),
                                    "tts_latency":   event.get("tts_latency"),
                                    "ttt_latency":   event.get("ttt_latency"),
                                }
                                tracker.on_llm_done(dg_latencies)  # print report

                            elif event_type == "AgentAudioDone":
                                tracker.reset()   # ready for next turn

                            # ── Route event ───────────────────────────
                            if event_type == "FunctionCallRequest":
                                await browser_ws.send_text(message)
                                await handle_function_call(dg_ws, event)
                            else:
                                await browser_ws.send_text(message)

                except Exception as e:
                    err_str = str(e)
                    if "no close frame" not in err_str and "1005" not in err_str:
                        print(f"[!] deepgram→browser: {type(e).__name__}: {e}", flush=True)
                        import traceback
                        traceback.print_exc()

            # ── Run both coroutines concurrently ──────────────────────────
            try:
                await asyncio.gather(
                    browser_to_deepgram(),
                    deepgram_to_browser(),
                )
            except Exception as e:
                print(f"[!] gather() error: {type(e).__name__}: {e}", flush=True)

    except websockets.exceptions.InvalidStatus as e:
        code = e.response.status_code
        print(f"[!] Deepgram rejected: HTTP {code}", flush=True)
        try:
            await browser_ws.send_text(
                json.dumps({"type": "Error",
                            "description": f"Deepgram rejected (HTTP {code})"})
            )
        except Exception: pass

    except Exception as e:
        print(f"[!] Proxy error: {type(e).__name__}: {e}", flush=True)

    finally:
        print("[-] Session ended\n", flush=True)