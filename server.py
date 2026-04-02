"""
server.py — Deepgram Voice Agent with Tool / Function Calling

HOW TOOLS WORK:
  1. You define tools in build_settings() → agent.think.functions
  2. User asks something that needs a tool (e.g. "what's the weather?")
  3. Deepgram LLM decides to call a tool → sends FunctionCallRequest
  4. Our server intercepts it, calls the REAL API
  5. We send FunctionCallResponse back to Deepgram
  6. Deepgram agent speaks the answer using the real data

Run:
    python -m uvicorn server:app --reload --port 8000
"""

import os
import json
import uuid
import asyncio
import httpx                  # pip install httpx  (for calling external APIs)
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

DEEPGRAM_API_KEY   = os.getenv("DEEPGRAM_API_KEY")
DEEPGRAM_AGENT_URL = "wss://agent.deepgram.com/v1/agent/converse"


# ════════════════════════════════════════════════════════════════════════
#  STEP 1 — DEFINE YOUR TOOLS HERE
#  Each tool needs:
#    name        → what the LLM calls it
#    description → tells the LLM WHEN to use this tool
#    parameters  → what arguments the LLM should collect from the user
# ════════════════════════════════════════════════════════════════════════

TOOLS = [

    # ── Tool 1: Check table availability ─────────────────────────────
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
                "date": {
                    "type": "string",
                    "description": "The date e.g. Saturday, December 14, or 2024-12-14",
                },
                "time": {
                    "type": "string",
                    "description": "The time e.g. 7pm, 7:30 PM, or 19:00",
                },
                "party_size": {
                    "type": "integer",
                    "description": "Number of guests",
                },
            },
            "required": ["date", "time", "party_size"],
        },
    },

    # ── Tool 2: Create a booking ──────────────────────────────────────
    {
        "name": "create_booking",
        "description": (
            "Create a confirmed restaurant table booking. Only call this "
            "after checking availability AND after collecting the customer's "
            "name and phone number. Always confirm details before booking."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name":       {"type": "string", "description": "Customer full name"},
                "phone":      {"type": "string", "description": "Customer phone number"},
                "date":       {"type": "string", "description": "Booking date"},
                "time":       {"type": "string", "description": "Booking time"},
                "party_size": {"type": "integer", "description": "Number of guests"},
            },
            "required": ["name", "phone", "date", "time", "party_size"],
        },
    },

    # ── Tool 3: Get weather (example of calling an external API) ──────
    {
        "name": "get_weather",
        "description": (
            "Get the current weather for a city. Call this when the user "
            "asks about weather in any location."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name e.g. London, New York, Mumbai",
                },
            },
            "required": ["city"],
        },
    },

    # ── ADD YOUR OWN TOOLS HERE ───────────────────────────────────────
    # {
    #     "name": "get_product_price",
    #     "description": "Get the price of a product when user asks",
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #             "product_name": {"type": "string", "description": "Product name"}
    #         },
    #         "required": ["product_name"],
    #     },
    # },

]


# ════════════════════════════════════════════════════════════════════════
#  STEP 2 — WRITE THE ACTUAL FUNCTIONS THAT DO THE REAL WORK
#  These are called when Deepgram requests a tool.
#  Replace the dummy logic with your real database / API calls.
# ════════════════════════════════════════════════════════════════════════

async def check_availability(date: str, time: str, party_size: int) -> dict:
    """
    Check table availability.
    """
    print(f"[TOOL] check_availability: {date} {time} for {party_size}", flush=True)

    # TODO: Replace with real DB query
    # e.g. rows = await db.fetch("SELECT * FROM slots WHERE date=$1 AND time=$2", date, time)

    # Dummy response:
    return {
        "available": True,
        "date": date,
        "time": time,
        "party_size": party_size,
        "available_slots": ["6:00 PM", "7:00 PM", "7:30 PM", "8:00 PM"],
        "message": f"Yes, we have availability on {date}. {time} works great!",
    }


async def create_booking(
    name: str, phone: str, date: str, time: str, party_size: int
) -> dict:
    """
    Create a confirmed booking.
    """
    print(f"[TOOL] create_booking: {name} {phone} {date} {time} x{party_size}", flush=True)

    ref = f"BOOK-{uuid.uuid4().hex[:6].upper()}"

    # TODO: Replace with real DB insert
    # await db.execute(
    #     "INSERT INTO bookings (ref, name, phone, date, time, party_size) VALUES ($1,$2,$3,$4,$5,$6)",
    #     ref, name, phone, date, time, party_size
    # )

    # TODO: Send confirmation SMS via Twilio
    # await twilio_client.messages.create(
    #     to=phone,
    #     from_=TWILIO_NUMBER,
    #     body=f"Booking confirmed! Ref: {ref}. Table for {party_size} on {date} at {time}."
    # )

    return {
        "success": True,
        "confirmation_ref": ref,
        "name": name,
        "date": date,
        "time": time,
        "party_size": party_size,
        "message": f"Booking confirmed! Your reference is {ref}. See you on {date} at {time}!",
    }


async def get_weather(city: str) -> dict:
    """
    Fetch real weather data from a free API.
    Uses wttr.in — no API key needed.
    """
    print(f"[TOOL] get_weather: {city}", flush=True)

    try:
        # wttr.in is free, no API key required
        url = f"https://wttr.in/{city}?format=j1"
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(url)
            data = resp.json()

        current = data["current_condition"][0]
        temp_c  = current["temp_C"]
        feels_c = current["FeelsLikeC"]
        desc    = current["weatherDesc"][0]["value"]
        humidity = current["humidity"]

        return {
            "city":      city,
            "temp_c":    temp_c,
            "feels_c":   feels_c,
            "condition": desc,
            "humidity":  f"{humidity}%",
            "message":   f"It's {temp_c}°C in {city} right now, {desc.lower()}. Feels like {feels_c}°C.",
        }

    except Exception as e:
        print(f"[TOOL] Weather API error: {e}", flush=True)
        return {
            "error":   str(e),
            "message": f"Sorry, I couldn't fetch the weather for {city} right now.",
        }


# ════════════════════════════════════════════════════════════════════════
#  STEP 3 — TOOL ROUTER
#  Maps tool names → functions. Add your new tools here too.
# ════════════════════════════════════════════════════════════════════════

async def execute_tool(name: str, arguments: dict) -> dict:
    """
    Routes a tool call to the right function.
    Add a new elif for every new tool you create.
    """
    if name == "check_availability":
        return await check_availability(**arguments)

    elif name == "create_booking":
        return await create_booking(**arguments)

    elif name == "get_weather":
        return await get_weather(**arguments)

    # ── Add your tools here ───────────────────────────────────────────
    # elif name == "get_product_price":
    #     return await get_product_price(**arguments)

    else:
        return {"error": f"Unknown tool: {name}"}


# ════════════════════════════════════════════════════════════════════════
#  STEP 4 — SETTINGS (tools are passed to LLM here)
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
            "think": {
                "provider": {"type": "open_ai", "model": "gpt-4o-mini"},
                "prompt": (
                    "You are a helpful voice assistant for Acme Restaurant. "
                    "You can check table availability, make bookings, and answer "
                    "weather questions. "
                    "Always confirm the date, time, party size, name, and phone "
                    "number before making a booking. "
                    "Keep responses short and conversational."
                ),
                # ── Tools are injected here ───────────────────────────
                "functions": TOOLS,
            },
            "speak": {
                "provider": {"type": "deepgram", "model": "aura-2-asteria-en"},
            },
            "greeting": "Hello! I can help you book a table or answer questions. How can I help?",
            "language": "en",
        },
    }


# ════════════════════════════════════════════════════════════════════════
#  STEP 5 — HANDLE FunctionCallRequest FROM DEEPGRAM
# ════════════════════════════════════════════════════════════════════════

async def handle_function_call(dg_ws, event: dict):
    """
    Called when Deepgram's LLM wants to use a tool.

    Event shape from Deepgram:
    {
        "type": "FunctionCallRequest",
        "functions": [
            {
                "id":        "func_abc123",
                "name":      "check_availability",
                "arguments": "{\"date\": \"Saturday\", \"time\": \"7pm\", \"party_size\": 4}",
                "client_side": true
            }
        ]
    }
    """
    for func in event.get("functions", []):
        func_id   = func.get("id",   "unknown")
        func_name = func.get("name", "unknown")
        raw_args  = func.get("arguments", "{}")

        # Parse JSON arguments string → dict
        try:
            arguments = json.loads(raw_args)
        except json.JSONDecodeError:
            arguments = {}

        print(f"\n[TOOL] ← Deepgram wants: {func_name}({arguments})", flush=True)

        # Execute the real function
        try:
            result = await execute_tool(func_name, arguments)
        except TypeError as e:
            result = {"error": f"Wrong arguments passed: {e}"}
        except Exception as e:
            result = {"error": f"Tool failed: {e}"}

        print(f"[TOOL] → Result: {result}", flush=True)

        # Send result back to Deepgram
        response = {
            "type":    "FunctionCallResponse",
            "id":      func_id,
            "name":    func_name,
            "content": json.dumps(result),
        }
        await dg_ws.send(json.dumps(response))
        print(f"[TOOL] ✓ Sent FunctionCallResponse for: {func_name}\n", flush=True)


# ════════════════════════════════════════════════════════════════════════
#  FASTAPI ROUTES
# ════════════════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    return {
        "status":       "running",
        "deepgram_key": f"{DEEPGRAM_API_KEY[:8]}..." if DEEPGRAM_API_KEY else "NOT SET ❌",
        "tools":        [t["name"] for t in TOOLS],
    }


@app.websocket("/ws")
async def proxy(browser_ws: WebSocket):
    await browser_ws.accept()
    print("\n[+] Browser connected", flush=True)

    if not DEEPGRAM_API_KEY:
        print("[!] DEEPGRAM_API_KEY not set", flush=True)
        await browser_ws.close(1008, "DEEPGRAM_API_KEY not configured")
        return

    print(f"[*] Deepgram key: {DEEPGRAM_API_KEY[:8]}...", flush=True)
    print(f"[*] Tools loaded: {[t['name'] for t in TOOLS]}", flush=True)
    print(f"[*] Connecting to Deepgram...", flush=True)

    try:
        async with websockets.connect(
            DEEPGRAM_AGENT_URL,
            additional_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"},
            open_timeout=10,
        ) as dg_ws:
            print("[+] Deepgram connected!\n", flush=True)

            settings_sent = False

            # ── Browser → Deepgram ────────────────────────────────────
            async def browser_to_deepgram():
                nonlocal settings_sent
                try:
                    while True:
                        msg = await browser_ws.receive()

                        if msg.get("bytes"):
                            # Raw mic audio — forward directly
                            await dg_ws.send(msg["bytes"])

                        elif msg.get("text"):
                            text = msg["text"]
                            try:
                                data = json.loads(text)
                                if data.get("type") == "Settings" and not settings_sent:
                                    # Replace browser settings with correct server settings
                                    payload = json.dumps(build_settings())
                                    print(f"[*] Sending settings with {len(TOOLS)} tools", flush=True)
                                    await dg_ws.send(payload)
                                    settings_sent = True
                                else:
                                    await dg_ws.send(text)
                            except json.JSONDecodeError:
                                await dg_ws.send(text)

                except WebSocketDisconnect:
                    print("[-] Browser disconnected", flush=True)
                except Exception as e:
                    if "no close frame" not in str(e) and "1005" not in str(e):
                        print(f"[!] browser→deepgram: {e}", flush=True)
                finally:
                    try: await dg_ws.close()
                    except Exception: pass

            # ── Deepgram → Browser ────────────────────────────────────
            async def deepgram_to_browser():
                try:
                    async for message in dg_ws:

                        if isinstance(message, bytes):
                            # TTS audio — forward directly to browser
                            await browser_ws.send_bytes(message)

                        else:
                            # JSON event — check if it's a tool call
                            print(f"[<] {message[:120]}", flush=True)

                            try:
                                event = json.loads(message)
                            except json.JSONDecodeError:
                                await browser_ws.send_text(message)
                                continue

                            if event.get("type") == "FunctionCallRequest":
                                # ── TOOL CALL — handle it here ────────
                                # Forward event to browser (so UI can show "agent is working...")
                                await browser_ws.send_text(message)
                                # Execute the tool and respond to Deepgram
                                await handle_function_call(dg_ws, event)
                            else:
                                # Everything else → forward to browser
                                await browser_ws.send_text(message)

                except Exception as e:
                    if "no close frame" not in str(e) and "1005" not in str(e):
                        print(f"[!] deepgram→browser: {e}", flush=True)

            await asyncio.gather(
                browser_to_deepgram(),
                deepgram_to_browser(),
            )

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