"""
create_azure_agent.py
──────────────────────
Create and chat with Azure AI Foundry agent using REST API + API key.
No Azure CLI needed. No SDK authentication issues.

Based on official Microsoft REST API docs.

Run:
    python create_azure_agent.py create   <- create agent
    python create_azure_agent.py list     <- list all agents
    python create_azure_agent.py chat     <- test chat
    python create_azure_agent.py delete   <- delete agent

.env needed:
    AZURE_OPENAI_KEY=your_azure_api_key
    AZURE_RESOURCE_NAME=crowdvoxpreprocessing-p-resource
    AZURE_PROJECT_NAME=crowdvoxpreprocessing-pipeline
    AGENT_NAME=voice-booking-agent
    AZURE_MODEL=gpt-4o
"""

import os
import sys
import json
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ── Date Calculation ──────────────────────────────────────────────────────────
today = datetime.now()
today_str = today.strftime("%B %d, %Y")
current_month = today.strftime("%B")
current_year = today.year

# ── Config ────────────────────────────────────────────────────────────────────
AZURE_KEY     = os.getenv("AZURE_OPENAI_KEY", "")
RESOURCE_NAME = os.getenv("AZURE_RESOURCE_NAME", "crowdvoxpreprocessing-p-resource")
PROJECT_NAME  = os.getenv("AZURE_PROJECT_NAME",  "crowdvoxpreprocessing-pipeline")
AGENT_NAME    = os.getenv("AGENT_NAME",  "voice-booking-agent")
AZURE_MODEL   = os.getenv("AZURE_MODEL", "gpt-4o")
API_VERSION   = "v1"

# Base URL
BASE_URL = f"https://{RESOURCE_NAME}.services.ai.azure.com/api/projects/{PROJECT_NAME}"

# ── Headers — uses API key directly as Bearer token ───────────────────────────
# Azure Foundry REST API accepts both:
#   Authorization: Bearer YOUR_API_KEY
#   api-key: YOUR_API_KEY
def get_headers() -> dict:
    return {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {AZURE_KEY}",
        "api-key":       AZURE_KEY,
    }

# ── System prompt ─────────────────────────────────────────────────────────────
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
    f"4. Call check_availability with the details collected. "
    f"5. If available, ask for their name only. Wait for answer. "
    f"6. Ask for their phone number only. Validate exactly 10 digits. Re-ask if invalid. "
    f"7. Confirm all details back to the customer in one sentence. "
    f"8. If they confirm, call create_booking. "

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


# ════════════════════════════════════════════════════════════════════════
#  TOOLS
# ════════════════════════════════════════════════════════════════════════

TOOLS = [
    {
        "type": "function",
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
        "type": "function",
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
        "type": "function",
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

def check_availability(date: str, time: str, party_size: int) -> dict:
    """Check if a restaurant table is available."""
    print(f"[TOOL] check_availability: {date} {time} x{party_size}", flush=True)
    return {
        "available": True,
        "date": date, "time": time, "party_size": party_size,
        "available_slots": ["6:00 PM", "7:00 PM", "7:30 PM", "8:00 PM"],
        "message": f"Yes, we have availability on {date}. {time} works great!",
    }


def create_booking(name: str, phone: str, date: str, time: str, party_size: int) -> dict:
    """Create a confirmed restaurant booking."""
    import uuid
    print(f"[TOOL] create_booking: {name} {phone} {date} {time} x{party_size}", flush=True)
    ref = f"BOOK-{uuid.uuid4().hex[:6].upper()}"
    return {
        "success": True,
        "confirmation_ref": ref,
        "message": f"Booking confirmed! Your reference is {ref}. See you on {date} at {time}!",
    }


def get_weather(city: str) -> dict:
    """Get current weather for a city."""
    print(f"[TOOL] get_weather: {city}", flush=True)
    try:
        response = requests.get(f"https://wttr.in/{city}?format=j1", timeout=5)
        data = response.json()
        c = data["current_condition"][0]
        return {
            "city": city, "temp_c": c["temp_C"], "feels_c": c["FeelsLikeC"],
            "condition": c["weatherDesc"][0]["value"], "humidity": f"{c['humidity']}%",
            "message": f"It's {c['temp_C']}°C in {city}, {c['weatherDesc'][0]['value'].lower()}. Feels like {c['FeelsLikeC']}°C.",
        }
    except Exception as e:
        return {"error": str(e), "message": f"Couldn't fetch weather for {city}."}


def check_config():
    if not AZURE_KEY:
        print("ERROR: AZURE_OPENAI_KEY not set in .env")
        sys.exit(1)
    print(f"[Azure] Resource: {RESOURCE_NAME}")
    print(f"[Azure] Project:  {PROJECT_NAME}")
    print(f"[Azure] Base URL: {BASE_URL}")
    print(f"[Azure] API Key:  {AZURE_KEY[:8]}...")


def handle_response(response: requests.Response, action: str) -> dict:
    """Handle API response and print errors clearly."""
    if response.status_code in (200, 201):
        return response.json()

    print(f"\nERROR {response.status_code} on {action}:")
    try:
        error = response.json()
        print(json.dumps(error, indent=2))
    except Exception:
        print(response.text)

    if response.status_code == 401:
        print("\nFix: Check AZURE_OPENAI_KEY in your .env")
    elif response.status_code == 403:
        print("\nFix: Your account needs Azure AI User role on this resource")
    elif response.status_code == 404:
        print("\nFix: Check AZURE_RESOURCE_NAME and AZURE_PROJECT_NAME in .env")
        print(f"     Current URL: {BASE_URL}")
    return None


# ════════════════════════════════════════════════════════════════════════
#  CREATE AGENT
# ════════════════════════════════════════════════════════════════════════

def create_agent():
    """
    Create a new agent via REST API.
    POST /api/projects/{project}/agents?api-version=v1
    """
    print(f"\n[Azure] Creating agent: {AGENT_NAME}")
    print(f"[Azure] Model: {AZURE_MODEL}\n")
    check_config()

    url = f"{BASE_URL}/agents?api-version={API_VERSION}"

    payload = {
        "name": AGENT_NAME,
        "definition": {
            "kind":         "prompt",
            "model":        AZURE_MODEL,
            "instructions": AGENT_INSTRUCTIONS,
            "tools":        TOOLS,
        }
    }

    print(f"\n[Azure] POST {url}")
    response = requests.post(url, headers=get_headers(), json=payload)
    data = handle_response(response, "create agent")

    if data:
        print(f"\nAgent created successfully!")
        print(f"   ID:      {data.get('id', 'N/A')}")
        print(f"   Name:    {data.get('name', 'N/A')}")
        print(f"   Version: {data.get('version', 'N/A')}")
        print(f"\nAdd to your .env:")
        print(f"   AGENT_NAME={data.get('name', AGENT_NAME)}")
        return data

    return None


# ════════════════════════════════════════════════════════════════════════
#  LIST AGENTS
# ════════════════════════════════════════════════════════════════════════

def list_agents():
    """
    List all agents in the project.
    GET /api/projects/{project}/agents?api-version=v1
    """
    print("\n[Azure] Listing agents...")
    check_config()

    url = f"{BASE_URL}/agents?api-version={API_VERSION}"
    response = requests.get(url, headers=get_headers())
    data = handle_response(response, "list agents")

    if data:
        agents = data.get("value", [data] if "name" in data else [])
        if not agents:
            print("No agents found.")
            return
        print(f"\nFound {len(agents)} agent(s):")
        for a in agents:
            print(f"  Name: {a.get('name')}  |  ID: {a.get('id')}  |  Version: {a.get('version', 'N/A')}")


# ════════════════════════════════════════════════════════════════════════
#  CHAT WITH AGENT
# ════════════════════════════════════════════════════════════════════════

def chat_with_agent():
    """
    Test chat with the agent via REST API.
    POST /api/projects/{project}/openai/v1/responses
    """
    print(f"\n[Azure] Chatting with: {AGENT_NAME}\n")
    check_config()

    # Step 1: Create a conversation
    conv_url = f"{BASE_URL}/openai/v1/conversations"
    conv_response = requests.post(
        conv_url,
        headers=get_headers(),
        json={
            "items": [{
                "type":    "message",
                "role":    "user",
                "content": [{"type": "input_text", "text": "I want to book a table"}]
            }]
        }
    )
    conv_data = handle_response(conv_response, "create conversation")
    conversation_id = conv_data.get("id") if conv_data else None

    if conversation_id:
        print(f"[Azure] Conversation ID: {conversation_id}\n")

    # Step 2: Ask questions
    questions = [
        "I want to book a table",
        "This Saturday at 7pm for 4 people",
    ]

    resp_url = f"{BASE_URL}/openai/v1/responses"

    for question in questions:
        print(f"You:   {question}")

        payload = {
            "agent_reference": {"type": "agent_reference", "name": AGENT_NAME},
            "input": [{"role": "user", "content": question}],
        }
        if conversation_id:
            payload["conversation"] = conversation_id

        response = requests.post(resp_url, headers=get_headers(), json=payload)
        data = handle_response(response, "chat")

        if data:
            # Extract text from response
            output = data.get("output_text") or data.get("output", "")
            if isinstance(output, list):
                for item in output:
                    if isinstance(item, dict) and item.get("type") == "message":
                        for c in item.get("content", []):
                            if c.get("type") == "output_text":
                                output = c.get("text", "")
            print(f"Agent: {output}\n")


# ════════════════════════════════════════════════════════════════════════
#  DELETE AGENT
# ════════════════════════════════════════════════════════════════════════

def delete_agent():
    """
    Delete agent by name.
    DELETE /api/projects/{project}/agents/{name}?api-version=v1
    """
    print(f"\n[Azure] Deleting agent: {AGENT_NAME}")
    check_config()

    url = f"{BASE_URL}/agents/{AGENT_NAME}?api-version={API_VERSION}"
    response = requests.delete(url, headers=get_headers())

    if response.status_code in (200, 204):
        print(f"Agent '{AGENT_NAME}' deleted.")
    else:
        handle_response(response, "delete agent")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 54)
    print("  Azure AI Foundry — REST API Agent Manager")
    print("=" * 54)

    command = sys.argv[1] if len(sys.argv) > 1 else "create"

    if   command == "create": create_agent()
    elif command == "list":   list_agents()
    elif command == "chat":   chat_with_agent()
    elif command == "delete": delete_agent()
    else:
        print("Commands:")
        print("  python create_azure_agent.py create  <- create agent")
        print("  python create_azure_agent.py list    <- list agents")
        print("  python create_azure_agent.py chat    <- test chat")
        print("  python create_azure_agent.py delete  <- delete agent")