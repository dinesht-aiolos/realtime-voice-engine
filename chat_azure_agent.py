"""
chat_azure_agent.py
────────────────────
Interactive chat matching official Azure docs format exactly.

Docs format used:
  POST /openai/v1/responses
  {
    "agent_reference": {"type": "agent_reference", "name": "AGENT_NAME"},
    "input": [{"role": "user", "content": "your message"}]
  }

  With conversation history:
  {
    "agent_reference": {"type": "agent_reference", "name": "AGENT_NAME", "version": "1"},
    "conversation": "conv_123456789",
    "input": [{"role": "user", "content": "follow up"}]
  }

Run:
    python chat_azure_agent.py           <- normal chat
    python chat_azure_agent.py --debug   <- shows full raw response

.env needed:
    AZURE_OPENAI_KEY=your_azure_api_key
    AZURE_RESOURCE_NAME=crowdvoxpreprocessing-p-resource
    AZURE_PROJECT_NAME=crowdvoxpreprocessing-pipeline
    AGENT_NAME=voice-booking-agent
"""

import os
import sys
import json
import requests
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
AZURE_KEY     = os.getenv("AZURE_OPENAI_KEY", "")
RESOURCE_NAME = os.getenv("AZURE_RESOURCE_NAME", "crowdvoxpreprocessing-p-resource")
PROJECT_NAME  = os.getenv("AZURE_PROJECT_NAME",  "crowdvoxpreprocessing-pipeline")
AGENT_NAME    = os.getenv("AGENT_NAME", "voice-booking-agent")
AGENT_VERSION = "1"
DEBUG         = "--debug" in sys.argv

BASE_URL = f"https://{RESOURCE_NAME}.services.ai.azure.com/api/projects/{PROJECT_NAME}"

HEADERS = {
    "Content-Type":  "application/json",
    "Authorization": f"Bearer {AZURE_KEY}",
    "api-key":       AZURE_KEY,
}


# ════════════════════════════════════════════════════════════════════════
#  STEP 1: Create conversation (from docs)
#  POST /openai/v1/conversations
#  Body: { "items": [{ "type": "message", "role": "user",
#                      "content": [{"type": "input_text", "text": "..."}] }] }
# ════════════════════════════════════════════════════════════════════════

def create_conversation(first_message: str = "") -> str:
    """Create a new conversation thread. Returns conversation ID."""
    url = f"{BASE_URL}/openai/v1/conversations"

    # Matches docs exactly
    if first_message:
        body = {
            "items": [{
                "type": "message",
                "role": "user",
                "content": [{
                    "type": "input_text",
                    "text": first_message
                }]
            }]
        }
    else:
        body = {"items": []}

    response = requests.post(url, headers=HEADERS, json=body)

    if DEBUG:
        print(f"\n[DEBUG] Create conversation response ({response.status_code}):")
        try:
            print(json.dumps(response.json(), indent=2))
        except Exception:
            print(response.text)
        print()

    if response.status_code in (200, 201):
        data = response.json()
        conv_id = data.get("id", "")
        print(f"[Azure] Conversation: {conv_id}\n")
        return conv_id
    else:
        print(f"[Azure] Could not create conversation ({response.status_code}): {response.text}")
        return ""


# ════════════════════════════════════════════════════════════════════════
#  STEP 2: Send message (from docs)
#  POST /openai/v1/responses
#  Body: {
#    "agent_reference": {"type": "agent_reference", "name": "...", "version": "1"},
#    "conversation": "conv_123456789",
#    "input": [{"role": "user", "content": "..."}]
#  }
# ════════════════════════════════════════════════════════════════════════

def send_message(conversation_id: str, user_input: str) -> str:
    """Send message and return agent response text."""
    url = f"{BASE_URL}/openai/v1/responses"

    # Matches docs exactly — same format as curl example
    payload = {
        "agent_reference": {
            "type":    "agent_reference",
            "name":    AGENT_NAME,
            "version": AGENT_VERSION,
        },
        "input": [{"role": "user", "content": user_input}],
    }

    # Add conversation ID for multi-turn memory
    if conversation_id:
        payload["conversation"] = conversation_id

    if DEBUG:
        print(f"\n[DEBUG] Sending payload:")
        print(json.dumps(payload, indent=2))

    response = requests.post(url, headers=HEADERS, json=payload)

    if DEBUG:
        print(f"\n[DEBUG] Response ({response.status_code}):")
        try:
            print(json.dumps(response.json(), indent=2))
        except Exception:
            print(response.text)
        print()

    if response.status_code in (200, 201):
        return extract_text(response.json())
    else:
        print(f"\n[Error {response.status_code}]")
        try:
            print(json.dumps(response.json(), indent=2))
        except Exception:
            print(response.text)
        if response.status_code == 401:
            print("Fix: Check AZURE_OPENAI_KEY in .env")
        elif response.status_code == 404:
            print(f"Fix: Agent '{AGENT_NAME}' not found — run: python create_azure_agent.py create")
        return ""


def extract_text(data: dict) -> str:
    """
    Extract agent reply text from Azure response.
    Tries every known field location.
    Run with --debug to see exact response structure.
    """
    # Try all known locations
    candidates = [
        # Direct string fields
        data.get("text"),
        data.get("output_text") if isinstance(data.get("output_text"), str) else None,
        data.get("content") if isinstance(data.get("content"), str) else None,
    ]

    for c in candidates:
        if c and isinstance(c, str) and c.strip():
            return c.strip()

    # output is a list of message objects
    output = data.get("output", [])
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            # item.content list
            content = item.get("content", [])
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict):
                        t = c.get("text") or c.get("value") or ""
                        if t and isinstance(t, str):
                            return t.strip()
            # item.text
            t = item.get("text")
            if t and isinstance(t, str):
                return t.strip()
            # item.message.content
            msg = item.get("message", {})
            if isinstance(msg, dict):
                t = msg.get("content", "")
                if t and isinstance(t, str):
                    return t.strip()

    # OpenAI choices format
    choices = data.get("choices", [])
    if choices:
        msg = choices[0].get("message", {})
        t = msg.get("content", "")
        if t:
            return t.strip()

    # Nothing worked — show top-level keys
    if not DEBUG:
        print(f"[!] Could not extract text. Run with --debug to see full response.")
        print(f"[!] Response keys: {list(data.keys())}")
    return ""


# ════════════════════════════════════════════════════════════════════════
#  INTERACTIVE CHAT
# ════════════════════════════════════════════════════════════════════════

def chat():
    print("=" * 54)
    print("  Azure AI Foundry — Interactive Chat")
    print("=" * 54)
    print(f"  Agent:    {AGENT_NAME}  (v{AGENT_VERSION})")
    print(f"  Resource: {RESOURCE_NAME}")
    print(f"  Project:  {PROJECT_NAME}")
    if DEBUG:
        print("  Mode:     DEBUG")
    print("=" * 54)
    print("\nCommands: 'new' = new conversation | 'quit' = exit\n")

    if not AZURE_KEY:
        print("ERROR: AZURE_OPENAI_KEY not set in .env")
        sys.exit(1)

    conversation_id = create_conversation()

    while True:
        try:
            user_input = input("You:   ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nChat ended.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("Chat ended.")
            break

        if user_input.lower() == "new":
            print("\n── New conversation ──\n")
            conversation_id = create_conversation()
            continue

        reply = send_message(conversation_id, user_input)
        if reply:
            print(f"Agent: {reply}\n")


if __name__ == "__main__":
    chat()