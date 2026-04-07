"""
test_azure_connection.py
─────────────────────────
Tests different Azure Voice Live URL formats to find which one works.
Run this before trying the full server.

Run:
    python test_azure_connection.py

.env needed:
    AZURE_OPENAI_KEY=your_key
    AZURE_RESOURCE_NAME=crowdvoxpreprocessing-p-resource
    AZURE_PROJECT_NAME=crowdvoxpreprocessing-pipeline
"""

import os
import asyncio
import json
import websockets
from dotenv import load_dotenv

load_dotenv()

AZURE_KEY     = os.getenv("AZURE_OPENAI_KEY", "")
RESOURCE_NAME = os.getenv("AZURE_RESOURCE_NAME", "crowdvoxpreprocessing-p-resource")
PROJECT_NAME  = os.getenv("AZURE_PROJECT_NAME", "crowdvoxpreprocessing-pipeline")
AGENT_ID      = os.getenv("AGENT_ID", "")
API_VERSION   = "2025-10-01"

# All URL formats to test
URLS_TO_TEST = [
    # New endpoint + model only
    f"wss://{RESOURCE_NAME}.services.ai.azure.com/voice-live/realtime?api-version={API_VERSION}&model=gpt-4o-realtime-preview",

    # Old endpoint + model only
    f"wss://{RESOURCE_NAME}.cognitiveservices.azure.com/voice-live/realtime?api-version={API_VERSION}&model=gpt-4o-realtime-preview",

    # New endpoint + agent
    f"wss://{RESOURCE_NAME}.services.ai.azure.com/voice-live/realtime?api-version={API_VERSION}&agent_id={AGENT_ID}&project_id={PROJECT_NAME}" if AGENT_ID else None,

    # Old endpoint + agent
    f"wss://{RESOURCE_NAME}.cognitiveservices.azure.com/voice-live/realtime?api-version={API_VERSION}&agent_id={AGENT_ID}&project_id={PROJECT_NAME}" if AGENT_ID else None,

    # Try without api-version
    f"wss://{RESOURCE_NAME}.services.ai.azure.com/voice-live/realtime?model=gpt-4o-realtime-preview",

    # Try older api version
    f"wss://{RESOURCE_NAME}.services.ai.azure.com/voice-live/realtime?api-version=2024-10-01&model=gpt-4o-realtime-preview",
]

HEADERS = {"api-key": AZURE_KEY}


async def test_url(url: str, index: int) -> bool:
    """Test if a URL connects successfully."""
    if not url:
        return False

    short = url[:80] + "..."
    print(f"\n[Test {index}] {short}")

    try:
        async with websockets.connect(
            url,
            additional_headers=HEADERS,
            open_timeout=8,
        ) as ws:
            print(f"  ✅ CONNECTED!")

            # Wait for first message
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=5)
                if isinstance(msg, bytes):
                    print(f"  First message: [binary {len(msg)} bytes]")
                else:
                    data = json.loads(msg)
                    print(f"  First message: type={data.get('type', 'unknown')}")
                    print(f"  ✅ THIS URL WORKS — use it in server_azure.py")
                    print(f"\n  WORKING URL:\n  {url}\n")
                    return True
            except asyncio.TimeoutError:
                print(f"  Connected but no message received (timeout) — may still work")
                return True

    except websockets.exceptions.InvalidStatus as e:
        print(f"  ❌ HTTP {e.response.status_code} — ", end="")
        if e.response.status_code == 401:
            print("Wrong API key")
        elif e.response.status_code == 403:
            print("No permission — need Cognitive Services User role")
        elif e.response.status_code == 404:
            print("Endpoint not found")
        else:
            print(f"Rejected")

    except Exception as e:
        err = str(e)
        if "did not receive a valid HTTP response" in err:
            print(f"  ❌ Invalid HTTP response — endpoint may not exist or wrong format")
        elif "Name or service not known" in err or "getaddrinfo" in err:
            print(f"  ❌ DNS error — resource name wrong")
        else:
            print(f"  ❌ {type(e).__name__}: {err[:80]}")

    return False


async def main():
    print("=" * 60)
    print("  Azure Voice Live — Connection Tester")
    print("=" * 60)
    print(f"  Resource: {RESOURCE_NAME}")
    print(f"  Key:      {AZURE_KEY[:8]}..." if AZURE_KEY else "  Key:      NOT SET ❌")
    print(f"  Agent ID: {AGENT_ID or 'not set'}")
    print("=" * 60)

    if not AZURE_KEY:
        print("\n❌ AZURE_OPENAI_KEY not set in .env")
        return

    found = False
    for i, url in enumerate(URLS_TO_TEST, 1):
        if url:
            result = await test_url(url, i)
            if result:
                found = True
                break
        else:
            print(f"\n[Test {i}] Skipped (AGENT_ID not set)")

    if not found:
        print("\n" + "=" * 60)
        print("No working URL found. Possible reasons:")
        print("  1. Voice Live API not enabled on your resource")
        print("  2. Wrong AZURE_RESOURCE_NAME in .env")
        print("  3. Wrong AZURE_OPENAI_KEY")
        print("  4. Your resource region does not support Voice Live")
        print("\nCheck supported regions:")
        print("  https://learn.microsoft.com/azure/ai-services/speech-service/voice-live-overview")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())