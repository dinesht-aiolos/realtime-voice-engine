# # Realtime Deepgram Voice Agent - Working with Microphone

# import os
# import sys
# import time
# import threading
# import numpy as np
# import sounddevice as sd
# from dotenv import load_dotenv

# from deepgram import DeepgramClient
# from deepgram.core.events import EventType
# from deepgram.agent.v1.types import (
#     AgentV1Settings,
#     AgentV1SettingsAgent,
#     AgentV1SettingsAudio,
#     AgentV1SettingsAudioInput,
#     AgentV1SettingsAudioOutput,
#     AgentV1SettingsAgentListen,
#     AgentV1SettingsAgentListenProvider_V1,
# )

# # Load env
# load_dotenv()

# # Enable UTF-8 console output on Windows
# if sys.platform == "win32":
#     import io
#     sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# SAMPLE_RATE = 24000
# CHUNK = 960
# CHANNELS = 1

# class VoiceAgent:
#     def __init__(self):
#         self.api_key = os.getenv("DEEPGRAM_API_KEY")
#         if not self.api_key:
#             raise ValueError("DEEPGRAM_API_KEY not set in .env")

#         self.client = DeepgramClient(api_key=self.api_key)
#         self.should_stop = False
#         self.audio_buffer = bytearray()

#     def mic_callback(self, indata, frames, time_info, status, connection):
#         """Callback for microphone input"""
#         if status:
#             print(f"Audio status: {status}", flush=True)

#         try:
#             audio_bytes = indata.tobytes()
#             connection.send_media(audio_bytes)
#         except Exception as e:
#             print(f"Send error: {e}", flush=True)

#     def run(self):
#         try:
#             print("Initializing Deepgram client...", flush=True)
#             # Use connection as a context manager
#             with self.client.agent.v1.connect() as connection:
#                 print("✓ Connected to Deepgram\n", flush=True)

#                 # Configure the Agent settings
#                 settings = AgentV1Settings(
#                     type="Settings",
#                     audio=AgentV1SettingsAudio(
#                         input=AgentV1SettingsAudioInput(
#                             encoding="linear16",
#                             sample_rate=SAMPLE_RATE,
#                         ),
#                         output=AgentV1SettingsAudioOutput(
#                             encoding="linear16",
#                             sample_rate=SAMPLE_RATE,
#                             container="wav",
#                         ),
#                     ),
#                     agent=AgentV1SettingsAgent(
#                         language="en",
#                         listen=AgentV1SettingsAgentListen(
#                             provider=AgentV1SettingsAgentListenProvider_V1(
#                                 type="deepgram",
#                                 model="nova-3",
#                             )
#                         ),
#                         think={
#                             "provider": {
#                                 "type": "open_ai",
#                                 "model": "gpt-4o-mini",
#                             },
#                             "prompt": "You are a helpful voice assistant. Keep responses short.",
#                         },
#                         speak=[{
#                             "provider": {
#                                 "type": "deepgram",
#                                 "model": "aura-2-thalia-en",
#                             }
#                         }],
#                         greeting="Hello! How can I help you today?",
#                     ),
#                 )

#                 # Event handlers
#                 def on_open(event):
#                     print("✓ Connection opened\n", flush=True)

#                 def on_message(message):
#                     # Handle binary audio data
#                     if isinstance(message, bytes):
#                         self.audio_buffer.extend(message)
#                         return

#                     msg_type = getattr(message, "type", "Unknown")

#                     if msg_type == "Welcome":
#                         print("✓ Welcome message received", flush=True)

#                     elif msg_type == "SettingsApplied":
#                         print("✓ Settings applied\n", flush=True)

#                     elif msg_type == "ConversationText":
#                         role = getattr(message, "role", "unknown")
#                         content = getattr(message, "content", "")
#                         if role == "user":
#                             print(f"👤 You: {content}", flush=True)
#                         else:
#                             print(f"🤖 Agent: {content}", flush=True)

#                     elif msg_type == "UserStartedSpeaking":
#                         print("👂 (Listening...)", flush=True)

#                     elif msg_type == "AgentThinking":
#                         print("🧠 (Thinking...)", flush=True)

#                     elif msg_type == "AgentStartedSpeaking":
#                         self.audio_buffer = bytearray()
#                         print("🗣️ (Speaking...)", flush=True)

#                     elif msg_type == "AgentAudioDone":
#                         if len(self.audio_buffer) > 0:
#                             try:
#                                 audio_np = np.frombuffer(bytes(self.audio_buffer), dtype="int16")
#                                 print("🔊 Playing response...", flush=True)
#                                 sd.play(audio_np, samplerate=SAMPLE_RATE)
#                                 sd.wait()
#                             except Exception as e:
#                                 print(f"Playback error: {e}", flush=True)
#                             finally:
#                                 self.audio_buffer = bytearray()
#                         print("", flush=True)

#                 def on_error(error):
#                     print(f"✗ Error: {error}", flush=True)
#                     self.should_stop = True

#                 def on_close(event):
#                     print(f"✗ Connection closed", flush=True)
#                     self.should_stop = True

#                 # Register event handlers
#                 connection.on(EventType.OPEN, on_open)
#                 connection.on(EventType.MESSAGE, on_message)
#                 connection.on(EventType.ERROR, on_error)
#                 connection.on(EventType.CLOSE, on_close)

#                 # Send settings
#                 print("Sending settings...", flush=True)
#                 connection.send_settings(settings)
#                 print("Settings sent\n", flush=True)

#                 # Start listener thread
#                 listener_thread = threading.Thread(
#                     target=connection.start_listening,
#                     daemon=True
#                 )
#                 listener_thread.start()

#                 # Wait for initialization
#                 time.sleep(2)

#                 # Start microphone
#                 print("Starting microphone...", flush=True)
#                 print("Speak now! (Ctrl+C to stop)\n", flush=True)

#                 with sd.InputStream(
#                     samplerate=SAMPLE_RATE,
#                     blocksize=CHUNK,
#                     channels=CHANNELS,
#                     dtype="int16",
#                     callback=lambda indata, frames, time_info, status: self.mic_callback(indata, frames, time_info, status, connection),
#                 ):
#                     while not self.should_stop:
#                         time.sleep(0.5)

#         except KeyboardInterrupt:
#             print("\nStopped by user", flush=True)
#             self.should_stop = True

#         except Exception as e:
#             print(f"Error: {e}", flush=True)
#             import traceback
#             traceback.print_exc()

#         finally:
#             print("\nChat ended", flush=True)


# if __name__ == "__main__":
#     print("🎤 Realtime Voice Agent (Deepgram)\n", flush=True)
#     VoiceAgent().run()
