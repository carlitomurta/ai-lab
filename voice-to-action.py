"""
Voice-to-Action AI Agent
Uses OpenAI Whisper for speech recognition and PyAudio for microphone input.
Recognizes spoken commands and dispatches them to registered action functions.
"""

import io
import os
import wave
import time
import struct
import functools
from datetime import datetime

import numpy as np
import pyaudio
import whisper
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────

SAMPLE_RATE     = 16_000   # Hz  (Whisper expects 16 kHz)
CHANNELS        = 1        # Mono
FORMAT          = pyaudio.paInt16
CHUNK           = 1_024    # Frames per buffer
SILENCE_THRESH  = 500      # RMS below this = silence
SILENCE_SECONDS = 1.5      # Seconds of silence before stopping recording
MAX_SECONDS     = 10       # Hard cap on recording length
WHISPER_MODEL   = os.getenv("WHISPER_MODEL_SIZE")  # tiny | base | small | medium | large


# ─────────────────────────────────────────────
#  Command Registry
# ─────────────────────────────────────────────

_commands: dict[str, dict] = {}   # keyword → {func, description, aliases}


def command(*keywords: str, description: str = ""):
    """
    Decorator to register a function as a voice command handler.

    Usage:
        @command("open browser", "launch browser", description="Opens the web browser")
        def open_browser():
            ...
    """
    def decorator(func):
        entry = {"func": func, "description": description, "aliases": list(keywords)}
        for kw in keywords:
            _commands[kw.lower()] = entry

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


def dispatch(transcript: str) -> bool:
    """
    Match transcript text against registered commands and call the handler.
    Returns True if a command was matched and executed.
    """
    text = transcript.lower().strip()
    for keyword, entry in _commands.items():
        if keyword in text:
            print(f"  ✅ Command matched: '{keyword}'")
            entry["func"]()
            return True
    return False


def list_commands():
    """Print all registered commands to the console."""
    seen = set()
    print("\n  Registered commands:")
    for entry in _commands.values():
        fid = id(entry["func"])
        if fid not in seen:
            seen.add(fid)
            aliases = " / ".join(f'"{a}"' for a in entry["aliases"])
            desc    = entry["description"] or "no description"
            print(f"    • {aliases}  →  {desc}")
    print()


# ─────────────────────────────────────────────
#  Built-in Action Handlers
# ─────────────────────────────────────────────

@command("what time is it", "current time", "tell me the time",
         description="Speaks the current time")
def get_time():
    now = datetime.now().strftime("%H:%M on %A, %B %d")
    print(f"  🕐 It is {now}.")


@command("open browser", "launch browser", "start browser",
         description="Opens the default web browser")
def open_browser():
    import webbrowser
    webbrowser.open("https://www.google.com")
    print("  🌐 Browser opened.")


@command("what commands", "help", "list commands",
         description="Lists all available commands")
def show_help():
    list_commands()


@command("stop listening", "quit", "exit", "goodbye",
         description="Stops the voice agent")
def stop_agent():
    # Handled specially in the main loop via a flag
    print("  👋 Stopping agent...")


@command("take a note", "add a note", "remember this",
         description="Prompts you to dictate a note (placeholder)")
def take_note():
    print("  📝 Note-taking triggered — extend this handler to save text.")


# ─────────────────────────────────────────────
#  Audio Capture
# ─────────────────────────────────────────────

def rms(data: bytes) -> float:
    """Compute RMS energy of a raw PCM chunk."""
    count  = len(data) // 2
    shorts = struct.unpack(f"{count}h", data)
    squares = sum(s * s for s in shorts)
    return (squares / count) ** 0.5 if count else 0.0


def record_until_silence(pa: pyaudio.PyAudio) -> bytes | None:
    """
    Open the microphone and record until SILENCE_SECONDS of silence
    or MAX_SECONDS total, whichever comes first.
    Returns raw PCM bytes, or None if nothing was captured.
    """
    stream = pa.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )

    print("  🎙  Listening…", end="", flush=True)
    frames: list[bytes] = []
    silence_frames = 0
    max_frames     = int(SAMPLE_RATE / CHUNK * MAX_SECONDS)
    silence_limit  = int(SAMPLE_RATE / CHUNK * SILENCE_SECONDS)
    got_voice      = False

    try:
        for _ in range(max_frames):
            chunk = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(chunk)
            energy = rms(chunk)

            if energy > SILENCE_THRESH:
                got_voice = True
                silence_frames = 0
                print(".", end="", flush=True)
            else:
                silence_frames += 1

            if got_voice and silence_frames >= silence_limit:
                break
    finally:
        stream.stop_stream()
        stream.close()

    print()
    if not got_voice:
        return None

    return b"".join(frames)


def pcm_to_wav_bytes(pcm: bytes) -> bytes:
    """Wrap raw PCM bytes in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)          # paInt16 = 2 bytes
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm)
    return buf.getvalue()


def pcm_to_float32(pcm: bytes) -> np.ndarray:
    """Convert raw PCM bytes to a float32 numpy array that Whisper expects."""
    audio_np = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    return audio_np / 32768.0


# ─────────────────────────────────────────────
#  Transcription
# ─────────────────────────────────────────────

def transcribe(model: whisper.Whisper, pcm: bytes) -> str:
    """Run Whisper inference on a PCM buffer and return transcript text."""
    audio = pcm_to_float32(pcm)
    result = model.transcribe(audio, fp16=False, language="en")
    return result["text"].strip()


# ─────────────────────────────────────────────
#  Main Agent Loop
# ─────────────────────────────────────────────

def run_agent():
    print("─" * 50)
    print("  Voice-to-Action Agent")
    print(f"  Loading Whisper model: '{WHISPER_MODEL}'…")
    model = whisper.load_model(WHISPER_MODEL)
    print("  Model ready.")
    list_commands()
    print("  Say a command, or 'quit' to exit.\n")
    print("─" * 50)

    pa = pyaudio.PyAudio()
    running = True

    try:
        while running:
            pcm = record_until_silence(pa)

            if pcm is None:
                print("  (no speech detected — try again)\n")
                continue

            print("  🔄 Transcribing…")
            transcript = transcribe(model, pcm)
            print(f"  📝 You said: \"{transcript}\"")

            # Check for stop command before dispatching
            stop_words = ("stop listening", "quit", "exit", "goodbye")
            if any(w in transcript.lower() for w in stop_words):
                print("  👋 Goodbye!")
                running = False
                continue

            matched = dispatch(transcript)
            if not matched:
                print("  ❓ No command matched. Try saying 'help' for a list.\n")
            else:
                print()

            time.sleep(0.3)   # brief pause before next listen cycle

    except KeyboardInterrupt:
        print("\n  Interrupted — shutting down.")
    finally:
        pa.terminate()


# ─────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    run_agent()