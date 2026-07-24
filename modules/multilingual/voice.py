"""
modules/multilingual/voice.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT TO ADD:
- No setup needed — Whisper automatically downloads
  the model on first run (100MB approx)

HOW TO USE:
   from modules.multilingual.voice import voice_to_text
   text = voice_to_text("path/to/audio.wav")
"""

import whisper

# ── Load Whisper (once at startup) ────────────────
# ADD: "base" model is fast and accurate enough
# Options: tiny, base, small, medium, large
# (larger = more accurate but slower)
_whisper_model = None

def _load():
    global _whisper_model
    if _whisper_model is None:
        print("Loading Whisper model...")
        _whisper_model = whisper.load_model("base")
        print("✅ Whisper loaded!")

def voice_to_text(audio_path: str) -> str:
    """
    Voice audio को text मध्ये convert करतो

    Input:
        audio_path (str) → .wav / .mp3 file path

    Output:
        str → transcribed text
    """
    _load()
    result = _whisper_model.transcribe(audio_path)
    return result["text"].strip()
