"""
modules/multilingual/translator.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT TO ADD:
- No setup needed — uses free Google Translate
- Internet connection आवश्यक आहे

HOW TO USE:
   from modules.multilingual.translator import (
       detect_language, translate_to_english,
       translate_to_original
   )
"""

from langdetect import detect
from deep_translator import GoogleTranslator

# ── Language Detection ────────────────────────────
def detect_language(text: str) -> str:
    """
    Text ची भाषा ओळखतो

    Output:
        str → 'mr' (Marathi), 'hi' (Hindi),
               'en' (English), etc.
    """
    try:
        return detect(text)
    except Exception:
        return "en"   # Default to English

# ── Translate to English ──────────────────────────
def translate_to_english(text: str) -> str:
    """
    कोणत्याही भाषेतून English मध्ये translate करतो
    (AI processing साठी)
    """
    try:
        if not text or not text.strip():
            return text
        return GoogleTranslator(
            source='auto', target='en'
        ).translate(text)
    except Exception:
        return text   # Return original if translation fails

# ── Translate Back to Original Language ──────────
def translate_to_original(text: str, lang: str) -> str:
    """
    English answer ला farmer च्या भाषेत translate करतो

    Input:
        text → English answer
        lang → 'mr', 'hi', 'en', etc.
    """
    try:
        if lang == "en" or not text or not text.strip():
            return text
        return GoogleTranslator(
            source='en', target=lang
        ).translate(text)
    except Exception:
        return text   # Return English if translation fails
