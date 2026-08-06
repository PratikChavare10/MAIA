from langdetect import detect
from deep_translator import GoogleTranslator

# ── Language Detection ────────────────────────────
def detect_language(text: str) -> str:

    try:
        lang= detect(text)
        if lang == "mr":
            lang="Marathi"
        elif lang  == "hi":
            lan = "Hindi"
        elif lang == "te":
            lan = "Telugu"
        else:
            lan = "English"
    except Exception:
        return "en"   # Default to English

# ── Translate to English ──────────────────────────
def translate_to_english(text: str) -> str:

    try:
        if not text or not text.strip():
            return text
        return GoogleTranslator(
            source='auto', target='en'
        ).translate(text)
    except Exception:
        return text   # Return original if translation fails


# text= "कसा आहेस तु"
# lang = translate_to_english(text)
# print(lang)
