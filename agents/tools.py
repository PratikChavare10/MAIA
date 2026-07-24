"""
agents/tools.py
━━━━━━━━━━━━━━
WHAT TO ADD:
- सगळे modules train + ready झाल्यावर
  हे file automatically सगळे import करतो
- नवीन module add केला तर येथे import add करा

HOW TO TEST:
   python agents/tools.py
"""

# ── Import All Modules ────────────────────────────
from modules.disease.predict             import predict_disease
from modules.crop_recommendation.predict import recommend_crop
from modules.yield_prediction.predict    import predict_yield
from modules.irrigation.calculator       import calculate_irrigation
from modules.weather.fetch               import get_weather
from modules.rag.retriever               import rag_search
from modules.multilingual.voice          import voice_to_text
from modules.multilingual.translator     import (
    detect_language,
    translate_to_english,
    translate_to_original
)
from modules.memory.memory               import (
    save_to_memory,
    get_history,
    clear_memory
)

print("✅ All tools loaded successfully!")

# ── Quick test ────────────────────────────────────
if __name__ == "__main__":
    print("\nTesting Weather Tool...")
    weather = get_weather("Pune")
    print(f"Weather: {weather}")

    print("\nTesting Irrigation Tool...")
    irrigation = calculate_irrigation(
        weather=weather,
        soil_moisture=35,
        crop_stage="flowering"
    )
    print(f"Irrigation: {irrigation}")

    print("\nTesting Translation Tool...")
    lang = detect_language("माझ्या टोमॅटोला रोग आहे")
    print(f"Language detected: {lang}")
    eng = translate_to_english("माझ्या टोमॅटोला रोग आहे")
    print(f"English: {eng}")

    print("\n✅ Basic tools working!")
