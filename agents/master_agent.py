"""
agents/master_agent.py
━━━━━━━━━━━━━━━━━━━━━
WHAT TO ADD:
- सगळे modules train + ready झाल्यावर हे run करा
- OPENAI_API_KEY .env मध्ये असणे आवश्यक आहे

HOW TO TEST:
   python agents/master_agent.py

FLOW:
   Farmer Input → weather → disease → crop_rec
   → yield → irrigation → rag → combine → Answer
"""

from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
import os

from config import OPENAI_API_KEY
from agents.tools import (
    predict_disease, recommend_crop, predict_yield,
    calculate_irrigation, get_weather, rag_search
)

# Set OpenAI key for LangGraph/LLM
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ── Shared State ──────────────────────────────────
# ADD: नवीन module add केला तर येथे नवीन field add करा
class FarmState(TypedDict):
    original_text:     str
    language:          str
    image_path:        Optional[str]
    city:              Optional[str]
    soil_data:         Optional[dict]

    # Module Results
    weather_data:      Optional[dict]
    disease_result:    Optional[dict]
    crop_rec_result:   Optional[dict]
    yield_result:      Optional[dict]
    irrigation_result: Optional[dict]
    rag_result:        Optional[dict]

    # Final
    final_answer:      Optional[str]


# ── Node Functions (each = one agent step) ────────

def weather_node(state: FarmState) -> FarmState:
    """Step 1: Weather fetch करतो"""
    city = state.get("city", "Pune")
    state["weather_data"] = get_weather(city)
    return state


def disease_node(state: FarmState) -> FarmState:
    """Step 2: Disease detect करतो (image असेल तर)"""
    if state.get("image_path"):
        state["disease_result"] = predict_disease(
            state["image_path"]
        )
    else:
        state["disease_result"] = {"disease": "No image provided",
                                    "confidence": "N/A",
                                    "treatment": "Upload crop photo for disease detection",
                                    "prevention": "",
                                    "severity": "N/A"}
    return state


def crop_rec_node(state: FarmState) -> FarmState:
    """Step 3: Crop recommendation देतो"""
    soil = state.get("soil_data") or {}
    w    = state.get("weather_data") or {}

    # ADD: Soil data form मधून घेणे — default values आत्ता
    state["crop_rec_result"] = recommend_crop(
        N=soil.get("N", 90),
        P=soil.get("P", 42),
        K=soil.get("K", 43),
        temperature=w.get("temperature", 25),
        humidity=w.get("humidity", 70),
        ph=soil.get("ph", 6.5),
        rainfall=soil.get("rainfall", 200)
    )
    return state


def yield_node(state: FarmState) -> FarmState:
    """Step 4: Yield prediction करतो"""
    soil = state.get("soil_data") or {}
    w    = state.get("weather_data") or {}

    state["yield_result"] = predict_yield(
        crop=soil.get("crop", "Cotton"),
        soil_type=soil.get("soil_type", "Black"),
        rainfall=w.get("temperature", 650),   # ADD: actual rainfall
        temperature=w.get("temperature", 32),
        area=soil.get("area", 2),
        fertilizer=soil.get("fertilizer", 50),
        humidity=w.get("humidity", 70),
        last_4_weeks=soil.get("last_4_weeks", [
            [650, 32, 70],
            [600, 33, 68],
            [700, 31, 75],
            [620, 32, 72]
        ])
    )
    return state


def irrigation_node(state: FarmState) -> FarmState:
    """Step 5: Irrigation advice देतो"""
    soil = state.get("soil_data") or {}
    state["irrigation_result"] = calculate_irrigation(
        weather=state.get("weather_data") or {},
        soil_moisture=soil.get("moisture", 45),
        crop_stage=soil.get("stage", "flowering")
    )
    return state


def rag_node(state: FarmState) -> FarmState:
    """Step 6: RAG knowledge base search करतो"""
    state["rag_result"] = rag_search(state["original_text"])
    return state


def combine_node(state: FarmState) -> FarmState:
    """Step 7: सगळे results एकत्र करतो"""
    parts = []

    # Disease Result
    d = state.get("disease_result")
    if d and d.get("disease") != "No image provided":
        parts.append(
            f"DISEASE DETECTED: {d['disease']} "
            f"(Confidence: {d['confidence']})\n"
            f"Treatment: {d['treatment']}\n"
            f"Prevention: {d['prevention']}"
        )

    # Crop Recommendation
    c = state.get("crop_rec_result")
    if c:
        top3 = ", ".join(
            [f"{x['crop']} ({x['confidence']})"
             for x in c.get("top_3", [])]
        )
        parts.append(
            f"RECOMMENDED CROP: {c['recommended_crop']}\n"
            f"Top 3 options: {top3}"
        )

    # Yield Prediction
    y = state.get("yield_result")
    if y:
        parts.append(
            f"EXPECTED YIELD: {y['final_yield']} {y['unit']}"
        )

    # Irrigation
    i = state.get("irrigation_result")
    if i:
        parts.append(
            f"IRRIGATION: {i['decision']}\n"
            f"Duration: {i['duration']} | "
            f"Method: {i['method']}"
        )

    # Weather
    w = state.get("weather_data")
    if w:
        parts.append(
            f"WEATHER: {w['temperature']}°C, "
            f"Humidity: {w['humidity']}%\n"
            f"Advisory: {w['farming_advice']}"
        )

    # RAG Knowledge
    r = state.get("rag_result")
    if r:
        parts.append(f"KNOWLEDGE BASE: {r['answer']}")

    state["final_answer"] = "\n\n".join(parts) if parts else \
        "I could not find relevant information. Please try again."

    return state


# ── Build LangGraph ───────────────────────────────
workflow = StateGraph(FarmState)

# Add all nodes
workflow.add_node("weather",    weather_node)
workflow.add_node("disease",    disease_node)
workflow.add_node("crop_rec",   crop_rec_node)
workflow.add_node("yield_pred", yield_node)
workflow.add_node("irrigation", irrigation_node)
workflow.add_node("rag",        rag_node)
workflow.add_node("combine",    combine_node)

# Define flow (edges)
workflow.set_entry_point("weather")
workflow.add_edge("weather",    "disease")
workflow.add_edge("disease",    "crop_rec")
workflow.add_edge("crop_rec",   "yield_pred")
workflow.add_edge("yield_pred", "irrigation")
workflow.add_edge("irrigation", "rag")
workflow.add_edge("rag",        "combine")
workflow.add_edge("combine",    END)

# Compile
graph = workflow.compile()


# ── Main Entry Function ───────────────────────────
def run_agent(text: str,
               image_path: str  = None,
               city: str        = "Pune",
               soil_data: dict  = None,
               language: str    = "en") -> str:
    """
    सगळे modules एकत्र connect करून answer देतो

    Input:
        text       → farmer's question (English)
        image_path → crop photo path (optional)
        city       → farmer's city for weather
        soil_data  → dict with N,P,K,ph,moisture,stage
        language   → detected language code

    Output:
        str → combined answer in English
              (caller will translate to farmer's language)
    """
    initial_state: FarmState = {
        "original_text":     text,
        "language":          language,
        "image_path":        image_path,
        "city":              city,
        "soil_data":         soil_data or {},
        "weather_data":      None,
        "disease_result":    None,
        "crop_rec_result":   None,
        "yield_result":      None,
        "irrigation_result": None,
        "rag_result":        None,
        "final_answer":      None
    }

    final_state = graph.invoke(initial_state)
    return final_state["final_answer"]


# ── Quick Test ────────────────────────────────────
if __name__ == "__main__":
    print("Testing Master Agent...\n")
    result = run_agent(
        text="My tomato leaves have brown spots. What disease is this?",
        city="Pune"
    )
    print("MASTER AGENT RESULT:")
    print(result)
