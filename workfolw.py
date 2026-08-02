

import os
import asyncio
from typing import TypedDict, Optional, Literal, Annotated

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from sympy.physics.units import days

os.environ["GROQ_API_KEY"] = ""
load_dotenv()

llm =  ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)


# ════════════════════════════
# STATE
# ════════════════════════════
class FarmerState(TypedDict):
    mode:              str
    user_query:        str
    city:              str
    image_path:        Optional[str]
    soil_data:         Optional[dict]
    yield_data:        Optional[dict]
    weather_data:      Optional[dict]
    disease_result:    Optional[dict]
    crop_rec_result:   Optional[dict]
    yield_result:      Optional[dict]
    irrigation_result: Optional[dict]
    rag_context:       Optional[str]
    llm_prompt:        Optional[str]
    messages:          Annotated[list[BaseMessage], add_messages]
    is_followup:       bool
    language:          str


# ════════════════════════════
# NODE — WEATHER (shared)
# ════════════════════════════
def weather_node(state: FarmerState) -> FarmerState:
    try:
        from modules.weather.fetch import get_weather
        state["weather_data"] = get_weather(state.get("city", "Pune"))
    except Exception:
        state["weather_data"] = {
            "temperature": "N/A", "humidity": "N/A",
            "rain_forecast": "N/A", "farming_advice": "Weather unavailable"
        }
    return state


# ════════════════════════════
# ENTRY ROUTER — first message in a thread vs a follow-up question
# ════════════════════════════
def route_entry(state: FarmerState) -> Literal["first_turn", "followup"]:
    return "followup" if state.get("is_followup") else "first_turn"


def followup_node(state: FarmerState) -> FarmerState:
    """
    Follow-up question in an existing thread — skip weather/prediction/prompt
    nodes entirely. Just append the new question; chat_node already has the
    full prior conversation (weather, prediction result, earlier Q&A) in
    `messages` from the checkpoint, so the LLM answers with that context.
    """
    query = state.get("user_query", "")
    state["messages"] = [HumanMessage(content=query)]
    return state


# ════════════════════════════
# ROUTER — decide which branch (first turn only)
# ════════════════════════════
def route_by_mode(state: FarmerState) -> Literal["disease", "crop", "yield"]:
    return state.get("mode", "disease")


# ════════════════════════════
# DISEASE BRANCH
# ════════════════════════════
def disease_detect_node(state: FarmerState) -> FarmerState:
    img = state.get("image_path")
    if img and os.path.exists(img):
        try:
            from modules.disease.predict import predict_disease
            state["disease_result"] = predict_disease(img)
        except Exception as e:
            state["disease_result"] = {"disease": "Unknown", "confidence": "N/A",
                                        "treatment": "Consult local expert", "error": str(e)}
    else:
        state["disease_result"] = {"disease": "No image provided",
                                    "confidence": "N/A", "treatment": "Please upload a crop photo"}
    return state


def disease_rag_node(state: FarmerState) -> FarmerState:
    disease_name = (state.get("disease_result") or {}).get("disease", "")
    query = f"{disease_name} plant disease treatment prevention management. {state.get('user_query','')}"
    try:
        from modules.rag.retriever import rag_search
        result = rag_search(query)
        state["rag_context"] = result.get("answer", "")
    except Exception as e:
        state["rag_context"] = f"RAG unavailable: {e}"
    return state


def disease_prompt_node(state: FarmerState) -> FarmerState:
    d   = state.get("disease_result") or {}
    w   = state.get("weather_data")   or {}
    rag = state.get("rag_context", "") or ""

    prompt = f"""You are MAIA, an expert AI assistant for Indian farmers. Respond clearly and practically.

=== DISEASE DETECTION RESULT ===
Disease: {d.get('disease','Unknown')}
Confidence: {d.get('confidence','N/A')}
Immediate Treatment: {d.get('treatment','N/A')}

=== KNOWLEDGE FROM AGRICULTURAL DOCUMENTS ===
{rag if rag else "No additional documents found."}

=== CURRENT WEATHER at {state.get('city','your location')} ===
Temperature: {w.get('temperature','N/A')}°C | Humidity: {w.get('humidity','N/A')}% | Rain: {w.get('rain_forecast','N/A')}
Advisory: {w.get('farming_advice','')}

=== FARMER'S QUESTION ===
{state.get('user_query','')}

=== YOUR TASK ===
1. Confirm disease name and what it means for the crop
2. Give step-by-step treatment with specific dosages
3. Consider weather — if rain is expected, advise not to spray
4. Give 3 preventive steps for future
5. Mention any government support (PMFBY crop insurance if loss is severe)
6. End with one key action the farmer should take TODAY

Be specific, practical, and farmer-friendly:
give me the answer in {state.get("language","English")}
"""

    state["llm_prompt"] = prompt
    state["messages"] = [HumanMessage(content=prompt)]
    return state


# ════════════════════════════
# CROP BRANCH
# ════════════════════════════
def crop_rec_node(state: FarmerState) -> FarmerState:
    soil = state.get("soil_data") or {}
    w    = state.get("weather_data") or {}
    try:
        from modules.crop_recommendation.predict import recommend_crop
        state["crop_rec_result"] = recommend_crop(
            N=float(soil.get("N", 90)),
            P=float(soil.get("P", 42)),
            K=float(soil.get("K", 43)),
            temperature=float(w.get("temperature", 25)),
            humidity=float(soil.get("humidity", 70)),
            ph=float(soil.get("ph", 6.5)),
            rainfall=float(soil.get("rainfall", 200))
        )
    except Exception as e:
        state["crop_rec_result"] = {"recommended_crop": "Unknown", "top_3": [], "error": str(e)}
    return state


def crop_prompt_node(state: FarmerState) -> FarmerState:
    cr   = state.get("crop_rec_result") or {}
    soil = state.get("soil_data") or {}
    w    = state.get("weather_data") or {}

    top3_str = ", ".join([
        f"{c['crop']} ({c['confidence']})"
        for c in cr.get("top_3", [])
    ]) or "N/A"

    prompt = f"""You are MAIA, an expert AI assistant for Indian farmers. Respond clearly and practically.

=== SOIL ANALYSIS ===
N={soil.get('N','N/A')} | P={soil.get('P','N/A')} | K={soil.get('K','N/A')} | pH={soil.get('ph','N/A')}
Humidity={soil.get('humidity','N/A')}% | Rainfall={soil.get('rainfall','N/A')}mm

=== AI CROP RECOMMENDATION ===
Best Crop: {cr.get('recommended_crop','Unknown')}
Top 3 Options: {top3_str}

=== CURRENT WEATHER at {state.get('city','your location')} ===
Temperature: {w.get('temperature','N/A')}°C | Humidity: {w.get('humidity','N/A')}% | Rain: {w.get('rain_forecast','N/A')}
Advisory: {w.get('farming_advice','')}

=== FARMER'S QUESTION ===
{state.get('user_query','')}

=== YOUR TASK ===
1. Explain why this crop is best for the given soil
2. Give expected yield range (quintal/acre)
3. Best sowing season for this crop
4. Top 3 things farmer should prepare before sowing
5. Basic fertilizer schedule (N-P-K timing)
6. Weather suitability for this crop right now
7. Estimated profit potential

Be encouraging, specific, and actionable:

give me the answer in {state.get("language","English")}

"""


    state["llm_prompt"] = prompt
    state["messages"] = [HumanMessage(content=prompt)]
    return state


# ════════════════════════════
# YIELD BRANCH
# ════════════════════════════
def yield_predict_node(state: FarmerState) -> FarmerState:
    yd = state.get("yield_data") or {}
    w  = state.get("weather_data") or {}
    try:
        from modules.yield_prediction.predict import predict_yield
        state["yield_result"] = predict_yield(
            crop=yd.get("crop", "Cotton"),
            soil_type=yd.get("soil_type", "Black"),
            rainfall=float(yd.get("rainfall", w.get("rainfall", 650))),
            temperature=float(w.get("temperature", 32)),
            area=float(yd.get("area", 1.0)),
            fertilizer=bool(yd.get("fertilizer", True)),
            region=str(yd.get("region", "North")),
            weather_condition=str(yd.get("weather_condition", "Sunny")),
            irrigation_used=bool(yd.get("irrigation_used", True)),
            days_to_harvest=int(yd.get("Days_to_Harvest", 120))
        )
        print('x')
    except Exception as e:
        print('y')
        state["yield_result"] = {"final_yield": "N/A", "rf_xgb_yield": "N/A",
                                  "lstm_yield": "N/A", "unit": "quintal/acre", "error": str(e)}

    # try:
    #     from modules.irrigation.calculator import calculate_irrigation
    #     state["irrigation_result"] = calculate_irrigation(
    #         weather=w, soil_moisture=45, crop_stage="flowering"
    #     )
    # except Exception:
    #     state["irrigation_result"] = {"decision": "Consult local expert", "duration": "N/A"}
    return state


def yield_prompt_node(state: FarmerState) -> FarmerState:
    yr = state.get("yield_result")      or {}
    ir = state.get("irrigation_result") or {}
    w  = state.get("weather_data")      or {}
    yd = state.get("yield_data")        or {}

    prompt = f"""You are MAIA, an expert AI assistant for Indian farmers. Respond clearly and practically.

=== YIELD PREDICTION RESULT ===
Crop: {yd.get('crop','N/A')} | Area: {yd.get('area','N/A')} acres | Soil: {yd.get('soil_type','N/A')}
RF+XGB Model Prediction: {yr.get('yield_per_hectare','N/A')} quintal/acre
For {yd.get("area")}: {yr.get("final_yield")} is expected.

FINAL ENSEMBLE YIELD:    {yr.get('final_yield','N/A')} quintal/acre

=== IRRIGATION ADVISORY ===
Decision: {ir.get('decision','N/A')}
Duration: {ir.get('duration','N/A')}
Water Need: {ir.get('water_need','N/A')}

=== CURRENT WEATHER at {state.get('city','your location')} ===
Temperature: {w.get('temperature','N/A')}°C | Humidity: {w.get('humidity','N/A')}% | Rain: {w.get('rain_forecast','N/A')}
Advisory: {w.get('farming_advice','')}

=== FARMER'S QUESTION ===
{state.get('user_query','')}

=== YOUR TASK ===
1. Tell the farmer their expected yield and if it's good/average/below average compared to district average
2. Give 5 SPECIFIC steps to INCREASE yield (with quantities and timing)
3. Irrigation advice based on current weather
4. Best fertilizer to apply NOW for higher yield
5. Weather risks that could reduce yield and how to protect
6. If yield seems low — explain 3 possible reasons
7. Is PMFBY crop insurance recommended?

Be specific with numbers and dates. Make the farmer feel confident:

give me the answer in {state.get("language","English")}
"""

    state["llm_prompt"] = prompt
    state["messages"] = [HumanMessage(content=prompt)]
    return state


# ════════════════════════════
# CHAT NODE — calls the LLM (shared by all branches)
# ════════════════════════════
def chat_node(state: FarmerState) -> FarmerState:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# ════════════════════════════
# BUILD GRAPH
# ════════════════════════════
def build_graph() -> StateGraph:
    g = StateGraph(FarmerState)

    g.add_node("weather",        weather_node)
    g.add_node("disease_detect", disease_detect_node)
    g.add_node("disease_rag",    disease_rag_node)
    g.add_node("disease_prompt", disease_prompt_node)
    g.add_node("crop_rec",       crop_rec_node)
    g.add_node("crop_prompt",    crop_prompt_node)
    g.add_node("yield_predict",  yield_predict_node)
    g.add_node("yield_prompt",   yield_prompt_node)
    g.add_node("followup_node",  followup_node)
    g.add_node("chat_node",      chat_node)

    # First turn in a thread -> full pipeline. Follow-up question -> chat_node directly.
    g.add_conditional_edges(
        START,
        route_entry,
        {
            "first_turn": "weather",
            "followup":   "followup_node",
        }
    )

    g.add_edge("followup_node", "chat_node")

    g.add_conditional_edges(
        "weather",
        route_by_mode,
        {
            "disease": "disease_detect",
            "crop":    "crop_rec",
            "yield":   "yield_predict",
        }
    )

    # Disease branch
    g.add_edge("disease_detect", "disease_rag")
    g.add_edge("disease_rag",    "disease_prompt")
    g.add_edge("disease_prompt", "chat_node")

    # Crop branch
    g.add_edge("crop_rec",    "crop_prompt")
    g.add_edge("crop_prompt", "chat_node")

    # Yield branch
    g.add_edge("yield_predict", "yield_prompt")
    g.add_edge("yield_prompt",  "chat_node")

    # All branches converge on chat_node -> END
    g.add_edge("chat_node", END)

    return g


# ════════════════════════════
# ASYNC SQLITE CHECKPOINTER (replaces MySQL save_checkpoint/load_checkpoint)
# ════════════════════════════
DB_PATH = "MAIA.db"

_graph_builder = build_graph()

_async_saver_cm  = None
_farmer_bot      = None
_init_lock       = asyncio.Lock()


async def _get_bot():
    """Compile the graph with AsyncSqliteSaver exactly once, reuse afterwards."""
    global _async_saver_cm, _farmer_bot
    if _farmer_bot is not None:
        return _farmer_bot

    async with _init_lock:
        if _farmer_bot is None:          # re-check inside the lock
            _async_saver_cm = AsyncSqliteSaver.from_conn_string(DB_PATH)
            async_checkpointer = await _async_saver_cm.__aenter__()
            _farmer_bot = _graph_builder.compile(checkpointer=async_checkpointer)
            print("✅ LangGraph FarmerState workflow compiled (AsyncSqliteSaver, once)!")
    return _farmer_bot



# ════════════════════════════
# HELPERS
# ════════════════════════════
async def _is_followup(thread_id) -> bool:
    """True if this thread already has prior conversation checkpointed."""
    if not thread_id:
        return False
    bot    = await _get_bot()
    config = {"configurable": {"thread_id": str(thread_id)}}
    snapshot = await bot.aget_state(config)
    return bool(snapshot and snapshot.values.get("messages"))


def _initial_state(mode,language, user_query, city, image_path, soil_data, yield_data,
                    is_followup: bool) -> FarmerState:
    state: FarmerState = {
        "mode":        mode,
        "user_query":  user_query,
        "city":        city,
        "image_path":  image_path,
        "soil_data":   soil_data or {},
        "yield_data":  yield_data or {},
        "messages":    [],
        "is_followup": is_followup,
        "language"    : language
    }
    # Only reset prediction/weather fields on a genuine first turn — on a
    # follow-up we leave them untouched so the previously checkpointed
    # values stay intact (LangGraph only overwrites keys present in input).
    if not is_followup:
        state.update({
            "weather_data":      None,
            "disease_result":    None,
            "crop_rec_result":   None,
            "yield_result":      None,
            "irrigation_result": None,
            "rag_context":       None,
            "llm_prompt":        None,
        })
    return state


async def stream_workflow(mode: str,user_query: str, language:str="English",city: str = "Pune",
                           image_path: str = None, soil_data: dict = None,
                           yield_data: dict = None, thread_id: str = None):
    """
    Async generator — yields the AI reply token by token as chat_node runs.
    Used by the FastAPI /api/stream SSE endpoint, and by test scripts.

    First message in a thread -> full pipeline (weather/prediction/prompt).
    Later messages in the same thread -> straight to chat_node, using the
    checkpointed conversation for context. State is auto-persisted to
    MAIA.db under thread_id via AsyncSqliteSaver.

    The graph is compiled ONCE (see _get_bot()) and reused on every call —
    no recompiling per request.
    """
    bot      = await _get_bot()
    followup = await _is_followup(thread_id)
    state    = _initial_state(mode,language, user_query, city, image_path, soil_data, yield_data, followup)
    config   = {"configurable": {"thread_id": str(thread_id or "default")}}

    async for msg_chunk, metadata in bot.astream(state, config=config, stream_mode="messages"):
        if metadata.get("langgraph_node") == "chat_node" and getattr(msg_chunk, "content", None):
            yield msg_chunk.content


async def retrieve_all_threads():
    """All thread_ids that have a saved LangGraph checkpoint."""
    bot     = await _get_bot()
    threads = set()
    async for checkpoint in bot.checkpointer.alist(None):
        threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(threads)


async def retrieve_thread_messages(thread_id):
    """Full chat message history for a thread — used to reload conversation on login."""
    bot    = await _get_bot()
    config = {"configurable": {"thread_id": str(thread_id)}}
    state  = await bot.aget_state(config)
    return state.values.get("messages", []) if state and state.values else []