

import os
import uuid
import json
from typing import Optional

from fastapi import FastAPI, Request, Form, UploadFile, File, Header, HTTPException, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from config import UPLOAD_FOLDER, HOST, PORT, DEBUG
from auth import create_access_token, decode_access_token
from db import (register_user, login_user, create_thread, get_threads,
                get_messages, save_message, update_thread_title, delete_thread)
from workfolw import stream_workflow
from modules.multilingual.voice       import voice_to_text
from modules.multilingual.translator  import detect_language, translate_to_english
from modules.weather.fetch            import get_weather

app = FastAPI(title="MAIA API")

# Streamlit runs on a different port — allow it to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten this to your Streamlit URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODE_LABELS = {
    "disease": "Disease Detection",
    "crop":    "Crop Recommendation",
    "yield":   "Yield Prediction",
}


# ────────────────────────────────────
# AUTH DEPENDENCY (JWT bearer token)
# ────────────────────────────────────
async def get_current_user(authorization: Optional[str] = Header(None)) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token   = authorization.split(" ", 1)[1]
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return payload   # {"user_id", "name", "city", "language", "exp"}


# ────────────────────────────────────
# AUTH ROUTES
# ────────────────────────────────────
@app.post("/api/auth/register")
async def api_register(
    name: str = Form(...), email: str = Form(...), password: str = Form(...),
    city: str = Form("Pune"), language: str = Form("English"),
):
    if len(password) < 6:
        return JSONResponse({"ok": False, "msg": "Password must be at least 6 characters."}, status_code=400)
    result = register_user(name.strip(), email.strip(), password.strip(), city.strip(), language)
    return JSONResponse(result)


@app.post("/api/auth/login")
async def api_login(email: str = Form(...), password: str = Form(...)):
    result = login_user(email.strip(), password.strip())
    if not result["ok"]:
        return JSONResponse({"ok": False, "msg": "Invalid email or password."}, status_code=401)

    user  = result["user"]
    token = create_access_token({
        "user_id":  user["id"],
        "name":     user["name"],
        "city":     user["city"],
        "language": user["language"],
    })
    return JSONResponse({
        "ok":    True,
        "token": token,
        "user":  {"id": user["id"], "name": user["name"], "email": user["email"],
                  "city": user["city"], "language": user["language"]},
    })


# ────────────────────────────────────
# THREADS API (sidebar)
# ────────────────────────────────────
@app.get("/api/threads")
async def api_threads(user: dict = Depends(get_current_user)):
    threads = get_threads(user["user_id"])
    result = []
    for t in threads:
        preview = (t.get("first_msg") or t.get("title") or "New Chat")[:60]
        result.append({
            "id":         t["id"],
            "title":      t["title"],
            "mode":       t["mode"],
            "mode_label": MODE_LABELS.get(t["mode"], "Chat"),
            "preview":    preview,
            "updated":    str(t.get("updated_at", ""))[:16],
        })
    return JSONResponse({"threads": result})


@app.get("/api/threads/{thread_id}")
async def api_thread(thread_id: int, user: dict = Depends(get_current_user)):
    msgs = get_messages(thread_id)
    return JSONResponse({"messages": [
        {"role": m["role"], "content": m["content"], "mode": m["mode"]} for m in msgs
    ]})


@app.post("/api/threads/new")
async def api_new_thread(request: Request, user: dict = Depends(get_current_user)):
    body = await request.json()
    mode = body.get("mode", "disease")
    tid  = create_thread(user["user_id"], mode=mode, title="New Chat")
    return JSONResponse({"thread_id": tid, "mode": mode})


@app.delete("/api/threads/{thread_id}")
async def api_delete_thread(thread_id: int, user: dict = Depends(get_current_user)):
    delete_thread(thread_id, user["user_id"])
    return JSONResponse({"ok": True})


# ────────────────────────────────────
# WEATHER API
# ────────────────────────────────────
@app.get("/api/weather")
async def api_weather(city: str = "Pune", user: dict = Depends(get_current_user)):
    try:
        return JSONResponse(get_weather(city))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ────────────────────────────────────
# CHAT — TEXT or AUDIO (mic) → SSE stream
# ────────────────────────────────────
@app.post("/api/chat/stream")
async def api_chat_stream(
    mode:       str            = Form(...),
    text:       str            = Form(""),
    city:       str            = Form("Pune"),
    thread_id:  Optional[str]  = Form(None),
    image:      Optional[UploadFile] = File(None),
    audio:      Optional[UploadFile] = File(None),
    soil_N:             Optional[str] = Form(None),
    soil_P:             Optional[str] = Form(None),
    soil_K:             Optional[str] = Form(None),
    soil_ph:            Optional[str] = Form(None),
    soil_humidity:      Optional[str] = Form(None),
    soil_rainfall:      Optional[str] = Form(None),
    yield_crop:         Optional[str] = Form(None),
    yield_soil:         Optional[str] = Form(None),
    yield_area:         Optional[str] = Form(None),
    yield_region: Optional[str] = Form(None),
    yield_weather: Optional[str] = Form(None),
    yield_rainfall: Optional[str] = Form(None),
    yield_fertilizer_used: Optional[str] = Form(None),
    yield_irrigation_used: Optional[str] = Form(None),
    yield_days_to_harvest: Optional[str] = Form(None),
    user: dict = Depends(get_current_user)
):
    """
    Flow:
      1. If `audio` given (mic) -> voice_to_text() gives (text, language).
      2. Else -> detect_language(text) on the typed text.
      3. If language != English -> translate_to_english() gives the query
         used to build the prediction prompt; `language` is passed through
         so chat_node instructs the LLM to answer back in that language.
      4. Save user message (raw text) to db.py, stream the workflow reply,
         save the AI reply (already in the target language) once done.
    """
    user_id = user["user_id"]
    text    = text.strip()

    soil_data = None
    if mode == "crop":
        try:
            soil_data = {
                "N":        float(soil_N or 90),
                "P":        float(soil_P or 42),
                "K":        float(soil_K or 43),
                "ph":       float(soil_ph or 6.5),
                "humidity": float(soil_humidity or 70),
                "rainfall": float(soil_rainfall or 200),
            }
        except Exception:
            pass

    yield_data = None
    if mode == "yield":
        try:
            yield_data = {
                "crop": yield_crop or "Cotton",
                "soil_type": yield_soil or "Black",
                "area": float(yield_area or 2),
                "fertilizer": yield_fertilizer_used,
                "region": str(yield_region or "North"),  # FIX: String ठेवायचे आहे, float नाही!
                "weather_condition": str(yield_weather or "Sunny"),
                "irrigation_used": yield_irrigation_used,
                "rainfall": float(yield_rainfall or 650.0),
                "Days_to_Harvest": int(yield_days_to_harvest or 120)
            }
        except Exception:
            pass

    # Image save (disease mode)
    image_path = None
    if image is not None:
        image_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}.jpg")
        with open(image_path, "wb") as f:
            f.write(await image.read())

    # ── Speech-to-text (mic) or typed-text language detection ──
    if audio is not None:
        audio_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}.webm")
        with open(audio_path, "wb") as f:
            f.write(await audio.read())
        try:
            text, language = voice_to_text(audio_path)
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)
    else:
        language = detect_language(text) if text else "English"

    if not text and not image_path:
        return JSONResponse({"error": "No input provided"}, status_code=400)

    # English query drives the prediction/prompt-building nodes; the LLM is
    # separately told (via `language`) to answer back in the user's language.
    if language and language.lower() not in ("en", "english"):
        user_query = translate_to_english(text) if text else text
    else:
        user_query = text

    print(language)

    # Thread bookkeeping (UI-facing db.py, separate from LangGraph's checkpoint)
    if not thread_id or thread_id == "null":
        thread_id = create_thread(user_id, mode=mode, title=(text[:60] if text else mode.title()))
    else:
        thread_id = int(thread_id)

    save_message(thread_id, user_id, "user", text or "Image uploaded", mode)

    async def generate():
        full_response = ""
        try:
            async for token in stream_workflow(
                mode       = mode,
                user_query = user_query,
                language   = language,
                city       = city,
                image_path = image_path,
                soil_data  = soil_data,
                yield_data = yield_data,
                thread_id  = thread_id,
            ):
                full_response += token
                yield f"data: {json.dumps({'token': token, 'thread_id': thread_id})}\n\n"

            yield f"data: {json.dumps({'done': True, 'thread_id': thread_id})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            if image_path and os.path.exists(image_path):
                os.remove(image_path)
            if full_response:
                save_message(thread_id, user_id, "assistant", full_response, mode)
                update_thread_title(thread_id, text[:60] or mode.title())

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=HOST, port=PORT, reload=DEBUG)