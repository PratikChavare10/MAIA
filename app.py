

import hashlib
import streamlit as st
from api_client import (register, login, get_threads, get_thread_messages,
                         new_thread, delete_thread, stream_chat)

st.set_page_config(page_title="MAIA — Farmer Assistant", page_icon="🌾", layout="wide")

MODE_LABELS = {
    "disease": "🩺 Disease Detection",
    "crop":    "🌱 Crop Recommendation",
    "yield":   "📊 Yield Prediction",
}

# ════════════════════════════
# SESSION STATE
# ════════════════════════════
defaults = {
    "token": None, "user": None, "page": "login",
    "thread_id": None, "mode": "disease", "messages": [],
    "last_audio_hash": None,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ════════════════════════════
# LOGIN / REGISTER
# ════════════════════════════
def login_page():
    st.title("🌾 MAIA — Login")
    with st.form("login_form"):
        email    = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

    if submitted:
        result = login(email, password)
        if result.get("ok"):
            st.session_state.token = result["token"]
            st.session_state.user  = result["user"]
            st.session_state.page  = "chat"
            st.rerun()
        else:
            st.error(result.get("msg", "Invalid email or password."))

    st.write("Don't have an account?")
    if st.button("Register instead"):
        st.session_state.page = "register"
        st.rerun()


def register_page():
    st.title("🌾 MAIA — Register")
    with st.form("register_form"):
        name     = st.text_input("Name")
        email    = st.text_input("Email")
        password = st.text_input("Password", type="password")
        city     = st.text_input("City", value="Pune")
        language = st.selectbox("Preferred language", ["English", "Marathi", "Hindi"])
        submitted = st.form_submit_button("Register")

    if submitted:
        result = register(name, email, password, city, language)
        if result.get("ok"):
            st.success("Registered! Please login.")
            st.session_state.page = "login"
            st.rerun()
        else:
            st.error(result.get("msg", "Registration failed."))

    if st.button("Back to login"):
        st.session_state.page = "login"
        st.rerun()


# ════════════════════════════
# SIDEBAR — mode picker + recent threads
# ════════════════════════════
def sidebar():
    with st.sidebar:
        st.subheader(f"👋 {st.session_state.user['name']}")

        mode_keys = list(MODE_LABELS.keys())
        st.session_state.mode = st.radio(
            "Mode", options=mode_keys, format_func=lambda m: MODE_LABELS[m],
            index=mode_keys.index(st.session_state.mode),
        )

        if st.button("➕ New Chat", use_container_width=True):
            st.session_state.thread_id = None
            st.session_state.messages  = []
            st.rerun()

        st.markdown("---")
        st.caption("Recent")

        threads = get_threads(st.session_state.token)
        for t in threads:
            cols = st.columns([5, 1])
            label = t["title"] or "New Chat"
            if cols[0].button(label, key=f"thread_{t['id']}", use_container_width=True):
                st.session_state.thread_id = t["id"]
                st.session_state.mode      = t["mode"]
                st.session_state.messages  = get_thread_messages(st.session_state.token, t["id"])
                st.rerun()
            if cols[1].button("🗑️", key=f"del_{t['id']}"):
                delete_thread(st.session_state.token, t["id"])
                if st.session_state.thread_id == t["id"]:
                    st.session_state.thread_id = None
                    st.session_state.messages  = []
                st.rerun()

        st.markdown("---")
        if st.button("Logout", use_container_width=True):
            for key in defaults:
                st.session_state[key] = defaults[key]
            st.rerun()


# ════════════════════════════
# MODE-SPECIFIC INPUT FIELDS
# ════════════════════════════
def mode_inputs():
    """Returns (extra_form_fields: dict, image_bytes: bytes|None)."""
    extra = {}
    image_bytes = None
    mode = st.session_state.mode

    if mode == "disease":
        uploaded = st.file_uploader("Upload a crop photo", type=["jpg", "jpeg", "png"])
        if uploaded:
            image_bytes = uploaded.read()

    elif mode == "crop":
        c1, c2, c3 = st.columns(3)
        extra["soil_N"] = c1.number_input("N", value=90.0)
        extra["soil_P"] = c2.number_input("P", value=42.0)
        extra["soil_K"] = c3.number_input("K", value=43.0)
        c4, c5, c6 = st.columns(3)
        extra["soil_ph"] = c4.number_input("pH", value=6.5)
        extra["soil_humidity"] = c5.number_input("Humidity %", value=70.0)
        extra["soil_rainfall"] = c6.number_input("Rainfall (mm)", value=200.0)

    elif mode == "yield":
        # Row 1: Crop Name & Land Area
        c1, c2 = st.columns(2)
        extra["yield_crop"] = c1.text_input("Crop", value="Cotton")
        extra["yield_area"] = c2.number_input("Area (acres)", value=1.0, min_value=0.1)

        # Row 2: Categorical Dropdowns (Region, Soil Type, Weather)
        c3, c4, c5 = st.columns(3)
        extra["yield_region"] = c3.selectbox(
            "Region",
            options=["North", "South", "East", "West"],
            index=0
        )
        extra["yield_soil"] = c4.selectbox(
            "Soil Type",
            options=["Sandy", "Peaty", "Clay", "Chalky", "Silt"],
            index=2  # Defaults to "Clay" (or change index as needed)
        )
        extra["yield_weather"] = c5.selectbox(
            "Weather Condition",
            options=["Sunny", "Rainy", "Cloudy"],
            index=0
        )

        # Row 3: Numerical Weather & Crop Inputs
        c6, c7, c8 = st.columns(3)
        extra["yield_rainfall"] = c6.number_input("Rainfall (mm)", value=650.0, min_value=0.0)
        extra["yield_days_to_harvest"] = c7.number_input("Days to Harvest", value=120, min_value=1)

        # Row 4: Fertilizer & Irrigation Details
        c8, c9, c10 = st.columns(3)
        extra["yield_fertilizer"] = c8.number_input("Fertilizer Amount (kg)", value=50.0, min_value=0.0)

        fertilizer_opt = c9.selectbox("Fertilizer Used?", options=["Yes", "No"], index=0)
        extra["yield_fertilizer_used"] = (fertilizer_opt == "Yes")

        irrigation_opt = c10.selectbox("Irrigation Used?", options=["Yes", "No"], index=0)
        extra["yield_irrigation_used"] = (irrigation_opt == "Yes")

    return extra, image_bytes


# ════════════════════════════
# SEND A QUERY (text or voice) AND STREAM THE REPLY
# ════════════════════════════
def send_query(extra: dict, image_bytes, text: str = None, audio_bytes: bytes = None):
    token = st.session_state.token

    form_data = {
        "mode":      st.session_state.mode,
        "text":      text or "",
        "city":      st.session_state.user.get("city", "Pune"),
        "thread_id": str(st.session_state.thread_id or ""),
        **{k: str(v) for k, v in extra.items()},
    }
    files = {}
    if image_bytes:
        files["image"] = ("photo.jpg", image_bytes, "image/jpeg")
    if audio_bytes:
        files["audio"] = ("voice.wav", audio_bytes, "audio/wav")
    if files:
        form_data["files"] = files

    display_text = text or ("🎤 Voice message" if audio_bytes else "Image uploaded")
    st.session_state.messages.append({"role": "user", "content": display_text})
    with st.chat_message("user"):
        st.write(display_text)

    result_holder = {}
    with st.chat_message("assistant"):
        full_reply = st.write_stream(stream_chat(token, result_holder=result_holder, **form_data))

    st.session_state.messages.append({"role": "assistant", "content": full_reply})

    # Keep our thread_id in sync with the backend's. Without this, thread_id
    # stayed None forever, so EVERY message looked like "start of a new
    # thread" to the backend (no real follow-up continuity), and the old
    # `if thread_id is None: st.rerun()` fired after every single message —
    # which combined with st.chat_input()'s submit-then-rerun timing caused
    # the same message to get reprocessed ("response keeps repeating").
    # No st.rerun() here at all now — the messages above already render in
    # this same script run, so one isn't needed. The sidebar's thread list
    # will pick up the new thread on the next natural rerun.
    backend_thread_id = result_holder.get("thread_id")
    if backend_thread_id is not None:
        st.session_state.thread_id = backend_thread_id


# ════════════════════════════
# MAIN CHAT PAGE
# ════════════════════════════
def chat_page():
    sidebar()
    st.title(f"🌾 MAIA — {MODE_LABELS[st.session_state.mode]}")

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    extra, image_bytes = mode_inputs()

    audio_value = st.audio_input("🎤 Or record your question")
    text_value  = st.chat_input("Type your question...")

    # st.audio_input keeps returning the SAME recording on every rerun until
    # the user re-records — it does NOT reset to None by itself. Without a
    # dedupe check here, that stale value keeps re-triggering send_query()
    # on every rerun (looks like it "runs continuously"), and because it's
    # checked first, it also blocks typed text from ever being processed.
    new_audio_bytes = None
    if audio_value is not None:
        audio_bytes = audio_value.getvalue()          # safe to call repeatedly, unlike .read()
        audio_hash  = hashlib.md5(audio_bytes).hexdigest()
        if audio_hash != st.session_state.get("last_audio_hash"):
            st.session_state.last_audio_hash = audio_hash
            new_audio_bytes = audio_bytes

    if new_audio_bytes is not None:
        send_query(extra, image_bytes, audio_bytes=new_audio_bytes)
    elif text_value:
        send_query(extra, image_bytes, text=text_value)


# ════════════════════════════
# ROUTER
# ════════════════════════════
if st.session_state.token is None:
    if st.session_state.page == "register":
        register_page()
    else:
        login_page()
else:
    chat_page()