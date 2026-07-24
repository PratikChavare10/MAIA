"""
app.py — Main Flask Application
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT TO ADD BEFORE RUNNING:
1. .env file बनवा (.env copy करा)
2. सगळे models train करा
3. RAG ingest करा: python modules/rag/ingest.py
4. MySQL database + tables बनवा

HOW TO RUN:
   python app.py
   → Open: http://localhost:5000
"""

from flask import Flask, request, jsonify, render_template
import pymysql
import os

from config import (MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD,
                     MYSQL_DATABASE, DEBUG, PORT, HOST,
                     check_config)
from agents.master_agent        import run_agent
from modules.multilingual.voice import voice_to_text
from modules.multilingual.translator import (
    detect_language,
    translate_to_english,
    translate_to_original
)
from modules.memory.memory import save_to_memory

# ── Flask App ─────────────────────────────────────
app = Flask(__name__)

# ── Check Config on Start ─────────────────────────
check_config()

# ── MySQL Connection ──────────────────────────────
def get_db():
    """MySQL connection return करतो"""
    try:
        return pymysql.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE,
            charset='utf8mb4'
        )
    except Exception as e:
        print(f"⚠️  MySQL connection failed: {e}")
        return None

# ── Create Tables (on first run) ──────────────────
def init_db():
    """Database tables बनवतो"""
    db = get_db()
    if not db:
        print("⚠️  Skipping DB init — MySQL not connected")
        return
    try:
        cursor = db.cursor()
        # Farmers table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS farmers (
                id         INT AUTO_INCREMENT PRIMARY KEY,
                name       VARCHAR(100),
                city       VARCHAR(100),
                language   VARCHAR(10) DEFAULT 'mr',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Chat history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id           INT AUTO_INCREMENT PRIMARY KEY,
                user_message TEXT,
                ai_response  TEXT,
                language     VARCHAR(10),
                city         VARCHAR(100),
                created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        db.commit()
        print("✅ MySQL tables ready!")
    except Exception as e:
        print(f"⚠️  DB init error: {e}")
    finally:
        db.close()

# ── Save chat to MySQL ────────────────────────────
def save_chat(user_msg, ai_response, lang, city):
    db = get_db()
    if not db:
        return
    try:
        cursor = db.cursor()
        cursor.execute(
            "INSERT INTO chat_history "
            "(user_message, ai_response, language, city) "
            "VALUES (%s, %s, %s, %s)",
            (user_msg, ai_response, lang, city)
        )
        db.commit()
    except Exception as e:
        print(f"⚠️  Save chat error: {e}")
    finally:
        db.close()


# ════════════════════════════════════════════════
# ROUTES
# ════════════════════════════════════════════════

@app.route("/")
def home():
    """Home page (index.html)"""
    return render_template("index.html")


@app.route("/chat")
def chat():
    """Chat page (chat.html)"""
    return render_template("chat.html")


@app.route("/api/query", methods=["POST"])
def query():
    """
    Main API — सगळे modules येथे connect होतात

    Form Data:
        text  → farmer's question (text or empty)
        city  → farmer's city
        image → crop photo file (optional)
        audio → voice file (optional)
    """
    text  = request.form.get("text", "")
    city  = request.form.get("city", "Pune")
    image = request.files.get("image")
    audio = request.files.get("audio")

    # ── Step 1: Voice → Text ──────────────────────
    if audio:
        audio_path = "temp_audio.wav"
        audio.save(audio_path)
        text = voice_to_text(audio_path)
        os.remove(audio_path)   # cleanup

    # ── Step 2: Language Detection ─────────────────
    lang = detect_language(text) if text.strip() else "en"

    # ── Step 3: Translate to English ───────────────
    english_text = translate_to_english(text) if text.strip() else ""

    # ── Step 4: Save Image ─────────────────────────
    image_path = None
    if image and image.filename:
        image_path = "temp_image.jpg"
        image.save(image_path)

    # ── Step 5: Run Master Agent ───────────────────
    # ADD: soil_data form मधून घेणे
    # (आत्ता default values वापरत आहे)
    soil_data = {
        "N":          float(request.form.get("N",          90)),
        "P":          float(request.form.get("P",          42)),
        "K":          float(request.form.get("K",          43)),
        "ph":         float(request.form.get("ph",         6.5)),
        "moisture":   float(request.form.get("moisture",   45)),
        "stage":      request.form.get("stage",  "flowering"),
        "crop":       request.form.get("crop",   "Cotton"),
        "soil_type":  request.form.get("soil_type", "Black"),
        "area":       float(request.form.get("area",       2)),
        "fertilizer": float(request.form.get("fertilizer", 50)),
        "rainfall":   float(request.form.get("rainfall",  650)),
        "last_4_weeks": [
            [650, 32, 70],
            [600, 33, 68],
            [700, 31, 75],
            [620, 32, 72]
        ]
    }

    answer_english = run_agent(
        text=english_text,
        image_path=image_path,
        city=city,
        soil_data=soil_data,
        language=lang
    )

    # ── Step 6: Cleanup temp image ─────────────────
    if image_path and os.path.exists(image_path):
        os.remove(image_path)

    # ── Step 7: Translate Answer Back ─────────────
    final_answer = translate_to_original(answer_english, lang)

    # ── Step 8: Save to Memory + MySQL ────────────
    save_to_memory(text, final_answer)
    save_chat(text, final_answer, lang, city)

    return jsonify({
        "answer":   final_answer,
        "language": lang
    })


# ── Run App ───────────────────────────────────────
if __name__ == "__main__":
    init_db()
    print(f"\n🌾 MAIA Starting on http://{HOST}:{PORT}\n")
    app.run(host=HOST, port=PORT, debug=DEBUG)
