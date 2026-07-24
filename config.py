"""
config.py — Central Configuration File for MAIA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO ADD YOUR KEYS:
1. Copy .env → .env
2. Fill in your API keys in .env
3. This file automatically loads them

WHAT TO ADD HERE:
- New API keys → add os.getenv("YOUR_KEY")
- New model paths → add a new PATH variable
- New settings → add at the bottom
"""

import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# ── LLM ──────────────────────────────────────────
# Get from: https://platform.openai.com/api-keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ── Weather API ───────────────────────────────────
# Get from: https://openweathermap.org → My API Keys
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
print(WEATHER_API_KEY)
# ── MySQL Database ────────────────────────────────
MYSQL_HOST     = os.getenv("MYSQL_HOST",     "localhost")
MYSQL_USER     = os.getenv("MYSQL_USER",     "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "maia_db")

# ── Model File Paths ──────────────────────────────
# ADD: After training each model, path automatically works
DISEASE_MODEL_PATH  = "../../models/disease_vit.keras"
CROP_REC_MODEL_PATH = "../../models/crop_rec_model.keras"
CROP_REC_SC_PATH    = "../../models/crop_rec_sc.pkl"
CROP_REC_LE_PATH    = "../../models/crop_rec_le.pkl"
YIELD_RF_PATH       = "models/yield_rf.pkl"
YIELD_XGB_PATH      = "models/yield_xgb.pkl"
YIELD_LSTM_PATH     = "models/yield_lstm.keras"
YIELD_SCALER_PATH   = "models/lstm_scaler.pkl"
LE_CROP_PATH        = "models/le_crop.pkl"
LE_SOIL_PATH        = "../../models/le_soil.pkl"
LE_WEATHER_PATH     = "../../models/le_weather.pkl"

# ── Data Paths ────────────────────────────────────
VECTORSTORE_PATH     = "data/vectorstore"
DOCUMENTS_PATH       = "data/documents"
PLANTVILLAGE_PATH    = "../../data/raw/plantvillage"
YIELD_DATASET_PATH   = "data/yield_dataset/crop_yield_data.csv"
WEEKLY_DATASET_PATH  = "data/yield_dataset/weekly_weather_yield.csv"
CROP_REC_DATASET     = "../../data/crop_recommendation/Crop_recommendation.csv"

# ── App Settings ──────────────────────────────────
DEBUG = True
PORT  = 5000
HOST  = "0.0.0.0"

# ── Validate Keys on Startup ──────────────────────
def check_config():
    missing = []
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if not WEATHER_API_KEY:
        missing.append("WEATHER_API_KEY")
    if not MYSQL_PASSWORD:
        missing.append("MYSQL_PASSWORD")
    if missing:
        print(f"⚠️  Missing in .env: {', '.join(missing)}")
    else:
        print("✅ All config keys loaded!")
