
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# ── LLM ──────────────────────────────────────────
# Get from: https://platform.openai.com/api-keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
# ── Weather API ───────────────────────────────────
# Get from: https://openweathermap.org → My API Keys
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

# ── Model File Paths ──────────────────────────────
# ADD: After training each model, path automatically works
DISEASE_MODEL_PATH  = "models/plant_vit_model.pth"
CROP_REC_MODEL_PATH = "models/crop_rec_model.keras"
CROP_REC_SC_PATH    = "models/crop_rec_sc.pkl"
CROP_REC_LE_PATH    = "models/crop_rec_le.pkl"

YIELD_VR_PATH       = "models/yield_rf.pkl"
YIELD_LSTM_PATH     = "models/yield_lstm.keras"
YIELD_SCALER_PATH   = "models/lstm_scaler.pkl"
LE_CROP_PATH        = "models/le_crop.pkl"
LE_SOIL_PATH        = "models/le_soil.pkl"
LE_WEATHER_PATH     = "models/le_weather.pkl"
LE_TRANS_PATH       = "models/le_trans.pkl"

# ── Data Paths ────────────────────────────────────
VECTORSTORE_PATH     = "data/vectorstore"
DOCUMENTS_PATH       = "../../data/documents"
PLANTVILLAGE_PATH    = "../../data/raw/plantvillage"
YIELD_DATASET_PATH   = "../../data/yield_dataset/crop_yield.csv"
WEEKLY_DATASET_PATH  = "data/yield_dataset/weekly_weather_yield.csv"
CROP_REC_DATASET     = "../../data/crop_recommendation/Crop_recommendation.csv"

# Upload
UPLOAD_FOLDER      = "uploads"
APP_DB_PATH   = os.getenv("APP_DB_PATH", "app.db")
# ── FastAPI ──
SECRET_KEY         = os.getenv("SECRET_KEY", "maia_super_secret_key_2025")
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7   # 7 days
FASTAPI_BASE_URL   = os.getenv("FASTAPI_BASE_URL", "http://127.0.0.1:8000")
JWT_ALGORITHM                = "HS256"
# ── App Settings ──────────────────────────────────

# SECRET_KEY  = os.getenv("SECRET_KEY", "maia_secret_2025_key")
HOST  = os.getenv("HOST", "0.0.0.0")
PORT  = int(os.getenv("PORT", 8000))
DEBUG = os.getenv("DEBUG", "true").lower() == "true"

# ── Validate Keys on Startup ──────────────────────

