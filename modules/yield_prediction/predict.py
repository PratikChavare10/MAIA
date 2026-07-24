"""
modules/yield_prediction/predict.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT TO ADD:
- train_rf_xgb.py + train_lstm.py run केल्यावर
  models automatically load होतात

HOW TO USE:
   from modules.yield_prediction.predict import predict_yield
   result = predict_yield(
       crop="Cotton", soil_type="Black",
       rainfall=650, temperature=32,
       area=2, fertilizer=50, humidity=70,
       last_4_weeks=[[650,32,70],[600,33,68],
                      [700,31,75],[620,32,72]]
   )
"""

import pickle
import numpy as np
import tensorflow as tf
from config import (YIELD_RF_PATH, YIELD_XGB_PATH,
                     YIELD_LSTM_PATH, YIELD_SCALER_PATH,
                     LE_CROP_PATH, LE_SOIL_PATH)

# ── Load Models (once at startup) ────────────────
_rf      = None
_xgb     = None
_lstm    = None
_scaler  = None
_le_crop = None
_le_soil = None

def _load():
    global _rf, _xgb, _lstm, _scaler, _le_crop, _le_soil
    if _rf is None:
        _rf      = pickle.load(open(YIELD_RF_PATH,    "rb"))
        _xgb     = pickle.load(open(YIELD_XGB_PATH,   "rb"))
        _lstm    = tf.keras.models.load_model(YIELD_LSTM_PATH)
        _scaler  = pickle.load(open(YIELD_SCALER_PATH, "rb"))
        _le_crop = pickle.load(open(LE_CROP_PATH,     "rb"))
        _le_soil = pickle.load(open(LE_SOIL_PATH,     "rb"))
        print("✅ Yield models loaded!")

# ── Predict Function (Ensemble) ───────────────────
def predict_yield(crop, soil_type, rainfall, temperature,
                   area, fertilizer, humidity,
                   last_4_weeks) -> dict:
    """
    RF+XGB (snapshot) + LSTM (sequence) ensemble

    Input:
        crop, soil_type  → string
        rainfall         → mm
        temperature      → Celsius
        area             → acres
        fertilizer       → kg/acre
        humidity         → percentage
        last_4_weeks     → list of 4 items:
                           [[rain,temp,humidity], ...]

    Output:
        dict → {rf_xgb_yield, lstm_yield,
                final_yield, unit}
    """
    _load()

    # ── Model 2: RF + XGBoost (Snapshot) ─────────
    crop_enc = _le_crop.transform([crop])[0]
    soil_enc = _le_soil.transform([soil_type])[0]
    snap = [[crop_enc, soil_enc, rainfall,
             temperature, area, fertilizer, humidity]]

    rf_out  = _rf.predict(snap)[0]
    xgb_out = _xgb.predict(snap)[0]
    model2  = (rf_out + xgb_out) / 2

    # ── Model 3: LSTM (Sequence) ──────────────────
    seq_scaled = _scaler.transform(last_4_weeks)
    seq_tensor = np.array([seq_scaled])
    model3     = float(_lstm.predict(seq_tensor, verbose=0)[0][0])

    # ── Ensemble (60% model2 + 40% model3) ───────
    final = (model2 * 0.6) + (model3 * 0.4)

    return {
        "rf_xgb_yield": round(float(model2), 2),
        "lstm_yield":   round(model3,          2),
        "final_yield":  round(final,            2),
        "unit":         "quintal/acre"
    }
