

import joblib
import numpy as np
from config import CROP_REC_MODEL_PATH, CROP_REC_LE_PATH, CROP_REC_SC_PATH
import tensorflow as tf

# ── Load Models (once at startup) ────────────────
_model = None
_le    = None
_sc    = None
def _load():
    global _model, _le
    if _model is None:
        _sc    = joblib.load(CROP_REC_SC_PATH)
        _model = tf.keras.models.load_model(CROP_REC_MODEL_PATH)
        _le    = joblib.load(CROP_REC_LE_PATH)
        print(" Crop Recommendation model loaded!")
        return _sc, _model, _le

_sc, _model, _le= _load()

# ── Predict Function ──────────────────────────────
def recommend_crop(N, P, K, temperature,
                    humidity, ph, rainfall) -> dict:

    # _load()


    inp   = [[N, P, K, temperature, humidity, ph, rainfall]]

    sc_inp=_sc.transform(inp)
    predictions = _model.predict(sc_inp)
    top3  = predictions.argsort()[-3:][0][::-1]
    return {
        "recommended_crop": _le.inverse_transform([top3[0]])[0],
        "top_3": [
            {
                "crop":       _le.inverse_transform([i])[0],
                "confidence": f"{predictions[0][i]*100:.1f}%"
            }
            for i in top3
        ]
    }
# print(recommend_crop(90,42,43,27,70,6.5,200))
