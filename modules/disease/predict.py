"""
modules/disease/predict.py
━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT TO ADD:
- TREATMENT_DB मध्ये तुमच्या dataset च्या
  class names नुसार treatments add करा
- train.py run केल्यावर models/disease_classes.json
  बनतो — त्यातील names वापरा

HOW TO USE:
   from modules.disease.predict import predict_disease
   result = predict_disease("path/to/image.jpg")
"""

import tensorflow as tf
import numpy as np
import json
from PIL import Image
from config import DISEASE_MODEL_PATH

# ── Load Model (once at startup) ─────────────────
# ADD: train.py run केल्यावरच हे load होईल
_model = None
_class_names = None

def _load_model():
    global _model, _class_names
    if _model is None:
        _model = tf.keras.models.load_model(DISEASE_MODEL_PATH)
        with open("models/disease_classes.json") as f:
            _class_names = json.load(f)
        print("✅ Disease model loaded!")

# ── Treatment Database ────────────────────────────
# ADD: आपल्या dataset च्या class names नुसार हे भरा
# Format: "ClassName": {"treatment": "...", "prevention": "..."}
TREATMENT_DB = {
    "Tomato___Early_blight": {
        "treatment":   "Spray Mancozeb 75% WP @ 2g/liter water",
        "prevention":  "Crop rotation, remove infected leaves",
        "severity":    "Moderate"
    },
    "Tomato___Late_blight": {
        "treatment":   "Spray Metalaxyl + Mancozeb @ 2.5g/liter",
        "prevention":  "Avoid overhead irrigation",
        "severity":    "High"
    },
    "Tomato___healthy": {
        "treatment":   "No treatment needed",
        "prevention":  "Continue regular monitoring",
        "severity":    "None"
    },
    "Potato___Early_blight": {
        "treatment":   "Spray Chlorothalonil @ 2g/liter",
        "prevention":  "Use certified disease-free seeds",
        "severity":    "Moderate"
    },
    "Potato___Late_blight": {
        "treatment":   "Spray Cymoxanil @ 1g/liter",
        "prevention":  "Destroy infected plants immediately",
        "severity":    "High"
    },
    "Corn_(maize)___Common_rust_": {
        "treatment":   "Spray Propiconazole @ 1ml/liter",
        "prevention":  "Use rust-resistant varieties",
        "severity":    "Moderate"
    },
    # ADD MORE: बाकी 38 classes साठी येथे add करा
    # train.py run केल्यावर disease_classes.json मध्ये
    # सगळ्या class names मिळतील
}

DEFAULT_TREATMENT = {
    "treatment":  "Consult local agricultural expert (KVK)",
    "prevention": "Regular field inspection recommended",
    "severity":   "Unknown"
}

# ── Predict Function ──────────────────────────────
def predict_disease(image_path: str) -> dict:
    """
    Crop photo घेऊन disease detect करतो

    Input:
        image_path (str) → uploaded image file path

    Output:
        dict → {
            disease, confidence, treatment,
            prevention, severity
        }
    """
    _load_model()

    # Preprocess image
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)

    # Predict
    preds      = _model.predict(arr, verbose=0)[0]
    idx        = np.argmax(preds)
    confidence = float(preds[idx]) * 100
    disease    = _class_names[idx]

    # Get treatment
    info = TREATMENT_DB.get(disease, DEFAULT_TREATMENT)

    return {
        "disease":    disease,
        "confidence": f"{confidence:.1f}%",
        "treatment":  info["treatment"],
        "prevention": info["prevention"],
        "severity":   info["severity"]
    }
