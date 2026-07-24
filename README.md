# 🌾 MAIA — Multimodal Agriculture Intelligence Assistant

## 📌 WHAT TO ADD BEFORE RUNNING (Step by Step)

---

### STEP 1 — Environment Setup
```bash
python -m venv maia_env
source maia_env/bin/activate        # Mac/Linux
maia_env\Scripts\activate           # Windows

pip install -r requirements.txt
```

---

### STEP 2 — API Keys Add करा
```bash
# .env copy करा
cp .env .env

# .env file उघडा आणि keys भरा:
OPENAI_API_KEY=sk-...
WEATHER_API_KEY=abc123...
MYSQL_PASSWORD=your_password
```

**Weather API Key कसे मिळवायचे:**
1. https://openweathermap.org वर जा
2. Account बनवा → Email verify करा
3. My API Keys → Key copy करा
4. .env मध्ये WEATHER_API_KEY= येथे paste करा

---

### STEP 3 — Datasets Download करा (Kaggle)
```
1. PlantVillage → data/raw/plantvillage/
   https://www.kaggle.com/datasets/emmarex/plantdisease

2. Crop Recommendation → data/crop_recommendation/Crop_recommendation.csv
   https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset

3. Yield Dataset → data/yield_dataset/crop_yield_data.csv
   https://www.kaggle.com/datasets/gurudathg/crop-yield-prediction-using-soil-and-weather
```

---

### STEP 4 — Models Train करा (Order important!)
```bash
# Disease Detection Model (ViT)
python modules/disease/train.py

# Crop Recommendation Model
python modules/crop_recommendation/train.py

# Yield Prediction Models
python modules/yield_prediction/train_rf_xgb.py
python modules/yield_prediction/train_lstm.py
```

---

### STEP 5 — PDF Documents Add करा (RAG साठी)
```
data/documents/ मध्ये हे PDFs download करून ठेवा:
- pm_kisan_scheme.pdf        → pmkisan.gov.in
- pmfby_insurance.pdf        → pmfby.gov.in
- crop_disease_manual.pdf    → icar.org.in
- fertilizer_guide.pdf       → kvk.icar.gov.in
```

---

### STEP 6 — RAG Vector Store बनवा
```bash
python modules/rag/ingest.py
```

---

### STEP 7 — MySQL Database Setup करा
```sql
CREATE DATABASE maia_db;
```
(app.py automatically tables बनवतो on first run)

---

### STEP 8 — App Run करा
```bash
python app.py
```
→ Open: http://localhost:5000

---

### STEP 9 — Docker मध्ये Run करा (Optional)
```bash
docker-compose up --build
```
→ Open: http://localhost:5000

---

## 📦 Project Structure
```
MAIA/
├── app.py                           ← Main Flask app
├── config.py                        ← API keys + settings
├── requirements.txt
├── .env                             ← ADD: Your API keys
├── Dockerfile
├── docker-compose.yml
│
├── modules/
│   ├── disease/
│   │   ├── train.py                 ← RUN: Step 4
│   │   └── predict.py
│   ├── crop_recommendation/
│   │   ├── train.py                 ← RUN: Step 4
│   │   └── predict.py
│   ├── yield_prediction/
│   │   ├── train_rf_xgb.py          ← RUN: Step 4
│   │   ├── train_lstm.py            ← RUN: Step 4
│   │   └── predict.py
│   ├── irrigation/
│   │   └── calculator.py
│   ├── weather/
│   │   └── fetch.py
│   ├── rag/
│   │   ├── ingest.py                ← RUN: Step 6
│   │   └── retriever.py
│   ├── multilingual/
│   │   ├── voice.py
│   │   └── translator.py
│   └── memory/
│       └── memory.py
│
├── agents/
│   ├── tools.py                     ← All modules imported here
│   └── master_agent.py              ← LangGraph orchestrator
│
├── data/
│   ├── documents/                   ← ADD: PDFs (Step 5)
│   ├── vectorstore/                 ← Auto-created (Step 6)
│   ├── raw/plantvillage/            ← ADD: Dataset (Step 3)
│   ├── yield_dataset/               ← ADD: Dataset (Step 3)
│   └── crop_recommendation/         ← ADD: Dataset (Step 3)
│
├── models/                          ← Auto-saved after training
│
└── templates/
    ├── index.html
    └── chat.html
```
