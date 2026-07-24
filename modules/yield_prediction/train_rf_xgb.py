"""
modules/yield_prediction/train_rf_xgb.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT TO ADD BEFORE RUNNING:
1. Download dataset from Kaggle:
   → https://www.kaggle.com/datasets/gurudathg/crop-yield-prediction-using-soil-and-weather
2. Save as: data/yield_dataset/crop_yield_data.csv

CSV must have these columns:
crop, soil_type, rainfall_mm, temperature,
area_acres, fertilizer_used, humidity,
yield_quintal_per_acre

HOW TO RUN:
   python modules/yield_prediction/train_rf_xgb.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import xgboost as xgb
import pickle
from sklearn.ensemble import VotingRegressor
from sklearn.compose import ColumnTransformer
from config import (YIELD_DATASET_PATH, YIELD_RF_PATH,
                     YIELD_XGB_PATH, LE_CROP_PATH, LE_SOIL_PATH, LE_WEATHER_PATH)

# ── Load Dataset ─────────────────────────────────
df = pd.read_csv(YIELD_DATASET_PATH)
print(f"Dataset shape: {df.shape}")

# ── Encode Categorical ───────────────────────────
# STEP 2: Encode categorical columns
le_crop = LabelEncoder()
le_soil = LabelEncoder()
le_weather=LabelEncoder()
df['crop_encoded'] = le_crop.fit_transform(df['Crop'])
df['soil_encoded'] = le_soil.fit_transform(df['Soil_Type'])
df['weather_encoded'] = le_weather.fit_transform(df['Weather_Condition'])

ohe=OneHotEncoder(drop="first",sparse_output=False,dtype=np.int32).set_output(transform='pandas')

# ── Features & Target ────────────────────────────
X=df.drop(columns=['Yield_tons_per_hectare','Crop','Soil_Type','Weather_Condition'])
y=df['Yield_tons_per_hectare']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

trans=ColumnTransformer(transformers=[('tranf1',ohe,['Region'])],remainder='passthrough',verbose_feature_names_out=False).set_output(transform='pandas')

X_train_trans=trans.fit_transform(X_train)
X_test_trans=trans.transform(X_test)

# ── Train Random Forest ──────────────────────────
# STEP 4: Train Random Forest
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)
rf.fit(X_train_trans, y_train)
rf_pred = rf.predict(X_test_trans)
print(f"Random Forest → R2: {r2_score(y_test, rf_pred):.3f} | MAE: {mean_absolute_error(y_test, rf_pred):.2f}")
print("Random Forest MAE:", mean_absolute_error(y_test, rf_pred))

# ── Train XGBoost ────────────────────────────────
xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)
xgb_model.fit(X_train_trans, y_train)
xgb_pred = xgb_model.predict(X_test_trans)
print(f"XGBoost       → R2: {r2_score(y_test, xgb_pred):.3f} | MAE: {mean_absolute_error(y_test, xgb_pred):.2f}")

# ── Save Models ──────────────────────────────────
pickle.dump(rf,        open(YIELD_RF_PATH,  "wb"))
pickle.dump(xgb_model, open(YIELD_XGB_PATH, "wb"))
print(f"\nRF Model saved:  {YIELD_RF_PATH}")
print(f"XGB Model saved: {YIELD_XGB_PATH}")

#by voting
vr=VotingRegressor(estimators=[('rf_model',rf),('xgb_model',xgb_model)])
vr.fit(X_train_trans, y_train)

vr_pred = xgb_model.predict(X_test_trans)
print("R2 Score:", r2_score(y_test, xgb_pred))
print("MAE:", mean_absolute_error(y_test, xgb_pred))

# Save encoders
pickle.dump(le_crop, open(LE_CROP_PATH, "wb"))
pickle.dump(le_soil, open(LE_SOIL_PATH, "wb"))
pickle.dump(le_weather, open(LE_WEATHER_PATH, "wb"))
pickle.dump(vr,        open(YIELD_RF_PATH,  "wb"))