"""
modules/crop_recommendation/train.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT TO ADD BEFORE RUNNING:
1. Download dataset from Kaggle:
   → https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset
2. Save as: data/crop_recommendation/Crop_recommendation.csv

HOW TO RUN:
   python modules/crop_recommendation/train.py
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Input
from tensorflow.keras.utils import to_categorical
import joblib
from config import CROP_REC_DATASET, CROP_REC_MODEL_PATH, CROP_REC_LE_PATH, CROP_REC_SC_PATH

# ── Load Dataset ─────────────────────────────────
# ADD: Dataset download केल्यावरच हे run करा
df = pd.read_csv(CROP_REC_DATASET)
print(f"Dataset shape: {df.shape}")
print(f"Crops: {df['label'].unique()}")

# ── Features & Target ────────────────────────────
X=df.drop(columns=['label'])
y=df['label']

# Encode crop labels
le = LabelEncoder()
y_enc = le.fit_transform(y)
print(len(set(y_enc)))
print(le.classes_)

#Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_enc,
    test_size=0.2,
    random_state=42
)

#Neural Network
#%%
model = Sequential()
model.add(Input(shape=(X_train.shape[1],)))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(len(set(y_enc)),activation='softmax'))

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# ── Train Model ──────────────────────────────────
history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=16
)

# ── Evaluate ─────────────────────────────────────
loss,acc=model.evaluate(X_test,y_test)
print("Accuracy :",acc)


# ── Save Model ───────────────────────────────────
joblib.dump(le, CROP_REC_LE_PATH)
joblib.dump(scaler, CROP_REC_SC_PATH)
model.save(CROP_REC_MODEL_PATH)


