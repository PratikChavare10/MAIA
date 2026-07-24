"""
modules/yield_prediction/train_lstm.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT TO ADD BEFORE RUNNING:
1. Weekly CSV file बनवा:
   data/yield_dataset/weekly_weather_yield.csv

CSV columns:
week, rainfall, temperature, humidity, yield_so_far

HOW TO RUN:
   python modules/yield_prediction/train_lstm.py
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pickle
from config import WEEKLY_DATASET_PATH, YIELD_LSTM_PATH, YIELD_SCALER_PATH

# ── Load Dataset ─────────────────────────────────
df = pd.read_csv(WEEKLY_DATASET_PATH)
print(f"Dataset shape: {df.shape}")

FEATURES = ['rainfall', 'temperature', 'humidity']
TARGET   = 'yield_so_far'

# ── Normalize ─────────────────────────────────────
scaler = MinMaxScaler()
df[FEATURES] = scaler.fit_transform(df[FEATURES])
pickle.dump(scaler, open(YIELD_SCALER_PATH, "wb"))
print(f"✅ Scaler saved: {YIELD_SCALER_PATH}")

# ── Create Sequences (last 4 weeks → predict) ────
SEQ_LEN = 4

def make_sequences(data, target, seq=SEQ_LEN):
    X, y = [], []
    for i in range(len(data) - seq):
        X.append(data[i:i+seq])
        y.append(target[i+seq])
    return np.array(X), np.array(y)

X, y = make_sequences(
    df[FEATURES].values,
    df[TARGET].values
)

split  = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Train: {X_train.shape} | Test: {X_test.shape}")

# ── Build LSTM Model ─────────────────────────────
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64,
        input_shape=(SEQ_LEN, len(FEATURES)),
        return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

model.summary()

# ── Train ────────────────────────────────────────
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    YIELD_LSTM_PATH,
    save_best_only=True,
    monitor='val_loss',
    verbose=1
)

model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=16,
    callbacks=[checkpoint],
    verbose=1
)

print(f"\n✅ LSTM Model saved: {YIELD_LSTM_PATH}")
