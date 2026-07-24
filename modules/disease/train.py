"""
modules/disease/train.py
━━━━━━━━━━━━━━━━━━━━━━━
WHAT TO ADD BEFORE RUNNING:
1. Download PlantVillage dataset from Kaggle
   → https://www.kaggle.com/datasets/emmarex/plantdisease
2. Extract to: data/raw/plantvillage/
   Structure should be:
   data/raw/plantvillage/
       Tomato_Early_blight/  ← folder per disease
       Tomato_Late_blight/
       ...

HOW TO RUN:
   python modules/disease/train.py
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import PLANTVILLAGE_PATH, DISEASE_MODEL_PATH
import os

# ── Config ──────────────────────────────────────
IMG_SIZE   = 224
BATCH_SIZE = 32
EPOCHS     = 20
NUM_CLASSES = 38   # PlantVillage has 38 disease classes

DEVICE = "GPU" if tf.config.list_physical_devices('GPU') else "CPU"
print(f"Training on: {DEVICE}")

# ── Data Generators ─────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    horizontal_flip=True,
    zoom_range=0.2,
    validation_split=0.2      # 80% train, 20% validation
)

train_gen = train_datagen.flow_from_directory(
    PLANTVILLAGE_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_gen = train_datagen.flow_from_directory(
    PLANTVILLAGE_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

NUM_CLASSES = len(train_gen.class_indices)
print(f"Total Classes Found: {NUM_CLASSES}")

# Save class names for predict.py
import json
class_names = list(train_gen.class_indices.keys())
with open("../../models/disease_classes.json", "w") as f:
    json.dump(class_names, f)
print(" Class names saved!")

# ── Vision Transformer Blocks ───────────────────
class PatchEmbedding(layers.Layer):
    def __init__(self, patch_size, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.proj = layers.Conv2D(embed_dim, patch_size,
                                   strides=patch_size)
        self.flat = layers.Reshape((-1, embed_dim))

    def call(self, x):
        x = self.proj(x)
        return self.flat(x)


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, mlp_dim):
        super().__init__()
        self.attn  = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads
        )
        self.ffn   = tf.keras.Sequential([
            layers.Dense(mlp_dim, activation='gelu'),
            layers.Dense(embed_dim)
        ])
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()

    def call(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


def build_vit(num_classes, img_size=224, patch_size=16,
               embed_dim=256, num_heads=8,
               num_layers=6, mlp_dim=512):

    inputs = layers.Input(shape=(img_size, img_size, 3))

    # Patch Embedding
    x = PatchEmbedding(patch_size, embed_dim)(inputs)

    # Positional Encoding
    num_patches = (img_size // patch_size) ** 2
    pos_emb = tf.Variable(
        tf.zeros((1, num_patches, embed_dim)), trainable=True
    )
    x = x + pos_emb

    # Transformer Blocks
    for _ in range(num_layers):
        x = TransformerBlock(embed_dim, num_heads, mlp_dim)(x)

    # Classification Head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(256, activation='gelu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs)


# ── Build Model ─────────────────────────────────
model = build_vit(num_classes=NUM_CLASSES)

model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=2e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ── Callbacks ───────────────────────────────────
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    DISEASE_MODEL_PATH,
    save_best_only=True,
    monitor='val_accuracy',
    verbose=1
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)

# ── Train ────────────────────────────────────────
print("\n Starting Training...\n")
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop]
)

print(f"\nDisease Model saved to: {DISEASE_MODEL_PATH}")
