# ================= IMPORTS =================
import os
import numpy as np
import random

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ================= SETTINGS =================
SEQUENCE_LENGTH = 30

# 🔥 CORRECT DATA PATH
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "../data/raw/Abhi")

print("Using dataset path:", DATASET_PATH)

# ================= LOAD DATA =================
X, y = [], []

class_names = sorted(os.listdir(DATASET_PATH))
print("CLASS ORDER:", class_names)

for label in class_names:
    label_path = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(label_path):
        continue

    for file in os.listdir(label_path):
        file_path = os.path.join(label_path, file)

        try:
            seq = np.load(file_path)

            # 🔥 FIX: convert (30, 21, 3) → (30, 63)
            if len(seq.shape) == 3 and seq.shape[1] == 21 and seq.shape[2] == 3:
                seq = seq.reshape(SEQUENCE_LENGTH, 63)

            # ✅ Accept valid format
            if len(seq.shape) == 2 and seq.shape[1] == 63:

                # Fix sequence length
                if seq.shape[0] < SEQUENCE_LENGTH:
                    seq = np.pad(seq, ((0, SEQUENCE_LENGTH - seq.shape[0]), (0, 0)), mode='edge')
                else:
                    seq = seq[:SEQUENCE_LENGTH]

                X.append(seq)
                y.append(label)

        except:
            continue

X = np.array(X)
y = np.array(y)

print("Loaded data:", X.shape, y.shape)

# ================= LABEL ENCODING =================
le = LabelEncoder()
y = le.fit_transform(y)

# 🔥 SAVE LABEL ORDER
class_names = list(le.classes_)
np.save(os.path.join(BASE_DIR, "labels.npy"), class_names)
print("Saved labels:", class_names)

# ================= AUGMENTATION =================
def augment_sequence(seq):
    seq = seq + np.random.normal(0, 0.008, seq.shape)

    if random.random() > 0.5:
        scale = np.random.uniform(0.85, 1.15)
        new_len = int(SEQUENCE_LENGTH * scale)
        idx = np.linspace(0, SEQUENCE_LENGTH - 1, new_len).astype(int)
        seq = seq[idx]

        if len(seq) < SEQUENCE_LENGTH:
            seq = np.pad(seq, ((0, SEQUENCE_LENGTH - len(seq)), (0, 0)), mode='edge')
        else:
            seq = seq[:SEQUENCE_LENGTH]

    return seq.astype(np.float32)

# ================= AUGMENT DATA =================
X_aug, y_aug = [], []

for seq, label in zip(X, y):
    X_aug.append(seq)
    y_aug.append(label)

    for _ in range(2):
        X_aug.append(augment_sequence(seq.copy()))
        y_aug.append(label)

X = np.array(X_aug)
y = np.array(y_aug)

print("After augmentation:", X.shape)

# ================= TRAIN TEST SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ================= MODEL =================
model = Sequential([
    Bidirectional(GRU(128, return_sequences=True), input_shape=(SEQUENCE_LENGTH, 63)),
    Dropout(0.3),

    Bidirectional(GRU(64, return_sequences=False)),
    Dropout(0.3),

    Dense(64, activation='relu'),
    Dropout(0.2),

    Dense(len(le.classes_), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ================= EARLY STOPPING =================
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=8,
    restore_best_weights=True
)

# ================= TRAIN =================
history = model.fit(
    X_train, y_train,
    epochs=60,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# ================= SAVE MODEL =================
model.save(os.path.join(BASE_DIR, "model.h5"))

print("✅ Training complete & labels saved!")