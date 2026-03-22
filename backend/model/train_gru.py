import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ================= PATH =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw")
MODEL_DIR = os.path.join(BASE_DIR, "model")

os.makedirs(MODEL_DIR, exist_ok=True)

# ================= LOAD DATA =================
X = []
y = []

for person in os.listdir(DATA_PATH):

    person_path = os.path.join(DATA_PATH, person)

    if not os.path.isdir(person_path):
        continue

    # 🔥 Skip empty folders
    if len(os.listdir(person_path)) == 0:
        print(f"Skipping empty person folder: {person}")
        continue

    for gesture in os.listdir(person_path):

        gesture_path = os.path.join(person_path, gesture)

        if not os.path.isdir(gesture_path):
            continue

        for file in os.listdir(gesture_path):

            if file.endswith(".npy"):

                file_path = os.path.join(gesture_path, file)

                data = np.load(file_path)

                # Validate shape
                if data.shape != (30, 21, 3):
                    print(f"Skipping invalid sample: {file_path}")
                    continue

                # Flatten 21x3 → 63 per frame
                data = data.reshape(30, 63)

                X.append(data)
                y.append(gesture)

# ================= CHECK DATA =================
if len(X) == 0:
    print("❌ No valid data found. Check dataset.")
    exit()

X = np.array(X)
y = np.array(y)

print("Dataset shape:", X.shape)
print("Labels:", np.unique(y))

# ================= LABEL ENCODING =================
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Save labels
labels_path = os.path.join(MODEL_DIR, "labels.npy")
np.save(labels_path, le.classes_)

print("Saved labels:", le.classes_)

# ================= TRAIN TEST SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_categorical,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# ================= MODEL =================
model = Sequential([
    GRU(128, return_sequences=True, input_shape=(30, 63)),
    Dropout(0.3),

    GRU(64, return_sequences=False),
    Dropout(0.3),

    Dense(64, activation='relu'),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ================= EARLY STOPPING =================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# ================= TRAIN =================
history = model.fit(
    X_train,
    y_train,
    epochs=25,
    batch_size=8,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# ================= SAVE MODEL =================
model_path = os.path.join(MODEL_DIR, "model.h5")

model.save(model_path)

print("\n✅ Model saved to:", model_path)
print("✅ Training complete.")

# ================= EVALUATE =================
loss, acc = model.evaluate(X_test, y_test)
print(f"\n🎯 Test Accuracy: {acc * 100:.2f}%")