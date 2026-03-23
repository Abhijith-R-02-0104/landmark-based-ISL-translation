# ================= ADD THIS after your imports =================
from tensorflow.keras.layers import GRU, Dense, Dropout, Bidirectional
import random

# ================= ADD augmentation function before the MODEL section =================
def augment_sequence(seq):
    """
    Jitter + time-warp to make model robust to recording variation.
    seq shape: (30, 63)
    """
    # Gaussian noise — simulates MediaPipe landmark jitter
    seq = seq + np.random.normal(0, 0.008, seq.shape)
    # Random time-scaling (speed up or slow down by ±15%)
    if random.random() > 0.5:
        scale   = np.random.uniform(0.85, 1.15)
        new_len = int(30 * scale)
        idx     = np.linspace(0, 29, new_len).astype(int)
        seq     = seq[idx]
        if len(seq) < 30:
            seq = np.pad(seq, ((0, 30 - len(seq)), (0, 0)), mode='edge')
        else:
            seq = seq[:30]
    return seq.astype(np.float32)

# ================= AFTER loading X, add augmented copies =================
# Add this right after your "X = np.array(X)" line:
X_aug, y_aug = [], []
for seq, label in zip(X, y):
    X_aug.append(seq)
    y_aug.append(label)
    # Add 2 augmented copies per sample — triples your effective dataset
    for _ in range(2):
        X_aug.append(augment_sequence(seq.copy()))
        y_aug.append(label)

X = np.array(X_aug)
y = np.array(y_aug)
print(f"After augmentation: {X.shape}")  # should be ~3x original

# ================= REPLACE your model definition with this =================
model = Sequential([
    # Bidirectional: reads sequence forward AND backward
    # The end of a gesture is often the most distinctive part —
    # a unidirectional GRU has already half-forgotten it by then
    Bidirectional(GRU(128, return_sequences=True), input_shape=(30, 63)),
    Dropout(0.3),

    Bidirectional(GRU(64, return_sequences=False)),
    Dropout(0.3),

    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(len(le.classes_), activation='softmax')
])

# ================= CHANGE early stopping patience =================
early_stop = EarlyStopping(
    monitor='val_accuracy',  # changed from val_loss — more intuitive for classification
    patience=8,              # was 5 — too aggressive for small datasets
    restore_best_weights=True
)

# ================= CHANGE epochs =================
history = model.fit(
    X_train, y_train,
    epochs=60,        # was 25 — too few with early stopping safety net
    batch_size=16,    # was 8 — 8 is too small, causes noisy gradient updates
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)