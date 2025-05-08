import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# === Create directory ===
save_dir = 'nn_models/Tsection'
os.makedirs(save_dir, exist_ok=True)

# === Load data ===
file_path = r'dataset_files\Tsection\Tsection_balanced2.parquet'
df = pd.read_parquet(file_path)

# === Select features and targets ===
features = ['MEd', 'beff', 'bw', 'h', 'hf', 'cnom', 'fi_str']
X = df[features]
y_fck = df['fck']
y_fi = df['fi']

# === Encode target variables ===
le_fck = LabelEncoder()
le_fi = LabelEncoder()
y_fck_enc = le_fck.fit_transform(y_fck)
y_fi_enc = le_fi.fit_transform(y_fi)

# === Train/test split ===
X_train, X_test, y_fck_train, y_fck_test, y_fi_train, y_fi_test = train_test_split(
    X, y_fck_enc, y_fi_enc, test_size=0.2, random_state=42
)

# === Feature scaling ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Build the enhanced model ===
input_layer = keras.Input(shape=(len(features),))
x = layers.Dense(512, activation='relu')(input_layer)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.2)(x)

fck_out = layers.Dense(len(le_fck.classes_), activation='softmax', name='fck_output')(x)
fi_out = layers.Dense(len(le_fi.classes_), activation='softmax', name='fi_output')(x)

model = keras.Model(inputs=input_layer, outputs=[fck_out, fi_out])

# === Learning rate scheduler ===
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=1000,
    decay_rate=0.9
)

# === Focal loss implementation ===
def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        ce = keras.losses.SparseCategoricalCrossentropy(reduction='none')(y_true, y_pred)
        pt = tf.exp(-ce)
        return alpha * tf.pow(1 - pt, gamma) * ce
    return loss

# === Compile model ===
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    loss={
        'fck_output': focal_loss(),
        'fi_output': focal_loss()
    },
    metrics={
        'fck_output': 'accuracy',
        'fi_output': 'accuracy'
    }
)

# === Enhanced callbacks ===
callbacks = [
    callbacks.EarlyStopping(patience=15, monitor='val_loss', restore_best_weights=True),
    callbacks.ModelCheckpoint(
        filepath=os.path.join(save_dir, 'best_model.keras'),
        monitor='val_loss',
        save_best_only=True
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6
    )
]

# === Train the model ===
history = model.fit(
    X_train_scaled,
    {'fck_output': y_fck_train, 'fi_output': y_fi_train},
    validation_split=0.2,
    epochs=200,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# === Training history visualization ===
def plot_history(history):
    plt.figure(figsize=(14, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['fck_output_accuracy'], label='Train Acc (fck)')
    plt.plot(history.history['val_fck_output_accuracy'], label='Val Acc (fck)')
    plt.plot(history.history['fi_output_accuracy'], label='Train Acc (fi)')
    plt.plot(history.history['val_fi_output_accuracy'], label='Val Acc (fi)')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_history(history)

# === Load best weights ===
model = keras.models.load_model(os.path.join(save_dir, 'best_model.keras'),
                              custom_objects={'loss': focal_loss()})

# === Evaluate ===
results = model.evaluate(
    X_test_scaled,
    {'fck_output': y_fck_test, 'fi_output': y_fi_test},
    verbose=0
)
print(f"\nTest Loss: {results[0]:.4f}")
print(f"Test Accuracy (fck): {results[3]:.4f}")
print(f"Test Accuracy (fi): {results[4]:.4f}")

# === Predictions ===
preds = model.predict(X_test_scaled)
fck_pred = np.argmax(preds[0], axis=1)
fi_pred = np.argmax(preds[1], axis=1)

# === Decode labels ===
y_fck_test_labels = le_fck.inverse_transform(y_fck_test)
fck_pred_labels = le_fck.inverse_transform(fck_pred)
y_fi_test_labels = le_fi.inverse_transform(y_fi_test)
fi_pred_labels = le_fi.inverse_transform(fi_pred)

# === Classification reports ===
print("\nClassification Report for fck:")
print(classification_report(y_fck_test_labels, fck_pred_labels))

print("\nClassification Report for fi:")
print(classification_report(y_fi_test_labels, fi_pred_labels))

# === Save artifacts ===
model.save(os.path.join(save_dir, 'final_model.keras'))
with open(os.path.join(save_dir, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
with open(os.path.join(save_dir, 'label_encoder_fck.pkl'), 'wb') as f:
    pickle.dump(le_fck, f)
with open(os.path.join(save_dir, 'label_encoder_fi.pkl'), 'wb') as f:
    pickle.dump(le_fi, f)

print(f"\nAll artifacts saved to: {save_dir}")