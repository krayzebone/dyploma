import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report

# Create save directory if it doesn't exist
save_dir = 'nn_models/Tsection'
os.makedirs(save_dir, exist_ok=True)

# 1. Load the dataset
file_path = 'dataset_files/Tsection/Tsection_optimized.parquet'
df = pd.read_parquet(file_path)

# 2. Preprocess the data
features = ['MEd', 'beff', 'bw', 'h', 'hf', 'cnom', 'fi_str']
targets = ['fck', 'fi']

X = df[features]
y_fck = df[targets[0]]
y_fi = df[targets[1]]

# Encode target variables
le_fck = LabelEncoder()
le_fi = LabelEncoder()
y_fck_encoded = le_fck.fit_transform(y_fck)
y_fi_encoded = le_fi.fit_transform(y_fi)

# Split data
X_train, X_test, y_fck_train, y_fck_test, y_fi_train, y_fi_test = train_test_split(
    X, y_fck_encoded, y_fi_encoded, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Build the neural network model
input_layer = keras.Input(shape=(len(features),))
x = layers.Dense(128, activation='relu')(input_layer)
x = layers.Dropout(0.3)(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.2)(x)

fck_output = layers.Dense(len(le_fck.classes_), activation='softmax', name='fck_output')(x)
fi_output = layers.Dense(len(le_fi.classes_), activation='softmax', name='fi_output')(x)

model = keras.Model(inputs=input_layer, outputs=[fck_output, fi_output], name="concrete_properties_classifier")

# 4. Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss={
        'fck_output': 'sparse_categorical_crossentropy',
        'fi_output': 'sparse_categorical_crossentropy'
    },
    metrics={
        'fck_output': 'accuracy',
        'fi_output': 'accuracy'
    }
)

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# 5. Train the model
history = model.fit(
    X_train_scaled,
    {'fck_output': y_fck_train, 'fi_output': y_fi_train},
    validation_split=0.15,
    epochs=100,
    batch_size=64,
    callbacks=[early_stopping],
    verbose=1
)

# 6. Evaluate
test_results = model.evaluate(
    X_test_scaled,
    {'fck_output': y_fck_test, 'fi_output': y_fi_test},
    verbose=0
)

print("\nTest Loss:", test_results[0])
print("Test Accuracy for fck:", test_results[3])
print("Test Accuracy for fi:", test_results[4])

# Predictions
predictions = model.predict(X_test_scaled)
fck_pred = np.argmax(predictions[0], axis=1)
fi_pred = np.argmax(predictions[1], axis=1)

# Decode labels
y_fck_test_names = le_fck.inverse_transform(y_fck_test)
fck_pred_names = le_fck.inverse_transform(fck_pred)

y_fi_test_names = le_fi.inverse_transform(y_fi_test)
fi_pred_names = le_fi.inverse_transform(fi_pred)

print("\nClassification Report for fck:")
print(classification_report(y_fck_test_names, fck_pred_names))

print("\nClassification Report for fi:")
print(classification_report(y_fi_test_names, fi_pred_names))

print("\nConfusion Matrix for fck:")
print(pd.crosstab(y_fck_test_names, fck_pred_names, 
                 rownames=['Actual'], colnames=['Predicted'], margins=True))

print("\nConfusion Matrix for fi:")
print(pd.crosstab(y_fi_test_names, fi_pred_names,
                 rownames=['Actual'], colnames=['Predicted'], margins=True))

# === Save model and scalers ===
model_path = os.path.join(save_dir, 'concrete_properties_classifier.keras')
model.save(model_path)

# Save scaler and encoders
with open(os.path.join(save_dir, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

with open(os.path.join(save_dir, 'label_encoder_fck.pkl'), 'wb') as f:
    pickle.dump(le_fck, f)

with open(os.path.join(save_dir, 'label_encoder_fi.pkl'), 'wb') as f:
    pickle.dump(le_fi, f)

print(f"\nModel and scalers saved to: {save_dir}")
