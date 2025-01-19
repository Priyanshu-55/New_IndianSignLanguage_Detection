import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import json

# Augmentation Functions
def apply_augmentations(sequence, sequence_length):
    augmented_sequences = []

    # Convert the sequence to numpy array for operations
    sequence = np.array(sequence)

    # Original Sequence
    augmented_sequences.append(sequence)  # Original sequence without changes

    # Mirrored Sequence
    augmented_sequences.append(mirror_sequence(sequence, sequence_length))

    # Noisy Sequence
    augmented_sequences.append(add_noise(sequence, sequence_length))

    # Time Warped Sequence
    augmented_sequences.append(time_warp(sequence, sequence_length))

    # Spatially Scaled Sequence
    augmented_sequences.append(spatial_scale(sequence, sequence_length))

    return augmented_sequences

# 1. Mirrored Sequences
def mirror_sequence(sequence, sequence_length):
    mirrored = [frame[::-1] for frame in sequence]  # Reverse each frame's landmarks
    return pad_or_trim_sequence(mirrored, sequence_length)

# 2. Noisy Sequences (with Bias)
def add_noise(sequence, sequence_length, bias=0.1):
    noise = np.random.normal(0, bias, sequence.shape)  # Add noise to the sequence
    noisy_sequence = sequence + noise
    return pad_or_trim_sequence(noisy_sequence, sequence_length)

# 3. Time Warping (speed up or slow down gesture sequences)
def time_warp(sequence, sequence_length, max_rate=1.5):
    rate = random.uniform(1.0, max_rate)
    length = len(sequence)
    indices = np.arange(length)
    new_indices = np.linspace(0, length-1, int(length * rate))
    warped_sequence = [sequence[int(i)] for i in new_indices]
    return pad_or_trim_sequence(warped_sequence, sequence_length)

# 4. Spatial Scaling
def spatial_scale(sequence, sequence_length, scale_factor=0.1):
    scaled_sequence = sequence * random.uniform(1 - scale_factor, 1 + scale_factor)
    return pad_or_trim_sequence(scaled_sequence, sequence_length)

# Helper function to pad or trim sequences to ensure consistent shape
def pad_or_trim_sequence(sequence, sequence_length):
    # Trim if the sequence is too long
    if len(sequence) > sequence_length:
        sequence = sequence[:sequence_length]
    # Pad if the sequence is too short
    elif len(sequence) < sequence_length:
        padding = [np.zeros_like(sequence[0])] * (sequence_length - len(sequence))
        sequence.extend(padding)
    return np.array(sequence)

# Paths and Parameters
DATA_PATH = "NewDataset"  # Root folder for dataset
actions = np.array(os.listdir(DATA_PATH))  # Assuming action names are folder names
sequence_length = 30  # Each sequence contains 30 frames
features_per_frame = 225  # Each frame has 225 features

# Create label map (gesture name to integer)
label_map = {label: num for num, label in enumerate(actions)}

# Load data
sequences, labels = [], []
print("Loading data...")

for action in tqdm(actions, desc="Processing actions"):
    action_path = os.path.join(DATA_PATH, action)

    for sequence in tqdm(os.listdir(action_path), desc=f"Processing sequences for {action}", leave=False):
        sequence_path = os.path.join(action_path, sequence)

        if not os.path.isdir(sequence_path):
            continue

        # Load each frame in the sequence
        window = []
        for frame_num in range(sequence_length):
            frame_path = os.path.join(sequence_path, f"{frame_num}.npy")
            if os.path.exists(frame_path):
                frame_features = np.load(frame_path)
                window.append(frame_features)
            else:
                print(f"Frame {frame_num} missing in {sequence_path}!")  # Debugging missing frames

        if len(window) == sequence_length:  # Ensure full sequences only
            augmented_sequences = apply_augmentations(window, sequence_length)  # Apply augmentations
            sequences.extend(augmented_sequences)
            labels.extend([label_map[action]] * len(augmented_sequences))  # Duplicate the label for each augmented sequence

# Convert to numpy arrays
X = np.array(sequences)
y = to_categorical(labels, num_classes=len(actions)).astype(int)

print(f"Data shape: X={X.shape}, y={y.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build the model
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length, features_per_frame)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(actions), activation='softmax'))  # Number of classes = len(actions)

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Callbacks (e.g., TensorBoard, EarlyStopping)
tb_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs')

# Train the model
print("Training the model...")
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[tb_callback])

# Save the model
model.save("gesture_recognition_model2.h5")
print("Model saved as 'gesture_recognition_model.h5'")

# Generate Report
print("Generating model report...")

# Accuracy and loss
accuracy = history.history['categorical_accuracy'][-1]
val_accuracy = history.history['val_categorical_accuracy'][-1]
loss = history.history['loss'][-1]
val_loss = history.history['val_loss'][-1]

# Predictions and Evaluation
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Classification report
class_report = classification_report(y_true_classes, y_pred_classes, target_names=actions, output_dict=True)
class_report_text = classification_report(y_true_classes, y_pred_classes, target_names=actions)

# Confusion Matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=actions)
disp.plot(cmap=plt.cm.Blues)

# Save confusion matrix
conf_matrix_path = "confusion_matrix2.png"
plt.savefig(conf_matrix_path)
plt.close()
print(f"Confusion matrix saved as {conf_matrix_path}")

# Save model details to a JSON file
model_report = {
    "model_architecture": [layer.get_config() for layer in model.layers],
    "input_shape": model.input_shape,
    "output_shape": model.output_shape,
    "num_parameters": model.count_params(),
    "training_accuracy": accuracy,
    "validation_accuracy": val_accuracy,
    "training_loss": loss,
    "validation_loss": val_loss,
    "classification_report": class_report,
    "confusion_matrix_path": conf_matrix_path
}

# Save the report to a JSON file
report_path = "model_report2.json"
with open(report_path, "w") as report_file:
    json.dump(model_report, report_file, indent=4)

print(f"Model report saved as {report_path}")

# Print classification report for reference
print("\nClassification Report:")
print(class_report_text)
