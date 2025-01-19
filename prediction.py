import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp

# Load the trained model
MODEL_PATH = "gesture_recognition_model2.h5"
model = load_model(MODEL_PATH)
print(f"Model loaded from '{MODEL_PATH}'")

# Parameters
sequence_length = 30  # Number of frames per sequence
features_per_frame = 225  # Number of features per frame
actions = ['Days', 'Monday', 'Sunday', 'Week']  # Gesture labels

# Initialize Mediapipe Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize a buffer to store a sequence of frames
sequence_buffer = []

# Function to extract features using Mediapipe
def extract_features(image):
    """
    Extract pose, left hand, and right hand landmarks using Mediapipe.
    Returns a flat feature vector of shape (features_per_frame,).
    """
    results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    features = []

    # Pose landmarks
    if results.pose_landmarks:
        features.extend([landmark.x for landmark in results.pose_landmarks.landmark])
        features.extend([landmark.y for landmark in results.pose_landmarks.landmark])
        features.extend([landmark.z for landmark in results.pose_landmarks.landmark])
    else:
        features.extend([0] * 99)  # 33 landmarks x 3 coordinates

    # Left hand landmarks
    if results.left_hand_landmarks:
        features.extend([landmark.x for landmark in results.left_hand_landmarks.landmark])
        features.extend([landmark.y for landmark in results.left_hand_landmarks.landmark])
        features.extend([landmark.z for landmark in results.left_hand_landmarks.landmark])
    else:
        features.extend([0] * 63)  # 21 landmarks x 3 coordinates

    # Right hand landmarks
    if results.right_hand_landmarks:
        features.extend([landmark.x for landmark in results.right_hand_landmarks.landmark])
        features.extend([landmark.y for landmark in results.right_hand_landmarks.landmark])
        features.extend([landmark.z for landmark in results.right_hand_landmarks.landmark])
    else:
        features.extend([0] * 63)  # 21 landmarks x 3 coordinates

    return np.array(features) if len(features) == features_per_frame else np.zeros(features_per_frame)

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to access the camera.")
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Extract features from the current frame
    features = extract_features(frame)
    if features is not None:
        sequence_buffer.append(features)

    # Ensure buffer size doesn't exceed the sequence length
    if len(sequence_buffer) > sequence_length:
        sequence_buffer.pop(0)

    # If the buffer is full, make a prediction
    if len(sequence_buffer) == sequence_length:
        input_sequence = np.array(sequence_buffer).reshape(1, sequence_length, features_per_frame)
        predictions = model.predict(input_sequence, verbose=0)
        predicted_label = np.argmax(predictions)
        confidence = np.max(predictions)

        # Display the predicted gesture
        cv2.putText(frame, f"Gesture: {actions[predicted_label]} ({confidence:.2f})",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the video feed
    cv2.imshow("Gesture Recognition", frame)

    # Break loop with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
holistic.close()
