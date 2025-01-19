# import sys
# import os

# # Add the path to your virtual environment's site-packages
# venv_path = os.path.join(os.path.dirname(__file__), '.venv', 'Lib', 'site-packages')
# sys.path.append(venv_path)

import cv2
import numpy as np
import mediapipe as mp

# Initialize Mediapipe models
mp_hands = mp.solutions.hands
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize hand tracking and holistic
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def mediapipe_detection(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(image_rgb)
    holistic_results = holistic.process(image_rgb)
    return image, hand_results, holistic_results

def draw_landmarks(image, hand_results, holistic_results):
    # Draw hand landmarks
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
    
    # Draw pose landmarks
    mp_drawing.draw_landmarks(
        image,
        holistic_results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
    )
    
    # Draw face mesh
    mp_drawing.draw_landmarks(
        image,
        holistic_results.face_landmarks,
        mp_holistic.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
    )

cap = cv2.VideoCapture(0)  # Use appropriate camera index
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1600)
cv2.namedWindow('Holistic Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Holistic Detection', 1200, 1600)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Resize the frame to 1200x1600
    frame = cv2.resize(frame, (1200, 1600))
    
    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)
    
    # Detection
    image, hand_results, holistic_results = mediapipe_detection(frame)
    draw_landmarks(image, hand_results, holistic_results)

    # Show the image with landmarks
    cv2.imshow('Holistic Detection', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
