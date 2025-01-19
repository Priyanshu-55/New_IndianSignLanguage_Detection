import numpy as np
import cv2
import os
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands
# mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize models
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
# face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

actions = np.array(["Days"])
start_folder = 1
no_sequences = 180
sequence_length = 30
DATA_PATH = os.path.join('WeekDays')

# Resolution 1600 x 1200

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
    if holistic_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            holistic_results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

def extract_keypoints(hand_results, holistic_results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in holistic_results.pose_landmarks.landmark]).flatten() if holistic_results.pose_landmarks else np.zeros(33*4)
    
    lh = np.zeros(21*3)
    rh = np.zeros(21*3)
    
    if hand_results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
            hand_array = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
            if handedness.classification[0].label == "Left":
                lh = hand_array
            else:
                rh = hand_array
    
    return np.concatenate([pose, lh, rh])

# Initialize Video Capture
cap = cv2.VideoCapture(0)
cv2.namedWindow('OpenCV Feed', cv2.WINDOW_NORMAL)

# Loop through actions
for action in actions:
    # Loop through sequences aka videos
    for sequence in range(start_folder, start_folder + no_sequences):
        # Ensure directories exist
        sequence_dir = os.path.join(DATA_PATH, action, str(sequence))
        if not os.path.exists(sequence_dir):
            os.makedirs(sequence_dir)
        
        # Loop through video length aka sequence length
        for frame_num in range(sequence_length):
            # Read feed
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from camera")
                break

            frame = cv2.flip(frame, 1)  # Flip the frame horizontally
            image, hand_results, holistic_results = mediapipe_detection(frame)
            draw_landmarks(image, hand_results, holistic_results)

            # Apply wait logic
            if frame_num == 0: 
                cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15,12), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', image)
                cv2.waitKey(2000)
            else: 
                cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15,12), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', image)

            # Export keypoints
            keypoints = extract_keypoints(hand_results, holistic_results)
            npy_path = os.path.join(sequence_dir, f"{frame_num}.npy")
            np.save(npy_path, keypoints)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()