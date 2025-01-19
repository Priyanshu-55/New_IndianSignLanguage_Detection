import os
import numpy as np

# Paths to original and new datasets
DATASET_PATH = "Dataset"
NEW_DATASET_PATH = "NewDataset"

# Ensure the new dataset folder exists
if not os.path.exists(NEW_DATASET_PATH):
    os.makedirs(NEW_DATASET_PATH)

# Process each gesture folder in the dataset
for gesture_folder in os.listdir(DATASET_PATH):
    gesture_path = os.path.join(DATASET_PATH, gesture_folder)
    
    # Skip if not a folder
    if not os.path.isdir(gesture_path):
        print(f"Skipping '{gesture_folder}', not a folder.")
        continue
    
    # Create corresponding folder in NewDataset
    new_gesture_path = os.path.join(NEW_DATASET_PATH, gesture_folder)
    if not os.path.exists(new_gesture_path):
        os.makedirs(new_gesture_path)
    
    print(f"Processing folder: {gesture_folder}")
    
    # Process each subfolder (e.g., 1, 2, 3, ...) in the gesture folder
    for subfolder in os.listdir(gesture_path):
        subfolder_path = os.path.join(gesture_path, subfolder)
        
        # Skip if not a folder
        if not os.path.isdir(subfolder_path):
            print(f"Skipping '{subfolder}', not a folder.")
            continue
        
        # Create corresponding subfolder in NewDataset
        new_subfolder_path = os.path.join(new_gesture_path, subfolder)
        if not os.path.exists(new_subfolder_path):
            os.makedirs(new_subfolder_path)
        
        print(f"Processing subfolder: {subfolder}")
        
        # Process each .npy file in the subfolder
        for npy_file in os.listdir(subfolder_path):
            print(f"Found file: '{npy_file}'")  # Debug: Print filenames
            
            # Ensure file ends with '.npy' extension
            if not npy_file.endswith('.npy'):
                print(f"Skipping non-npy file: {npy_file}")
                continue

            npy_path = os.path.join(subfolder_path, npy_file)
            print(f"Processing file: {npy_path}")
            
            try:
                # Load landmarks
                landmarks = np.load(npy_path)
                print(f"Loaded landmarks shape: {landmarks.shape}")
            except Exception as e:
                print(f"Error loading file '{npy_file}': {e}")
                continue
            
            # Process pose landmarks: Remove every 4th value (visibility)
            pose_landmarks = landmarks[:33 * 4].reshape(-1, 4)  # Extract pose with visibility
            pose_landmarks_no_visibility = pose_landmarks[:, :3].flatten()  # Remove visibility
            
            # Keep hand landmarks unchanged
            left_hand_landmarks = landmarks[33 * 4:33 * 4 + 21 * 3]  # Left hand
            right_hand_landmarks = landmarks[33 * 4 + 21 * 3:]  # Right hand
            
            # Combine processed landmarks
            new_landmarks = np.concatenate([pose_landmarks_no_visibility, left_hand_landmarks, right_hand_landmarks])
            
            # Save to NewDataset
            new_npy_path = os.path.join(new_subfolder_path, npy_file)
            np.save(new_npy_path, new_landmarks)
            print(f"Saved updated landmarks to '{new_npy_path}'")

print("Visibility removed and updated dataset saved in 'NewDataset' folder.")
