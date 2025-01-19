import os
import numpy as np
import json

# Paths to dataset
DATASET_PATH = "NewDataset"

# Function to load and inspect the data
def inspect_data():
    insights = {}  # Dictionary to store insights

    X = []  # Features
    y = []  # Labels

    # Load all files in the dataset
    for gesture_folder in os.listdir(DATASET_PATH):
        gesture_path = os.path.join(DATASET_PATH, gesture_folder)

        if not os.path.isdir(gesture_path):
            continue
        
        for subfolder in os.listdir(gesture_path):
            subfolder_path = os.path.join(gesture_path, subfolder)

            if not os.path.isdir(subfolder_path):
                continue

            for npy_file in os.listdir(subfolder_path):
                if npy_file.endswith('.npy'):
                    npy_path = os.path.join(subfolder_path, npy_file)
                    landmarks = np.load(npy_path)
                    X.append(landmarks)
                    y.append(gesture_folder)  # Folder name is the label
    
    # Convert lists to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Inspect the shape and content of X
    insights["X_shape"] = X.shape
    insights["num_samples"] = X.shape[0]
    insights["features_per_sample"] = X.shape[1]


    # Check if data is consistent with desired features (e.g., 33*3 + 21*3 + 21*3 = 225)
    num_features = X.shape[1]
    insights["features_per_sample_value"] = num_features

    # Check the total number of elements
    total_elements = X.size
    insights["total_elements"] = total_elements

    # Display the shape of the first sample to understand how data is structured
    insights["first_sample_shape"] = X[0].shape

    # Checking how many frames and features are in the data
    frames_per_sequence = 30  # assuming you want 30 frames per sequence
    features_per_frame = 225  # total features per frame (33*3 + 21*3 + 21*3)
    
    # Ensure that reshaping is possible
    if total_elements % (frames_per_sequence * features_per_frame) == 0:
        insights["reshaping_feasible"] = True
    else:
        insights["reshaping_feasible"] = False

    return insights

# Run the inspection function
insights = inspect_data()

# Save insights as JSON
insights_file_path = "dataset_insights.json"
with open(insights_file_path, "w") as json_file:
    json.dump(insights, json_file, indent=4)

print(f"Dataset insights saved to {insights_file_path}")
