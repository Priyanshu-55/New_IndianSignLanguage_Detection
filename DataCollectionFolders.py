import os
import numpy as np
import shutil

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('WeekDays')

# Actions that we try to detect
actions = np.array(["Wednesday"])

# Thirty videos worth of data
no_sequences = 180

# Videos are going to be 30 frames in length(1 sec)
sequence_length = 30

# Folder start
start_folder = 1

# Ensure the main DATA_PATH directory exists
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

# Create or update folders for each action
for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    
    if os.path.exists(action_path):
        # If the action folder already exists, delete and recreate it
        shutil.rmtree(action_path)
        os.makedirs(action_path)
    else:
        # If it's a new action, just create the folder
        os.makedirs(action_path)
    
    # Create subdirectories for sequences
    for sequence in range(1, no_sequences + 1):
        try:
            os.makedirs(os.path.join(action_path, str(sequence)))
        except Exception as e:
            print(f"Could not create directory: {e}")
