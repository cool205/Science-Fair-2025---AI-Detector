import os
import shutil
import pandas as pd

# Paths
csv_path = 'path_to_labels.csv'
source_dir = 'path_to_extracted/test_data_v2'
target_dir = 'Ai Training/data/test'

# Create target folders
os.makedirs(f'{target_dir}/FAKE', exist_ok=True)
os.makedirs(f'{target_dir}/REAL', exist_ok=True)

# Load labels
df = pd.read_csv(csv_path)

# Loop through each row in the CSV
for _, row in df.iterrows():
    filename = row['filename']
    label = row['label']

    # Determine destination folder
    if label.lower() == 'ai':
        dest_folder = 'FAKE'
    elif label.lower() == 'human':
        dest_folder = 'REAL'
    else:
        continue  # Skip unknown labels

    # Move the file
    src_path = os.path.join(source_dir, filename)
    dst_path = os.path.join(target_dir, dest_folder, filename)

    if os.path.exists(src_path):
        shutil.move(src_path, dst_path)
