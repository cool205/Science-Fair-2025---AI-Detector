import os

def count_images_in_folders(root_dir):
    folder_counts = {}
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        if os.path.isdir(folder_path):
            image_count = sum(
                1 for file in os.listdir(folder_path)
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
            )
            folder_counts[folder_name] = image_count
    return folder_counts

# Example usage
DATASET_PATH = r'AI Training/data/train'  # Change this to your dataset path
counts = count_images_in_folders(DATASET_PATH)
for folder, count in counts.items():
    print(f"{folder}: {count} images")
