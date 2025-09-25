from datasets import load_dataset

# Set your custom dataset storage folder
dataset_folder = "C:/data2"

# Load the dataset into that folder
ds = load_dataset("Hemg/AI-Generated-vs-Real-Images-Datasets", cache_dir=dataset_folder)

# Optional: print out dataset structure
print(ds)
