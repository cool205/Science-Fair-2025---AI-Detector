from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

# Define the image transformation with the requested transforms
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize to 32x32
    transforms.RandomRotation(30),  # Randomly rotate the image by 30 degrees
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),  # Adjust brightness, contrast, saturation, and hue
    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
    transforms.RandomVerticalFlip(),  # Randomly flip the image vertically
    transforms.ToTensor()  # Convert the image to a tensor
])

# Load an image manually (replace with actual image path)
image_path = r'C:\Users\Hannah\OneDrive\Desktop\Science-Fair-2025---AI-Detector\AI Training\data\easy\ai_images\1009 (5).jpg'  # Adjust the path
img = Image.open(image_path)

# Apply the transformation to the image
transformed_img = transform(img)

# Plot the original and transformed image
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Original Image (before transformation)
ax[0].imshow(img)
ax[0].set_title("Original Image")
ax[0].axis('off')  # Hide axes for better display

# Transformed Image (after transformation)
# Convert tensor to numpy for visualization
transformed_img = transformed_img.permute(1, 2, 0).numpy()  # Change the dimensions from (C, H, W) to (H, W, C)
ax[1].imshow(transformed_img)
ax[1].set_title("Transformed Image")
ax[1].axis('off')  # Hide axes for better display
plt.show()