
import torch
import onnxruntime as ort
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
from model import curriculum

# Load ONNX model
onnx_path = 'ai_detector_model_quantized.onnx'
ort_session = ort.InferenceSession(onnx_path)

# Function to test and visualize a sample image for a given curriculum stage
def test_and_visualize(image_path, stage=0):
    transform = curriculum[stage]['transform']
    img = Image.open(image_path).convert('RGB')
    transformed_img = transform(img)
    # Show transformed image
    plt.imshow(transformed_img.permute(1, 2, 0))
    plt.title(f'Transformed Image (Stage {stage+1})')
    plt.axis('off')
    plt.show()
    # Model expects batch dimension
    input_tensor = transformed_img.unsqueeze(0).numpy()
    # ONNX expects float32
    input_tensor = input_tensor.astype('float32')
    output = ort_session.run(None, {'input': input_tensor})[0]
    probs = torch.softmax(torch.from_numpy(output), dim=1).squeeze().cpu().numpy()
    pred = int(probs.argmax())
    print(f'Prediction: {"AI" if pred == 1 else "Real"}')
    print(f'Confidence (Real): {probs[0]*100:.2f}%')
    print(f'Confidence (AI): {probs[1]*100:.2f}%')

if __name__ == "__main__":
    # Example usage: test one image from each stage
    sample_images = [
        'data/easy/ai_images/sample.jpg',
        'data/medium/ai_images/sample.jpg',
        'data/hard/ai_images/sample.jpg'
    ]
    for i, img_path in enumerate(sample_images):
        if os.path.exists(img_path):
            print(f'--- Stage {i+1} ---')
            test_and_visualize(img_path, stage=i)
        else:
            print(f'Image not found: {img_path}')
