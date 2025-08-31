#comment
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim
import torch.quantization
import torch.onnx

# Use MobileNetV2 as the model architecture
def get_mobilenetv2(num_classes=2):
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model

curriculum = [
    {
        'ai_dataset_path': 'data/easy/ai_images',
        'real_dataset_path': 'data/easy/real_images',
        'transform': transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])
    },
    {
        'ai_dataset_path': 'data/medium/ai_images',
        'real_dataset_path': 'data/medium/real_images',
        'transform': transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    },
    {
        'ai_dataset_path': 'data/hard/ai_images',
        'real_dataset_path': 'data/hard/real_images',
        'transform': transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor()
        ])
    }
]

def train(model, dataloader, criterion, optimizer, device, epoch=None, stage=None):
    model.train()
    total = len(dataloader)
    desc = f"Stage {stage+1} Epoch {epoch+1}" if stage is not None and epoch is not None else "Training"
    for images, labels in tqdm(dataloader, desc=desc, leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_mobilenetv2(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for stage, config in enumerate(curriculum):
        print(f"Stage {stage+1}: Training with curriculum dataset")
        ai_dataset = datasets.ImageFolder(config['ai_dataset_path'], transform=config['transform'])
        real_dataset = datasets.ImageFolder(config['real_dataset_path'], transform=config['transform'])
        # Combine datasets (label 0: real, label 1: AI)
        combined_dataset = torch.utils.data.ConcatDataset([real_dataset, ai_dataset])
        dataloader = DataLoader(combined_dataset, batch_size=32, shuffle=True)

        for epoch in range(3):  # epochs per curriculum stage
            train(model, dataloader, criterion, optimizer, device, epoch=epoch, stage=stage)
            print(f"Epoch {epoch+1} completed for stage {stage+1}")

    # Quantize the model for size reduction
    model.eval()
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )

    # Export to ONNX
    dummy_input = torch.randn(1, 3, 32, 32, device=device)
    onnx_path = 'quack_scan_model.onnx'
    torch.onnx.export(
        quantized_model,
        dummy_input,
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        opset_version=12,
        do_constant_folding=True
    )
    print(f"Quantized ONNX model saved to {onnx_path}")

if __name__ == "__main__":
    main()
