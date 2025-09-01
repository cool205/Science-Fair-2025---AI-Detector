
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim
import torch.quantization
import torch.onnx

import os

print("Program Running")

# Use MobileNetV2 as the model architecture
def get_mobilenetv2(num_classes=2):
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model

curriculum = [
    {
        'dataset_path': r'C:\Users\Hannah\OneDrive\Desktop\Science-Fair-2025---AI-Detector\AI Training\data\easy',
        'transform': transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])
    },
    {
        'dataset_path': r'C:\Users\Hannah\OneDrive\Desktop\Science-Fair-2025---AI-Detector\AI Training\data\medium',
        'transform': transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    },
    {
        'dataset_path': r'C:\Users\Hannah\OneDrive\Desktop\Science-Fair-2025---AI-Detector\AI Training\data\hard',
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


CHECKPOINT_FN = "model_checkpoint.pth"

def save_checkpoint(epoch, model, optimizer, scheduler=None, scaler=None, filename=CHECKPOINT_FN):
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "opt_state": optimizer.state_dict()
    }
    if scheduler is not None:
        checkpoint["sched_state"] = scheduler.state_dict()
    if scaler is not None:
        checkpoint["scaler_state"] = scaler.state_dict()
    torch.save(checkpoint, filename)

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
        dataset = datasets.ImageFolder(config['dataset_path'], transform=config['transform'])
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


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
