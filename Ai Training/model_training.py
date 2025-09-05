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
def get_mobilenetv2(num_classes=2, dropout_rate=0.5):
    model = models.mobilenet_v2(weights=None)
    # Add dropout after the classifier layer
    model.classifier[1] = nn.Sequential(
        nn.Linear(model.last_channel, 1280),  # Original output size
        nn.Dropout(dropout_rate),  # Add dropout layer with configurable rate
        nn.ReLU(inplace=True),
        nn.Linear(1280, num_classes)  # Final classification layer
    )
    return model

# Hyperparameter Tuning Variables
learning_rate = 0.001
batch_size = 32
optimizer_choice = "adam"  # Options: "adam", "sgd"
dropout_rate = 0.5  # Dropout rate (can be adjusted)

# Curriculum configuration
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
    print(f"Checkpoint saved at epoch {epoch}")

def train(model, dataloader, criterion, optimizer, device, epoch=None, stage=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    desc = f"Stage {stage+1} Epoch {epoch+1}" if stage is not None and epoch is not None else "Training"
    for images, labels in tqdm(dataloader, desc=desc, leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_accuracy = correct / total
    avg_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch+1} Training Loss: {avg_loss:.4f}, Training Accuracy: {train_accuracy*100:.2f}%")
    return train_accuracy, avg_loss

def validate(model, dataloader, criterion, device, epoch=None, stage=None):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"Validation Stage {stage+1} Epoch {epoch+1}", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = correct / total
    avg_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch+1} Validation Loss: {avg_loss:.4f}, Validation Accuracy: {val_accuracy*100:.2f}%")
    return val_accuracy, avg_loss

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_mobilenetv2(num_classes=2, dropout_rate=dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss()

    if optimizer_choice == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_choice == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_choice}")

    for stage, config in enumerate(curriculum):
        print(f"Stage {stage+1}: Training with curriculum dataset")
        dataset = datasets.ImageFolder(config['dataset_path'], transform=config['transform'])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(5):  # 5 epochs per stage
            print(f"Training epoch {epoch+1}")
            train_accuracy, train_loss = train(model, dataloader, criterion, optimizer, device, epoch=epoch, stage=stage)
            val_accuracy, val_loss = validate(model, dataloader, criterion, device, epoch=epoch, stage=stage)

            # Save checkpoint after each epoch
            save_checkpoint(epoch, model, optimizer)

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
