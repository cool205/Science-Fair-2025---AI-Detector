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
LOG_FILE = "accuracy_log.txt"  # File to save training and validation accuracies

def get_model_size(model, model_filename="model.pth"):
    torch.save(model, model_filename)  # Save the entire model
    model_size = os.path.getsize(model_filename) / (1024 ** 2)  # MB
    print(f"Model size: {model_size:.2f} MB")

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
    print(f"Checkpoint saved at epoch {epoch+1}")

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

def log_accuracies(epoch, train_accuracy, val_accuracy, log_file=LOG_FILE):
    with open(log_file, 'a') as f:
        f.write(f"Epoch {epoch+1}, Train Accuracy: {train_accuracy*100:.2f}%, Validation Accuracy: {val_accuracy*100:.2f}%\n")
    print(f"Accuracies logged for epoch {epoch+1}.")

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

    # Load checkpoint if exists
    if os.path.exists(CHECKPOINT_FN):
        print(f"Loading checkpoint from {CHECKPOINT_FN}")
        checkpoint = torch.load(CHECKPOINT_FN)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["opt_state"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming from epoch {start_epoch}")
    else:
        start_epoch = 0
        print("No checkpoint found, starting from scratch.")

    # Log header to file
    with open(LOG_FILE, 'w') as f:
        f.write("Epoch, Train Accuracy, Validation Accuracy\n")

    for stage, config in enumerate(curriculum):
        dataset = datasets.ImageFolder(config['dataset_path'], transform=config['transform'])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(start_epoch, 15):  # 15 epochs per stage
            train_accuracy, train_loss = train(model, dataloader, criterion, optimizer, device, epoch=epoch, stage=stage)
            val_accuracy, val_loss = validate(model, dataloader, criterion, device, epoch=epoch, stage=stage)

            # Log accuracies to file
            log_accuracies(epoch, train_accuracy, val_accuracy)

            # Save checkpoint after each epoch
            save_checkpoint(epoch, model, optimizer)
            get_model_size(model, model_filename=CHECKPOINT_FN)  # Model size after checkpoint

    # Quantize the model for size reduction
    dummy_input = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 channels, 32x32 image size
    model.eval()
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )

    # Ensure the model is on the CPU before quantization
    device = torch.device('cpu')
    model.to(device)

    # Now apply the quantization (example for dynamic quantization)
    quantized_model = torch.quantization.quantize_dynamic(
        model,  # Your model
        {torch.nn.Linear},  # Layers to quantize (e.g., Linear layers)
        dtype=torch.qint8  # The dtype for quantization
    )

    # Export the quantized model to ONNX
    torch.onnx.export(
        quantized_model,
        dummy_input,  # Example input
        'quantized_model.onnx',
        export_params=True,
        opset_version=12,
        do_constant_folding=True
    )

if __name__ == "__main__":
    main()
