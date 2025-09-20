
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np

print("Program Running")

# Use MobileNetV2 as the model architecture with adjustments for overfitting prevention
def get_mobilenetv2(num_classes=2, dropout_rate=0.5):  # Increased dropout rate for regularization
    model = models.mobilenet_v2(weights=None)
    # Add dropout after the classifier layer
    model.classifier[1] = nn.Sequential(
        nn.Linear(model.last_channel, 1280),
        nn.Dropout(dropout_rate),
        nn.ReLU(inplace=True),
        nn.Linear(1280, num_classes)
    )
    return model

# Hyperparameter Tuning Variables
learning_rate = 0.001
batch_size = 64
optimizer_choice = "adam"  # Options: "adam", "sgd"
dropout_rate = 0.5  # Increased dropout rate
epochs_per_stage = 15

# Curriculum configuration (with data augmentation changes)
curriculum = [
    {
        'dataset_path': r'C:\Users\Hannah\OneDrive\Desktop\Science-Fair-2025---AI-Detector\AI Training\data\easy',
        'train_transform': transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),  # Added horizontal flip for data augmentation
            transforms.RandomRotation(10),  # Added small rotation to make the data more robust
            transforms.ToTensor()
        ]),
        'val_transform': transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])
    },
    {
        'dataset_path': r'C:\Users\Hannah\OneDrive\Desktop\Science-Fair-2025---AI-Detector\AI Training\data\medium',
        'train_transform': transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),  # Increased rotation range for better augmentation
            transforms.ToTensor()
        ]),
        'val_transform': transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])  
    },
    {
        'dataset_path': r'C:\Users\Hannah\OneDrive\Desktop\Science-Fair-2025---AI-Detector\AI Training\data\hard',
        'train_transform': transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomRotation(20),  # Moderate rotation range for more generalization
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),  # Added vertical flip
            transforms.ToTensor()
        ]),
        'val_transform': transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])  
    }
]

LOG_FILE = "accuracy_log.txt"  # File to save training and validation accuracies

def get_model_size(model, model_filename="model.pth"):
    torch.save(model.state_dict(), model_filename)  # Save model state dict
    model_size = os.path.getsize(model_filename) / (1024 ** 2)  # MB
    print(f"Model size: {model_size:.2f} MB")

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

# Early stopping
def early_stopping(val_accuracy, best_val_accuracy, patience, counter):
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        counter = 0  # Reset counter when improvement occurs
    else:
        counter += 1
    if counter >= patience:
        print("Early stopping triggered")
        return True, best_val_accuracy, counter
    return False, best_val_accuracy, counter

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_mobilenetv2(num_classes=2, dropout_rate=dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss()

    if optimizer_choice == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)  # Added weight decay
    elif optimizer_choice == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)  # Added weight decay
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_choice}")

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # Learning rate scheduler

    # Log header to file
    with open(LOG_FILE, 'w') as f:
        f.write("Epoch, Train Accuracy, Validation Accuracy\n")

    best_val_accuracy = 0.0
    patience = 3  # Early stopping patience
    counter = 0

    for stage, config in enumerate(curriculum):
        # Load the dataset and split into train and validation
        dataset = datasets.ImageFolder(config['dataset_path'], transform=config['train_transform'])
        train_size = int(0.8 * len(dataset))  # 80% for training
        val_size = len(dataset) - train_size  # 20% for validation
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        for epoch in range(epochs_per_stage):  # 7 epochs per stage
            train_accuracy, train_loss = train(model, train_dataloader, criterion, optimizer, device, epoch=epoch, stage=stage)
            val_accuracy, val_loss = validate(model, val_dataloader, criterion, device, epoch=epoch, stage=stage)

            # Log accuracies to file
            log_accuracies(epoch, train_accuracy, val_accuracy)

            # Early stopping check
            stop, best_val_accuracy, counter = early_stopping(val_accuracy, best_val_accuracy, patience, counter)
            if stop:
                break

            # Step the learning rate scheduler
            scheduler.step()

        # Optional: Save the model after each stage (if needed)
        torch.save(model.state_dict(), f"model_stage_{stage+1}.pth")

    # Fine-tune the model after curriculum learning if needed
    for param in model.parameters():
        param.requires_grad = False  # Freeze all layers
    for param in model.classifier.parameters():  # Only train the classifier
        param.requires_grad = True

    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate / 10)  # Lower learning rate for fine-tuning
    for epoch in range(5):  # Fine-tune for a few more epochs
        train_accuracy, train_loss = train(model, train_dataloader, criterion, optimizer, device, epoch=epoch)
        val_accuracy, val_loss = validate(model, val_dataloader, criterion, device, epoch=epoch)

    # Save the final model
    torch.save(model.state_dict(), "final_model.pth")
    print("Model saved after fine-tuning.")

    # Quantize the model for size reduction
    dummy_input = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 channels, 32x32 image size
    model.eval()
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    
    # Fine-tune after quantization
    optimizer = optim.Adam(quantized_model.parameters(), lr=learning_rate / 10)  # Lower learning rate for fine-tuning
    for epoch in range(5):  # Fine-tune the quantized model for a few epochs
        train_accuracy, train_loss = train(quantized_model, train_dataloader, criterion, optimizer, device, epoch=epoch)
        val_accuracy, val_loss = validate(quantized_model, val_dataloader, criterion, device, epoch=epoch)

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

