import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim
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
batch_size = 64
optimizer_choice = "adam"  # Options: "adam", "sgd"
dropout_rate = 0.6  # Dropout rate (can be adjusted)
epochs_per_stage = 5  # Number of epochs per curriculum stage

# Curriculum configuration
curriculum = [
    {
        'dataset_path': r'C:\Users\Hannah\OneDrive\Desktop\Science-Fair-2025---AI-Detector\AI Training\data\easy',
        'train_transform': transforms.Compose([  # Augmentations for training
            transforms.Resize((32, 32)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.0),
            transforms.ToTensor()
        ]),
        'val_transform': transforms.Compose([  # No augmentations for validation
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ]) 
    },
    {
        'dataset_path': r'C:\Users\Hannah\OneDrive\Desktop\Science-Fair-2025---AI-Detector\AI Training\data\medium',
        'train_transform': transforms.Compose([  # Augmentations for training
            transforms.Resize((32, 32)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]),
        'val_transform': transforms.Compose([  # No augmentations for validation
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])  
    },
    {
        'dataset_path': r'C:\Users\Hannah\OneDrive\Desktop\Science-Fair-2025---AI-Detector\AI Training\data\hard',
        'train_transform': transforms.Compose([  # Augmentations for training
            transforms.Resize((32, 32)),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor()
        ]),
        'val_transform': transforms.Compose([  # No augmentations for validation
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

    # Log header to file
    with open(LOG_FILE, 'w') as f:
        f.write("Epoch, Train Accuracy, Validation Accuracy\n")

    for stage, config in enumerate(curriculum):
        dataset = datasets.ImageFolder(config['dataset_path'], transform=config['train_transform'])
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        for epoch in range(epochs_per_stage):
            train_accuracy, train_loss = train(model, train_dataloader, criterion, optimizer, device, epoch=epoch, stage=stage)
            val_accuracy, val_loss = validate(model, val_dataloader, criterion, device, epoch=epoch, stage=stage)
            log_accuracies(epoch, train_accuracy, val_accuracy)

        if stage == len(curriculum) - 1 and epoch == 4:
            torch.save(model.state_dict(), "final_model.pth")
            print(f"Model saved after final epoch: Epoch {epoch+1}")

    # Export non-quantized model to ONNX
    dummy_input = torch.randn(1, 3, 32, 32)
    model.eval()
    model.cpu()

    torch.onnx.export(
        model,
        dummy_input,
        "model_fp32.onnx",
        export_params=True,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        do_constant_folding=True
    )

    print("FP32 model exported to ONNX.")

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
