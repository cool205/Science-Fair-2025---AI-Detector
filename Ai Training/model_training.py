import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import os

print("Program Running")

# --- Hyperparameters ---
learning_rate = 0.0001
batch_size = 64
dropout_rate = 0.5
epochs = 30
optimizer_choice = "adam"

# --- Dataset Path ---
dataset_path = r'C:\Users\Hannah\OneDrive\Desktop\Science-Fair-2025---AI-Detector\AI Training\data\easy'

# --- Transforms ---
train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

# --- Model ---
def get_mobilenetv2(num_classes=2, dropout_rate=0.5):
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Sequential(
        nn.Linear(model.last_channel, 1280),
        nn.Dropout(dropout_rate),
        nn.ReLU(inplace=True),
        nn.Linear(1280, num_classes)
    )
    return model

# --- Logger ---
LOG_FILE = "accuracy_log.txt"

def train(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1} Training", leave=False):
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
    print(f"Epoch {epoch+1} Training Loss: {avg_loss:.4f}, Accuracy: {train_accuracy*100:.2f}%")
    return train_accuracy, avg_loss

def validate(model, dataloader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1} Validation", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_accuracy = correct / total
    avg_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch+1} Validation Loss: {avg_loss:.4f}, Accuracy: {val_accuracy*100:.2f}%")
    return val_accuracy, avg_loss

def log_accuracies(epoch, train_accuracy, val_accuracy):
    with open(LOG_FILE, 'a') as f:
        f.write(f"Epoch {epoch+1}, Train Accuracy: {train_accuracy*100:.2f}%, Validation Accuracy: {val_accuracy*100:.2f}%\n")
    print(f"Logged accuracies for epoch {epoch+1}.")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Dataset
    full_dataset = datasets.ImageFolder(dataset_path, transform=train_transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Use separate transform for validation
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = get_mobilenetv2(num_classes=2, dropout_rate=dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss()

    if optimizer_choice == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    with open(LOG_FILE, 'w') as f:
        f.write("Epoch, Train Accuracy, Validation Accuracy\n")

    for epoch in range(epochs):
        train_acc, train_loss = train(model, train_loader, criterion, optimizer, device, epoch)
        val_acc, val_loss = validate(model, val_loader, criterion, device, epoch)
        log_accuracies(epoch, train_acc, val_acc)
        scheduler.step()

    torch.save(model.state_dict(), "final_model.pth")
    print("Final model saved.")

    # Export to ONNX
    model.eval()
    dummy_input = torch.randn(1, 3, 32, 32)
    torch.onnx.export(
        model,
        dummy_input,
        "final_model.onnx",
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("Model exported to final_model.onnx.")

if __name__ == "__main__":
    main()
