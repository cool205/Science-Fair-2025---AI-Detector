import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import os

print("Program Running")

# Model setup
def get_mobilenetv2(num_classes=2, dropout_rate=0.6):
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Sequential(
        nn.Linear(model.last_channel, 1280),
        nn.Dropout(dropout_rate),
        nn.ReLU(inplace=True),
        nn.Linear(1280, num_classes)
    )
    return model

# Hyperparameters
learning_rate = 0.0001
batch_size = 64
dropout_rate = 0.6
epochs = 20
LOG_FILE = "accuracy_log.txt"
DATASET_PATH = r'data/train'

# Transforms
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    normalize
])

val_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    normalize
])

# Training loop
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

    accuracy = correct / total
    avg_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch+1} Training Loss: {avg_loss:.4f}, Accuracy: {accuracy*100:.2f}%")
    return accuracy, avg_loss

# Validation loop
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

    accuracy = correct / total
    avg_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch+1} Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy*100:.2f}%")
    return accuracy, avg_loss

# Logging
def log_accuracies(epoch, train_acc, val_acc):
    with open(LOG_FILE, 'a') as f:
        f.write(f"Epoch {epoch+1}, Train Accuracy: {train_acc*100:.2f}%, Validation Accuracy: {val_acc*100:.2f}%\n")
    print(f"Logged accuracies for epoch {epoch+1}")

# Main training pipeline
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    model = get_mobilenetv2(num_classes=2, dropout_rate=dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # Load dataset
    full_dataset = datasets.ImageFolder(DATASET_PATH, transform=train_transform)
    targets = [label for _, label in full_dataset.imgs]

    # Stratified split
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(full_dataset.imgs, targets))
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    with open(LOG_FILE, 'w') as f:
        f.write("Epoch, Train Accuracy, Validation Accuracy\n")

    for epoch in range(epochs):
        train_acc, train_loss = train(model, train_loader, criterion, optimizer, device, epoch)
        val_acc, val_loss = validate(model, val_loader, criterion, device, epoch)
        log_accuracies(epoch, train_acc, val_acc)
        scheduler.step(val_loss)

    # Fine-tune classifier
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate / 10)
    for epoch in range(5):
        train_acc, _ = train(model, train_loader, criterion, optimizer, device, epoch)
        val_acc, _ = validate(model, val_loader, criterion, device, epoch)

    torch.save(model.state_dict(), "final_model.pth")
    print("Model saved to final_model.pth")

if __name__ == "__main__":
    main()
