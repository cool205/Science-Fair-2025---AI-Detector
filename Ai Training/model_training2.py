import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import os

"""Single-run training script based on model_training.py
Hyperparameters are set from the requested list (inferred mapping):
  epochs=29, dropout=0.6, batch_size=64, learning_rate=0.001,
  color_jitter=0.2, num_workers=17
This script trains once and saves the final model to disk.
"""

print("Program Running: single run")

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

# Inferred hyperparameters (from user input)
epochs = 29
dropout_rate = 0.6
batch_size = 64
learning_rate = 0.001
color_jitter_strength = 0.2
num_workers = 17

LOG_FILE = "accuracy_log_run2.txt"
DATASET_PATH = r'AI Training/data/train'
MODEL_OUT_PATH = 'model_training2_final.pth'

# Transforms
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

def make_train_transform(color_jitter_strength=0.2):
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=color_jitter_strength, contrast=color_jitter_strength, saturation=color_jitter_strength, hue=0.1),
        transforms.ToTensor(),
        normalize
    ])

val_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    normalize
])

# Training/validation loops (same as model_training.py)
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

    accuracy = correct / total if total > 0 else 0.0
    avg_loss = running_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    print(f"Epoch {epoch+1} Training Loss: {avg_loss:.4f}, Accuracy: {accuracy*100:.2f}%")
    return accuracy, avg_loss

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

    accuracy = correct / total if total > 0 else 0.0
    avg_loss = running_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    print(f"Epoch {epoch+1} Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy*100:.2f}%")
    return accuracy, avg_loss

def log_accuracies(epoch, train_acc, val_acc):
    with open(LOG_FILE, 'a') as f:
        f.write(f"Epoch {epoch+1}, Train Accuracy: {train_acc*100:.2f}%, Validation Accuracy: {val_acc*100:.2f}%\n")
    print(f"Logged accuracies for epoch {epoch+1}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # Prepare dataset
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset path not found: {DATASET_PATH}. Exiting.")
        return

    base_dataset = datasets.ImageFolder(DATASET_PATH)
    targets = [label for _, label in base_dataset.imgs]

    # Single run transforms/datasets
    train_transform = make_train_transform(color_jitter_strength=color_jitter_strength)
    full_dataset = datasets.ImageFolder(DATASET_PATH, transform=train_transform)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(full_dataset.imgs, targets))
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=min(num_workers, os.cpu_count() or 0))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=min(num_workers, os.cpu_count() or 0))

    # Model, loss, optimizer
    model = get_mobilenetv2(num_classes=2, dropout_rate=dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # Prepare log
    with open(LOG_FILE, 'w') as f:
        f.write('epoch,train_acc,val_acc\n')

    # Run training
    for epoch in range(epochs):
        train_acc, train_loss = train(model, train_loader, criterion, optimizer, device, epoch)
        val_acc, val_loss = validate(model, val_loader, criterion, device, epoch)
        log_accuracies(epoch, train_acc, val_acc)
        with open(LOG_FILE, 'a') as f:
            f.write(f"{epoch+1},{train_acc:.4f},{val_acc:.4f}\n")
        scheduler.step(val_loss)

    # Save final model
    torch.save({'model_state_dict': model.state_dict(), 'hyperparams': {
        'epochs': epochs,
        'dropout': dropout_rate,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'color_jitter': color_jitter_strength,
        'num_workers': num_workers
    }}, MODEL_OUT_PATH)
    print(f"Model saved to {MODEL_OUT_PATH}")

    # Optionally export to ONNX (attempt but don't fail if unsupported)
    try:
        dummy_input = torch.randn(1, 3, 32, 32, device=device)
        onnx_path = 'model_training2_final.onnx'
        torch.onnx.export(model, dummy_input, onnx_path, input_names=['input'], output_names=['output'], opset_version=12)
        print(f"ONNX model exported to {onnx_path}")
    except Exception as e:
        print("ONNX export failed:", e)


if __name__ == '__main__':
    main()
