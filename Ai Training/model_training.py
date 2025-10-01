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
# Defaults (will be overridden by grid search)
learning_rate = 0.0001
batch_size = 64
dropout_rate = 0.6
epochs = 10
LOG_FILE = "accuracy_log.txt"
LOG_FILE_2 = "accuracy_log2.txt"
DATASET_PATH = r'data/train'

# Transforms
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

def make_train_transform(color_jitter_strength=0.2):
    # color_jitter_strength controls brightness/contrast/saturation and hue small value
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
    # Grid search settings
    dropout_rates = [0.5, 0.6, 0.7, 0.8]
    batch_sizes = [32, 64]
    learning_rates = [1e-3, 1e-4, 1e-5]
    color_jitters = [0.1, 0.2, 0.3]

    # Prepare dataset once (we will recreate dataset objects with different transforms per run)
    base_dataset = datasets.ImageFolder(DATASET_PATH)
    targets = [label for _, label in base_dataset.imgs]

    # Prepare log files and write headers
    with open(LOG_FILE, 'w') as f:
        f.write('dropout,batch_size,learning_rate,color_jitter,epoch,train_acc,val_acc\n')
    with open(LOG_FILE_2, 'w') as f:
        f.write('dropout,batch_size,learning_rate,color_jitter,epoch,train_acc,val_acc\n')

    total_runs = len(dropout_rates) * len(batch_sizes) * len(learning_rates) * len(color_jitters)
    run_idx = 0

    for d in dropout_rates:
        for b in batch_sizes:
            for lr in learning_rates:
                for cj in color_jitters:
                    run_idx += 1
                    print(f"Starting run {run_idx}/{total_runs}: dropout={d}, batch={b}, lr={lr}, color_jitter={cj}")

                    # Build transforms and datasets for this run
                    train_transform = make_train_transform(color_jitter_strength=cj)
                    full_dataset = datasets.ImageFolder(DATASET_PATH, transform=train_transform)
                    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
                    train_idx, val_idx = next(splitter.split(full_dataset.imgs, targets))
                    train_dataset = Subset(full_dataset, train_idx)
                    val_dataset = Subset(full_dataset, val_idx)
                    val_dataset.dataset.transform = val_transform

                    train_loader = DataLoader(train_dataset, batch_size=b, shuffle=True)
                    val_loader = DataLoader(val_dataset, batch_size=b, shuffle=False)

                    # Model, criterion, optimizer
                    model = get_mobilenetv2(num_classes=2, dropout_rate=d).to(device)
                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

                    # Train for configured epochs
                    for epoch in range(epochs):
                        train_acc, train_loss = train(model, train_loader, criterion, optimizer, device, epoch)
                        val_acc, val_loss = validate(model, val_loader, criterion, device, epoch)

                        # choose which log file to write to (split runs roughly in half)
                        log_line = f"{d},{b},{lr},{cj},{epoch+1},{train_acc:.4f},{val_acc:.4f}\n"
                        if run_idx <= total_runs // 2:
                            with open(LOG_FILE, 'a') as f:
                                f.write(log_line)
                        else:
                            with open(LOG_FILE_2, 'a') as f:
                                f.write(log_line)

                        scheduler.step(val_loss)

                    # Note: do not save models per request
                    print(f"Completed run {run_idx}/{total_runs}")

    print("All runs completed. Logs written to", LOG_FILE, "and", LOG_FILE_2)

if __name__ == "__main__":
    main()
