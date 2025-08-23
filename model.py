import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import torch.nn as nn
import torch.optim as optim

# Example model (replace with your own architecture)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Curriculum: list of dataset difficulties (easy to hard)
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

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN().to(device)
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
            train(model, dataloader, criterion, optimizer, device)
            print(f"Epoch {epoch+1} completed for stage {stage+1}")

    torch.save(model.state_dict(), 'ai_detector_model.pth')

if __name__ == "__main__":
    main()