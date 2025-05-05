# Preston Mann
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from DartSimple import DartSimple
from DartSimple import DartNoWeights
from customDataLoaderDart import CustomDartDataset
from FlatDartDataset import FlatDartDataset

from tqdm import tqdm


def train(model, train_loader, val_loader, epochs=10, lr=1e-4, device="cuda"):
    model.to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        print(f"\nEpoch {epoch+1}/{epochs}")
        for images, keypoints_gt in tqdm(train_loader, desc="Training", leave=False):
            images = images.to(device)
            keypoints_gt = keypoints_gt.to(device)
            optimizer.zero_grad()
            keypoints_pred = model(images)
            loss = criterion(keypoints_pred, keypoints_gt)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"Train Loss: {epoch_loss:.3f}  | Val Loss: {val_loss:.3f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "models/DartNoWeights.pth")


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, keypoints_gt in tqdm(dataloader, desc="Validating", leave=False):
            images = images.to(device)
            keypoints_gt = keypoints_gt.to(device)
            keypoints_pred = model(images)
            loss = criterion(keypoints_pred, keypoints_gt)
            total_loss += loss.item() * images.size(0)
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss


if __name__ == "__main__":
    root_data_dir = "/home/preston/Public/DART3/"

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    # full_dataset = CustomDartDataset(root_data_dir, transform=transform)
    full_dataset = FlatDartDataset(root_data_dir, transform=transform)
    val_size = int(0.1 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = DartNoWeights()
    train(model, train_loader, val_loader, epochs=20, lr=1e-4)
