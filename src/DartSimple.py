# preston mann
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import pickle
from PIL import Image


class DartSimple(nn.Module):
    def __init__(self, num_keypoints=21):
        super(DartSimple, self).__init__()

        self.backbone = models.resnet18(pretrained=True)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, num_keypoints * 3),
        )

    def forward(self, x):
        features = self.backbone(x)
        keypoints = self.fc(features)
        return keypoints.view(-1, 21, 3)


class DartNoWeights(nn.Module):
    def __init__(self, num_keypoints=21):
        super(DartNoWeights, self).__init__()

        self.backbone = models.resnet18(weights=None)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, num_keypoints * 3),
        )

    def forward(self, x):
        features = self.backbone(x)
        keypoints = self.fc(features)
        return keypoints.view(-1, 21, 3)


def load_model(model_path):
    model = DartSimple()
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cuda")))
    model.eval()
    return model


class DartDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        print(self.image_dir)
        with open(label_file, "rb") as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data["img"])

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.data["img"][idx])
        image = Image.open(image_path).convert("RGB")
        keypoints = torch.tensor(self.data["joint3d"][idx], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, keypoints


class SimplePoseCNN(nn.Module):
    def __init__(self, num_keypoints=21, output_dim=3):
        super(SimplePoseCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, num_keypoints * output_dim),
        )

        self.num_keypoints = num_keypoints
        self.output_dim = output_dim

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x.view(-1, self.num_keypoints, self.output_dim)


def evaluate_model(model, dataloader):
    model.eval()
    total_loss = 0.0
    num_samples = 0
    criterion = nn.L1Loss()
    with torch.no_grad():
        for images, keypoints_gt in dataloader:
            keypoints_pred = model(images)
            loss = criterion(keypoints_pred, keypoints_gt)
            total_loss += loss.item() * images.size(0)
            num_samples += images.size(0)
    avg_loss_mm = total_loss / num_samples
    print(f"Average loss: {avg_loss_mm:.2f}")
    return avg_loss_mm


if __name__ == "__main__":
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    dataset = DartDataset(
        "data/DARTset/test/0/", "data/DARTset/test/part_0.pkl", transform=transform
    )
    print(len(dataset))
    model = DartSimple()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    evaluate_model(model, dataloader)
