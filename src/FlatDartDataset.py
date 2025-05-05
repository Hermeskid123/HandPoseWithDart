# Preston Mann
import os
import pickle
from PIL import Image
from torch.utils.data import Dataset


class FlatDartDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, "images")
        self.transform = transform

        label_file = os.path.join(root_dir, "labels.pkl")
        with open(label_file, "rb") as f:
            data = pickle.load(f)

        self.labels = data["joint3d"]
        self.samples = []

        for img_name in data["img"]:
            img_path = os.path.join(self.image_dir, img_name)
            label_index = int(img_name.split("_")[0])
            self.samples.append((img_path, label_index))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_index = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        keypoints = self.labels[label_index]

        if self.transform:
            image = self.transform(image)

        return image, keypoints
