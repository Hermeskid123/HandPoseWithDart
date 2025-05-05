# Preston Mann

import os
import pickle
from PIL import Image
import torch
from torch.utils.data import Dataset


class CustomDartDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples = []

        for folder in sorted(os.listdir(root_dir)):
            folder_path = os.path.join(root_dir, folder)
            if not os.path.isdir(folder_path):
                continue

            label_path = os.path.join(folder_path, "output.pkl")
            if not os.path.exists(label_path):
                continue

            with open(label_path, "rb") as f:
                label_data = pickle.load(f)
                joint3d = torch.tensor(label_data["joint_3d"], dtype=torch.float32)

            image_filenames = sorted(
                [
                    fname
                    for fname in os.listdir(folder_path)
                    if fname.endswith(".png")
                    and os.path.isfile(os.path.join(folder_path, fname))
                ]
            )

            for img_name in image_filenames:
                img_path = os.path.join(folder_path, img_name)
                self.samples.append((img_path, joint3d.clone()))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, keypoints = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, keypoints
