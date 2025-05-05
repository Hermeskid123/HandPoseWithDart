# Preston Mann
import os
import pickle
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from DartSimple import DartSimple
from DartSimple import SimplePoseCNN
from FlatDartDataset import FlatDartDataset
import torchvision.transforms as transforms


def evaluate_and_save(model_path, image_dir, output_path, device="cuda"):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    dataset = FlatDartDataset(image_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # model = DartSimple().to(device)
    model = SimplePoseCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    criterion = torch.nn.L1Loss(reduction="mean")

    results = {
        "loss_mm": [],
        "predicted_3d": [],
    }

    total_loss = 0
    counter = 0
    for i, (image, gt_3d) in enumerate(tqdm(dataloader, desc="Evaluating")):
        counter = counter + 1
        image = image.to(device)
        gt_3d = gt_3d.to(device)

        with torch.no_grad():
            pred_3d = model(image)
            loss = criterion(pred_3d, gt_3d).item() * 1000  # mm
            # print("loss: ", loss)
        total_loss = loss + total_loss
        results["loss_mm"].append(loss)
        results["predicted_3d"].append(pred_3d.squeeze(0).cpu().numpy())

    print("total_loss no div ", total_loss)
    total_loss = total_loss / counter
    print("total_loss ", total_loss)
    print("samples, ", counter)
    with open(output_path, "wb") as f:
        pickle.dump(results, f)

    print(f"\nSaved evaluation results to {output_path}")


if __name__ == "__main__":
    model_path = "3DART/SimplePoseCNNBest.pth"
    image_dir = "/home/preston/Public/DART3"
    output_path = "results.pkl"
    evaluate_and_save(model_path, image_dir, output_path)
