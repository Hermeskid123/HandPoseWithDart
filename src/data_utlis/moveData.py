import os
import pickle
import shutil
from pathlib import Path

import torch
from tqdm import tqdm


def flatten_dataset(source_root, dest_root):
    source_root = Path(source_root)
    dest_root = Path(dest_root)
    image_out_dir = dest_root / "images"
    label_out_path = dest_root / "labels.pkl"

    image_out_dir.mkdir(parents=True, exist_ok=True)

    all_image_names = []
    all_joint3d = []
    all_joint2d = []

    folder_dirs = sorted([d for d in source_root.iterdir() if d.is_dir()])

    print(f"Found {len(folder_dirs)} folders.")

    for folder_idx, folder in enumerate(tqdm(folder_dirs, desc="Processing Folders")):
        output_pkl = folder / "output.pkl"
        if not output_pkl.exists():
            continue

        with open(output_pkl, "rb") as f:
            label_data = pickle.load(f)
            joint_3d = torch.tensor(label_data["joint_3d"], dtype=torch.float32).clone()
            joint_2d = torch.tensor(label_data["joint_2d"], dtype=torch.float32).clone()

        all_joint3d.append(joint_3d)
        all_joint2d.append(joint_2d)

        image_filenames = sorted(
            [
                fname
                for fname in os.listdir(folder)
                if fname.endswith(".png") and os.path.isfile(folder / fname)
            ]
        )

        for img_idx, fname in enumerate(image_filenames):
            print(fname)
            if "mask" in fname:
                pass
            else:
                new_filename = f"{folder_idx}_{img_idx}.png"
                src_path = folder / fname
                dst_path = image_out_dir / new_filename
                shutil.copy(src_path, dst_path)
                all_image_names.append(new_filename)

    label_dict = {
        "img": all_image_names,
        "joint3d": all_joint3d,
        "joint2d": all_joint2d,
    }

    with open(label_out_path, "wb") as f:
        pickle.dump(label_dict, f)

    print(f"\nFinished flattening dataset.")
    print(f"  - Total images: {len(all_image_names)}")
    print(f"  - Total joint3d labels: {len(all_joint3d)}")
    print(f"  - Total joint2d labels: {len(all_joint2d)}")
    print(f"  - Output saved to: {label_out_path}")


if __name__ == "__main__":
    source = "/home/preston/Public/2degree"
    destination = "/home/preston/Public/testDelLater"
    flatten_dataset(source, destination)
