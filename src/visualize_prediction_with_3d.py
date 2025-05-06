# preston mann
import os
import pickle
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from DartSimple import DartSimple

COLOR_JOINTS = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.4, 0.0],
        [0.0, 0.6, 0.0],
        [0.0, 0.8, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.6],
        [0.0, 0.0, 1.0],
        [0.2, 0.2, 1.0],
        [0.4, 0.4, 1.0],
        [0.0, 0.4, 0.4],
        [0.0, 0.6, 0.6],
        [0.0, 0.8, 0.8],
        [0.0, 1.0, 1.0],
        [0.4, 0.4, 0.0],
        [0.6, 0.6, 0.0],
        [0.8, 0.8, 0.0],
        [1.0, 1.0, 0.0],
        [0.4, 0.0, 0.4],
        [0.6, 0.0, 0.6],
        [0.8, 0.0, 0.8],
        [1.0, 0.0, 1.0],
    ]
)[:, ::-1]

MANO_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]


def plot_hand(image, coords_hw, vis=None, linewidth=2):
    if vis is None:
        vis = np.ones_like(coords_hw[:, 0], dtype=bool)
    for (i, j), color in zip(MANO_EDGES, COLOR_JOINTS[1:]):
        if vis[i] and vis[j]:
            pt1 = tuple(np.round(coords_hw[i]).astype(int))
            pt2 = tuple(np.round(coords_hw[j]).astype(int))
            cv2.line(image, pt1, pt2, color=color * 255.0, thickness=linewidth)
    for i in range(coords_hw.shape[0]):
        if vis[i]:
            pt = tuple(np.round(coords_hw[i]).astype(int))
            cv2.circle(
                image,
                pt,
                radius=2 * linewidth,
                thickness=-1,
                color=COLOR_JOINTS[i] * 255.0,
            )


def preprocess_image(img_path):
    img = cv2.imread(img_path)[..., ::-1]  # BGR to RGB
    img_resized = cv2.resize(img, (224, 224))
    tensor = img_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
    return img_resized, torch.from_numpy(tensor).unsqueeze(0), img_resized.copy()


def fit_ortho_param(joints_3d, joints_2d):
    x3d = joints_3d[:, :2]
    x2d = joints_2d
    mean_3d = x3d.mean(0)
    mean_2d = x2d.mean(0)
    x3d_centered = x3d - mean_3d
    x2d_centered = x2d - mean_2d
    scale = (x2d_centered * x3d_centered).sum() / (x3d_centered**2).sum()
    trans = mean_2d / scale - mean_3d
    return {"scale": scale, "trans": trans}


def ortho_project(points_3d, cam_params):
    scale, trans = cam_params["scale"], cam_params["trans"]
    return scale * (points_3d[:, :2] + trans)


def project_3d_like_unity(
    joints_3d_pred, joints_3d_gt, joints_2d_gt, image_size=224, original_size=512
):
    joints_2d_gt_scaled = (np.array(joints_2d_gt) / original_size) * image_size
    cam_params = fit_ortho_param(joints_3d_gt, joints_2d_gt_scaled)
    return ortho_project(joints_3d_pred, cam_params), joints_2d_gt_scaled


def plot_hand_3d(joints_3d, ax, color="green", title=""):
    ax.clear()
    ax.scatter(joints_3d[:, 0], joints_3d[:, 1], joints_3d[:, 2], c=color, s=25)
    for i, j in MANO_EDGES:
        x = [joints_3d[i, 0], joints_3d[j, 0]]
        y = [joints_3d[i, 1], joints_3d[j, 1]]
        z = [joints_3d[i, 2], joints_3d[j, 2]]
        ax.plot(x, y, z, c=color)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=200, azim=60)
    ax.set_box_aspect([1, 1, 1])


def draw_2d(image, joints_2d):
    image = image.copy()
    plot_hand(image, joints_2d, linewidth=2)
    return image


def load_sample(index, model, image_dir, data, device):
    img_name = data["img"][index]
    joints_3d_gt = np.array(data["joint3d"][index])
    joints_2d_gt = np.array(data["joint2d"][index])
    image_path = os.path.join(image_dir, img_name)
    _, input_tensor, image_rgb = preprocess_image(image_path)

    with torch.no_grad():
        pred_3d = model(input_tensor.to(device)).squeeze(0).cpu().numpy()

    loss = (
        torch.nn.functional.l1_loss(
            torch.tensor(pred_3d), torch.tensor(joints_3d_gt)
        ).item()
        * 1000
    )

    pred_2d, gt_2d_scaled = project_3d_like_unity(pred_3d, joints_3d_gt, joints_2d_gt)

    image_2d_gt = draw_2d(image_rgb, gt_2d_scaled)
    image_2d_pred = draw_2d(image_rgb, pred_2d)

    return joints_3d_gt, pred_3d, loss, image_2d_gt, image_2d_pred


def visualize_with_slider(
    model_path="test_run_fulldata/dart_model_best.pth",
    image_dir="data/DARTset/test/0/",
    label_file="data/DARTset/test/part_0.pkl",
    device="cuda",
):
    with open(label_file, "rb") as f:
        data = pickle.load(f)

    num_samples = len(data["img"])
    model = DartSimple().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    fig = plt.figure(figsize=(18, 6))
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    ax3 = fig.add_subplot(2, 2, 3)

    def update(idx):
        gt_joints, pred_joints, loss, image_gt, image_pred = load_sample(
            idx, model, image_dir, data, device
        )
        plot_hand_3d(gt_joints, ax1, color="green", title="Ground Truth (3D)")
        plot_hand_3d(
            pred_joints, ax2, color="red", title=f"Prediction (3D)\nLoss: {loss:.2f} mm"
        )
        ax3.clear()
        ax3.imshow(image_gt)
        ax3.axis("off")
        ax3.set_title("Ground Truth (2D)")
        fig.canvas.draw_idle()
        print(f"\n[Sample {idx}] L1 Loss: {loss:.2f} mm")

    ax_slider = plt.axes([0.25, 0.01, 0.5, 0.03])
    slider = Slider(ax_slider, "Sample Index", 0, num_samples - 1, valinit=0, valstep=1)
    slider.on_changed(update)

    update(0)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()


if __name__ == "__main__":
    visualize_with_slider()
