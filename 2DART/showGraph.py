import argparse
import re
from pathlib import Path
import matplotlib.pyplot as plt


def parse_log(file_path: Path):
    lines = file_path.read_text().splitlines()
    model_name = lines[0].strip()
    train_losses, val_losses = [], []

    for line in lines:
        if line.startswith("Train Loss"):
            train_losses.append(
                1000 * float(re.search(r"Train Loss:\s*([0-9eE.+-]+)", line).group(1))
            )
        elif line.startswith("Val Loss"):
            val_losses.append(
                1000 * float(re.search(r"Val Loss:\s*([0-9eE.+-]+)", line).group(1))
            )

    return model_name, train_losses, val_losses


def main():
    parser = argparse.ArgumentParser(description="Plot train/validation losses in mm")
    parser.add_argument("logs", nargs=3, help="Paths to the three log files.")
    parser.add_argument(
        "--output", default="loss_plot3.png", help="Filename for the saved plot."
    )
    args = parser.parse_args()

    for log_path in args.logs:
        model, train, val = parse_log(Path(log_path))
        epochs = range(1, len(train) + 1)
        plt.plot(epochs, train, label=f"{model} Train")
        plt.plot(epochs, val, linestyle="--", label=f"{model} Val")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=600)
    plt.show()


if __name__ == "__main__":
    main()
