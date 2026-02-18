import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from train_cnn2d_rms import SimpleCNN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Dataset root with test folder")
    ap.add_argument("--weights", required=True, help="Path to model weights")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    data_root = Path(args.data)
    test_dir = data_root / "test"
    if not test_dir.exists():
        raise SystemExit("Expected test/ in data root")

    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ]
    )

    test_ds = datasets.ImageFolder(str(test_dir), transform=transform)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes=len(test_ds.classes)).to(device)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    acc = correct / total if total else 0.0
    print(f"Test acc: {acc:.3f}")


if __name__ == "__main__":
    main()
