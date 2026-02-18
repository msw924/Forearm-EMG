import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Dataset root with test/ folder")
    ap.add_argument("--weights", required=True, help="Path to .pt weights")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    test_dir = Path(args.data) / "test"
    if not test_dir.exists():
        raise SystemExit("Expected test/ in dataset root")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    test_ds = datasets.ImageFolder(str(test_dir), transform=transform)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(test_ds.classes))
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model = model.to(device)
    model.eval()

    correct = 0
    total = 0
    num_classes = len(test_ds.classes)
    conf = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            for t, p in zip(y.view(-1), pred.view(-1)):
                conf[t.long(), p.long()] += 1

    acc = correct / total if total else 0.0
    print(f"Test acc: {acc:.3f}")
    print("Confusion matrix (rows=true, cols=pred):")
    header = " " * 12 + " ".join(f"{c[:8]:>8}" for c in test_ds.classes)
    print(header)
    for i, cls in enumerate(test_ds.classes):
        row = " ".join(f"{int(v):8d}" for v in conf[i].tolist())
        print(f"{cls[:10]:>10} {row}")
    per_class_acc = []
    for i in range(num_classes):
        denom = conf[i].sum().item()
        acc_i = conf[i, i].item() / denom if denom else 0.0
        per_class_acc.append(acc_i)
    print("Per-class acc:")
    for cls, acc_i in zip(test_ds.classes, per_class_acc):
        print(f"{cls}: {acc_i:.3f}")


if __name__ == "__main__":
    main()
