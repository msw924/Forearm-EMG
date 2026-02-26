import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Dataset root with train/val folders")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--out", default="runs", help="Output folder for model + classes")
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    data_root = Path(args.data)
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise SystemExit("Expected train/ and val/ in data root")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_ds = datasets.ImageFolder(str(train_dir), transform=transform)
    val_ds = datasets.ImageFolder(str(val_dir), transform=transform)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(train_ds.classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    best_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = correct / total if total else 0.0
        scheduler.step(acc)
        print(f"Epoch {epoch + 1}/{args.epochs} | val acc: {acc:.3f}")

        if acc > best_acc:
            best_acc = acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print("Early stopping: no improvement.")
                break

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "resnet18_graphs.pt")
    (out_dir / "classes.txt").write_text("\n".join(train_ds.classes), encoding="utf-8")


if __name__ == "__main__":
    main()
