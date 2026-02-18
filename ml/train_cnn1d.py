import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class EMGWindowDataset(Dataset):
    def __init__(self, root):
        self.root = Path(root)
        self.samples = []
        self.classes = []
        for class_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            self.classes.append(class_dir.name)
            for f in sorted(class_dir.glob("*.npy")):
                self.samples.append((f, class_dir.name))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        x = np.load(path)
        y = self.class_to_idx[label]
        return torch.from_numpy(x), y


def pad_collate(batch):
    xs, ys = zip(*batch)
    max_len = max(x.shape[1] for x in xs)
    chans = xs[0].shape[0]
    out = torch.zeros(len(xs), chans, max_len, dtype=torch.float32)
    for i, x in enumerate(xs):
        out[i, :, : x.shape[1]] = x
    return out, torch.tensor(ys, dtype=torch.long)


class CNN1D(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.net(x)
        x = x.squeeze(-1)
        return self.fc(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Dataset root with train/val folders")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--out", default="runs_cnn1d")
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    data_root = Path(args.data)
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise SystemExit("Expected train/ and val/ in data root")

    train_ds = EMGWindowDataset(train_dir)
    val_ds = EMGWindowDataset(val_dir)
    if not train_ds.samples:
        raise SystemExit("No training samples found.")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, collate_fn=pad_collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, collate_fn=pad_collate
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNN1D(in_channels=32, num_classes=len(train_ds.classes)).to(device)
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
    torch.save(model.state_dict(), out_dir / "cnn1d_emg.pt")
    (out_dir / "classes.txt").write_text("\n".join(train_ds.classes), encoding="utf-8")


if __name__ == "__main__":
    main()
