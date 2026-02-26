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


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        res = x if self.downsample is None else self.downsample(x)
        if res.shape[-1] > out.shape[-1]:
            res = res[..., : out.shape[-1]]
        elif res.shape[-1] < out.shape[-1]:
            out = out[..., : res.shape[-1]]
        out = out + res
        out = self.relu(out)
        return out


class TCN(nn.Module):
    def __init__(self, in_channels, num_classes, channels=(64, 128, 256), kernel_size=5, dropout=0.1):
        super().__init__()
        layers = []
        prev = in_channels
        for i, ch in enumerate(channels):
            layers.append(TCNBlock(prev, ch, kernel_size, dilation=2 ** i, dropout=dropout))
            prev = ch
        self.tcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(prev, num_classes)

    def forward(self, x):
        x = self.tcn(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Dataset root with train/val folders")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--out", default="runs_tcn")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--kernel-size", type=int, default=5)
    ap.add_argument("--dropout", type=float, default=0.1)
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
    model = TCN(32, len(train_ds.classes), kernel_size=args.kernel_size, dropout=args.dropout).to(device)

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
    torch.save(model.state_dict(), out_dir / "tcn_emg.pt")
    (out_dir / "classes.txt").write_text("\n".join(train_ds.classes), encoding="utf-8")


if __name__ == "__main__":
    main()
