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
    ap.add_argument("--data", required=True, help="Dataset root with test/ folder")
    ap.add_argument("--weights", required=True, help="Path to .pt weights")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    test_dir = Path(args.data) / "test"
    if not test_dir.exists():
        raise SystemExit("Expected test/ in dataset root")

    test_ds = EMGWindowDataset(test_dir)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, collate_fn=pad_collate
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNN1D(in_channels=32, num_classes=len(test_ds.classes))
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
    print("Per-class acc:")
    for i, cls in enumerate(test_ds.classes):
        denom = conf[i].sum().item()
        acc_i = conf[i, i].item() / denom if denom else 0.0
        print(f"{cls}: {acc_i:.3f}")


if __name__ == "__main__":
    main()
