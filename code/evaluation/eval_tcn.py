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
    model = TCN(32, len(test_ds.classes)).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
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
    header = " " * 12 + " ".join(f\"{c[:8]:>8}\" for c in test_ds.classes)
    print(header)
    for i, cls in enumerate(test_ds.classes):
        row = " ".join(f\"{int(v):8d}\" for v in conf[i].tolist())
        print(f\"{cls[:10]:>10} {row}\")
    print("Per-class acc:")
    for i, cls in enumerate(test_ds.classes):
        denom = conf[i].sum().item()
        acc_i = conf[i, i].item() / denom if denom else 0.0
        print(f\"{cls}: {acc_i:.3f}\")


if __name__ == "__main__":
    main()
