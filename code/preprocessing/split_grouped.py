import argparse
import json
import random
import re
import shutil
from pathlib import Path


SHIFT_RE = re.compile(r"_shift\\d{2}$")
JITTER_RE = re.compile(r"_j\\d+$")


def iter_samples(src_root, exts):
    for cls_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        for ext in exts:
            for f in sorted(cls_dir.glob(f"*{ext}")):
                yield cls_dir.name, f


def base_key(name):
    stem = Path(name).stem
    stem = SHIFT_RE.sub("", stem)
    stem = JITTER_RE.sub("", stem)
    return stem


def copy_or_link(src, dst, link):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if link:
        if dst.exists():
            return
        dst.symlink_to(src)
    else:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Root folder with class subfolders")
    ap.add_argument("--dst", required=True, help="Output folder for train/val/test")
    ap.add_argument("--train", type=float, default=0.7)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--link", action="store_true", help="Symlink instead of copy")
    ap.add_argument("--exts", type=str, default=".png,.npy", help="Comma-separated extensions to include")
    ap.add_argument("--out-splits", type=str, default=None, help="Optional JSON summary path")
    args = ap.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)
    exts = [e.strip() for e in args.exts.split(",") if e.strip()]
    samples = list(iter_samples(src_root, exts))
    if not samples:
        raise SystemExit("No samples found in src with the requested extensions.")

    groups = {}
    for cls_name, f in samples:
        key = (cls_name, base_key(f.name))
        groups.setdefault(key, []).append((cls_name, f))

    keys = list(groups.keys())
    rng = random.Random(args.seed)
    rng.shuffle(keys)

    n = len(keys)
    n_train = int(round(n * args.train))
    n_val = int(round(n * args.val))
    if n_train + n_val > n:
        n_val = max(0, n - n_train)
    n_test = n - n_train - n_val

    train_keys = set(keys[:n_train])
    val_keys = set(keys[n_train:n_train + n_val])
    test_keys = set(keys[n_train + n_val :])

    for key, items in groups.items():
        if key in train_keys:
            split = "train"
        elif key in val_keys:
            split = "val"
        else:
            split = "test"
        for cls_name, f in items:
            dst = dst_root / split / cls_name / f.name
            copy_or_link(f, dst, args.link)

    if args.out_splits:
        out_path = Path(args.out_splits)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "train_groups": len(train_keys),
                    "val_groups": len(val_keys),
                    "test_groups": len(test_keys),
                    "seed": args.seed,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
