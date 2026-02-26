import argparse
import random
import shutil
from pathlib import Path


def iter_images(class_dir):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy"}
    return [p for p in class_dir.iterdir() if p.suffix.lower() in exts and p.is_file()]


def copy_or_link(src, dst, link):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if link:
        if dst.exists():
            return
        dst.symlink_to(src)
    else:
        shutil.copy2(src, dst)


def split_class(images, train_frac, val_frac, seed):
    rng = random.Random(seed)
    imgs = images[:]
    rng.shuffle(imgs)
    n = len(imgs)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)
    train = imgs[:n_train]
    val = imgs[n_train:n_train + n_val]
    test = imgs[n_train + n_val:]
    return train, val, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Root folder with class subfolders")
    ap.add_argument("--dst", required=True, help="Output folder for train/val/test")
    ap.add_argument("--train", type=float, default=0.7)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--link", action="store_true", help="Symlink instead of copy")
    args = ap.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)
    classes = [p for p in src_root.iterdir() if p.is_dir()]
    if not classes:
        raise SystemExit("No class folders found in src")

    for cls_dir in classes:
        images = iter_images(cls_dir)
        if not images:
            continue
        train, val, test = split_class(images, args.train, args.val, args.seed)
        for split_name, split_imgs in [("train", train), ("val", val), ("test", test)]:
            for img in split_imgs:
                dst = dst_root / split_name / cls_dir.name / img.name
                copy_or_link(img, dst, args.link)


if __name__ == "__main__":
    main()
