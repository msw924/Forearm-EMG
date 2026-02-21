import argparse
import json
import random
import shutil
from pathlib import Path


def iter_samples(src_root, exts):
    for cls_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        for ext in exts:
            for f in sorted(cls_dir.glob(f"*{ext}")):
                yield cls_dir.name, f


def subject_from_name(name):
    return name.split("-", 1)[0]


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
    ap.add_argument("--test-subjects", type=int, default=7)
    ap.add_argument("--val-subjects", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--link", action="store_true", help="Symlink instead of copy")
    ap.add_argument("--out-splits", type=str, default=None, help="Optional JSON summary path")
    ap.add_argument("--exts", type=str, default=".png,.npy", help="Comma-separated extensions to include")
    args = ap.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)
    exts = [e.strip() for e in args.exts.split(",") if e.strip()]
    samples = list(iter_samples(src_root, exts))
    if not samples:
        raise SystemExit("No samples found in src with the requested extensions.")

    subjects = sorted({subject_from_name(f.name) for _, f in samples})
    rng = random.Random(args.seed)
    rng.shuffle(subjects)

    n_test = min(args.test_subjects, len(subjects))
    n_val = min(args.val_subjects, max(0, len(subjects) - n_test))
    test_subjects = sorted(subjects[:n_test])
    val_subjects = sorted(subjects[n_test:n_test + n_val])
    train_subjects = sorted(subjects[n_test + n_val:])

    for cls_name, f in samples:
        subj = subject_from_name(f.name)
        if subj in test_subjects:
            split = "test"
        elif subj in val_subjects:
            split = "val"
        else:
            split = "train"
        dst = dst_root / split / cls_name / f.name
        copy_or_link(f, dst, args.link)

    if args.out_splits:
        out_path = Path(args.out_splits)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "train": train_subjects,
                    "val": val_subjects,
                    "test": test_subjects,
                    "seed": args.seed,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
