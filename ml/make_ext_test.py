import argparse
import shutil
from pathlib import Path


def copy_tree(src, dst):
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.is_dir():
            copy_tree(item, dst / item.name)
        elif item.is_file():
            shutil.copy2(item, dst / item.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="External raw dataset root (emg/polar)")
    ap.add_argument("--dst", required=True, help="Output dataset root with test/")
    ap.add_argument("--kind", choices=["emg", "polar"], required=True)
    args = ap.parse_args()

    src_root = Path(args.src) / args.kind
    dst_root = Path(args.dst) / "test"
    if not src_root.exists():
        raise SystemExit(f"Missing source: {src_root}")

    copy_tree(src_root, dst_root)
    print(f"Copied {args.kind} external test set to {dst_root}")


if __name__ == "__main__":
    main()
