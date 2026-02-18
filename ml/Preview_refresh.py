from pathlib import Path
import argparse
import random
import math
import os

os.environ['MPLCONFIGDIR'] = '/tmp/mplconfig'
Path('/tmp/mplconfig').mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEFAULT_ROOT = Path('/Users/maxwilliams/Library/CloudStorage/Dropbox/MW_GR_data/Data/dataset_raw')


def pick_one_per_subject(imgs):
    by_subject = {}
    for img in imgs:
        # filename starts with subject name before the first "-"
        subject = img.name.split('-', 1)[0]
        by_subject.setdefault(subject, []).append(img)
    picks = []
    for subject, group in by_subject.items():
        picks.append(random.choice(group))
    return picks


def make_grid(root, kind, n=9, one_per_subject=False):
    imgs = list((root / kind).rglob('*.png'))
    if not imgs:
        print(f'No images found for {kind}')
        return
    if one_per_subject:
        imgs = pick_one_per_subject(imgs)
    random.shuffle(imgs)
    imgs = imgs[:n]
    cols = int(math.sqrt(n))
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6, 6))
    axes = axes.flatten()
    for ax, img in zip(axes, imgs):
        ax.imshow(plt.imread(img))
        ax.set_title(img.parent.name, fontsize=8)
        ax.axis('off')
    for ax in axes[len(imgs):]:
        ax.axis('off')
    suffix = 'subject' if one_per_subject else 'random'
    out = root / f'preview_{kind}_{suffix}.png'
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Saved {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=str(DEFAULT_ROOT), help="dataset root")
    ap.add_argument("--n", type=int, default=9)
    ap.add_argument("--one-per-subject", action="store_true")
    ap.add_argument("--kind", type=str, default=None, help="Folder name to preview (e.g. emg, polar, srf, ts)")
    args = ap.parse_args()

    root = Path(args.root)
    if args.kind:
        make_grid(root, args.kind, args.n, one_per_subject=args.one_per_subject)
    else:
        make_grid(root, 'emg', args.n, one_per_subject=args.one_per_subject)
        make_grid(root, 'polar', args.n, one_per_subject=args.one_per_subject)


if __name__ == '__main__':
    main()
