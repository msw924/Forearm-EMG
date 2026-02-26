import argparse
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def fit_circle(xs, ys):
    x = xs.astype(float)
    y = ys.astype(float)
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x * x + y * y
    coeff, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    a, b2, c = coeff
    cx = a / 2.0
    cy = b2 / 2.0
    r = math.sqrt(max(c + cx * cx + cy * cy, 0.0))
    return cx, cy, r


def channel_angles_deg(num_channels, step_deg, ch15_deg, clockwise=True, ch20_up=False):
    angles = []
    for ch in range(1, num_channels + 1):
        delta = (ch - 15) * step_deg
        ang = ch15_deg + (delta if clockwise else -delta)
        angles.append(ang)
    if ch20_up:
        ch20_angle = angles[19]
        rot = -90.0 - ch20_angle
        angles = [a + rot for a in angles]
    return np.radians(angles)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Forearm cross-section image")
    ap.add_argument("--out", required=True, help="Output image path")
    ap.add_argument("--radius", type=float, default=0.55, help="Radius as fraction of image min dim")
    ap.add_argument("--step-deg", type=float, default=360.0 / 32.0)
    ap.add_argument("--ch15-deg", type=float, default=0.0, help="Angle (deg) for channel 15")
    ap.add_argument("--ch15-shift", type=int, default=0, help="Shift channel 15 by N steps of step-deg")
    ap.add_argument("--clockwise", action="store_true", help="Channel numbers increase clockwise")
    ap.add_argument("--ch20-up", action="store_true", help="Rotate so channel 20 is up")
    ap.add_argument("--rotate-180", action="store_true", help="Rotate image and electrodes 180 degrees")
    args = ap.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        raise SystemExit(f"Image not found: {img_path}")
    img = plt.imread(img_path)
    if args.rotate_180:
        img = np.rot90(img, 2)
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    radius = args.radius * min(w, h)

    ch15_deg = args.ch15_deg + args.ch15_shift * args.step_deg
    angles = channel_angles_deg(
        32,
        step_deg=args.step_deg,
        ch15_deg=ch15_deg,
        clockwise=args.clockwise,
        ch20_up=args.ch20_up,
    )
    xs = center[0] + radius * np.cos(angles)
    ys = center[1] + radius * np.sin(angles)
    if args.rotate_180:
        xs = 2 * center[0] - xs
        ys = 2 * center[1] - ys

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img)
    colors = plt.cm.hsv(np.linspace(0.0, 1.0, len(xs), endpoint=False))
    ax.scatter(xs, ys, c=colors, s=30, alpha=0.95, edgecolors="black", linewidths=0.5)
    for i, (x, y, c) in enumerate(zip(xs, ys, colors), start=1):
        ax.text(
            x,
            y,
            f"{i}",
            color="black",
            fontsize=7,
            ha="center",
            va="center",
            bbox=dict(boxstyle="circle,pad=0.2", facecolor=c, edgecolor="black", alpha=1.0),
        )
    ax.set_title("Electrode positions (rest)")
    ax.axis("off")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


if __name__ == "__main__":
    main()
