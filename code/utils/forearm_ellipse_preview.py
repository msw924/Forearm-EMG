"""
Interactive electrode ellipse placement tool.
Drag the sliders to position/resize the ellipse in real time.
When done, the final values are printed to the terminal — copy them
straight into the forearm_avg_heatmap.py command.

Usage:
    python forearm_ellipse_preview.py \
        --image ../../reports/figures/screenshot_2026-02-16_08-12-56.png \
        --clockwise --ch-top 1 --rotate-180
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def channel_angles_deg(num_channels, step_deg=360.0 / 32.0, ch_top=None, clockwise=True):
    angles = []
    for ch in range(1, num_channels + 1):
        delta = (ch - 15) * step_deg
        ang = (delta if clockwise else -delta)
        angles.append(ang)
    if ch_top is not None:
        top_angle = angles[ch_top - 1]
        rot = -90.0 - top_angle
        angles = [a + rot for a in angles]
    return np.radians(angles)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--clockwise", action="store_true")
    ap.add_argument("--ch-top", type=int, default=None)
    ap.add_argument("--rotate-180", action="store_true")
    ap.add_argument("--rx", type=float, default=0.42)
    ap.add_argument("--ry", type=float, default=0.50)
    ap.add_argument("--cx", type=float, default=0.50)
    ap.add_argument("--cy", type=float, default=0.50)
    args = ap.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        raise SystemExit(f"Image not found: {img_path}")
    img = plt.imread(img_path)
    if args.rotate_180:
        img = np.rot90(img, 2)
    h, w = img.shape[:2]

    angles = channel_angles_deg(32, ch_top=args.ch_top, clockwise=args.clockwise)
    if args.rotate_180:
        angles = angles + np.pi
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    # ── layout: image axes + 4 sliders below ──────────────────────────────
    fig = plt.figure(figsize=(7, 8))
    ax_img = fig.add_axes([0.05, 0.25, 0.90, 0.72])
    ax_img.imshow(img)
    ax_img.axis("off")

    scat = ax_img.scatter([], [], c="cyan", s=120, alpha=0.9,
                          edgecolors="black", linewidths=0.5, zorder=3)
    texts = [ax_img.text(0, 0, f"{i}", color="black", fontsize=7,
                         ha="center", va="center", fontweight="bold", zorder=4)
             for i in range(1, 33)]
    title = ax_img.set_title("")

    ax_rx = fig.add_axes([0.15, 0.17, 0.70, 0.03])
    ax_ry = fig.add_axes([0.15, 0.12, 0.70, 0.03])
    ax_cx = fig.add_axes([0.15, 0.07, 0.70, 0.03])
    ax_cy = fig.add_axes([0.15, 0.02, 0.70, 0.03])

    s_rx = Slider(ax_rx, "rx", 0.10, 0.70, valinit=args.rx)
    s_ry = Slider(ax_ry, "ry", 0.10, 0.70, valinit=args.ry)
    s_cx = Slider(ax_cx, "cx", 0.20, 0.80, valinit=args.cx)
    s_cy = Slider(ax_cy, "cy", 0.20, 0.80, valinit=args.cy)

    def update(_=None):
        rx = s_rx.val * w
        ry = s_ry.val * h
        cx = s_cx.val * w
        cy = s_cy.val * h
        xs = cx + rx * cos_a
        ys = cy + ry * sin_a
        scat.set_offsets(np.column_stack([xs, ys]))
        for i, (x, y, t) in enumerate(zip(xs, ys, texts)):
            t.set_position((x, y))
        title.set_text(
            f"rx={s_rx.val:.3f}  ry={s_ry.val:.3f}  "
            f"cx={s_cx.val:.3f}  cy={s_cy.val:.3f}"
        )
        fig.canvas.draw_idle()

    s_rx.on_changed(update)
    s_ry.on_changed(update)
    s_cx.on_changed(update)
    s_cy.on_changed(update)
    update()

    plt.show()

    # Print final values when window is closed
    print("\n── Final ellipse values ──")
    print(f"  --rx {s_rx.val:.3f} --ry {s_ry.val:.3f} "
          f"--cx {s_cx.val:.3f} --cy {s_cy.val:.3f}")


if __name__ == "__main__":
    main()
