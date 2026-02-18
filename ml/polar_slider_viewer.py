import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from EMG_opener import (
    base_data_dir,
    sensor_unit,
    fs,
    load_emg,
)


def find_trial(subject_name, trial_name):
    subject_dir = base_data_dir / subject_name
    if not subject_dir.exists():
        raise SystemExit(f"Subject not found: {subject_name}")
    trial_dir = subject_dir / trial_name
    if not trial_dir.exists():
        raise SystemExit(f"Trial not found: {trial_dir}")
    emg_files = sorted(trial_dir.glob(f"*_M02{sensor_unit}_EMG_raw.sig"))
    if not emg_files:
        raise SystemExit(f"No EMG files in {trial_dir}")
    return emg_files[0]


def stacked_emg_plot(signal, ds=10):
    t = np.arange(signal.shape[1]) / fs
    sig = signal[:, ::ds]
    tt = t[::ds]
    amp = np.max(np.abs(sig))
    offset = 1.2 * amp if amp > 0 else 1.0
    y = sig + offset * np.arange(sig.shape[0])[:, None]
    return tt, y, offset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True, help="Subject folder name, e.g. (03)Manon")
    ap.add_argument("--trial", required=True, help="Trial folder name, e.g. trial 3")
    ap.add_argument("--ds", type=int, default=10)
    ap.add_argument("--window-sec", type=float, default=0.5, help="Averaging window around slider time")
    ap.add_argument("--max-radius", type=float, default=None, help="Optional fixed max radius")
    ap.add_argument("--scale", type=float, default=1.0, help="Scale factor for radius")
    args = ap.parse_args()

    emg_path = find_trial(args.subject, args.trial)
    signal = load_emg(emg_path)
    t = np.arange(signal.shape[1]) / fs

    if args.ds <= 0:
        raise SystemExit("Invalid downsample value.")

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 2], hspace=0.35)
    ax_polar = fig.add_subplot(gs[0], projection="polar")
    ax_emg = fig.add_subplot(gs[1])
    ax_slider = fig.add_axes([0.12, 0.05, 0.76, 0.03])

    tt, y, offset = stacked_emg_plot(signal, ds=args.ds)
    ax_emg.plot(tt, y.T, linewidth=0.3)
    ax_emg.set_xlabel("Time (s)")
    line = ax_emg.axvline(tt[0], color="g", linewidth=1.2)
    yticks = offset * np.arange(signal.shape[0])
    ylabels = [f"Ch{idx + 1:02d}" for idx in range(signal.shape[0])]
    ax_emg.set_yticks(yticks)
    ax_emg.set_yticklabels(ylabels, fontsize=7)

    angles = np.linspace(0, 2 * np.pi, signal.shape[0], endpoint=False)
    ax_polar.set_facecolor("black")
    polar_line, = ax_polar.plot([], [], color="cyan", linewidth=1.2)
    ax_polar.tick_params(colors="white")

    ax_slider.set_facecolor("0.95")
    slider = Slider(
        ax=ax_slider,
        label="Time (s)",
        valmin=float(tt[0]),
        valmax=float(tt[-1]),
        valinit=float(tt[0]),
        valstep=1.0 / fs,
    )

    if args.max_radius is None:
        max_radius = float(np.percentile(np.abs(signal), 95))
    else:
        max_radius = float(args.max_radius)
    max_radius = max_radius * args.scale if max_radius > 0 else 1.0

    def update_polar(xpos):
        half = args.window_sec / 2.0
        start_t = max(0.0, xpos - half)
        end_t = min(t[-1], xpos + half)
        start_idx = int(round(start_t * fs))
        end_idx = int(round(end_t * fs))
        start_idx = max(0, min(start_idx, signal.shape[1] - 1))
        end_idx = max(start_idx + 1, min(end_idx, signal.shape[1]))
        seg = signal[:, start_idx:end_idx]
        vals = np.mean(np.abs(seg), axis=1)
        polar_line.set_data(np.r_[angles, angles[0]], np.r_[vals, vals[0]])
        ax_polar.set_ylim(0.0, max_radius)
        ax_polar.set_title(
            f"{args.subject} | {args.trial} | t={xpos:.2f}s | window={args.window_sec:.2f}s",
            fontsize=10,
        )

    def on_slide(val):
        xpos = float(val)
        line.set_xdata([xpos, xpos])
        update_polar(xpos)
        fig.canvas.draw_idle()

    slider.on_changed(on_slide)
    update_polar(tt[0])
    degrees = np.degrees(angles)
    labels = [f"{idx + 1}" for idx in range(len(angles))]
    ax_polar.set_thetagrids(degrees, labels, fontsize=6, color="white")
    ax_polar.set_rgrids(
        np.linspace(0, max_radius, 5)[1:],
        angle=0,
        color="white",
        alpha=0.3,
        fontsize=7,
    )
    plt.tight_layout(rect=[0.0, 0.08, 1.0, 1.0])
    plt.show()


if __name__ == "__main__":
    main()
