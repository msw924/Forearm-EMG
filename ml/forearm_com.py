import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from EMG_opener import (
    base_data_dir,
    sensor_unit,
    fs,
    load_emg,
    load_aux,
    compute_move_markers,
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
    emg_path = emg_files[0]
    aux_path = emg_path.with_name(emg_path.name.replace("_EMG_raw.sig", "_AUX1_raw.sig"))
    log_candidates = list(trial_dir.glob("*.rtf")) + list(trial_dir.glob("*.txt")) + list(trial_dir.glob("*.pdf"))
    log_path = log_candidates[0] if log_candidates else None
    return emg_path, aux_path, log_path


def load_offsets(path):
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Offsets not found: {path}")
    offsets = {}
    with p.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["subject"], row["trial"], row["emg_file"], row["label"])
            offsets[key] = float(row["offset_sec"])
    return offsets


def load_manual_windows(path):
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Manual windows not found: {path}")
    windows = {}
    with p.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["subject"], row["trial"], row["emg_file"], row["label"])
            windows[key] = (float(row["start_t"]), float(row["end_t"]))
    return windows


def channel_angles_deg(num_channels, step_deg=11.5, ch15_deg=0.0, clockwise=True, ch20_up=True):
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


def compute_centroid(weights, angles, center, radius):
    xs = center[0] + radius * np.cos(angles)
    ys = center[1] + radius * np.sin(angles)
    w = np.maximum(weights, 0.0)
    if np.sum(w) == 0:
        return center, (xs, ys)
    cx = np.sum(xs * w) / np.sum(w)
    cy = np.sum(ys * w) / np.sum(w)
    return (cx, cy), (xs, ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--trial", required=True)
    ap.add_argument("--label", required=True, help="Movement label, e.g. wrist extension")
    ap.add_argument("--manual", type=str, default=None, help="Manual windows CSV")
    ap.add_argument("--offsets", type=str, default=None, help="Offsets CSV (old UI)")
    ap.add_argument("--image", required=True, help="Forearm cross-section image path")
    ap.add_argument("--out", required=True, help="Output image path")
    ap.add_argument("--radius", type=float, default=0.55, help="Radius as fraction of image min dim")
    ap.add_argument("--step-deg", type=float, default=360.0 / 32.0)
    ap.add_argument("--ch15-deg", type=float, default=0.0, help="Angle (deg) for channel 15")
    ap.add_argument("--ch15-shift", type=int, default=0, help="Shift channel 15 by N steps of step-deg")
    ap.add_argument("--clockwise", action="store_true", help="Channel numbers increase clockwise")
    ap.add_argument("--ch20-up", action="store_true", help="Rotate so channel 20 is up")
    ap.add_argument("--rotate-180", action="store_true", help="Rotate image and electrodes 180 degrees")
    ap.add_argument("--label-channels", action="store_true", help="Draw channel number labels")
    ap.add_argument("--log-scale", action="store_true", help="Use log scaling for color mapping")
    ap.add_argument("--point-size", type=float, default=60.0, help="Electrode marker size")
    args = ap.parse_args()

    emg_path, aux_path, log_path = find_trial(args.subject, args.trial)
    signal = load_emg(emg_path)
    manual = load_manual_windows(args.manual)
    offsets = load_offsets(args.offsets)
    key = (args.subject, args.trial, emg_path.name, args.label)
    if key in manual:
        start_t, end_t = manual[key]
    else:
        aux = load_aux(aux_path)
        aux_start, aux_end, move_starts, labels, move_duration, _, _, per_label_offsets = compute_move_markers(
            signal, aux, log_path
        )
        if not move_starts or not labels or move_duration is None:
            raise SystemExit("No movement windows found.")
        label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
        label_map = {lbl.strip().lower(): i for lbl, i in label_to_idx.items()}
        key_norm = args.label.strip().lower()
        if key_norm not in label_map:
            available = ", ".join(sorted(label_to_idx.keys()))
            raise SystemExit(f"Label not found in log: {args.label}. Available: {available}")
        idx = label_map[key_norm]
        off = per_label_offsets[idx]
        if key in offsets:
            off = offsets[key]
        start_t = move_starts[idx] + off
        end_t = start_t + max(move_duration - off, 0.0)
    start_idx = int(round(start_t * fs))
    end_idx = int(round(end_t * fs))
    start_idx = max(start_idx, 0)
    end_idx = min(end_idx, signal.shape[1])
    if end_idx <= start_idx:
        raise SystemExit("Invalid window")

    seg = signal[:, start_idx:end_idx]
    rms = np.sqrt(np.mean(seg * seg, axis=1))

    img_path = Path(args.image)
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
    if args.rotate_180:
        angles = angles + np.pi
    (cx, cy), (xs, ys) = compute_centroid(rms, angles, center, radius)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img)
    # Plot activation per electrode (red=high, blue=low)
    rms_for_color = rms
    if args.log_scale:
        eps = max(float(np.max(rms)) * 1e-6, 1e-12)
        rms_for_color = np.log10(rms + eps)
    norm = plt.Normalize(vmin=float(np.min(rms_for_color)), vmax=float(np.max(rms_for_color)))
    colors = plt.cm.coolwarm(norm(rms_for_color))
    ax.scatter(
        xs,
        ys,
        c=colors,
        s=args.point_size,
        alpha=0.95,
        edgecolors="black",
        linewidths=0.5,
    )
    if args.label_channels:
        for i, (x, y, c) in enumerate(zip(xs, ys, colors), start=1):
            ax.text(
                x,
                y,
                f"{i}",
                color="grey",
                fontsize=7,
                ha="center",
                va="center",
                fontweight="bold",
            )
    ax.scatter([cx], [cy], c="black", s=90, marker="x")
    sm = plt.cm.ScalarMappable(norm=norm, cmap="coolwarm")
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
    cbar_label = "log10(RMS) (activation)" if args.log_scale else "RMS (activation)"
    cbar.set_label(cbar_label)
    ax.set_title(f"{args.subject} | {args.trial} | {args.label}")
    ax.axis("off")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


if __name__ == "__main__":
    main()
