import argparse
import csv
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


def load_windows(path):
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    windows = {}
    with p.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["subject"], row["trial"], row["emg_file"])
            label = row.get("label", "").strip()
            try:
                start_t = float(row["start_t"])
                end_t = float(row["end_t"])
            except (TypeError, ValueError):
                continue
            windows.setdefault(key, []).append((label, start_t, end_t))
    return windows


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
    log_candidates = list(trial_dir.glob("*.rtf")) + list(trial_dir.glob("*.txt")) + list(trial_dir.glob("*.pdf"))
    log_path = log_candidates[0] if log_candidates else None
    return emg_files[0], emg_files[0].with_name(emg_files[0].name.replace("_EMG_raw.sig", "_AUX1_raw.sig")), log_path


def compute_com(vals, angle_offset_rad=0.0):
    angles = np.linspace(0, 2 * np.pi, len(vals), endpoint=False) + angle_offset_rad
    weights = np.clip(vals, 0.0, None)
    total = float(weights.sum())
    if total <= 0:
        return 0.0, 0.0, angles
    x = float(np.sum(weights * np.cos(angles)) / total)
    y = float(np.sum(weights * np.sin(angles)) / total)
    return x, y, angles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True, help="Subject folder name, e.g. (03)Manon")
    ap.add_argument("--trial", required=True, help="Trial folder name, e.g. trial 3")
    ap.add_argument("--label", required=True, help="Movement label to analyze")
    ap.add_argument("--windows", type=str, default=None, help="Manual windows CSV (start/end)")
    ap.add_argument("--angle-offset", type=float, default=0.0, help="Degrees to rotate channel angles")
    ap.add_argument("--image", type=str, default=None, help="Forearm cross-section image path")
    ap.add_argument("--center-x", type=float, default=None, help="Image center X in pixels")
    ap.add_argument("--center-y", type=float, default=None, help="Image center Y in pixels")
    ap.add_argument("--radius-px", type=float, default=None, help="Radius in pixels for channel ring")
    ap.add_argument("--out", type=str, default=None, help="Output image path")
    ap.add_argument("--out-csv", type=str, default=None, help="Optional CSV to save COM result")
    args = ap.parse_args()

    emg_path, aux_path, log_path = find_trial(args.subject, args.trial)
    signal = load_emg(emg_path)

    start_t = None
    end_t = None
    windows = load_windows(args.windows)
    win_key = (args.subject, args.trial, emg_path.name)
    if win_key in windows:
        for label, s_t, e_t in windows[win_key]:
            if label == args.label:
                start_t, end_t = s_t, e_t
                break
    if start_t is None or end_t is None:
        if log_path is None:
            raise SystemExit("No manual window found and no log file available.")
        aux = load_aux(aux_path)
        _, _, move_starts, labels, move_duration, _, _, per_label_offsets = compute_move_markers(
            signal, aux, log_path
        )
        if not move_starts or not labels or move_duration is None:
            raise SystemExit("No movement markers found.")
        for t0, label, off in zip(move_starts, labels, per_label_offsets):
            if label == args.label:
                start_t = t0 + off
                end_t = start_t + max(move_duration - off, 0.0)
                break

    if start_t is None or end_t is None:
        raise SystemExit("Target label not found for this trial.")

    start_idx = int(round(start_t * fs))
    end_idx = int(round(end_t * fs))
    start_idx = max(start_idx, 0)
    end_idx = min(end_idx, signal.shape[1])
    if end_idx <= start_idx:
        raise SystemExit("Invalid window after clamping.")

    seg = signal[:, start_idx:end_idx]
    vals = np.sqrt(np.mean(seg * seg, axis=1))
    angle_offset_rad = np.deg2rad(args.angle_offset)
    com_x, com_y, angles = compute_com(vals, angle_offset_rad=angle_offset_rad)
    print(f"COM (unit circle): x={com_x:.4f}, y={com_y:.4f}")

    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        header = not out_csv.exists()
        with out_csv.open("a", newline="") as f:
            writer = csv.writer(f)
            if header:
                writer.writerow(["subject", "trial", "emg_file", "label", "com_x", "com_y"])
            writer.writerow([args.subject, args.trial, emg_path.name, args.label, f"{com_x:.6f}", f"{com_y:.6f}"])

    if not args.image:
        return

    img = plt.imread(args.image)
    h, w = img.shape[0], img.shape[1]
    cx = args.center_x if args.center_x is not None else w / 2.0
    cy = args.center_y if args.center_y is not None else h / 2.0
    radius = args.radius_px if args.radius_px is not None else min(w, h) * 0.45

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img)
    ax.set_axis_off()

    ring_x = cx + radius * np.cos(angles)
    ring_y = cy - radius * np.sin(angles)
    ax.plot(ring_x, ring_y, "o", markersize=3, color="cyan", alpha=0.8)

    com_px = cx + radius * com_x
    com_py = cy - radius * com_y
    ax.plot([com_px], [com_py], "o", markersize=8, color="yellow")
    ax.plot([cx, com_px], [cy, com_py], "-", color="yellow", linewidth=1.5, alpha=0.8)
    ax.set_title(f"{args.subject} | {args.trial} | {args.label}", fontsize=10)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Saved {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
