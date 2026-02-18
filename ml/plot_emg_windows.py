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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--trial", required=True)
    ap.add_argument("--offsets", type=str, default=None, help="Offsets CSV (old UI)")
    ap.add_argument("--out", required=True, help="Output image path")
    ap.add_argument("--ds", type=int, default=10, help="Downsample for display")
    args = ap.parse_args()

    emg_path, aux_path, log_path = find_trial(args.subject, args.trial)
    if log_path is None:
        raise SystemExit("No log file found for this trial.")

    signal = load_emg(emg_path)
    aux = load_aux(aux_path)
    aux_start, aux_end, move_starts, labels, move_duration, _, _, per_label_offsets = compute_move_markers(
        signal, aux, log_path
    )
    if not move_starts or not labels or move_duration is None:
        raise SystemExit("No movement windows found.")

    offsets = load_offsets(args.offsets)
    ds = max(args.ds, 1)
    sig = signal[:, ::ds]
    t = np.arange(sig.shape[1]) / (fs / ds)

    amp = float(np.max(np.abs(sig)))
    offset = 1.2 * amp if amp > 0 else 1.0
    y = sig + offset * np.arange(sig.shape[0])[:, None]

    fig, ax = plt.subplots(figsize=(14, 6))
    for ch in range(y.shape[0]):
        ax.plot(t, y[ch], linewidth=0.3, color="black", alpha=0.7)

    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
    for idx, (t0, label, off) in enumerate(zip(move_starts, labels, per_label_offsets)):
        key = (args.subject, args.trial, emg_path.name, label)
        if key in offsets:
            off = offsets[key]
        start_t = t0 + off
        end_t = start_t + max(move_duration - off, 0.0)
        ax.axvline(start_t, color=colors[idx], linewidth=1.2)
        ax.axvline(end_t, color=colors[idx], linewidth=1.2, linestyle="--")
        ax.text(
            start_t,
            y[-1, -1] + offset * 0.2,
            label,
            rotation=90,
            color=colors[idx],
            fontsize=8,
            va="bottom",
            ha="center",
        )

    ax.set_title(f"{args.subject} | {args.trial} | {emg_path.name}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channels (stacked)")
    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
