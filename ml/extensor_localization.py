import argparse
import csv
import sys
from pathlib import Path

import numpy as np

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


def apply_drop_channel(signal, excluded_channel):
    if not excluded_channel:
        return signal
    if isinstance(excluded_channel, int):
        excluded = [excluded_channel]
    else:
        excluded = list(excluded_channel)
    out = signal.copy()
    for ch in sorted(set(excluded)):
        idx = ch - 1
        if idx < 0 or idx >= out.shape[0]:
            continue
        if idx == 0:
            out[idx, :] = out[idx + 1, :]
        elif idx == out.shape[0] - 1:
            out[idx, :] = out[idx - 1, :]
        else:
            out[idx, :] = 0.5 * (out[idx - 1, :] + out[idx + 1, :])
    return out


def load_offsets(path):
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    offsets = {}
    with p.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["subject"], row["trial"], row["emg_file"], row["label"])
            offsets[key] = float(row["offset_sec"])
    return offsets


def load_noise(path):
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    noise = {}
    with p.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["subject"], row["trial"], row["emg_file"])
            ch_raw = row.get("excluded_channels", "").strip()
            if ch_raw:
                excluded = [int(c) for c in ch_raw.split(",") if c.strip().isdigit()]
            else:
                excluded = []
            noise[key] = (float(row["noise_score"]), excluded)
    return noise


def select_subject_dirs(subjects_list=None, subjects_range=None):
    if subjects_list or subjects_range:
        subject_dirs = sorted([p for p in base_data_dir.iterdir() if p.is_dir()])
    else:
        subject_dirs = sorted(base_data_dir.glob("(??)*"))
    if subjects_list:
        name_set = set(subjects_list)
        subject_dirs = [p for p in subject_dirs if p.name in name_set]
    elif subjects_range:
        start, end = subjects_range
        want = set([f"({i:02d})" for i in range(start, end + 1)])
        subject_dirs = [p for p in subject_dirs if p.name[:4] in want]
    return subject_dirs


def iter_trials(subjects_list=None, subjects_range=None):
    subject_dirs = select_subject_dirs(subjects_list, subjects_range)
    for subject_dir in subject_dirs:
        trial_dirs = sorted([p for p in subject_dir.iterdir() if p.is_dir() and p.name.lower().startswith("trial")])
        for trial_dir in trial_dirs:
            emg_files = sorted(trial_dir.glob(f"*_M02{sensor_unit}_EMG_raw.sig"))
            if not emg_files:
                continue
            log_candidates = list(trial_dir.glob("*.rtf")) + list(trial_dir.glob("*.txt")) + list(trial_dir.glob("*.pdf"))
            log_path = log_candidates[0] if log_candidates else None
            for emg_path in emg_files:
                aux_path = emg_path.with_name(emg_path.name.replace("_EMG_raw.sig", "_AUX1_raw.sig"))
                yield subject_dir, trial_dir, emg_path, aux_path, log_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--subjects", type=str, default=None, help="Comma-separated subject folder names")
    ap.add_argument("--subjects-range", type=str, default=None, help="Range like 1-10")
    ap.add_argument("--offsets", type=str, default=None, help="Manual offsets CSV")
    ap.add_argument("--noise", type=str, default=None, help="Trial noise CSV")
    ap.add_argument("--label", type=str, default="Wrist Extension")
    ap.add_argument("--top-k", type=int, default=5, help="Number of top channels to summarize")
    ap.add_argument("--z-thresh", type=float, default=2.0, help="Z-score threshold for 'significant' channels")
    args = ap.parse_args()

    subjects_list = [v.strip() for v in args.subjects.split(",")] if args.subjects else None
    subjects_range = None
    if args.subjects_range:
        a, b = args.subjects_range.split("-")
        subjects_range = (int(a), int(b))

    offsets = load_offsets(args.offsets)
    noise = load_noise(args.noise)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for subject_dir, trial_dir, emg_path, aux_path, log_path in iter_trials(subjects_list, subjects_range):
        if log_path is None:
            continue
        signal = load_emg(emg_path)
        noise_key = (subject_dir.name, trial_dir.name, emg_path.name)
        if noise_key in noise:
            _, excluded = noise[noise_key]
            signal = apply_drop_channel(signal, excluded)
        aux = load_aux(aux_path)

        aux_start, aux_end, move_starts, labels, move_duration, _, _, per_label_offsets = compute_move_markers(
            signal, aux, log_path
        )
        if not move_starts or not labels or move_duration is None:
            continue

        label_map = {lbl.strip().lower(): i for i, lbl in enumerate(labels)}
        key_norm = args.label.strip().lower()
        if key_norm not in label_map:
            continue
        idx = label_map[key_norm]
        off = per_label_offsets[idx]
        key = (subject_dir.name, trial_dir.name, emg_path.name, labels[idx])
        if key in offsets:
            off = offsets[key]

        start_t = move_starts[idx] + off
        window_sec = max(move_duration - off, 0.0)
        if window_sec <= 0:
            continue
        start_idx = int(round(start_t * fs))
        end_idx = int(round((start_t + window_sec) * fs))
        start_idx = max(start_idx, 0)
        end_idx = min(end_idx, signal.shape[1])
        if end_idx <= start_idx:
            continue

        seg = signal[:, start_idx:end_idx]
        rms = np.sqrt(np.mean(seg * seg, axis=1))
        total = float(np.sum(rms)) if np.sum(rms) > 0 else 1.0

        order = np.argsort(rms)[::-1]
        top_k = min(args.top_k, rms.size)
        top_idx = (order[:top_k] + 1).tolist()
        top_sum = float(np.sum(rms[order[:top_k]]))
        top_ratio = top_sum / total

        mean = float(np.mean(rms))
        std = float(np.std(rms))
        z = (rms - mean) / (std if std > 0 else 1.0)
        sig_idx = (np.flatnonzero(z >= args.z_thresh) + 1).tolist()
        sig_count = len(sig_idx)
        sig_ratio = float(np.sum(rms[[i - 1 for i in sig_idx]])) / total if sig_idx else 0.0

        top_mean = float(np.mean(rms[order[:top_k]])) if top_k else 0.0
        rest_idx = order[top_k:]
        rest_mean = float(np.mean(rms[rest_idx])) if rest_idx.size else 0.0
        mean_ratio = top_mean / (rest_mean if rest_mean > 0 else 1.0)
        mean_to_median = float(mean / (np.median(rms) if np.median(rms) > 0 else 1.0))

        rows.append(
            {
                "subject": subject_dir.name,
                "trial": trial_dir.name,
                "emg_file": emg_path.name,
                "label": labels[idx],
                "top_k": top_k,
                "top_channels": ";".join(str(i) for i in top_idx),
                "top_ratio": top_ratio,
                "sig_count": sig_count,
                "sig_channels": ";".join(str(i) for i in sig_idx),
                "sig_ratio": sig_ratio,
                "top_mean": top_mean,
                "rest_mean": rest_mean,
                "mean_ratio": mean_ratio,
                "mean_to_median": mean_to_median,
            }
        )

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "subject",
                "trial",
                "emg_file",
                "label",
                "top_k",
                "top_channels",
                "top_ratio",
                "sig_count",
                "sig_channels",
                "sig_ratio",
                "top_mean",
                "rest_mean",
                "mean_ratio",
                "mean_to_median",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
