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
    subject_glob,
    trial_glob,
    sensor_unit,
    fs,
    load_emg,
    load_aux,
    compute_move_markers,
)


def safe_label(label):
    return label.lower().replace(" ", "_").replace("/", "_")


def parse_subject_list(val):
    if not val:
        return None
    return [v.strip() for v in val.split(",") if v.strip()]


def select_subject_dirs(subjects_list=None, subjects_range=None):
    if subjects_list or subjects_range:
        subject_dirs = sorted([p for p in base_data_dir.iterdir() if p.is_dir()])
    else:
        subject_dirs = sorted(base_data_dir.glob(subject_glob))
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
        trial_dirs = sorted([p for p in subject_dir.glob(trial_glob) if p.is_dir()])
        for trial_dir in trial_dirs:
            emg_files = sorted(trial_dir.glob(f"*_M02{sensor_unit}_EMG_raw.sig"))
            if not emg_files:
                continue
            log_candidates = list(trial_dir.glob("*.rtf")) + list(trial_dir.glob("*.txt")) + list(trial_dir.glob("*.pdf"))
            log_path = log_candidates[0] if log_candidates else None
            for emg_path in emg_files:
                aux_path = emg_path.with_name(emg_path.name.replace("_EMG_raw.sig", "_AUX1_raw.sig"))
                yield subject_dir, trial_dir, emg_path, aux_path, log_path


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
            ch_raw = row.get("excluded_channel", "").strip()
            excluded = int(ch_raw) if ch_raw else None
            noise[key] = (float(row["noise_score"]), excluded)
    return noise


def load_skip(path):
    if not path:
        return set()
    p = Path(path)
    if not p.exists():
        return set()
    skip = set()
    with p.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            subject = row.get("subject", "").strip()
            trial = row.get("trial", "").strip()
            if subject and trial:
                skip.add((subject, trial))
    return skip


def apply_drop_channel(signal, excluded_channel):
    if excluded_channel is None:
        return signal
    idx = excluded_channel - 1
    if idx < 0 or idx >= signal.shape[0]:
        return signal
    out = signal.copy()
    if idx == 0:
        out[idx, :] = out[idx + 1, :]
    elif idx == signal.shape[0] - 1:
        out[idx, :] = out[idx - 1, :]
    else:
        out[idx, :] = 0.5 * (out[idx - 1, :] + out[idx + 1, :])
    return out


def resample_segment(seg, target_len):
    if seg.shape[1] == target_len:
        return seg
    x_old = np.linspace(0.0, 1.0, seg.shape[1])
    x_new = np.linspace(0.0, 1.0, target_len)
    out = np.empty((seg.shape[0], target_len), dtype=seg.dtype)
    for ch in range(seg.shape[0]):
        out[ch] = np.interp(x_new, x_old, seg[ch])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output root for .npy windows")
    ap.add_argument("--subjects", type=str, default=None)
    ap.add_argument("--subjects-range", type=str, default=None)
    ap.add_argument("--offsets", type=str, default=None)
    ap.add_argument("--windows", type=str, default=None, help="Manual windows CSV (start/end)")
    ap.add_argument("--require-windows", action="store_true", help="Skip trials missing manual windows")
    ap.add_argument("--noise", type=str, default=None)
    ap.add_argument("--skip", type=str, default=None)
    ap.add_argument("--ds", type=int, default=1)
    ap.add_argument("--target-sec", type=float, default=None, help="Resample to fixed duration")
    args = ap.parse_args()

    subjects_list = parse_subject_list(args.subjects)
    subjects_range = None
    if args.subjects_range:
        a, b = args.subjects_range.split("-")
        subjects_range = (int(a), int(b))

    offsets = load_offsets(args.offsets)
    windows = load_windows(args.windows)
    noise = load_noise(args.noise)
    skip = load_skip(args.skip)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    meta_path = out_root / "metadata.csv"

    target_len = None
    if args.target_sec is not None:
        target_len = int(round(args.target_sec * fs / args.ds))
        if target_len <= 0:
            raise SystemExit("Invalid target-sec.")

    with meta_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "subject", "trial", "emg_file", "label", "start_t", "end_t"])

        for subject_dir, trial_dir, emg_path, aux_path, log_path in iter_trials(subjects_list, subjects_range):
            if (subject_dir.name, trial_dir.name) in skip:
                continue
            signal = load_emg(emg_path)
            noise_key = (subject_dir.name, trial_dir.name, emg_path.name)
            if noise_key in noise:
                signal = apply_drop_channel(signal, noise[noise_key][1])
            if args.ds > 1:
                signal = signal[:, :: args.ds]
            manual_key = (subject_dir.name, trial_dir.name, emg_path.name)
            manual_windows = windows.get(manual_key)
            if manual_windows:
                window_iter = manual_windows
            else:
                if args.require_windows:
                    continue
                if log_path is None:
                    continue
                aux = load_aux(aux_path)
                aux_start, aux_end, move_starts, labels, move_duration, _, _, per_label_offsets = compute_move_markers(
                    signal, aux, log_path
                )
                if not move_starts or not labels or move_duration is None:
                    continue
                window_iter = []
                for t0, label, off in zip(move_starts, labels, per_label_offsets):
                    key = (subject_dir.name, trial_dir.name, emg_path.name, label)
                    if key in offsets:
                        off = offsets[key]
                    start_t = t0 + off
                    end_t = start_t + max(move_duration - off, 0.0)
                    window_iter.append((label, start_t, end_t))

            for label, start_t, end_t in window_iter:
                if not label:
                    continue
                if end_t <= start_t:
                    continue
                start_idx = int(round(start_t * fs / args.ds))
                end_idx = int(round(end_t * fs / args.ds))
                start_idx = max(start_idx, 0)
                end_idx = min(end_idx, signal.shape[1])
                if end_idx <= start_idx:
                    continue

                seg = signal[:, start_idx:end_idx]
                if target_len is not None:
                    seg = resample_segment(seg, target_len)

                label_dir = safe_label(label)
                out_dir = out_root / label_dir
                out_dir.mkdir(parents=True, exist_ok=True)

                base = f"{subject_dir.name}-{trial_dir.name}-{emg_path.stem}-{label_dir}.npy"
                out_path = out_dir / base
                np.save(out_path, seg.astype(np.float32))
                writer.writerow([out_path, subject_dir.name, trial_dir.name, emg_path.name, label, start_t, end_t])


if __name__ == "__main__":
    main()
