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


def iter_trials(subjects_list=None, subjects_range=None, trials_list=None):
    subject_dirs = select_subject_dirs(subjects_list, subjects_range)
    for subject_dir in subject_dirs:
        trial_dirs = sorted([p for p in subject_dir.glob(trial_glob) if p.is_dir()])
        if trials_list:
            name_set = set(trials_list)
            trial_dirs = [p for p in trial_dirs if p.name in name_set]
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


def load_layout(path, num_channels):
    if not path:
        rows = 4
        cols = 8
        if rows * cols != num_channels:
            raise SystemExit("Default 4x8 layout does not match channel count.")
        grid = np.arange(num_channels).reshape(rows, cols)
        return grid
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Layout file not found: {path}")
    mapping = {}
    with p.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ch = int(row["channel"]) - 1
            r = int(row["row"])
            c = int(row["col"])
            mapping[(r, c)] = ch
    max_r = max(r for r, _ in mapping.keys())
    max_c = max(c for _, c in mapping.keys())
    grid = np.full((max_r + 1, max_c + 1), -1, dtype=int)
    for (r, c), ch in mapping.items():
        grid[r, c] = ch
    return grid


def compute_global_rms_range(subjects_list, subjects_range, offsets, noise, skip, slice_len):
    rms_min = None
    rms_max = None
    for subject_dir, trial_dir, emg_path, aux_path, log_path in iter_trials(subjects_list, subjects_range):
        if (subject_dir.name, trial_dir.name) in skip:
            continue
        if log_path is None:
            continue
        signal = load_emg(emg_path)
        noise_key = (subject_dir.name, trial_dir.name, emg_path.name)
        if noise_key in noise:
            signal = apply_drop_channel(signal, noise[noise_key][1])
        aux = load_aux(aux_path)
        aux_start, aux_end, move_starts, labels, move_duration, _, _, per_label_offsets = compute_move_markers(
            signal, aux, log_path
        )
        if not move_starts or not labels or move_duration is None:
            continue
        for t0, label, off in zip(move_starts, labels, per_label_offsets):
            key = (subject_dir.name, trial_dir.name, emg_path.name, label)
            if key in offsets:
                off = offsets[key]
            start_t = t0 + off
            end_t = start_t + max(move_duration - off, 0.0)
            if end_t <= start_t:
                continue
            start_idx = int(round(start_t * fs))
            end_idx = int(round(end_t * fs))
            start_idx = max(start_idx, 0)
            end_idx = min(end_idx, signal.shape[1])
            for idx in range(start_idx, end_idx - slice_len + 1, slice_len):
                seg = signal[:, idx:idx + slice_len]
                rms = np.sqrt(np.mean(seg * seg, axis=1))
                rmin = float(np.min(rms))
                rmax = float(np.max(rms))
                rms_min = rmin if rms_min is None else min(rms_min, rmin)
                rms_max = rmax if rms_max is None else max(rms_max, rmax)
    if rms_min is None or rms_max is None:
        rms_min, rms_max = 0.0, 1.0
    return rms_min, rms_max


def save_srf_image(rms_stack, out_path, vmin, vmax):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(2.24, 2.24), dpi=100)
    ax = fig.add_subplot(111)
    ax.imshow(rms_stack, aspect="auto", cmap="magma", vmin=vmin, vmax=vmax, origin="lower")
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(out_path, dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_ts_image(rms_vals, grid, out_path, vmin, vmax):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros(grid.shape, dtype=float)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            ch = grid[r, c]
            if ch >= 0:
                img[r, c] = rms_vals[ch]
    fig = plt.figure(figsize=(2.24, 2.24), dpi=100)
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap="magma", vmin=vmin, vmax=vmax)
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(out_path, dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output root for images")
    ap.add_argument("--mode", choices=["srf", "ts"], required=True)
    ap.add_argument("--subjects", type=str, default=None)
    ap.add_argument("--subjects-range", type=str, default=None)
    ap.add_argument("--trials", type=str, default=None, help="Comma-separated trial folder names")
    ap.add_argument("--offsets", type=str, default=None)
    ap.add_argument("--noise", type=str, default=None)
    ap.add_argument("--skip", type=str, default=None)
    ap.add_argument("--slice-sec", type=float, default=0.25)
    ap.add_argument("--step-sec", type=float, default=0.05)
    ap.add_argument("--history-slices", type=int, default=10)
    ap.add_argument("--layout", type=str, default=None, help="CSV with channel,row,col")
    args = ap.parse_args()

    subjects_list = parse_subject_list(args.subjects)
    subjects_range = None
    if args.subjects_range:
        a, b = args.subjects_range.split("-")
        subjects_range = (int(a), int(b))
    trials_list = parse_subject_list(args.trials)

    offsets = load_offsets(args.offsets)
    noise = load_noise(args.noise)
    skip = load_skip(args.skip)

    slice_len = int(round(args.slice_sec * fs))
    step_len = int(round(args.step_sec * fs))
    if slice_len <= 0 or step_len <= 0:
        raise SystemExit("Invalid slice/step length.")

    rms_min, rms_max = compute_global_rms_range(
        subjects_list, subjects_range, offsets, noise, skip, slice_len
    )

    grid = None
    if args.mode == "ts":
        grid = load_layout(args.layout, num_channels=32)

    out_root = Path(args.out)
    for subject_dir, trial_dir, emg_path, aux_path, log_path in iter_trials(
        subjects_list, subjects_range, trials_list
    ):
        if (subject_dir.name, trial_dir.name) in skip:
            continue
        if log_path is None:
            continue
        signal = load_emg(emg_path)
        noise_key = (subject_dir.name, trial_dir.name, emg_path.name)
        if noise_key in noise:
            signal = apply_drop_channel(signal, noise[noise_key][1])
        aux = load_aux(aux_path)
        aux_start, aux_end, move_starts, labels, move_duration, _, _, per_label_offsets = compute_move_markers(
            signal, aux, log_path
        )
        if not move_starts or not labels or move_duration is None:
            continue

        for t0, label, off in zip(move_starts, labels, per_label_offsets):
            key = (subject_dir.name, trial_dir.name, emg_path.name, label)
            if key in offsets:
                off = offsets[key]
            start_t = t0 + off
            end_t = start_t + max(move_duration - off, 0.0)
            if end_t <= start_t:
                continue
            start_idx = int(round(start_t * fs))
            end_idx = int(round(end_t * fs))
            start_idx = max(start_idx, 0)
            end_idx = min(end_idx, signal.shape[1])
            if end_idx - start_idx < slice_len:
                continue

            label_dir = safe_label(label)
            subject_name = subject_dir.name
            trial_name = trial_dir.name.replace("trial", "(00)trial")
            label_name = f"(00){label_dir}"
            base_root = f"{subject_name}-{trial_name}-{label_name}"

            slice_idx = 0
            for idx in range(start_idx, end_idx - slice_len + 1, step_len):
                seg = signal[:, idx:idx + slice_len]
                rms = np.sqrt(np.mean(seg * seg, axis=1))
                if args.mode == "ts":
                    out_path = out_root / "ts" / label_dir / f"{base_root}_s{slice_idx:04d}.png"
                    save_ts_image(rms, grid, out_path, rms_min, rms_max)
                else:
                    history = []
                    for k in range(args.history_slices):
                        h_end = idx - k * step_len
                        h_start = h_end - slice_len
                        if h_start < 0:
                            break
                        h_seg = signal[:, h_start:h_end]
                        h_rms = np.sqrt(np.mean(h_seg * h_seg, axis=1))
                        history.append(h_rms)
                    if not history:
                        continue
                    history = history[::-1]
                    rms_stack = np.stack(history, axis=1)
                    out_path = out_root / "srf" / label_dir / f"{base_root}_s{slice_idx:04d}.png"
                    save_srf_image(rms_stack, out_path, rms_min, rms_max)
                slice_idx += 1


if __name__ == "__main__":
    main()
