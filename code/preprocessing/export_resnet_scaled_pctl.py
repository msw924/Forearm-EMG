import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib.image as mpimg

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


def channel_base_color(ch_idx):
    anchors = [
        (4.0, (1.0, 0.0, 0.0)),   # red
        (12.0, (0.0, 0.0, 1.0)),  # blue band
        (20.0, (0.0, 1.0, 0.0)),  # green peak
        (28.0, (0.0, 0.0, 1.0)),  # blue band
        (32.0, (0.0, 0.0, 1.0)),  # blue tail
    ]
    ch = float(ch_idx)
    if ch <= anchors[0][0]:
        return np.array(anchors[0][1], dtype=float)
    for (x0, c0), (x1, c1) in zip(anchors, anchors[1:]):
        if ch <= x1:
            t = (ch - x0) / (x1 - x0) if x1 > x0 else 0.0
            c0 = np.array(c0, dtype=float)
            c1 = np.array(c1, dtype=float)
            return c0 * (1.0 - t) + c1 * t
    return np.array(anchors[-1][1], dtype=float)


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


def save_emg_rms_image(seg, out_path, bins=224, shifts=None, scale_pctl=95.0):
    n = seg.shape[1]
    if n <= 0:
        return
    bins_src = min(bins, n)
    edges = np.linspace(0, n, bins_src + 1, dtype=int)
    rms_bins = np.zeros((seg.shape[0], bins_src), dtype=float)
    for i in range(bins_src):
        s = edges[i]
        e = edges[i + 1]
        if e <= s:
            e = min(n, s + 1)
        chunk = seg[:, s:e]
        rms_bins[:, i] = np.sqrt(np.mean(chunk * chunk, axis=1))
    if bins_src != bins:
        x_src = np.linspace(0, 1, bins_src)
        x_dst = np.linspace(0, 1, bins)
        rms_bins = np.vstack([np.interp(x_dst, x_src, row) for row in rms_bins])

    ref = float(np.percentile(rms_bins, scale_pctl))
    denom = ref if ref > 0 else 1.0
    norm = np.clip(rms_bins / denom, 0.0, 1.0)

    height = 32 * 7
    img = np.ones((height, bins, 3), dtype=float)
    row_order = [3, 2, 4, 1, 5, 0, 6]
    for ch in range(32):
        base = channel_base_color(ch + 1)
        row_base = ch * 7
        for t_idx in range(bins):
            level = norm[ch, t_idx]
            n_rows = int(round(1 + level * 6))
            n_rows = max(1, min(7, n_rows))
            for i in range(n_rows):
                rr = row_base + row_order[i]
                img[rr, t_idx, :] = base

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not shifts:
        mpimg.imsave(out_path, img, vmin=0.0, vmax=1.0)
        return

    for frac in shifts:
        shift_cols = int(round(frac * bins))
        shifted = np.roll(img, shift_cols, axis=1)
        suffix = f"_shift{int(round(frac*100)):02d}"
        out_shift = out_path.with_name(out_path.stem + suffix + out_path.suffix)
        mpimg.imsave(out_shift, shifted, vmin=0.0, vmax=1.0)


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


def safe_label(label):
    return label.lower().replace(" ", "_").replace("/", "_")


def parse_subject_list(val):
    if not val:
        return None
    return [v.strip() for v in val.split(",") if v.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output root for RMS images")
    ap.add_argument("--subjects", type=str, default=None, help="Comma-separated subject folder names")
    ap.add_argument("--subjects-range", type=str, default=None, help="Range like 1-10")
    ap.add_argument("--offsets", type=str, default=None, help="CSV of manual offsets")
    ap.add_argument("--noise", type=str, default=None, help="CSV of trial noise info")
    ap.add_argument("--rms-bins", type=int, default=224, help="Time bins for RMS image width")
    ap.add_argument("--rms-pctl", type=float, default=95.0, help="Percentile for per-trial scaling")
    ap.add_argument("--rms-shifts", type=str, default=None, help="Comma-separated fractional shifts")
    ap.add_argument("--trial", type=str, default=None, help="Only process a specific trial directory name")
    ap.add_argument("--jitters", type=int, default=1, help="Number of jittered windows per movement")
    ap.add_argument("--jitter-sec", type=float, default=0.1, help="Jitter size in seconds (+/-)")
    args = ap.parse_args()

    subjects_list = parse_subject_list(args.subjects)
    subjects_range = None
    if args.subjects_range:
        a, b = args.subjects_range.split("-")
        subjects_range = (int(a), int(b))

    out_root = Path(args.out)
    emg_root = out_root / "emg_rms"

    offsets = load_offsets(args.offsets)
    noise = load_noise(args.noise)

    jitter_offsets = [0.0]
    if args.jitters > 1:
        jitter_offsets = np.linspace(-args.jitter_sec, args.jitter_sec, args.jitters).tolist()

    rms_shifts = None
    if args.rms_shifts:
        rms_shifts = [float(v.strip()) for v in args.rms_shifts.split(",") if v.strip()]

    for subject_dir, trial_dir, emg_path, aux_path, log_path in iter_trials(subjects_list, subjects_range):
        if args.trial and trial_dir.name != args.trial:
            continue
        signal = load_emg(emg_path)
        noise_key = (subject_dir.name, trial_dir.name, emg_path.name)
        if noise_key in noise:
            _, excluded = noise[noise_key]
            signal = apply_drop_channel(signal, excluded)
        aux = load_aux(aux_path)
        if log_path is None:
            continue

        aux_start, aux_end, move_starts, labels, move_duration, _, _, per_label_offsets = compute_move_markers(
            signal, aux, log_path
        )
        if not move_starts or not labels or move_duration is None:
            continue

        for t0, label, off in zip(move_starts, labels, per_label_offsets):
            for j_idx, jitter in enumerate(jitter_offsets):
                key = (subject_dir.name, trial_dir.name, emg_path.name, label)
                if key in offsets:
                    off = offsets[key]
                start_t = t0 + off + jitter
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
                label_dir = safe_label(label)
                subject_name = subject_dir.name
                trial_name = trial_dir.name.replace("trial", "(00)trial")
                label_name = f"(00){label_dir}"
                base = f"{subject_name}-{trial_name}-{label_name}_j{j_idx}.png"
                rms_out = emg_root / label_dir / base
                save_emg_rms_image(
                    seg,
                    rms_out,
                    bins=args.rms_bins,
                    shifts=rms_shifts,
                    scale_pctl=args.rms_pctl,
                )


if __name__ == "__main__":
    main()
