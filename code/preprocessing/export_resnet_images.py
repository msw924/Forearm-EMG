import argparse
import csv
import math
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BASE_DATA_DIR = Path("/Users/maxwilliams/Library/CloudStorage/Dropbox/MW_GR_data/Data")
SENSOR_UNIT = "DA1"
FS = 2049.122
ADC_RES = 16
DIN = 2.4
GAIN = 192
MAX_LEV = 2**ADC_RES - 1
NUM_CHANNELS = 32


def load_emg(emg_path):
    print(f"Reading EMG from: {emg_path}")
    val = np.fromfile(emg_path, dtype="<u2")
    n_frames = val.size // NUM_CHANNELS
    val = val[: n_frames * NUM_CHANNELS]
    signal = val.reshape(n_frames, NUM_CHANNELS).T
    scaled = (signal.astype(np.float64) / MAX_LEV) * (DIN / GAIN)
    return scaled - scaled.mean(axis=1, keepdims=True)


def load_aux(aux_path):
    if not aux_path.exists():
        return np.array([], dtype=np.float64)
    print(f"Reading AUX from: {aux_path}")
    aux = np.fromfile(aux_path, dtype="<u2")
    return aux.astype(np.float64) / MAX_LEV


def detect_aux_pulse(aux):
    if aux.size == 0:
        return None, None
    aux_1d = aux.reshape(-1)
    aux_thresh = 0.5 * np.max(aux_1d)
    above = np.flatnonzero(aux_1d > aux_thresh)
    if above.size == 0:
        return None, None
    return above[0] / FS, above[-1] / FS


def parse_log(log_path):
    if log_path is None or not log_path.exists():
        return None, None, None, []
    log_text = ""
    if log_path.suffix.lower() == ".rtf":
        rtf_raw = log_path.read_text(encoding="utf-8", errors="ignore")
        rtf_text = re.sub(r"\\(par|line)\\b", "\n", rtf_raw)
        rtf_text = rtf_text.replace("\\\n", "\n")
        rtf_text = re.sub(r"\\[a-zA-Z]+-?\\d* ?", "", rtf_text)
        rtf_text = re.sub(r"[{}]", "", rtf_text)
        rtf_text = re.sub(r"\n+", "\n", rtf_text)
        log_text = rtf_text.strip()
    else:
        log_text = log_path.read_text(encoding="utf-8", errors="ignore")

    pulse_start_time_sec = None
    pre_move_wait_sec = None
    post_move_wait_sec = None
    move_labels = []
    if log_text:
        pulse_match = re.search(r"Sending pulse \\(start\\) at ([0-9.]+) s", log_text)
        if pulse_match:
            pulse_start_time_sec = float(pulse_match.group(1))
        wait_match = re.search(r"Waiting ([0-9.]+) s before text prompts", log_text)
        if wait_match:
            pre_move_wait_sec = float(wait_match.group(1))
        post_match = re.search(r"Waiting ([0-9.]+) s before final pulse", log_text)
        if post_match:
            post_move_wait_sec = float(post_match.group(1))
        lines = [ln.strip() for ln in log_text.splitlines() if ln.strip()]
        start_idx = -1
        try:
            start_idx = lines.index("Waiting 10 s before text prompts...")
        except ValueError:
            for i, ln in enumerate(lines):
                if ln.startswith("Waiting ") and "before text prompts" in ln:
                    start_idx = i
                    break
        if start_idx != -1:
            for ln in lines[start_idx + 1 :]:
                if ln.startswith("Waiting ") or "Sending pulse" in ln:
                    break
                move_labels.append(ln)
    return pulse_start_time_sec, pre_move_wait_sec, post_move_wait_sec, move_labels


def compute_move_markers(signal, aux, log_path, move_duration_override=None):
    aux_start, aux_end = detect_aux_pulse(aux)
    pulse_start, pre_wait, post_wait, labels = parse_log(log_path)
    move_duration = move_duration_override
    if move_duration is None and aux_start is not None and aux_end is not None and pre_wait is not None and post_wait is not None and labels:
        total_span = aux_end - aux_start
        active_span = total_span - pre_wait - post_wait
        move_duration = active_span / len(labels) if active_span > 0 else 10.0
    if pre_wait is None or move_duration is None or not labels:
        return aux_start, aux_end, None, None, None, None, None, None
    pulse_start_rel = aux_start if aux_start is not None else 0.0
    move_starts = [pulse_start_rel + pre_wait + i * move_duration for i in range(len(labels))]
    per_label_offsets = [1.0 for _ in labels]
    return aux_start, aux_end, move_starts, labels, move_duration, 1.0, None, per_label_offsets


def normalize_label(label):
    return label.strip().lower().replace(" ", "_")


def parse_subjects(args):
    subjects_list = [v.strip() for v in args.subjects.split(",")] if args.subjects else None
    subjects_range = None
    if args.subjects_range:
        a, b = args.subjects_range.split("-")
        subjects_range = (int(a), int(b))
    return subjects_list, subjects_range


def select_subject_dirs(subjects_list=None, subjects_range=None):
    subject_dirs = sorted([p for p in BASE_DATA_DIR.glob("(??)*") if p.is_dir()])
    if subjects_list:
        name_set = set(subjects_list)
        subject_dirs = [p for p in subject_dirs if p.name in name_set]
    elif subjects_range:
        start, end = subjects_range
        want = set([f"({i:02d})" for i in range(start, end + 1)])
        subject_dirs = [p for p in subject_dirs if p.name[:4] in want]
    return subject_dirs


def iter_trials(subjects_list=None, subjects_range=None, trial_name=None):
    subject_dirs = select_subject_dirs(subjects_list, subjects_range)
    for subject_dir in subject_dirs:
        trial_dirs = sorted([p for p in subject_dir.iterdir() if p.is_dir() and p.name.lower().startswith("trial")])
        if trial_name:
            trial_dirs = [p for p in trial_dirs if p.name.lower() == trial_name.lower()]
        for trial_dir in trial_dirs:
            emg_files = sorted(trial_dir.glob(f"*_M02{SENSOR_UNIT}_EMG_raw.sig"))
            if not emg_files:
                continue
            log_candidates = list(trial_dir.glob("*.rtf")) + list(trial_dir.glob("*.txt")) + list(trial_dir.glob("*.pdf"))
            log_path = log_candidates[0] if log_candidates else None
            for emg_path in emg_files:
                aux_path = emg_path.with_name(emg_path.name.replace("_EMG_raw.sig", "_AUX1_raw.sig"))
                yield subject_dir, trial_dir, emg_path, aux_path, log_path


def load_offsets(path):
    if not path or not Path(path).exists():
        return {}
    offsets = {}
    with Path(path).open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["subject"], row["trial"], row["emg_file"], row["label"])
            offsets[key] = float(row["offset_sec"])
    return offsets


def load_noise(path):
    if not path or not Path(path).exists():
        return {}
    noise = {}
    with Path(path).open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["subject"], row["trial"], row["emg_file"])
            ch_raw = row.get("excluded_channels", "").strip()
            excluded = [int(c) for c in ch_raw.split(",") if c.strip().isdigit()] if ch_raw else []
            noise[key] = excluded
    return noise


def load_skip(path):
    if not path or not Path(path).exists():
        return set()
    skip = set()
    with Path(path).open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            skip.add((row["subject"], row["trial"]))
    return skip


def apply_filters(signal, bandpass_low=None, bandpass_high=None, notch=None, notch_width=2.0):
    if notch:
        b, a = iirnotch(notch, notch_width, FS)
        signal = filtfilt(b, a, signal, axis=1)
    if bandpass_low and bandpass_high:
        nyq = FS / 2.0
        low = bandpass_low / nyq
        high = bandpass_high / nyq
        b, a = butter(4, [low, high], btype="bandpass")
        signal = filtfilt(b, a, signal, axis=1)
    return signal


def replace_dropped_channels(signal, dropped):
    if not dropped:
        return signal
    fixed = signal.copy()
    channels = fixed.shape[0]
    for ch in sorted(set(dropped)):
        idx = ch - 1
        if idx < 0 or idx >= channels:
            continue
        left = max(idx - 1, 0)
        right = min(idx + 1, channels - 1)
        if left == idx and right == idx:
            continue
        fixed[idx] = 0.5 * (fixed[left] + fixed[right]) if left != right else fixed[left]
    return fixed


def rms_bins(signal, start_idx, end_idx, bins):
    seg = signal[:, start_idx:end_idx]
    if seg.size == 0:
        return None
    length = seg.shape[1]
    edges = np.linspace(0, length, bins + 1, dtype=int)
    out = np.zeros((seg.shape[0], bins), dtype=np.float64)
    for i in range(bins):
        a = edges[i]
        b = edges[i + 1]
        if b <= a:
            out[:, i] = 0.0
        else:
            window = seg[:, a:b]
            out[:, i] = np.sqrt(np.mean(window * window, axis=1))
    return out


def stripe7_image(rms_matrix, rgb_colors, rms_min, rms_max):
    rows = rms_matrix.shape[0]
    bins = rms_matrix.shape[1]
    img = np.ones((rows * 7, bins, 3), dtype=np.float32)
    denom = max(rms_max - rms_min, 1e-12)
    norm = np.clip((rms_matrix - rms_min) / denom, 0.0, 1.0)
    for ch in range(rows):
        base = ch * 7
        row_color = rgb_colors[ch]
        for i in range(bins):
            level = norm[ch, i]
            if level <= 0:
                continue
            if level >= 1.0:
                img[base: base + 7, i, :] = row_color
            else:
                mid = base + 3
                img[mid, i, :] = row_color
                if level > 0.33:
                    img[mid - 1, i, :] = row_color
                    img[mid + 1, i, :] = row_color
                if level > 0.66:
                    img[base: base + 7, i, :] = row_color
    return img


def gray_image(rms_matrix, rms_min, rms_max):
    denom = max(rms_max - rms_min, 1e-12)
    norm = np.clip((rms_matrix - rms_min) / denom, 0.0, 1.0)
    return np.repeat(norm[:, :, None], 3, axis=2)


def channel_colors(highlight_channels=None):
    # highlight_channels: set of 1-indexed channels to use as the green gradient region.
    # Keeps the same red/green/blue gradient style as the default but focuses green on exactly those channels.
    # Default: red gradient centered at ch4 (1-9), green gradient centered at ch18 (15-24), blue elsewhere.
    colors = np.zeros((32, 3), dtype=np.float32)
    if highlight_channels:
        ch_list = sorted(highlight_channels)
        center = sum(ch_list) / len(ch_list)
        half_span = max(abs(c - center) for c in ch_list) or 1.0
        for ch in range(1, 33):
            if ch in highlight_channels:
                t = 1.0 - abs(ch - center) / half_span
                g = float(np.clip(t, 0.0, 1.0))
                colors[ch - 1] = (0.0, g, 1.0 - g)
            elif 1 <= ch <= 9:
                t = 1.0 - abs(ch - 4) / 5.0
                r = float(np.clip(t, 0.0, 1.0))
                colors[ch - 1] = (r, 0.0, 1.0 - r)
            else:
                colors[ch - 1] = (0.0, 0.0, 1.0)
        return colors
    for ch in range(1, 33):
        if 15 <= ch <= 24:
            t = 1.0 - abs(ch - 18) / 5.0
            g = float(np.clip(t, 0.0, 1.0))
            b = 1.0 - g
            r = 0.0
        elif 1 <= ch <= 9:
            t = 1.0 - abs(ch - 4) / 5.0
            r = float(np.clip(t, 0.0, 1.0))
            b = 1.0 - r
            g = 0.0
        else:
            r, g, b = 0.0, 0.0, 1.0
        colors[ch - 1] = (r, g, b)
    return colors


def compute_global_rms_range(items, offsets, noise, window_sec, bandpass_low, bandpass_high, notch, notch_width, bins):
    rms_min = math.inf
    rms_max = -math.inf
    for subject_dir, trial_dir, emg_path, aux_path, log_path in items:
        if log_path is None:
            continue
        signal = load_emg(emg_path)
        signal = apply_filters(signal, bandpass_low, bandpass_high, notch, notch_width)
        excluded = noise.get((subject_dir.name, trial_dir.name, emg_path.name), [])
        signal = replace_dropped_channels(signal, excluded)
        aux = load_aux(aux_path)
        _, _, move_starts, labels, move_duration, _, _, per_label_offsets = compute_move_markers(
            signal, aux, log_path
        )
        if move_starts is None or labels is None or move_duration is None:
            continue
        for idx, label in enumerate(labels):
            key = (subject_dir.name, trial_dir.name, emg_path.name, label)
            offset = offsets.get(key, per_label_offsets[idx])
            start = move_starts[idx] + offset
            end = start + window_sec
            start_idx = max(int(round(start * FS)), 0)
            end_idx = min(int(round(end * FS)), signal.shape[1])
            rms = rms_bins(signal, start_idx, end_idx, bins)
            if rms is None:
                continue
            rms_min = min(rms_min, float(np.min(rms)))
            rms_max = max(rms_max, float(np.max(rms)))
    if not math.isfinite(rms_min) or not math.isfinite(rms_max):
        return 0.0, 1.0
    return rms_min, rms_max


def save_image(img, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(out_path, img, vmin=0.0, vmax=1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Legacy output path (kept for compatibility)")
    ap.add_argument("--subjects", type=str, default=None, help="Comma-separated subject folder names")
    ap.add_argument("--subjects-range", type=str, default=None, help="Range like 1-10")
    ap.add_argument("--trial", type=str, default=None, help="Trial folder name filter (e.g., 'trial 3')")
    ap.add_argument("--offsets", type=str, default=None)
    ap.add_argument("--noise", type=str, default=None)
    ap.add_argument("--skip", type=str, default=None)
    ap.add_argument("--emg-rms-out", type=str, default=None)
    ap.add_argument("--rms-bins", type=int, default=224)
    ap.add_argument("--emg-rms-style", type=str, default="stripe7", choices=["stripe7", "gray"])
    ap.add_argument("--rms-shifts", type=str, default=None, help="Comma-separated shift fractions, e.g. 0,0.2,0.4")
    ap.add_argument("--jitters", type=int, default=1)
    ap.add_argument("--jitter-sec", type=float, default=0.0)
    ap.add_argument("--window-sec", type=float, default=8.0)
    ap.add_argument("--bandpass-low", type=float, default=None)
    ap.add_argument("--bandpass-high", type=float, default=None)
    ap.add_argument("--notch", type=float, default=None)
    ap.add_argument("--notch-width", type=float, default=2.0)
    ap.add_argument("--highlight-channels", type=str, default=None, help="Comma-separated 1-indexed channel numbers to color pure green (all others blue), e.g. 16,17,18,19,20")
    args = ap.parse_args()

    subjects_list, subjects_range = parse_subjects(args)
    offsets = load_offsets(args.offsets)
    noise = load_noise(args.noise)
    skip = load_skip(args.skip)

    items = []
    for subject_dir, trial_dir, emg_path, aux_path, log_path in iter_trials(
        subjects_list, subjects_range, args.trial
    ):
        if (subject_dir.name, trial_dir.name) in skip:
            continue
        items.append((subject_dir, trial_dir, emg_path, aux_path, log_path))

    if not args.emg_rms_out:
        print("No --emg-rms-out specified; nothing to do.")
        return

    rms_min, rms_max = compute_global_rms_range(
        items,
        offsets,
        noise,
        args.window_sec,
        args.bandpass_low,
        args.bandpass_high,
        args.notch,
        args.notch_width,
        args.rms_bins,
    )

    highlight_channels = None
    if args.highlight_channels:
        highlight_channels = set(int(c) for c in args.highlight_channels.split(",") if c.strip().isdigit())
    rgb_colors = channel_colors(highlight_channels)
    shift_vals = [0.0]
    if args.rms_shifts:
        shift_vals = [float(v) for v in args.rms_shifts.split(",") if v.strip() != ""]
    jitter_vals = [0.0]
    if args.jitters and args.jitters > 1:
        jitter_vals = np.linspace(-args.jitter_sec, args.jitter_sec, args.jitters).tolist()

    out_root = Path(args.emg_rms_out) / "emg_rms"

    for subject_dir, trial_dir, emg_path, aux_path, log_path in items:
        if log_path is None:
            continue
        signal = load_emg(emg_path)
        signal = apply_filters(signal, args.bandpass_low, args.bandpass_high, args.notch, args.notch_width)
        excluded = noise.get((subject_dir.name, trial_dir.name, emg_path.name), [])
        signal = replace_dropped_channels(signal, excluded)
        aux = load_aux(aux_path)
        _, _, move_starts, labels, move_duration, _, _, per_label_offsets = compute_move_markers(
            signal, aux, log_path
        )
        if move_starts is None or labels is None or move_duration is None:
            continue

        for idx, label in enumerate(labels):
            key = (subject_dir.name, trial_dir.name, emg_path.name, label)
            offset = offsets.get(key, per_label_offsets[idx])
            for j_idx, jitter in enumerate(jitter_vals):
                start = move_starts[idx] + offset + jitter
                end = start + args.window_sec
                start_idx = max(int(round(start * FS)), 0)
                end_idx = min(int(round(end * FS)), signal.shape[1])
                rms = rms_bins(signal, start_idx, end_idx, args.rms_bins)
                if rms is None:
                    continue
                if args.emg_rms_style == "gray":
                    img = gray_image(rms, rms_min, rms_max)
                else:
                    img = stripe7_image(rms, rgb_colors, rms_min, rms_max)

                label_slug = normalize_label(label)
                trial_tag = f"(00){trial_dir.name}"
                move_tag = f"(00){label_slug}"
                base = f"{subject_dir.name}-{trial_tag}-{move_tag}_j{j_idx}"
                out_path = out_root / label_slug / f"{base}.png"
                save_image(img, out_path)

                # Shifted variants (column roll)
                for shift in shift_vals:
                    if abs(shift) < 1e-9:
                        continue
                    cols = img.shape[1]
                    shift_cols = int(round(shift * cols))
                    if shift_cols == 0:
                        continue
                    shifted = np.roll(img, shift_cols, axis=1)
                    out_shift = out_root / label_slug / f"{base}_shift{int(round(shift * 100)):02d}.png"
                    save_image(shifted, out_shift)


if __name__ == "__main__":
    main()
