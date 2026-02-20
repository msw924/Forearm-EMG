import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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


def save_emg_segment_image(seg, fs, out_path, amp_ref, ds=1):
    t = np.arange(seg.shape[1]) / fs
    sig = seg[:, ::ds]
    tt = t[::ds]
    amp = amp_ref if amp_ref > 0 else np.max(np.abs(sig))
    offset = 1.2 * amp if amp > 0 else 1.0
    y = sig + offset * np.arange(sig.shape[0])[:, None]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(2.24, 2.24), dpi=100)
    ax = fig.add_subplot(111)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for ch in range(y.shape[0]):
        group = ch // 8
        ax.plot(tt, y[ch], linewidth=0.3, color=colors[group])
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(out_path, dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def add_polar_rings(ax, rings=(0.25, 0.5, 0.75, 1.0)):
    theta = np.linspace(0, 2 * np.pi, 256)
    for r in rings:
        ax.plot(theta, np.full_like(theta, r), color="white", alpha=0.2, linewidth=0.6)


def normalize_rms(rms_vals, rms_vmin, rms_vmax):
    denom = rms_vmax - rms_vmin if rms_vmax > rms_vmin else 1.0
    norm = (rms_vals - rms_vmin) / denom
    return np.clip(norm, 0.0, 1.0)


def save_polar_heatmap(rms_vals, out_path, rms_vmin, rms_vmax, r_bins=64):
    angles = np.linspace(0, 2 * np.pi, len(rms_vals) + 1)
    r_edges = np.linspace(0.0, 1.0, r_bins + 1)

    norm = normalize_rms(rms_vals, rms_vmin, rms_vmax)

    Z = np.zeros((r_bins, len(rms_vals)))
    for i, v in enumerate(norm):
        r_idx = int(round(v * (r_bins - 1)))
        if r_idx >= 0:
            Z[: r_idx + 1, i] = v

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(2.24, 2.24), dpi=100)
    ax = fig.add_subplot(111, projection="polar")
    Theta, R = np.meshgrid(angles, r_edges)
    ax.pcolormesh(Theta, R, Z, cmap="magma", shading="auto")
    add_polar_rings(ax)
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(out_path, dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_polar_line(rms_vals, out_path, rms_vmin, rms_vmax):
    angles = np.linspace(0, 2 * np.pi, len(rms_vals), endpoint=False)
    norm = normalize_rms(rms_vals, rms_vmin, rms_vmax)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(2.24, 2.24), dpi=100)
    ax = fig.add_subplot(111, projection="polar")
    ax.plot(np.r_[angles, angles[0]], np.r_[norm, norm[0]], color="white", linewidth=1.2)
    add_polar_rings(ax)
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(out_path, dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def channel_base_color(ch_idx):
    # Piecewise gradient with blue bands at ch10-14 and ch24-30.
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


def save_emg_rms_image(seg, out_path, rms_vmin, rms_vmax, bins=224, style="stripe7", shifts=None):
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
    denom = rms_vmax - rms_vmin if rms_vmax > rms_vmin else 1.0
    norm = (rms_bins - rms_vmin) / denom
    norm = np.clip(norm, 0.0, 1.0)

    if style == "stripe7":
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
        else:
            for frac in shifts:
                shift_cols = int(round(frac * bins))
                shifted = np.roll(img, shift_cols, axis=1)
                suffix = f"_shift{int(round(frac*100)):02d}"
                out_shift = out_path.with_name(out_path.stem + suffix + out_path.suffix)
                mpimg.imsave(out_shift, shifted, vmin=0.0, vmax=1.0)
        return

    img = norm
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not shifts:
        mpimg.imsave(out_path, img, cmap="gray", vmin=0.0, vmax=1.0)
    else:
        for frac in shifts:
            shift_cols = int(round(frac * bins))
            shifted = np.roll(img, shift_cols, axis=1)
            suffix = f"_shift{int(round(frac*100)):02d}"
            out_shift = out_path.with_name(out_path.stem + suffix + out_path.suffix)
            mpimg.imsave(out_shift, shifted, cmap="gray", vmin=0.0, vmax=1.0)


def safe_label(label):
    return label.lower().replace(" ", "_").replace("/", "_")


def parse_subject_list(val):
    if not val:
        return None
    return [v.strip() for v in val.split(",") if v.strip()]


def parse_notch_list(val):
    if not val:
        return []
    parts = [p.strip() for p in val.split(",") if p.strip()]
    return [float(p) for p in parts]


def apply_fft_filter(signal, bandpass, notch_freqs, notch_width):
    if bandpass is None and not notch_freqs:
        return signal
    low, high = bandpass if bandpass else (0.0, fs / 2.0)
    n = signal.shape[1]
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mask = (freqs >= low) & (freqs <= high)
    if notch_freqs:
        for f0 in notch_freqs:
            mask &= ~((freqs >= f0 - notch_width) & (freqs <= f0 + notch_width))
    spec = np.fft.rfft(signal, axis=1)
    spec *= mask[None, :]
    return np.fft.irfft(spec, n=n, axis=1)


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


def select_subject_dirs(subjects_list=None, subjects_range=None):
    # If a list/range is provided, ignore subject_glob and scan all subject folders.
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


def compute_global_calibration(subjects_list=None, subjects_range=None, noise=None, skip=None):
    amp_ref = 0.0
    rms_min = None
    rms_max = None

    skip = skip or set()
    for subject_dir, trial_dir, emg_path, aux_path, log_path in iter_trials(subjects_list, subjects_range):
        if (subject_dir.name, trial_dir.name) in skip:
            continue
        signal = load_emg(emg_path)
        noise_key = (subject_dir.name, trial_dir.name, emg_path.name)
        excluded_idx = None
        if noise and noise_key in noise:
            excluded = noise[noise_key][1]
            signal = apply_drop_channel(signal, excluded)
        amp_ref = max(amp_ref, float(np.max(np.abs(signal))))

        if log_path is None:
            continue
        aux = load_aux(aux_path)
        aux_start, aux_end, move_starts, labels, move_duration, _, _, per_label_offsets = compute_move_markers(
            signal, aux, log_path
        )
        if not move_starts or not labels or move_duration is None:
            continue

        for t0, off in zip(move_starts, per_label_offsets):
            start_t = t0 + off
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
            rms_vals = np.sqrt(np.mean(seg * seg, axis=1))
            rmin = float(np.min(rms_vals))
            rmax = float(np.max(rms_vals))
            rms_min = rmin if rms_min is None else min(rms_min, rmin)
            rms_max = rmax if rms_max is None else max(rms_max, rmax)

    if rms_min is None or rms_max is None:
        rms_min = 0.0
        rms_max = 1.0

    return amp_ref, rms_min, rms_max


def compute_filtered_amp_ref(subjects_list, subjects_range, noise, skip, bandpass, notch_freqs, notch_width):
    amp_ref = 0.0
    skip = skip or set()
    for subject_dir, trial_dir, emg_path, aux_path, log_path in iter_trials(subjects_list, subjects_range):
        if (subject_dir.name, trial_dir.name) in skip:
            continue
        signal = load_emg(emg_path)
        noise_key = (subject_dir.name, trial_dir.name, emg_path.name)
        if noise and noise_key in noise:
            signal = apply_drop_channel(signal, noise[noise_key][1])
        filt = apply_fft_filter(signal, bandpass, notch_freqs, notch_width)
        amp_ref = max(amp_ref, float(np.max(np.abs(filt))))
    return amp_ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output root for images")
    ap.add_argument("--ds", type=int, default=1, help="Downsample for EMG plots")
    ap.add_argument("--subjects", type=str, default=None, help="Comma-separated subject folder names")
    ap.add_argument("--subjects-range", type=str, default=None, help="Range like 1-10")
    ap.add_argument("--offsets", type=str, default=None, help="CSV of manual offsets")
    ap.add_argument("--noise", type=str, default=None, help="CSV of trial noise info")
    ap.add_argument("--skip", type=str, default=None, help="CSV of trial skip info")
    ap.add_argument("--filtered-out", type=str, default=None, help="Output root for filtered EMG images")
    ap.add_argument("--emg-rms-out", type=str, default=None, help="Output root for EMG RMS images")
    ap.add_argument("--rms-bins", type=int, default=224, help="Time bins for RMS image width")
    ap.add_argument("--emg-rms-style", type=str, default="stripe7", choices=["stripe7", "gray"])
    ap.add_argument("--rms-shifts", type=str, default=None, help="Comma-separated fractional shifts (e.g., 0,0.2,0.4)")
    ap.add_argument("--trial", type=str, default=None, help="Only process a specific trial directory name")
    ap.add_argument("--bandpass-low", type=float, default=20.0)
    ap.add_argument("--bandpass-high", type=float, default=250.0)
    ap.add_argument("--notch", type=str, default="50,60")
    ap.add_argument("--notch-width", type=float, default=1.0)
    ap.add_argument("--polar-style", type=str, default="line", choices=["line", "heatmap"])
    ap.add_argument("--jitters", type=int, default=3, help="Number of jittered windows per movement")
    ap.add_argument("--jitter-sec", type=float, default=0.1, help="Jitter size in seconds (+/-)")
    args = ap.parse_args()

    subjects_list = parse_subject_list(args.subjects)
    subjects_range = None
    if args.subjects_range:
        a, b = args.subjects_range.split("-")
        subjects_range = (int(a), int(b))

    out_root = Path(args.out)
    emg_root = out_root / "emg"
    polar_root = out_root / "polar"
    filtered_root = Path(args.filtered_out) if args.filtered_out else None
    filtered_emg_root = filtered_root / "emg" if filtered_root else None
    rms_root = Path(args.emg_rms_out) if args.emg_rms_out else None
    rms_emg_root = rms_root / "emg_rms" if rms_root else None

    offsets = {}
    if args.offsets:
        with open(args.offsets, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["subject"], row["trial"], row["emg_file"], row["label"])
                offsets[key] = float(row["offset_sec"])
    noise = {}
    if args.noise:
        with open(args.noise, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["subject"], row["trial"], row["emg_file"])
            ch_raw = row.get("excluded_channels", "").strip()
            if ch_raw:
                excluded = [int(c) for c in ch_raw.split(",") if c.strip().isdigit()]
            else:
                excluded = []
            noise[key] = (float(row["noise_score"]), excluded)
    skip = load_skip(args.skip)
    bandpass = (args.bandpass_low, args.bandpass_high)
    notch_freqs = parse_notch_list(args.notch)
    amp_ref, rms_min, rms_max = compute_global_calibration(
        subjects_list, subjects_range, noise=noise, skip=skip
    )
    filtered_amp_ref = None
    if filtered_emg_root:
        filtered_amp_ref = compute_filtered_amp_ref(
            subjects_list, subjects_range, noise, skip, bandpass, notch_freqs, args.notch_width
        )
    print(f"Global EMG amp_ref={amp_ref:.6g}, rms_min={rms_min:.6g}, rms_max={rms_max:.6g}")

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
        noise_score = None
        excluded_channel = None
        if noise_key in noise:
            noise_score, excluded_channel = noise[noise_key]
            signal = apply_drop_channel(signal, excluded_channel)
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
                rms_vals = np.sqrt(np.mean(seg * seg, axis=1))

                label_dir = safe_label(label)
                subject_name = subject_dir.name
                trial_name = trial_dir.name.replace("trial", "(00)trial")
                label_name = f"(00){label_dir}"
                noise_tag = ""
                if noise_score is not None:
                    if excluded_channel is None:
                        noise_tag = f"_noise{noise_score:.4g}"
                    else:
                        noise_tag = f"_noise{noise_score:.4g}_ch{excluded_channel:02d}"
                base = f"{subject_name}-{trial_name}-{label_name}_j{j_idx}{noise_tag}.png"

                emg_out = emg_root / label_dir / base
                polar_out = polar_root / label_dir / base
                rms_out = rms_emg_root / label_dir / base if rms_emg_root else None

                save_emg_segment_image(seg, fs, emg_out, amp_ref, ds=args.ds)
                if rms_out:
                    save_emg_rms_image(
                        seg,
                        rms_out,
                        rms_min,
                        rms_max,
                        bins=args.rms_bins,
                        style=args.emg_rms_style,
                        shifts=rms_shifts,
                    )
                if filtered_emg_root and filtered_amp_ref:
                    filt_seg = apply_fft_filter(seg, bandpass, notch_freqs, args.notch_width)
                    filt_out = filtered_emg_root / label_dir / base
                    save_emg_segment_image(filt_seg, fs, filt_out, filtered_amp_ref, ds=args.ds)
                if args.polar_style == "heatmap":
                    save_polar_heatmap(rms_vals, polar_out, rms_min, rms_max)
                else:
                    save_polar_line(rms_vals, polar_out, rms_min, rms_max)


if __name__ == "__main__":
    main()
