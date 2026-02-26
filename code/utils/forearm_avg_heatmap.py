"""
Compute the cross-subject/trial average RMS for a movement label and
render it as a colour-mapped heatmap on a forearm cross-section image.

Example usage:
    python forearm_avg_heatmap.py \
        --image ../../reports/figures/forearm_electrodes_rest.png \
        --label "Wrist Extension" \
        --out ../../reports/figures/forearm_avg_wrist_extension.png \
        --clockwise --ch20-up --label-channels
"""

import argparse
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
from extensor_localization import (
    load_offsets,
    load_noise,
    apply_drop_channel,
)


def channel_angles_deg(num_channels, step_deg=360.0 / 32.0, ch15_deg=0.0,
                       clockwise=True, ch_top=None):
    angles = []
    for ch in range(1, num_channels + 1):
        delta = (ch - 15) * step_deg
        ang = ch15_deg + (delta if clockwise else -delta)
        angles.append(ang)
    if ch_top is not None:
        top_angle = angles[ch_top - 1]
        rot = -90.0 - top_angle
        angles = [a + rot for a in angles]
    return np.radians(angles)


def iter_trials():
    subject_dirs = sorted(base_data_dir.glob("(??)*"))
    for subject_dir in subject_dirs:
        trial_dirs = sorted([
            p for p in subject_dir.iterdir()
            if p.is_dir() and p.name.lower().startswith("trial")
        ])
        for trial_dir in trial_dirs:
            emg_files = sorted(trial_dir.glob(f"*_M02{sensor_unit}_EMG_raw.sig"))
            if not emg_files:
                continue
            log_candidates = (
                list(trial_dir.glob("*.rtf"))
                + list(trial_dir.glob("*.txt"))
                + list(trial_dir.glob("*.pdf"))
            )
            log_path = log_candidates[0] if log_candidates else None
            for emg_path in emg_files:
                aux_path = emg_path.with_name(
                    emg_path.name.replace("_EMG_raw.sig", "_AUX1_raw.sig")
                )
                yield subject_dir, trial_dir, emg_path, aux_path, log_path


def collect_rms(label, offsets, noise):
    rms_list = []
    for subject_dir, trial_dir, emg_path, aux_path, log_path in iter_trials():
        if log_path is None:
            continue

        signal = load_emg(emg_path)

        noise_key = (subject_dir.name, trial_dir.name, emg_path.name)
        if noise_key in noise:
            score, excluded = noise[noise_key]
            signal = apply_drop_channel(signal, excluded)

        aux = load_aux(aux_path)
        (_, _, move_starts, labels, move_duration,
         _, _, per_label_offsets) = compute_move_markers(signal, aux, log_path)

        if not move_starts or not labels or move_duration is None:
            continue

        label_map = {lbl.strip().lower(): i for i, lbl in enumerate(labels)}
        key_norm = label.strip().lower()
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

        start_idx = max(0, int(round(start_t * fs)))
        end_idx = min(signal.shape[1], int(round((start_t + window_sec) * fs)))
        if end_idx <= start_idx:
            continue

        seg = signal[:, start_idx:end_idx]
        rms = np.sqrt(np.mean(seg * seg, axis=1))
        rms_list.append(rms)
        print(f"  {subject_dir.name} / {trial_dir.name}: OK")

    return rms_list


def main():
    ap = argparse.ArgumentParser(
        description="Average-RMS forearm heatmap across all subjects/trials."
    )
    ap.add_argument("--image", required=True, help="Forearm cross-section image path")
    ap.add_argument("--out", required=True, help="Output image path")
    ap.add_argument("--label", default="Wrist Extension",
                    help="Movement label (default: 'Wrist Extension')")
    ap.add_argument("--offsets", default=None, help="Offsets CSV path")
    ap.add_argument("--noise", default=None, help="Trial noise CSV path")
    ap.add_argument("--radius", type=float, default=0.55,
                    help="Electrode ring radius as fraction of image min dim")
    ap.add_argument("--step-deg", type=float, default=360.0 / 32.0)
    ap.add_argument("--ch15-deg", type=float, default=0.0)
    ap.add_argument("--clockwise", action="store_true",
                    help="Channel numbers increase clockwise")
    ap.add_argument("--ch-top", type=int, default=None,
                    help="Which channel number sits at 12 o'clock (e.g. 2 or 20)")
    ap.add_argument("--rotate-180", action="store_true")
    ap.add_argument("--label-channels", action="store_true",
                    help="Draw channel number labels on electrodes")
    ap.add_argument("--log-scale", action="store_true")
    ap.add_argument("--point-size", type=float, default=60.0)
    args = ap.parse_args()

    offsets = load_offsets(args.offsets)
    noise = load_noise(args.noise)

    print(f"Collecting RMS for '{args.label}' across all subjects/trials...")
    rms_list = collect_rms(args.label, offsets, noise)

    if not rms_list:
        raise SystemExit(f"No trials found with label '{args.label}'.")

    rms_avg = np.mean(np.vstack(rms_list), axis=0)
    print(f"Averaged {len(rms_list)} trials.")

    img_path = Path(args.image)
    if not img_path.exists():
        raise SystemExit(f"Image not found: {img_path}")
    img = plt.imread(img_path)
    if args.rotate_180:
        img = np.rot90(img, 2)
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    radius = args.radius * min(w, h)

    angles = channel_angles_deg(
        32,
        step_deg=args.step_deg,
        ch15_deg=args.ch15_deg,
        clockwise=args.clockwise,
        ch_top=args.ch_top,
    )
    if args.rotate_180:
        angles = angles + np.pi

    xs = center[0] + radius * np.cos(angles)
    ys = center[1] + radius * np.sin(angles)

    rms_for_color = rms_avg
    if args.log_scale:
        eps = max(float(np.max(rms_avg)) * 1e-6, 1e-12)
        rms_for_color = np.log10(rms_avg + eps)

    norm = plt.Normalize(vmin=float(np.min(rms_for_color)),
                         vmax=float(np.max(rms_for_color)))
    colors = plt.cm.coolwarm(norm(rms_for_color))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img)
    ax.scatter(xs, ys, c=colors, s=args.point_size, alpha=0.95,
               edgecolors="black", linewidths=0.5)
    if args.label_channels:
        for i, (x, y, c) in enumerate(zip(xs, ys, colors), start=1):
            ax.text(x, y, f"{i}", color="white", fontsize=7,
                    ha="center", va="center", fontweight="bold")

    sm = plt.cm.ScalarMappable(norm=norm, cmap="coolwarm")
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
    cbar_label = "log10(RMS)" if args.log_scale else "RMS (activation)"
    cbar.set_label(cbar_label)

    n = len(rms_list)
    ax.set_title(f"Average {args.label} | n={n} trials")
    ax.axis("off")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
