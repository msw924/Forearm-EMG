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


def load_offsets(path):
    if not path.exists():
        return {}
    offsets = {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["subject"], row["trial"], row["emg_file"], row["label"])
            offsets[key] = float(row["offset_sec"])
    return offsets


def save_offsets(path, offsets):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "trial", "emg_file", "label", "offset_sec"])
        for (subject, trial, emg_file, label), offset in offsets.items():
            writer.writerow([subject, trial, emg_file, label, f"{offset:.6f}"])


def select_subject_dirs(subjects_list=None, subjects_range=None):
    subject_dirs = sorted([p for p in base_data_dir.glob("(??)*") if p.is_dir()])
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
        trial_dirs = sorted([p for p in subject_dir.iterdir() if p.is_dir() and p.name.lower().startswith("trial ")])
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


def stacked_emg_plot(signal, ds=10):
    T = signal.shape[1]
    t = np.arange(T) / fs
    sig = signal[:, ::ds]
    tt = t[::ds]
    amp = np.max(np.abs(sig))
    offset = 1.2 * amp if amp > 0 else 1.0
    y = sig + offset * np.arange(sig.shape[0])[:, None]
    return tt, y


def compute_trial_noise(signal):
    per_channel_rms = np.sqrt(np.mean(signal * signal, axis=1))
    noise_score = float(np.mean(per_channel_rms))
    return noise_score, None


def load_noise(path):
    if not path.exists():
        return {}
    noise = {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["subject"], row["trial"], row["emg_file"])
            ch_raw = row.get("excluded_channel", "").strip()
            excluded = int(ch_raw) if ch_raw else None
            noise[key] = (float(row["noise_score"]), excluded)
    return noise


def save_noise(path, noise):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "trial", "emg_file", "noise_score", "excluded_channel"])
        for (subject, trial, emg_file), (score, ch) in noise.items():
            ch_val = "" if ch is None else ch
            writer.writerow([subject, trial, emg_file, f"{score:.6f}", ch_val])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="CSV output for offsets")
    ap.add_argument("--subjects", type=str, default=None, help="Comma-separated subject folder names")
    ap.add_argument("--subjects-range", type=str, default=None, help="Range like 1-10")
    ap.add_argument("--trials", type=str, default=None, help="Comma-separated trial folder names")
    ap.add_argument("--noise-out", type=str, default=None, help="CSV output for trial noise info")
    ap.add_argument("--ds", type=int, default=10)
    ap.add_argument("--step", type=float, default=0.3)
    ap.add_argument("--big-step", type=float, default=0.5)
    ap.add_argument("--pre", type=float, default=15.0, help="Seconds before label to show")
    ap.add_argument("--post", type=float, default=15.0, help="Seconds after label to show")
    ap.add_argument("--window-sec", type=float, default=8.0, help="Window length after start (seconds)")
    ap.add_argument("--reset", action="store_true", help="Ignore existing offsets and start fresh")
    args = ap.parse_args()

    subjects_list = [v.strip() for v in args.subjects.split(",")] if args.subjects else None
    subjects_range = None
    if args.subjects_range:
        a, b = args.subjects_range.split("-")
        subjects_range = (int(a), int(b))
    trials_list = [v.strip() for v in args.trials.split(",")] if args.trials else None

    offsets_path = Path(args.out)
    offsets = {} if args.reset else load_offsets(offsets_path)
    noise_path = Path(args.noise_out) if args.noise_out else offsets_path.with_name("trial_noise.csv")
    noise = load_noise(noise_path)

    items = []
    for subject_dir, trial_dir, emg_path, aux_path, log_path in iter_trials(
        subjects_list, subjects_range, trials_list
    ):
        if log_path is None:
            continue
        items.append((subject_dir, trial_dir, emg_path, aux_path, log_path))

    idx = 0
    label_idx = 0
    current = {}

    def load_current():
        nonlocal current, idx
        while True:
            subject_dir, trial_dir, emg_path, aux_path, log_path = items[idx]
            signal = load_emg(emg_path)
            aux = load_aux(aux_path)
            aux_start, aux_end, move_starts, labels, move_duration, _, _, per_label_offsets = compute_move_markers(
                signal, aux, log_path
            )
            if move_starts is None or labels is None or move_duration is None:
                idx_next = idx + 1
                if idx_next >= len(items):
                    raise SystemExit("No usable trials with labels.")
                idx = idx_next
                continue
            noise_key = (subject_dir.name, trial_dir.name, emg_path.name)
            if noise_key not in noise:
                noise[noise_key] = compute_trial_noise(signal)
            current = {
                "subject": subject_dir.name,
                "trial": trial_dir.name,
                "emg_file": emg_path.name,
                "signal": signal,
                "move_starts": move_starts,
                "labels": labels,
                "move_duration": move_duration,
                "offsets": per_label_offsets,
                "noise": noise[noise_key],
                "channels": signal.shape[0],
            }
            break

    def get_offset(idx=None):
        if not current.get("labels"):
            return 0.0
        use_idx = label_idx if idx is None else idx
        label = current["labels"][use_idx]
        key = (current["subject"], current["trial"], current["emg_file"], label)
        return offsets.get(key, current["offsets"][use_idx])

    def set_offset(value):
        if not current.get("labels"):
            return
        label = current["labels"][label_idx]
        key = (current["subject"], current["trial"], current["emg_file"], label)
        offsets[key] = value

    def persist_current_offsets():
        if not current.get("labels"):
            return
        for i, label in enumerate(current["labels"]):
            key = (current["subject"], current["trial"], current["emg_file"], label)
            offsets[key] = get_offset(i)

    def render():
        plt.clf()
        fig = plt.gcf()
        ax = plt.gca()
        tt, y = stacked_emg_plot(current["signal"], ds=args.ds)
        ax.plot(tt, y.T, linewidth=0.3)
        for i, (t0, label) in enumerate(zip(current["move_starts"], current["labels"])):
            offset = get_offset(i)
            start_t = t0 + offset
            end_t = start_t + max(args.window_sec, 0.0)
            is_current = i == label_idx
            base_color = "k" if is_current else "0.4"
            start_color = "g" if is_current else "0.6"
            z = 6 if is_current else 3
            ax.axvline(t0, color=base_color, linewidth=1.2 if is_current else 0.8, alpha=0.7, zorder=z)
            ax.axvline(start_t, color=start_color, linewidth=2.0 if is_current else 0.8, alpha=1.0 if is_current else 0.6, zorder=z)
            ax.axvline(end_t, color=start_color, linewidth=1.4 if is_current else 0.6, alpha=0.6, linestyle="--", zorder=z)
            ax.text(
                t0,
                1.02,
                label,
                transform=ax.get_xaxis_transform(),
                rotation=90,
                fontsize=8,
                va="bottom",
                ha="center",
                alpha=0.7,
                clip_on=False,
            )
        title = f"{current['subject']} | {current['trial']} | {current['emg_file']}"
        label = current["labels"][label_idx]
        offset = get_offset()
        noise_score, excluded_channel = current["noise"]
        drop_text = "None" if excluded_channel is None else f"Ch{excluded_channel:02d}"
        fig.suptitle(
            f"{title}\nactive: {label} | offset={offset:.3f}s | noise={noise_score:.4g}",
            fontsize=11,
            y=0.98,
        )
        drop_banner = f"DROP: {drop_text}"
        fig.text(
            0.99,
            0.93,
            drop_banner,
            ha="right",
            va="top",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.9),
        )
        fig.text(
            0.01,
            0.93,
            "Set drop: up/down | Clear: 0",
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.7),
        )
        ax.set_xlim(tt[0], tt[-1])
        fig.subplots_adjust(top=0.86)
        plt.draw()

    def on_key(event):
        nonlocal idx, label_idx
        if event.key == "right":
            set_offset(get_offset() + args.step)
        elif event.key == "left":
            set_offset(get_offset() - args.step)
        elif event.key == "shift+right":
            set_offset(get_offset() + args.big_step)
        elif event.key == "shift+left":
            set_offset(get_offset() - args.big_step)
        elif event.key in ("enter", "n"):
            persist_current_offsets()
            label_idx += 1
            if label_idx >= len(current["labels"]):
                label_idx = 0
                idx += 1
                if idx >= len(items):
                    save_offsets(offsets_path, offsets)
                    save_noise(noise_path, noise)
                    plt.close()
                    return
                load_current()
        elif event.key == "p":
            label_idx -= 1
            if label_idx < 0:
                label_idx = 0
        elif event.key == "up":
            score, ch = current["noise"]
            ch = 1 if ch is None else min(ch + 1, current["channels"])
            noise_key = (current["subject"], current["trial"], current["emg_file"])
            noise[noise_key] = (score, ch)
            current["noise"] = noise[noise_key]
        elif event.key == "down":
            score, ch = current["noise"]
            if ch is None:
                ch = current["channels"]
            else:
                ch = max(ch - 1, 1)
            noise_key = (current["subject"], current["trial"], current["emg_file"])
            noise[noise_key] = (score, ch)
            current["noise"] = noise[noise_key]
        elif event.key == "0":
            score, _ = current["noise"]
            noise_key = (current["subject"], current["trial"], current["emg_file"])
            noise[noise_key] = (score, None)
            current["noise"] = noise[noise_key]
        elif event.key == "s":
            persist_current_offsets()
            save_offsets(offsets_path, offsets)
            save_noise(noise_path, noise)
        elif event.key == "q":
            persist_current_offsets()
            save_offsets(offsets_path, offsets)
            save_noise(noise_path, noise)
            plt.close()
            return
        render()

    if not items:
        raise SystemExit("No trials found with logs")

    load_current()
    fig = plt.figure(figsize=(12, 8))
    fig.canvas.mpl_connect("key_press_event", on_key)
    render()
    plt.show()


if __name__ == "__main__":
    main()
