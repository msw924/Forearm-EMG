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
    parse_log,
)


DEFAULT_LABELS = [
    "Index Extension",
    "Index Flexion",
    "Wrist Extension",
    "Wrist Flexion",
    "3-Finger Extension",
    "3-Finger Flexion",
]


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
            log_candidates = list(trial_dir.glob("*.rtf")) + list(trial_dir.glob("*.txt")) + list(
                trial_dir.glob("*.pdf")
            )
            log_path = log_candidates[0] if log_candidates else None
            for emg_path in emg_files:
                yield subject_dir, trial_dir, emg_path, log_path


def load_windows(path):
    if not path.exists():
        return {}
    windows = {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["subject"], row["trial"], row["emg_file"], row["label"])
            try:
                start_t = float(row["start_t"])
                end_t = float(row["end_t"])
            except (TypeError, ValueError):
                continue
            windows[key] = (start_t, end_t)
    return windows


def save_windows(path, windows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "trial", "emg_file", "label", "start_t", "end_t"])
        for (subject, trial, emg_file, label), (start_t, end_t) in windows.items():
            writer.writerow([subject, trial, emg_file, label, f"{start_t:.6f}", f"{end_t:.6f}"])


def stacked_emg_plot(signal, ds=10):
    t = np.arange(signal.shape[1]) / fs
    sig = signal[:, ::ds]
    tt = t[::ds]
    amp = np.max(np.abs(sig))
    offset = 1.2 * amp if amp > 0 else 1.0
    y = sig + offset * np.arange(sig.shape[0])[:, None]
    return tt, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="CSV output for manual windows")
    ap.add_argument("--subjects", type=str, default=None)
    ap.add_argument("--subjects-range", type=str, default=None)
    ap.add_argument("--trials", type=str, default=None)
    ap.add_argument("--labels", type=str, default=None, help="Comma-separated label list (overrides log)")
    ap.add_argument("--ds", type=int, default=10)
    ap.add_argument("--step", type=float, default=0.2)
    ap.add_argument("--big-step", type=float, default=0.5)
    ap.add_argument("--default-window", type=float, default=None, help="Seconds to auto-set end after start")
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()

    subjects_list = parse_subject_list(args.subjects)
    subjects_range = None
    if args.subjects_range:
        a, b = args.subjects_range.split("-")
        subjects_range = (int(a), int(b))
    trials_list = parse_subject_list(args.trials)

    windows_path = Path(args.out)
    windows = {} if args.reset else load_windows(windows_path)

    items = list(iter_trials(subjects_list, subjects_range, trials_list))
    if not items:
        raise SystemExit("No trials found.")

    idx = 0
    label_idx = 0
    mode = "start"
    current = {}

    def trial_labels(log_path):
        if args.labels:
            return [v.strip() for v in args.labels.split(",") if v.strip()]
        if log_path:
            _, _, _, labels = parse_log(log_path)
            if labels:
                return labels
        return DEFAULT_LABELS

    def load_current():
        nonlocal current, label_idx
        subject_dir, trial_dir, emg_path, log_path = items[idx]
        signal = load_emg(emg_path)
        labels = trial_labels(log_path)
        label_idx = min(label_idx, len(labels) - 1)
        current = {
            "subject": subject_dir.name,
            "trial": trial_dir.name,
            "emg_file": emg_path.name,
            "signal": signal,
            "labels": labels,
        }

    def get_key(label):
        return (current["subject"], current["trial"], current["emg_file"], label)

    def get_window(label):
        return windows.get(get_key(label))

    def set_window(label, start_t=None, end_t=None):
        if start_t is None and end_t is None:
            windows.pop(get_key(label), None)
            return
        if start_t is None or end_t is None:
            existing = windows.get(get_key(label))
            if existing:
                start_t = existing[0] if start_t is None else start_t
                end_t = existing[1] if end_t is None else end_t
            else:
                return
        start_t, end_t = normalize_window(start_t, end_t)
        windows[get_key(label)] = (start_t, end_t)

    def normalize_window(start_t, end_t):
        if start_t > end_t:
            start_t, end_t = end_t, start_t
        start_t = max(start_t, 0.0)
        end_t = min(end_t, current["signal"].shape[1] / fs)
        return start_t, end_t

    def render():
        plt.clf()
        fig = plt.gcf()
        ax = plt.gca()
        tt, y = stacked_emg_plot(current["signal"], ds=args.ds)
        ax.plot(tt, y.T, linewidth=0.3)
        for i, label in enumerate(current["labels"]):
            win = get_window(label)
            is_current = i == label_idx
            color = "g" if is_current else "0.5"
            if win:
                start_t, end_t = win
                ax.axvline(start_t, color=color, linewidth=1.2 if is_current else 0.8)
                ax.axvline(end_t, color=color, linewidth=1.0 if is_current else 0.6, linestyle="--")
                ax.axvspan(start_t, end_t, color=color, alpha=0.08 if is_current else 0.04)
                ax.text(
                    start_t,
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
        label = current["labels"][label_idx]
        win = get_window(label)
        win_text = "unset" if not win else f"{win[0]:.3f}s → {win[1]:.3f}s"
        title = f"{current['subject']} | {current['trial']} | {current['emg_file']}"
        fig.suptitle(f"{title}\nactive: {label} | window: {win_text}", fontsize=11, y=0.98)
        fig.text(
            0.01,
            0.93,
            "Click to set start/end | [ start ] end | left/right step | n/p next/prev | r reset | s save | q quit",
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.7),
        )
        fig.text(
            0.99,
            0.93,
            f"Mode: {mode}",
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.7),
        )
        ax.set_xlim(tt[0], tt[-1])
        fig.subplots_adjust(top=0.86)
        plt.draw()

    def move_active(delta, target):
        label = current["labels"][label_idx]
        win = get_window(label)
        if not win:
            return
        start_t, end_t = win
        if target == "start":
            start_t += delta
        else:
            end_t += delta
        set_window(label, start_t=start_t, end_t=end_t)

    def on_key(event):
        nonlocal idx, label_idx, mode
        if event.key == "left":
            move_active(-args.step, mode)
        elif event.key == "right":
            move_active(args.step, mode)
        elif event.key == "shift+left":
            move_active(-args.big_step, mode)
        elif event.key == "shift+right":
            move_active(args.big_step, mode)
        elif event.key == "[":
            mode = "start"
        elif event.key == "]":
            mode = "end"
        elif event.key in ("n", "enter"):
            label_idx += 1
            if label_idx >= len(current["labels"]):
                label_idx = 0
                idx += 1
                if idx >= len(items):
                    save_windows(windows_path, windows)
                    plt.close()
                    return
                load_current()
        elif event.key == "p":
            label_idx = max(0, label_idx - 1)
        elif event.key == "r":
            label = current["labels"][label_idx]
            set_window(label, None, None)
        elif event.key == "s":
            save_windows(windows_path, windows)
        elif event.key == "q":
            save_windows(windows_path, windows)
            plt.close()
            return
        render()

    def on_click(event):
        if event.inaxes is None or event.xdata is None:
            return
        label = current["labels"][label_idx]
        t_click = float(event.xdata)
        win = get_window(label)
        if mode == "start":
            start_t = t_click
            end_t = win[1] if win else None
            if end_t is None and args.default_window:
                end_t = start_t + args.default_window
            if end_t is not None:
                start_t, end_t = normalize_window(start_t, end_t)
            set_window(label, start_t=start_t, end_t=end_t)
        else:
            end_t = t_click
            start_t = win[0] if win else None
            if start_t is None and args.default_window:
                start_t = end_t - args.default_window
            if start_t is not None:
                start_t, end_t = normalize_window(start_t, end_t)
            set_window(label, start_t=start_t, end_t=end_t)
        render()

    load_current()
    fig = plt.figure(figsize=(12, 8))
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("button_press_event", on_click)
    render()
    plt.show()


if __name__ == "__main__":
    main()
