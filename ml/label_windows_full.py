import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

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
    t = np.arange(signal.shape[1]) / fs
    sig = signal[:, ::ds]
    tt = t[::ds]
    amp = np.max(np.abs(sig))
    offset = 1.2 * amp if amp > 0 else 1.0
    y = sig + offset * np.arange(sig.shape[0])[:, None]
    return tt, y, offset


def load_manual(path):
    if not path.exists():
        return {}
    out = {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["subject"], row["trial"], row["emg_file"], row["label"])
            out[key] = (float(row["start_t"]), float(row["end_t"]))
    return out


def save_manual(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "trial", "emg_file", "label", "start_t", "end_t"])
        for row in rows:
            writer.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="CSV output for manual windows")
    ap.add_argument("--subjects", type=str, default=None)
    ap.add_argument("--subjects-range", type=str, default=None)
    ap.add_argument("--trials", type=str, default=None, help="Comma-separated trial folder names")
    ap.add_argument("--ds", type=int, default=10)
    ap.add_argument("--reset", action="store_true", help="Ignore existing manual timings")
    args = ap.parse_args()

    subjects_list = parse_subject_list(args.subjects)
    subjects_range = None
    if args.subjects_range:
        a, b = args.subjects_range.split("-")
        subjects_range = (int(a), int(b))
    trials_list = parse_subject_list(args.trials)

    out_path = Path(args.out)
    manual = {} if args.reset else load_manual(out_path)

    items = []
    for subject_dir, trial_dir, emg_path, aux_path, log_path in iter_trials(
        subjects_list, subjects_range, trials_list
    ):
        if log_path is None:
            continue
        items.append((subject_dir, trial_dir, emg_path, aux_path, log_path))

    if not items:
        raise SystemExit("No trials found with logs.")

    idx = 0
    current = {}
    state = {"cursor_t": 0.0}

    def load_current():
        nonlocal current
        subject_dir, trial_dir, emg_path, aux_path, log_path = items[idx]
        signal = load_emg(emg_path)
        aux = load_aux(aux_path)
        aux_start, aux_end, move_starts, labels, move_duration, _, _, per_label_offsets = compute_move_markers(
            signal, aux, log_path
        )
        if not labels:
            raise SystemExit(f"No labels in log: {log_path}")
        current = {
            "subject": subject_dir.name,
            "trial": trial_dir.name,
            "emg_file": emg_path.name,
            "signal": signal,
            "labels": labels,
            "starts": [None for _ in labels],
            "last_end": None,
        }
        # preload any existing manual values
        for i, label in enumerate(labels):
            key = (current["subject"], current["trial"], current["emg_file"], label)
            if key in manual:
                start_t, end_t = manual[key]
                current["starts"][i] = start_t
                if i == len(labels) - 1:
                    current["last_end"] = end_t
        state["cursor_t"] = 0.0

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 0.08], hspace=0.35)
    ax_emg = fig.add_subplot(gs[0])
    ax_slider = fig.add_subplot(gs[1])

    def render():
        ax_emg.clear()
        tt, y, offset = stacked_emg_plot(current["signal"], ds=args.ds)
        ax_emg.plot(tt, y.T, linewidth=0.3)
        ax_emg.set_xlim(tt[0], tt[-1])
        ax_emg.set_xlabel("Time (s)")
        yticks = offset * np.arange(current["signal"].shape[0])
        ylabels = [f"Ch{idx + 1:02d}" for idx in range(current["signal"].shape[0])]
        ax_emg.set_yticks(yticks)
        ax_emg.set_yticklabels(ylabels, fontsize=7)

        # movement markers
        for i, label in enumerate(current["labels"]):
            start_t = current["starts"][i]
            if start_t is None:
                continue
            ax_emg.axvline(start_t, color="g", linewidth=1.0)
            ax_emg.text(start_t, 1.02, f"{i + 1}:{label}", transform=ax_emg.get_xaxis_transform(),
                        rotation=90, fontsize=8, va="bottom", ha="center", clip_on=False)
        if current["last_end"] is not None:
            ax_emg.axvline(current["last_end"], color="m", linewidth=1.0, linestyle="--")

        ax_emg.set_title(
            f"{current['subject']} | {current['trial']} | {current['emg_file']}\n"
            f"Keys: 1-6 set start | e set last end | n/p trial | s save | q quit"
        )

        line = ax_emg.axvline(state["cursor_t"], color="k", linewidth=1.0, alpha=0.7)
        fig.canvas.draw_idle()
        return line

    slider = Slider(
        ax=ax_slider,
        label="Time (s)",
        valmin=0.0,
        valmax=1.0,
        valinit=0.0,
    )

    cursor_line = None

    def update_slider_limits():
        t = np.arange(current["signal"].shape[1]) / fs
        slider.valmin = float(t[0])
        slider.valmax = float(t[-1])
        slider.set_val(float(t[0]))

    def on_slide(val):
        state["cursor_t"] = float(val)
        if cursor_line is not None:
            cursor_line.set_xdata([state["cursor_t"], state["cursor_t"]])
        fig.canvas.draw_idle()

    slider.on_changed(on_slide)

    def set_start(idx_label):
        if 0 <= idx_label < len(current["starts"]):
            current["starts"][idx_label] = state["cursor_t"]
            render()

    def on_key(event):
        nonlocal idx, cursor_line
        if event.key in [str(i) for i in range(1, 10)]:
            set_start(int(event.key) - 1)
        elif event.key == "e":
            current["last_end"] = state["cursor_t"]
            render()
        elif event.key in ("n", "enter"):
            save_all()
            idx += 1
            if idx >= len(items):
                plt.close()
                return
            load_current()
            update_slider_limits()
            cursor_line = render()
        elif event.key == "p":
            save_all()
            idx = max(idx - 1, 0)
            load_current()
            update_slider_limits()
            cursor_line = render()
        elif event.key == "s":
            save_all()
        elif event.key == "q":
            save_all()
            plt.close()

    def save_all():
        rows = []
        for key, (start_t, end_t) in manual.items():
            rows.append([*key, start_t, end_t])
        # include current trial
        for i, label in enumerate(current["labels"]):
            start_t = current["starts"][i]
            if start_t is None:
                continue
            if i < len(current["labels"]) - 1:
                # end = next start
                next_start = current["starts"][i + 1]
                if next_start is None:
                    continue
                end_t = next_start
            else:
                if current["last_end"] is None:
                    continue
                end_t = current["last_end"]
            key = (current["subject"], current["trial"], current["emg_file"], label)
            manual[key] = (start_t, end_t)
            rows.append([*key, start_t, end_t])
        save_manual(out_path, rows)

    load_current()
    update_slider_limits()
    cursor_line = render()
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()


if __name__ == "__main__":
    main()
