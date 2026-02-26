import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from EMG_opener import (
    base_data_dir,
    subject_glob,
    trial_glob,
    sensor_unit,
    load_emg,
    load_aux,
    compute_move_markers,
)


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
        trial_dirs = sorted([p for p in subject_dir.iterdir() if p.is_dir() and p.name.lower().startswith("trial ")])
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
    offsets = {}
    if not path.exists():
        return offsets
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Offsets CSV to update")
    ap.add_argument("--subjects", type=str, default=None)
    ap.add_argument("--subjects-range", type=str, default=None)
    args = ap.parse_args()

    subjects_list = parse_subject_list(args.subjects)
    subjects_range = None
    if args.subjects_range:
        a, b = args.subjects_range.split("-")
        subjects_range = (int(a), int(b))

    out_path = Path(args.out)
    offsets = load_offsets(out_path)
    added = 0

    for subject_dir, trial_dir, emg_path, aux_path, log_path in iter_trials(subjects_list, subjects_range):
        if log_path is None:
            continue
        signal = load_emg(emg_path)
        aux = load_aux(aux_path)
        aux_start, aux_end, move_starts, labels, move_duration, _, _, per_label_offsets = compute_move_markers(
            signal, aux, log_path
        )
        if move_starts is None or labels is None:
            continue
        for label, off in zip(labels, per_label_offsets):
            key = (subject_dir.name, trial_dir.name, emg_path.name, label)
            if key in offsets:
                continue
            offsets[key] = off
            added += 1

    save_offsets(out_path, offsets)
    print(f"Backfilled {added} missing offsets.")


if __name__ == "__main__":
    main()
