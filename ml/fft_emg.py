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


def find_trial(subject_name, trial_name):
    subject_dir = base_data_dir / subject_name
    if not subject_dir.exists():
        raise SystemExit(f"Subject not found: {subject_name}")
    trial_dir = subject_dir / trial_name
    if not trial_dir.exists():
        raise SystemExit(f"Trial not found: {trial_dir}")
    emg_files = sorted(trial_dir.glob(f"*_M02{sensor_unit}_EMG_raw.sig"))
    if not emg_files:
        raise SystemExit(f"No EMG files in {trial_dir}")
    emg_path = emg_files[0]
    aux_path = emg_path.with_name(emg_path.name.replace("_EMG_raw.sig", "_AUX1_raw.sig"))
    log_candidates = list(trial_dir.glob("*.rtf")) + list(trial_dir.glob("*.txt")) + list(trial_dir.glob("*.pdf"))
    log_path = log_candidates[0] if log_candidates else None
    return emg_path, aux_path, log_path


def compute_mean_psd(seg, nfft, max_freq):
    if seg.size == 0:
        return None, None
    seg = seg - np.mean(seg, axis=1, keepdims=True)
    spec = np.fft.rfft(seg, n=nfft, axis=1)
    power = (np.abs(spec) ** 2) / nfft
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    if max_freq is not None:
        mask = freqs <= max_freq
        freqs = freqs[mask]
        power = power[:, mask]
    mean_power = np.mean(power, axis=0)
    return freqs, mean_power


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True, help="Subject folder name, e.g. (03)Manon")
    ap.add_argument("--trial", required=True, help="Trial folder name, e.g. trial 3")
    ap.add_argument("--nfft", type=int, default=4096)
    ap.add_argument("--max-freq", type=float, default=500.0)
    ap.add_argument("--out", type=str, default=None, help="Optional PNG output path")
    args = ap.parse_args()

    emg_path, aux_path, log_path = find_trial(args.subject, args.trial)
    if log_path is None:
        raise SystemExit("No log found for trial; cannot locate movement windows.")

    signal = load_emg(emg_path)
    aux = load_aux(aux_path)
    aux_start, aux_end, move_starts, labels, move_duration, _, _, per_label_offsets = compute_move_markers(
        signal, aux, log_path
    )
    if not move_starts or not labels or move_duration is None:
        raise SystemExit("No movement windows found.")

    all_move_psd = []
    all_base_psd = []
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
        freqs, mean_power = compute_mean_psd(seg, args.nfft, args.max_freq)
        if mean_power is not None:
            all_move_psd.append(mean_power)

        base_end = start_idx
        base_start = max(0, base_end - (end_idx - start_idx))
        base_seg = signal[:, base_start:base_end]
        freqs_b, base_power = compute_mean_psd(base_seg, args.nfft, args.max_freq)
        if base_power is not None:
            all_base_psd.append(base_power)

    if not all_move_psd or not all_base_psd:
        raise SystemExit("Not enough data to compute spectra.")

    move_mean = np.mean(np.vstack(all_move_psd), axis=0)
    base_mean = np.mean(np.vstack(all_base_psd), axis=0)

    plt.figure(figsize=(10, 5))
    plt.plot(freqs, 10 * np.log10(move_mean + 1e-12), label="Movement (avg)")
    plt.plot(freqs, 10 * np.log10(base_mean + 1e-12), label="Baseline (avg)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (dB)")
    plt.title(f"{args.subject} | {args.trial} | Avg FFT")
    plt.legend()
    plt.tight_layout()

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150)
    else:
        plt.show()


if __name__ == "__main__":
    main()
