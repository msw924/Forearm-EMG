from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pickle
import re
from scipy.signal import iirnotch, filtfilt
from plots_emg import plot_emg_time_series
from plots_emg_auto import compute_auto_delays
from plots_polar import plot_polar


# %%%%%%%%%%% CONFIG %%%%%%%%%%
# Top-level settings for batch selection, plotting, and caching.
base_data_dir = Path("/Users/maxwilliams/Library/CloudStorage/Dropbox/MW_GR_data/Data")
subject_glob = "(03)Manon"
trial_glob = "trial *"
sensor_unit = "DA1"
move_duration_sec = None  # set to float to override; otherwise infer from AUX + log

plot_emg = True
plot_polar_enabled = True
use_auto_delay = True
auto_delay_k = 5.0
auto_delay_buffer_sec = 0.5
auto_delay_channels = list(range(1, 32))
use_cache = True
cache_dir = base_data_dir / "_cache_emg"
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Acquisition constants and sampling rate.
adcRes = 16
din = 2.4
gain = 192
maxLev = 2**adcRes - 1
numChannels = 32
fs = 2049.122


# Load and preprocess EMG (scale, demean, notch filter).
def load_emg(emg_path):
    print(f"Reading EMG from: {emg_path}")
    val = np.fromfile(emg_path, dtype="<u2")
    n_frames = val.size // numChannels
    val = val[:n_frames * numChannels]
    signal = val.reshape(n_frames, numChannels).T
    scaled = (signal.astype(np.float64) / maxLev) * (din / gain)
    scaled = scaled - scaled.mean(axis=1, keepdims=True)
    b, a = iirnotch(50.0, 30.0, fs)
    return filtfilt(b, a, scaled, axis=1)


# Load AUX channel (scaled).
def load_aux(aux_path):
    if not aux_path.exists():
        return np.array([], dtype=np.float64)
    print(f"Reading AUX from: {aux_path}")
    aux = np.fromfile(aux_path, dtype="<u2")
    aux = aux.astype(np.float64) / maxLev
    return aux


# Find AUX pulse start/end for alignment.
def detect_aux_pulse(aux):
    if aux.size == 0:
        return None, None
    aux_1d = aux.reshape(-1)
    aux_thresh = 0.5 * np.max(aux_1d)
    above = np.flatnonzero(aux_1d > aux_thresh)
    if above.size == 0:
        return None, None
    return above[0] / fs, above[-1] / fs


# Parse movement log for timing and labels.
def parse_log(log_path):
    txt_fallback_path = log_path.with_suffix(".txt")
    log_text = ""
    if log_path.suffix.lower() == ".pdf":
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(str(log_path))
            log_text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            print(f"Could not read PDF log: {e}. Trying .txt fallback.")
            if txt_fallback_path.exists():
                try:
                    log_text = txt_fallback_path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    log_text = txt_fallback_path.read_text(encoding="latin-1")
            else:
                print("No .txt fallback found.")
    elif log_path.suffix.lower() == ".rtf":
        try:
            rtf_raw = log_path.read_text(encoding="utf-8", errors="ignore")
            rtf_text = re.sub(r"\\(par|line)\\b", "\n", rtf_raw)
            rtf_text = rtf_text.replace("\\\n", "\n")
            rtf_text = re.sub(r"\\[a-zA-Z]+-?\\d* ?", "", rtf_text)
            rtf_text = re.sub(r"[{}]", "", rtf_text)
            rtf_text = re.sub(r"\n+", "\n", rtf_text)
            log_text = rtf_text.strip()
        except Exception as e:
            print(f"Could not read RTF log: {e}")
    else:
        try:
            log_text = log_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            log_text = log_path.read_text(encoding="latin-1")
        except Exception as e:
            print(f"Could not read log file: {e}")

    pulse_start_time_sec = None
    pre_move_wait_sec = None
    post_move_wait_sec = None
    move_labels = []
    if log_text:
        pulse_match = re.search(r"Sending pulse \(start\) at ([0-9.]+) s", log_text)
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
        if start_idx == -1:
            print("Could not find 'Waiting ... before text prompts' line in log.")
        else:
            for ln in lines[start_idx + 1 :]:
                if ln.startswith("Waiting ") or "Sending pulse" in ln:
                    break
                move_labels.append(ln)

    return pulse_start_time_sec, pre_move_wait_sec, post_move_wait_sec, move_labels


# Locate a per-trial log file (RTF/TXT/PDF).
def find_log_file(trial_dir, fallback_name=None):
    for ext in (".rtf", ".txt", ".pdf"):
        matches = sorted(trial_dir.glob(f"*{ext}"))
        if matches:
            return matches[0]
    if fallback_name:
        cand = trial_dir / fallback_name
        if cand.exists():
            return cand
    return None


# Compute movement start times from AUX + log (with optional auto delay).
def compute_move_markers(signal, aux, log_path, move_duration_override=None, use_auto_delay_override=None):
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
    true_start_offset_sec = 1.0
    auto_delay_val = None
    per_label_offsets = [true_start_offset_sec for _ in labels]
    use_auto = use_auto_delay if use_auto_delay_override is None else use_auto_delay_override
    if use_auto:
        offsets, avg_delay, _ = compute_auto_delays(
            signal,
            fs,
            move_starts,
            labels,
            move_duration,
            aux_start,
            pre_wait,
            k=auto_delay_k,
            pre_start_buffer=auto_delay_buffer_sec,
            channels=auto_delay_channels,
        )
        if offsets is not None:
            per_label_offsets = offsets
            auto_delay_val = avg_delay
            true_start_offset_sec = avg_delay if avg_delay is not None else true_start_offset_sec
    return (
        aux_start,
        aux_end,
        move_starts,
        labels,
        move_duration,
        true_start_offset_sec,
        auto_delay_val,
        per_label_offsets,
    )


# Compute per-movement RMS windows for polar plots.
def compute_rms_windows(signal, aux, log_path, move_duration_override=None):
    aux_start, aux_end, move_starts, labels, move_duration, true_start_offset_sec, _, per_label_offsets = compute_move_markers(
        signal, aux, log_path, move_duration_override
    )
    if move_starts is None or move_duration is None:
        return None, None

    window_duration_sec = max(move_duration - true_start_offset_sec, 0.0)
    if window_duration_sec <= 0:
        return None, None

    rms_rows = []
    valid_labels = []
    for t0, label, off in zip(move_starts, labels, per_label_offsets):
        t_true = t0 + off
        start_idx = int(round(t_true * fs))
        end_idx = int(round((t_true + window_duration_sec) * fs))
        start_idx = max(start_idx, 0)
        end_idx = min(end_idx, signal.shape[1])
        if start_idx >= end_idx:
            continue
        seg = signal[:, start_idx:end_idx]
        upper = np.percentile(seg, 95, axis=1, keepdims=True)
        lower = np.percentile(seg, 5, axis=1, keepdims=True)
        seg = np.clip(seg, lower, upper)
        rms = np.sqrt(np.mean(seg * seg, axis=1))
        rms_rows.append(rms)
        valid_labels.append(label)

    if not rms_rows:
        return None, None
    return np.vstack(rms_rows), valid_labels


# Plot stacked EMG time series with AUX + labels.


def cache_paths(emg_path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = emg_path.stem
    return cache_dir / f"{stem}_signal.npy", cache_dir / f"{stem}_aux.npy"


# Load from cache or compute + save.
def load_or_cache(emg_path, aux_path):
    sig_cache, aux_cache = cache_paths(emg_path)
    if use_cache and sig_cache.exists() and aux_cache.exists():
        signal = np.load(sig_cache)
        aux = np.load(aux_cache)
        return signal, aux

    signal = load_emg(emg_path)
    aux = load_aux(aux_path)
    if use_cache:
        np.save(sig_cache, signal)
        np.save(aux_cache, aux)
    return signal, aux


# Process one EMG file: load, plot EMG, plot polar.
def run_trial(trial_dir, emg_path, aux_path, log_path):
    signal, aux = load_or_cache(emg_path, aux_path)

    if plot_emg:
        # Fixed 1.0s delay version
        plot_emg_time_series(
            signal,
            f"EMG Time Series | Fixed 1.0s | {trial_dir.name} | {emg_path.stem}",
            fs,
            detect_aux_pulse,
            compute_move_markers,
            aux=aux,
            log_path=log_path,
            move_duration_override=move_duration_sec,
            use_auto_delay_override=False,
        )
        # Auto-delay version
        plot_emg_time_series(
            signal,
            f"EMG Time Series | Auto Delay | {trial_dir.name} | {emg_path.stem}",
            fs,
            detect_aux_pulse,
            compute_move_markers,
            aux=aux,
            log_path=log_path,
            move_duration_override=move_duration_sec,
            use_auto_delay_override=True,
        )

    if plot_polar_enabled and log_path is not None:
        rms_mat, valid_labels = compute_rms_windows(signal, aux, log_path, move_duration_sec)
        if rms_mat is not None:
            plot_polar(rms_mat, valid_labels, f"Polar Plot | {trial_dir.name} | {emg_path.stem}")


# Batch driver: iterate subject/trials and process each EMG file.
def main():
    subject_dirs = sorted(base_data_dir.glob(subject_glob))
    if not subject_dirs:
        print(f"No subject dirs matched: {subject_glob}")
        return

    for subject_dir in subject_dirs:
        trial_dirs = sorted([p for p in subject_dir.glob(trial_glob) if p.is_dir()])
        if not trial_dirs:
            print(f"No trials found in {subject_dir}")
            continue

        for trial_dir in trial_dirs:
            emg_files = sorted(trial_dir.glob(f"*_M02{sensor_unit}_EMG_raw.sig"))
            if not emg_files:
                print(f"No EMG files found in {trial_dir}")
                continue
            log_path = find_log_file(trial_dir)
            if log_path is None:
                print(f"No log file found in {trial_dir}")
            for emg_path in emg_files:
                aux_path = emg_path.with_name(emg_path.name.replace("_EMG_raw.sig", "_AUX1_raw.sig"))
                run_trial(trial_dir, emg_path, aux_path, log_path)


if __name__ == "__main__":
    main()
