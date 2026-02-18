import numpy as np


"""Auto-delay utilities for EMG plots based on wrist/3-finger slope."""


def compute_auto_delays(
    signal,
    fs,
    move_starts,
    labels,
    move_duration,
    aux_start,
    pre_wait,
    k=5.0,
    min_delay=0.01,
    max_delay=3.0,
    smooth_sec=0.05,
    pre_start_buffer=0.5,
    channels=None,
):
    if not move_starts or not labels:
        return None, None, None
    if move_duration is None or move_duration <= 0:
        return None, None, None

    # Envelope of EMG across selected channels
    if channels is None:
        sig = signal
    else:
        sig = signal[channels, :]
    env = np.sqrt(np.mean(sig * sig, axis=0))
    win = max(1, int(round(smooth_sec * fs)))
    if win > 1:
        kernel = np.ones(win, dtype=np.float64) / win
        env = np.convolve(env, kernel, mode="same")

    # Derivative of envelope (slope)
    deriv = np.gradient(env) * fs

    # Baseline from the entire signal
    base = deriv
    med = float(np.median(base))
    mad = float(np.median(np.abs(base - med)))
    thr = med + k * mad

    def delay_for_label(target_label):
        if target_label not in labels:
            return None
        idx = labels.index(target_label)
        t0 = move_starts[idx]
        start_idx = int(round(t0 * fs))
        end_idx = int(round((t0 + move_duration) * fs))
        start_idx = max(start_idx, 0)
        end_idx = min(end_idx, len(deriv))
        if end_idx <= start_idx:
            return None

        seg = deriv[start_idx:end_idx]
        peak_idx = int(np.argmax(seg))
        peak_val = seg[peak_idx]
        if peak_val <= thr:
            return None

        delay = (start_idx + peak_idx) / fs - t0
        delay = delay - pre_start_buffer
        delay = max(min_delay, min(max_delay, delay))
        return delay

    wrist_ext_delay = delay_for_label("Wrist Extension")
    wrist_flex_delay = delay_for_label("Wrist Flexion")
    three_ext_delay = delay_for_label("3-Finger Extension")
    three_flex_delay = delay_for_label("3-Finger Flexion")

    valid = [
        d
        for d in (
            wrist_ext_delay,
            wrist_flex_delay,
            three_ext_delay,
            three_flex_delay,
        )
        if d is not None
    ]
    avg_delay = float(np.mean(valid)) if valid else None

    per_label_offsets = [1.0 for _ in labels]
    for i, label in enumerate(labels):
        if label == "Wrist Extension" and wrist_ext_delay is not None:
            per_label_offsets[i] = wrist_ext_delay
        elif label == "Wrist Flexion" and wrist_flex_delay is not None:
            per_label_offsets[i] = wrist_flex_delay
        elif label == "3-Finger Extension" and three_ext_delay is not None:
            per_label_offsets[i] = three_ext_delay
        elif label == "3-Finger Flexion" and three_flex_delay is not None:
            per_label_offsets[i] = three_flex_delay
        elif "Index" in label:
            per_label_offsets[i] = avg_delay if avg_delay is not None else 1.0

    # Fallback: if nothing valid, keep defaults
    if avg_delay is None:
        return per_label_offsets, None, None

    return per_label_offsets, avg_delay, {
        "wrist_ext": wrist_ext_delay,
        "wrist_flex": wrist_flex_delay,
        "three_ext": three_ext_delay,
        "three_flex": three_flex_delay,
    }
