import numpy as np
import matplotlib.pyplot as plt


"""EMG plotting utilities."""


# Plot stacked EMG time series with optional AUX + labels.
def plot_emg_time_series(
    signal,
    title,
    fs,
    detect_aux_pulse,
    compute_move_markers,
    aux=None,
    log_path=None,
    move_duration_override=None,
    use_auto_delay_override=None,
    ds=10,
):
    T = signal.shape[1]
    t = np.arange(T) / fs
    sig = signal[:, ::ds]
    tt = t[::ds]
    amp = np.max(np.abs(sig))
    offset = 1.2 * amp
    y = sig + offset * np.arange(sig.shape[0])[:, None]

    plt.figure(figsize=(12, 8))
    plt.plot(tt, y.T, linewidth=0.3)
    plt.yticks(offset * np.arange(sig.shape[0]), [f"Ch{i+1}" for i in range(sig.shape[0])])
    plt.xlabel("Time (s)")
    plt.title(title)
    if aux is not None:
        aux_start, aux_end = detect_aux_pulse(aux)
        if aux_start is not None:
            plt.axvline(aux_start, color="red", linewidth=1.0, alpha=0.8)
        if aux_end is not None:
            plt.axvline(aux_end, color="red", linewidth=1.0, alpha=0.8)
    if log_path is not None and aux is not None:
        aux_start, aux_end, move_starts, labels, move_duration, true_start_offset_sec, auto_delay_val, per_label_offsets = compute_move_markers(
            signal, aux, log_path, move_duration_override, use_auto_delay_override
        )
        if move_starts and labels:
            for t0, label, off in zip(move_starts, labels, per_label_offsets):
                if t0 <= tt[-1]:
                    plt.axvline(t0, color="k", linewidth=0.8, alpha=0.4)
                    t_true = t0 + off
                    if t_true <= tt[-1]:
                        plt.axvline(t_true, color="k", linewidth=0.8, alpha=0.4, linestyle="--")
                    plt.text(t0, y.max(), label, rotation=90, va="bottom", ha="right", fontsize=8)
    if use_auto_delay_override and auto_delay_val is not None:
        plt.title(f"{title} | delay={auto_delay_val:.3f}s")
    plt.tight_layout()
    plt.show()


# Plot polar RMS summary for selected movements.
