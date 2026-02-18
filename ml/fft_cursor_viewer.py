import argparse
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
    sensor_unit,
    fs,
    load_emg,
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
    return emg_files[0]


def stacked_emg_plot(signal, ds=10):
    t = np.arange(signal.shape[1]) / fs
    sig = signal[:, ::ds]
    tt = t[::ds]
    amp = np.max(np.abs(sig))
    offset = 1.2 * amp if amp > 0 else 1.0
    y = sig + offset * np.arange(sig.shape[0])[:, None]
    return tt, y


def compute_fft(signal, start_idx, end_idx, nfft, max_freq, channel=None):
    if channel is not None:
        seg = signal[channel:channel + 1, start_idx:end_idx]
    else:
        seg = signal[:, start_idx:end_idx]
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
    ap.add_argument("--ds", type=int, default=10)
    ap.add_argument("--window-sec", type=float, default=2.0)
    ap.add_argument("--nfft", type=int, default=4096)
    ap.add_argument("--max-freq", type=float, default=500.0)
    ap.add_argument("--channel", type=int, default=None, help="1-based channel for FFT only")
    args = ap.parse_args()

    emg_path = find_trial(args.subject, args.trial)
    signal = load_emg(emg_path)
    t = np.arange(signal.shape[1]) / fs
    chan_idx = None
    if args.channel is not None:
        chan_idx = args.channel - 1
        if chan_idx < 0 or chan_idx >= signal.shape[0]:
            raise SystemExit(f"Invalid channel: {args.channel}")

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 2], hspace=0.35)
    ax_fft = fig.add_subplot(gs[0])
    ax_emg = fig.add_subplot(gs[1])
    ax_slider = fig.add_axes([0.12, 0.05, 0.76, 0.03])

    tt, y = stacked_emg_plot(signal, ds=args.ds)
    ax_emg.plot(tt, y.T, linewidth=0.3)
    ax_emg.set_xlim(tt[0], tt[-1])
    ax_emg.set_xlabel("Time (s)")

    line = ax_emg.axvline(tt[0], color="g", linewidth=1.2)
    fft_line, = ax_fft.plot([], [], color="k")
    ax_fft.set_xlabel("Frequency (Hz)")
    ax_fft.set_ylabel("Power (dB)")

    state = {"x": tt[0]}

    # Precompute global FFT scale using the whole trial (average over channels)
    freqs_all, power_all = compute_fft(signal, 0, signal.shape[1], args.nfft, args.max_freq, channel=chan_idx)
    if power_all is None:
        raise SystemExit("Could not compute FFT scale.")
    global_db = 10 * np.log10(power_all + 1e-12)
    db_min = float(np.min(global_db))
    db_max = float(np.max(global_db))

    def update_fft(xpos):
        half = args.window_sec / 2.0
        start_t = max(0.0, xpos - half)
        end_t = min(t[-1], xpos + half)
        start_idx = int(round(start_t * fs))
        end_idx = int(round(end_t * fs))
        freqs, power = compute_fft(signal, start_idx, end_idx, args.nfft, args.max_freq, channel=chan_idx)
        if freqs is None:
            return
        fft_line.set_data(freqs, 10 * np.log10(power + 1e-12))
        ax_fft.set_xlim(freqs[0], freqs[-1])
        ax_fft.set_ylim(db_min, db_max)
        chan_title = "all channels" if chan_idx is None else f"Ch{args.channel:02d}"
        ax_fft.set_title(
            f"{args.subject} | {args.trial} | {chan_title} | t={xpos:.2f}s | window={args.window_sec:.2f}s"
        )

    ax_slider.set_facecolor("0.95")
    slider = Slider(
        ax=ax_slider,
        label="Time (s)",
        valmin=float(tt[0]),
        valmax=float(tt[-1]),
        valinit=float(tt[0]),
        valstep=(tt[1] - tt[0]) if len(tt) > 1 else 0.01,
    )

    def on_slide(val):
        state["x"] = float(val)
        line.set_xdata([state["x"], state["x"]])
        update_fft(state["x"])
        fig.canvas.draw_idle()

    slider.on_changed(on_slide)

    update_fft(state["x"])
    plt.tight_layout(rect=[0.0, 0.08, 1.0, 1.0])
    plt.show()


if __name__ == "__main__":
    main()
