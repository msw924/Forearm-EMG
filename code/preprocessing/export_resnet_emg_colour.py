"""
export_resnet_emg_colour.py

Export 224×224 colour-coded EMG images optimised for ResNet18.

Layout
------
  32 channels × 7 pixel rows = 224 rows (height)
  224 time bins               = 224 cols  (width)

Per-channel pixel fill (centre-out within the 7-row strip):
  level = 0  →  1 centre pixel coloured, 3 white either side
  level = 1  →  all 7 pixels coloured

Colour scheme (1-indexed channels, smooth gradient transitions):
  Red   (flexor)   : ch 1-3  and ch 30-32
  Blue  (neutral)  : ch 3-15 and ch 21-30
  Green (extensor) : ch 15-21
  Boundaries between zones are gradients, not hard cuts.

Amplitude scaling
-----------------
  --scale-mode global  (default)
      Two-pass: collect all RMS bins → compute a single percentile reference
      → apply uniformly across all images. Ensures consistent brightness.
  --scale-mode per-trial
      Each image scaled independently (legacy behaviour).

Window source (priority order per movement)
-------------------------------------------
  1. --manual-windows CSV  (subject, trial, emg_file, label, start_t, end_t)
  2. --offsets CSV         (subject, trial, emg_file, label, offset_sec)
  3. Auto-delay from compute_move_markers
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib.image as mpimg

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _subdir in ("utils", "visualization"):
    _p = str(PROJECT_ROOT / _subdir)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from EMG_opener import (        # noqa: E402
    base_data_dir,
    sensor_unit,
    fs,
    load_emg,
    load_aux,
    compute_move_markers,
)


# ── Colour map ─────────────────────────────────────────────────────────────────
#
# Anchor points drive smooth interpolation around the forearm ring.
# Centres of each functional zone carry the purest colour; boundaries
# transition gradually into the neighbouring zone.
#
#   ch 1  – 3  : red   (flexor belly, low channels)
#   ch 3  – 15 : blue  (lateral / intermediate)
#   ch 15 – 21 : green (extensor belly, dorsal)   ← pure green plateau
#   ch 21 – 30 : blue  (lateral / intermediate)
#   ch 30 – 32 : red   (flexor belly, high channels)
#
# The green zone has a flat plateau from ch 15-21 (pure green across all
# 7 extensor channels, centred on ch 18).  Transitions occur over ch 12-15
# (blue→green) and ch 21-24 (green→blue), keeping ch 18 as visual centre.
#
_COLOUR_ANCHORS = [
    (1,  (1.0, 0.0, 0.0)),   # red   – flexor low
    (3,  (1.0, 0.0, 0.0)),   # red   – flexor/neutral boundary
    (9,  (0.0, 0.0, 1.0)),   # blue  – neutral centre (3-12)
    (12, (0.0, 0.0, 1.0)),   # blue  – start of green transition
    (15, (0.0, 1.0, 0.0)),   # green – extensor plateau start
    (18, (0.0, 1.0, 0.0)),   # green – extensor centre (ch 18)
    (21, (0.0, 1.0, 0.0)),   # green – extensor plateau end
    (24, (0.0, 0.0, 1.0)),   # blue  – end of green transition
    (27, (0.0, 0.0, 1.0)),   # blue  – neutral centre (24-30)
    (30, (0.0, 0.0, 1.0)),   # blue  – neutral/flexor boundary
    (32, (1.0, 0.0, 0.0)),   # red   – flexor high
]


def channel_base_color(ch_idx: int) -> np.ndarray:
    """Linearly interpolated RGB colour for 1-indexed channel ch_idx."""
    ch = float(ch_idx)
    if ch <= _COLOUR_ANCHORS[0][0]:
        return np.array(_COLOUR_ANCHORS[0][1], dtype=float)
    for (x0, c0), (x1, c1) in zip(_COLOUR_ANCHORS, _COLOUR_ANCHORS[1:]):
        if ch <= x1:
            t = (ch - x0) / (x1 - x0) if x1 > x0 else 0.0
            return np.array(c0, dtype=float) * (1.0 - t) + np.array(c1, dtype=float) * t
    return np.array(_COLOUR_ANCHORS[-1][1], dtype=float)


# ── Image rendering ────────────────────────────────────────────────────────────

# Fill order within each 7-row strip: centre first, then expanding outward.
# Row 3 (0-indexed within the strip) is the centre.
#   n_rows=1  → row 3 only      (1 pixel, 3 white either side)
#   n_rows=3  → rows 2, 3, 4
#   n_rows=7  → all rows filled  (maximum activation)
_ROW_ORDER = [3, 2, 4, 1, 5, 0, 6]


def compute_rms_bins(seg: np.ndarray, bins: int = 224) -> np.ndarray:
    """Compute per-channel RMS in *bins* equal-duration time slots.

    Returns array of shape (n_channels, bins).
    """
    n = seg.shape[1]
    bins_src = min(bins, n)
    edges = np.linspace(0, n, bins_src + 1, dtype=int)
    rms = np.zeros((seg.shape[0], bins_src), dtype=float)
    for i in range(bins_src):
        s = edges[i]
        e = min(max(edges[i + 1], s + 1), n)
        chunk = seg[:, s:e]
        rms[:, i] = np.sqrt(np.mean(chunk * chunk, axis=1))
    if bins_src < bins:
        x_src = np.linspace(0, 1, bins_src)
        x_dst = np.linspace(0, 1, bins)
        rms = np.vstack([np.interp(x_dst, x_src, row) for row in rms])
    return rms


def render_image(rms_bins: np.ndarray, amp_ref: float, bins: int = 224, grayscale: bool = False) -> np.ndarray:
    """Render a 224×224 RGB float image from pre-computed RMS bins.

    White background; each channel's strip fills centre-out according
    to normalised amplitude.  If grayscale=True, all channels render black.
    """
    denom = amp_ref if amp_ref > 0.0 else 1.0
    norm = np.clip(rms_bins / denom, 0.0, 1.0)

    img = np.ones((32 * 7, bins, 3), dtype=float)   # white background
    for ch in range(32):
        colour = np.array([0.0, 0.0, 0.0]) if grayscale else channel_base_color(ch + 1)
        row_base = ch * 7
        for t in range(bins):
            level = float(norm[ch, t])
            n_rows = max(1, min(7, int(round(1.0 + level * 6.0))))
            for i in range(n_rows):
                img[row_base + _ROW_ORDER[i], t] = colour
    return img


def save_image(img: np.ndarray, out_path: Path, shifts=None, row_shifts=None, bins: int = 224) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    col_fracs = shifts if shifts else [0.0]
    row_fracs = row_shifts if row_shifts else [0.0]
    n_rows = img.shape[0]
    for cf in col_fracs:
        for rf in row_fracs:
            shifted = np.roll(img, int(round(cf * bins)), axis=1)
            shifted = np.roll(shifted, int(round(rf * n_rows)), axis=0)
            suffix = f"_shift{int(round(cf * 100)):02d}_rshift{int(round(rf * 100)):02d}"
            out_shift = out_path.with_name(out_path.stem + suffix + out_path.suffix)
            mpimg.imsave(str(out_shift), shifted, vmin=0.0, vmax=1.0)


# ── CSV helpers ────────────────────────────────────────────────────────────────

def load_manual_windows(path) -> dict:
    """Load manual window CSV → {(subject, trial, emg_file, label): (start_t, end_t)}."""
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        print(f"Warning: manual windows CSV not found: {p}")
        return {}
    out = {}
    with p.open("r", newline="") as f:
        for row in csv.DictReader(f):
            key = (row["subject"], row["trial"], row["emg_file"], row["label"])
            out[key] = (float(row["start_t"]), float(row["end_t"]))
    print(f"Loaded {len(out)} manual windows from {p}")
    return out


def load_offsets(path) -> dict:
    """Load offsets CSV → {(subject, trial, emg_file, label): offset_sec}."""
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    out = {}
    with p.open("r", newline="") as f:
        for row in csv.DictReader(f):
            key = (row["subject"], row["trial"], row["emg_file"], row["label"])
            out[key] = float(row["offset_sec"])
    return out


def load_noise(path) -> dict:
    """Load noise CSV → {(subject, trial, emg_file): (noise_score, [excluded_channels])}."""
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    out = {}
    with p.open("r", newline="") as f:
        for row in csv.DictReader(f):
            key = (row["subject"], row["trial"], row["emg_file"])
            ch_raw = row.get("excluded_channels", "").strip()
            excluded = [int(c) for c in ch_raw.split(",") if c.strip().isdigit()]
            out[key] = (float(row["noise_score"]), excluded)
    return out


# ── Channel drop ───────────────────────────────────────────────────────────────

def apply_drop_channel(signal: np.ndarray, excluded: list) -> np.ndarray:
    if not excluded:
        return signal
    out = signal.copy()
    for ch in sorted(set(excluded)):
        idx = ch - 1
        if not (0 <= idx < out.shape[0]):
            continue
        if idx == 0:
            out[idx] = out[idx + 1]
        elif idx == out.shape[0] - 1:
            out[idx] = out[idx - 1]
        else:
            out[idx] = 0.5 * (out[idx - 1] + out[idx + 1])
    return out


# ── Trial iteration ────────────────────────────────────────────────────────────

def iter_trials(subjects_list=None, subjects_range=None, data_dir=None, include_da2=False):
    root = Path(data_dir) if data_dir else base_data_dir
    if subjects_list or subjects_range:
        subject_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    else:
        subject_dirs = sorted(root.glob("(??)*"))

    if subjects_list:
        name_set = set(subjects_list)
        subject_dirs = [p for p in subject_dirs if p.name in name_set]
    elif subjects_range:
        start, end = subjects_range
        want = {f"({i:02d})" for i in range(start, end + 1)}
        subject_dirs = [p for p in subject_dirs if p.name[:4] in want]

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
                yield subject_dir, trial_dir, emg_path, aux_path, log_path, False
            if include_da2:
                da2_files = sorted(trial_dir.glob("*_M02DA2_EMG_raw.sig"))
                for emg_path in da2_files:
                    aux_path = emg_path.with_name(
                        emg_path.name.replace("_EMG_raw.sig", "_AUX1_raw.sig")
                    )
                    yield subject_dir, trial_dir, emg_path, aux_path, log_path, True


def safe_label(label: str) -> str:
    return label.lower().replace(" ", "_").replace("/", "_")


def parse_list(val):
    if not val:
        return None
    return [v.strip() for v in val.split(",") if v.strip()]


# ── Window collector ───────────────────────────────────────────────────────────

def collect_windows(args, manual_windows, offsets, noise):
    """Generator yielding (rms_bins, out_path) for every valid movement window.

    Window priority:
      1. manual_windows CSV  → use start_t / end_t directly
      2. offsets CSV         → apply offset_sec to move_start
      3. auto-delay          → use compute_move_markers per-label offset
    """
    subjects_list = parse_list(args.subjects)
    subjects_range = None
    if args.subjects_range:
        a, b = args.subjects_range.split("-")
        subjects_range = (int(a), int(b))

    jitter_offsets = [0.0]
    if args.jitters > 1:
        jitter_offsets = np.linspace(-args.jitter_sec, args.jitter_sec, args.jitters).tolist()

    out_root = Path(args.out)
    emg_root = out_root / "emg_rms"

    for subject_dir, trial_dir, emg_path, aux_path, log_path, is_da2 in iter_trials(
        subjects_list, subjects_range,
        data_dir=getattr(args, "data_dir", None),
        include_da2=getattr(args, "include_da2", False),
    ):
        if args.trial and trial_dir.name != args.trial:
            continue
        if log_path is None:
            continue

        noise_key = (subject_dir.name, trial_dir.name, emg_path.name)
        if noise_key in noise:
            noise_score, excluded = noise[noise_key]
            if args.max_noise is not None and noise_score > args.max_noise:
                print(f"  Skipping {subject_dir.name} / {trial_dir.name} – noise {noise_score:.6f} > {args.max_noise}")
                continue
        else:
            excluded = []

        print(f"  Loading {subject_dir.name} / {trial_dir.name} / {emg_path.name}")
        signal = load_emg(emg_path)
        if is_da2:
            # DA2 channels are physically reversed relative to DA1.
            # Flip the channel axis so DA2 channel ordering matches DA1
            # (DA1[17]=DA2_local[16], DA1[18]=DA2_local[15] → flip maps both correctly).
            signal = signal[::-1, :].copy()
        if excluded:
            signal = apply_drop_channel(signal, excluded)

        aux = load_aux(aux_path)
        (_, _, move_starts, labels, move_duration, _, _, per_label_offsets) = (
            compute_move_markers(signal, aux, log_path)
        )
        if not move_starts or not labels or move_duration is None:
            print(f"    Skipping – could not compute move markers")
            continue

        for t0, label, auto_off in zip(move_starts, labels, per_label_offsets):
            key = (subject_dir.name, trial_dir.name, emg_path.name, label)
            for j_idx, jitter in enumerate(jitter_offsets):
                if key in manual_windows:
                    # Absolute start/end from manual labelling tool
                    start_t = manual_windows[key][0] + jitter
                    end_t   = manual_windows[key][1] + jitter
                else:
                    off = offsets.get(key, auto_off)
                    start_t = t0 + off + jitter
                    end_t   = start_t + max(move_duration - off, 0.0)

                if end_t <= start_t:
                    continue

                start_idx = max(int(round(start_t * fs)), 0)
                end_idx   = min(int(round(end_t   * fs)), signal.shape[1])
                if end_idx <= start_idx:
                    continue

                seg = signal[:, start_idx:end_idx]

                # Build output path
                label_dir    = safe_label(label)
                subject_name = subject_dir.name + ("_da2" if is_da2 else "")
                trial_name   = trial_dir.name.replace("trial", "(00)trial")
                label_name   = f"(00){label_dir}"
                fname        = f"{subject_name}-{trial_name}-{label_name}_j{j_idx}.png"
                out_path     = emg_root / label_dir / fname

                rms_bins = compute_rms_bins(seg, bins=args.rms_bins)
                yield rms_bins, out_path


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Export 224×224 colour-coded EMG RMS images for ResNet18."
    )
    ap.add_argument("--out", required=True,
                    help="Output root directory")
    ap.add_argument("--data-dir", type=str, default=None,
                    help="Override base data directory (e.g. ~/Downloads/Data 2)")
    ap.add_argument("--include-da2", action="store_true", default=False,
                    help="Also process DA2 files (channels reversed to match DA1 ordering)")
    ap.add_argument("--grayscale", action="store_true", default=False,
                    help="Render images in black and white instead of colour")
    ap.add_argument("--subjects", type=str, default=None,
                    help="Comma-separated subject folder names")
    ap.add_argument("--subjects-range", type=str, default=None,
                    help="Inclusive subject number range, e.g. 1-10")
    ap.add_argument("--trial", type=str, default=None,
                    help="Process only this trial folder name")
    ap.add_argument("--manual-windows", type=str, default=None,
                    help="CSV with manually labelled windows "
                         "(subject,trial,emg_file,label,start_t,end_t)")
    ap.add_argument("--offsets", type=str, default=None,
                    help="CSV of per-label start offsets (offset_sec)")
    ap.add_argument("--noise", type=str, default=None,
                    help="CSV of trial noise scores / excluded channels")
    ap.add_argument("--max-noise", type=float, default=None,
                    help="Skip trials whose noise_score exceeds this threshold (e.g. 0.0004)")
    ap.add_argument("--rms-bins", type=int, default=224,
                    help="Number of time bins = image width (default 224)")
    ap.add_argument("--rms-pctl", type=float, default=95.0,
                    help="Percentile used for amplitude reference (default 95)")
    ap.add_argument("--scale-mode", choices=["global", "per-trial"], default="global",
                    help="'global' (default): one reference across all images, preserving "
                         "inter-class amplitude differences; 'per-trial': each image self-normalised")
    ap.add_argument("--rms-shifts", type=str, default="0,0.2,0.4",
                    help="Comma-separated fractional column-shift values for augmentation "
                         "(default '0,0.2,0.4' → _shift00, _shift20, _shift40)")
    ap.add_argument("--rms-row-shifts", type=str, default="0,0.2,0.4",
                    help="Comma-separated fractional row-shift values for augmentation "
                         "(default '0,0.2,0.4' → _rshift00, _rshift20, _rshift40)")
    ap.add_argument("--jitters", type=int, default=1,
                    help="Number of jittered copies per window (default 1 = no jitter)")
    ap.add_argument("--jitter-sec", type=float, default=0.1,
                    help="Max jitter magnitude in seconds (default 0.1)")
    args = ap.parse_args()

    manual_windows = load_manual_windows(args.manual_windows)
    offsets        = load_offsets(args.offsets)
    noise          = load_noise(args.noise)

    rms_shifts = None
    if args.rms_shifts:
        rms_shifts = [float(v.strip()) for v in args.rms_shifts.split(",") if v.strip()]

    rms_row_shifts = None
    if args.rms_row_shifts:
        rms_row_shifts = [float(v.strip()) for v in args.rms_row_shifts.split(",") if v.strip()]

    # ── Pass 1: collect all RMS bins ──────────────────────────────────────────
    print("Pass 1: loading EMG data and computing RMS bins…")
    cache = []   # list of (rms_bins, out_path)
    for rms_bins, out_path in collect_windows(args, manual_windows, offsets, noise):
        cache.append((rms_bins, out_path))

    if not cache:
        print("No valid windows found. Check subjects/trials/logs.")
        return

    print(f"  Collected {len(cache)} windows.")

    # ── Amplitude reference ───────────────────────────────────────────────────
    if args.scale_mode == "global":
        # Use the 95th-percentile of per-window peak values as the global reference.
        # Taking a flat percentile of all RMS values (including the many near-zero
        # quiet-channel bins) yields a reference near the noise floor → saturation.
        # Instead: for each window compute its own high-percentile peak, then take
        # the 95th percentile of those peaks across all windows.  This anchors the
        # scale to "what does a typical strong activation look like?" so that weak
        # movements (finger extension) show little colour and strong movements (wrist
        # extension) show much more — preserving inter-class amplitude information.
        window_peaks = [float(np.percentile(r, args.rms_pctl)) for r, _ in cache]
        amp_ref = float(np.percentile(window_peaks, 95.0))
        print(f"Global amp_ref (95th-pctl of per-window {args.rms_pctl}th-pctl peaks): {amp_ref:.6f} V")
    else:
        amp_ref = None   # computed per-image in pass 2

    # ── Pass 2: render and save images ────────────────────────────────────────
    print(f"Pass 2: rendering and saving {len(cache)} images…")
    for i, (rms_bins, out_path) in enumerate(cache):
        if amp_ref is None:
            ref = float(np.percentile(rms_bins, args.rms_pctl))
            ref = ref if ref > 0.0 else 1.0
        else:
            ref = amp_ref

        img = render_image(rms_bins, ref, bins=args.rms_bins, grayscale=args.grayscale)
        save_image(img, out_path, shifts=rms_shifts, row_shifts=rms_row_shifts, bins=args.rms_bins)

        if (i + 1) % 50 == 0 or (i + 1) == len(cache):
            print(f"  {i + 1}/{len(cache)}")

    print("Done.")


if __name__ == "__main__":
    main()
