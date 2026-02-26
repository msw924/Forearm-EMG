# EMG Processing + ML Pipeline Specification

This document sets out the important aspects of this project and should be refered to when making changes. it looks at the main documents but more importantly set outs the pipeline of how things should work. 

## 1) Data Layout

Root data directory:
- `base_data_dir`: `/Users/maxwilliams/Library/CloudStorage/Dropbox/MW_GR_data/Data`

Folder structure (typical):
- `(NN)Name/`
  - `trial N/`
    - `*_M02DA1_EMG_raw.sig` (32‑channel EMG, uint16, interleaved)
    - `*_M02DA1_AUX1_raw.sig` (AUX trigger channel, uint16)
    - `*.rtf` / `*.txt` / `*.pdf` (movement log; optional)

Outputs and datasets (examples):
- `_cache_emg/`: cached numpy signals and aux
- `dataset_raw/`, `dataset_manual/`, `dataset_emg_numeric/`, `dataset_rms/`, etc.
- `runs_*`: model outputs and class lists

## 2) Core EMG Processing (`EMG_opener.py`)

Purpose:
- Read raw EMG/AUX, parse logs, compute movement windows, and generate plots.

Key constants:
- 32 channels, `fs = 2049.122`, scaling from raw uint16 to volts.
- `sensor_unit = "DA1"` default; it is used to locate files.

Main functions:
- `load_emg(emg_path)`:
  - Reads `<u2` raw signal, reshapes to `(32, n_samples)`.
  - Scales to volts using ADC parameters.
  - Demeans per channel.
  - Applies 50 Hz notch filter (IIR + filtfilt).
- `load_aux(aux_path)`:
  - Reads AUX and scales to [0..1].
- `detect_aux_pulse(aux)`:
  - Finds first/last threshold crossing for AUX alignment.
- `parse_log(log_path)`:
  - Supports `.rtf`, `.txt`, `.pdf` (RTF stripped to plain text).
  - Extracts pre/post wait, pulse time, and movement labels.
- `compute_move_markers(...)`:
  - Combines AUX, log, and inferred move duration.
  - Computes movement start times.
  - Applies auto‑delay if enabled (via `plots_emg_auto.compute_auto_delays`).
  - Returns `move_starts`, `labels`, `move_duration`, `per_label_offsets`.
- `compute_rms_windows(...)`:
  - For each movement, applies its offset and computes RMS over its window.
  - Uses percentile clipping before RMS (5/95).

Batch driver:
- `main()` uses `subject_glob` + `trial_glob` to iterate trials.
- For each file:
  - `plot_emg_time_series` (fixed delay + auto delay)
  - `plot_polar` (RMS per movement)

## 3) EMG Plotting

### `plots_emg.py`
- Plots stacked EMG time series with labels/markers.
- Supports fixed delay and auto delay via arguments.

### `plots_emg_auto.py`
- Computes per‑label auto delays using RMS/threshold logic.
- Used by `EMG_opener.compute_move_markers`.

### `plots_polar.py`
- Polar plots from per‑movement RMS vectors.

## 4) Manual Labeling / QA

### `ml/label_windows.py`
- Old UI for manually adjusting movement start offsets (per label).
- Uses `offsets.csv` to store `offset_sec` per movement.
- Supports manual drop channel + noise recording.
- Keys: arrow keys move offset, `n/p` move label, `s` save, `q` quit.

### `ml/label_windows_manual.py` + `ml/label_windows_full.py`
- Experimental/manual UI variants for full‑window start/end labeling.

### `ml/plot_emg_windows.py`
- Generates a static plot with start (solid) and end (dashed) lines for each movement.
- Uses `offsets.csv` to apply manual offsets.
- Used for sanity checks without UI.

### Offsets / noise files
- `ml/offsets.csv`: per movement offsets per trial.
- `ml/trial_noise.csv`: optional per‑trial noise score and excluded channel.
- `ml/trial_skip.csv`: legacy, not used in current labeling UI.

### `ml/backfill_offsets.py`
- Fills missing offsets with default per‑label values.
- Intended to avoid revisiting old trials.

## 5) Image Dataset Export

### `ml/export_resnet_images.py`
Exports EMG and polar images from movement windows.

Inputs:
- `--offsets` (optional manual offsets)
- `--noise` (optional drop channels)
- `--subjects` or `--subjects-range`
- Jitter options (multiple windows per movement)

Outputs:
- `emg/`: line plots (stacked channels)
- `polar/`: polar plots (line/heatmap)
- optional `filtered` EMG plots (bandpass + notch)

Global calibration:
- Computes `amp_ref`, `rms_min`, `rms_max` across subjects for consistent scaling.

### RMS image export (32×256 grayscale)
- `--emg-rms-out`: enables grayscale RMS “strip” images.
- `--rms-bins` (default 256): number of time bins.
- Produces `dataset_rms/emg_rms/<label>/...png`.
- Each image: 32 rows (channels), 256 columns (time), grayscale RMS in each bin.

## 6) Realtime Image Export (Experimental)

### `ml/export_realtime_images.py`
- Generates “sliding RMS frame” or “topographic snapshot” datasets.
- Uses manual offsets, window slicing, and history.
- Intended for near real‑time ML experimentation.

## 7) Dataset Splitting

### `ml/split_dataset.py`
- Splits image datasets into `train/val/test` folders.
- Supports `.png` and `.npy`.
- Used for EMG/polar/RMS images.

### `ml/split_by_subject.py`
- Subject‑level split for subject‑held‑out testing.

## 8) ML Training / Evaluation

### ResNet (image models)
- `ml/train_resnet18.py` / `ml/eval_resnet18.py`
- Resizes to 224×224, converts grayscale→RGB, normalizes with ImageNet stats.
- Expects `train/val/test` folders.

### 1D CNN (numeric)
- `ml/train_cnn1d.py` / `ml/eval_cnn1d.py`
- Uses `.npy` windows (exported via `ml/export_emg_windows.py`).
- Pads variable length sequences.

### TCN
- `ml/train_tcn.py` / `ml/eval_tcn.py`
- Temporal convolution for `.npy` windows.

### 2D CNN for RMS strips
- `ml/train_cnn2d_rms.py` / `ml/eval_cnn2d_rms.py`
- Uses 1‑channel 32×256 grayscale images without resizing.

## 9) Visualization / UI Tools

### FFT viewers
- `ml/fft_cursor_viewer.py`: EMG plot + FFT spectrum at slider cursor.
- `ml/fft_emg.py`: FFT utilities.

### Polar slider viewer
- `ml/polar_slider_viewer.py`: EMG plot + polar map of channel values at cursor.
- Uses 0.5s averaging window to reduce jitter.

### Forearm visualization
- `ml/forearm_electrodes_viewer.py`: plots electrode positions on forearm image.
- `ml/forearm_com.py`:
  - Takes movement window and computes RMS per channel.
  - Colors electrodes by activation and marks center of mass (COM).
  - Supports rotation, channel mapping, label‑per‑electrode.
  - Styles: scatter (default), ring, voronoi (experimental).

## 10) Key Outputs / Artifacts

- `offsets.csv`: manual movement offsets (per label).
- `trial_noise.csv`: noise score + excluded channel.
- `dataset_rms/emg_rms`: 32×256 grayscale RMS images.
- `dataset_*_split`: train/val/test splits.
- `runs_*`: model weights + classes list.
- `forearm_com_*.png`: activation overlays.
- `plot_windows_*.png`: sanity‑check plots.

## 11) Current Defaults and Assumptions

- `sensor_unit = "DA1"` used across scripts.
- Movement windows are inferred from AUX + log, then shifted by per‑label offsets.
- If offsets missing: fall back to default per‑label offsets (via `backfill_offsets.py` or default in `compute_move_markers`).
- RMS windows use full movement duration minus offset.

## 12) How to Validate a Trial’s Windows (Current Best Practice)

- `ml/plot_emg_windows.py` for static verification using offsets.
- `ml/label_windows.py` for manual correction and saving to `offsets.csv`.

## 13) Suggested Next Steps (if modifying)

- Keep a single “source of truth” for offsets and apply across exporters.
- Decide whether RMS images or raw numeric windows are the primary ML input.
- Consider subject‑held‑out splits for generalization.
