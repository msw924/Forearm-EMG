import numpy as np
import matplotlib.pyplot as plt


"""Polar plotting utilities."""


# Plot polar RMS summary for selected movements.
def plot_polar(rms_mat, valid_labels, title):
    polar_targets = [
        "Index Extension",
        "Index Flexion",
        "Wrist Extension",
        "Wrist Flexion",
        "3-Finger Extension",
        "3-Finger Flexion",
    ]
    target_labels = [lbl for lbl in polar_targets if lbl in valid_labels]
    if not target_labels:
        return

    target_indices = [valid_labels.index(lbl) for lbl in target_labels]
    rms_subset = rms_mat[target_indices]
    vmin = np.min(rms_subset)
    vmax = np.max(rms_subset)
    denom = vmax - vmin if vmax > vmin else 1.0
    base_radius = 1.0
    scale = 0.3
    angles = np.linspace(0, 2 * np.pi, rms_mat.shape[1], endpoint=False)
    angles_closed = np.append(angles, angles[0])

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="polar")
    ax.plot(np.linspace(0, 2 * np.pi, 200), [base_radius] * 200, color="gray", linewidth=0.8)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    for label, idx, color in zip(target_labels, target_indices, colors):
        rms_vals = rms_mat[idx]
        norm = (rms_vals - vmin) / denom
        radii = base_radius + scale * norm
        radii_closed = np.append(radii, radii[0])
        ax.plot(angles_closed, radii_closed, linewidth=1.2, color=color, label=label)
        ax.scatter(angles, radii, s=14, color=color)
    ax.set_title(title)
    ax.set_rticks([])
    ax.set_xticks(angles)
    ax.set_xticklabels([f"Ch{i+1}" for i in range(len(angles))], fontsize=8)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


# Build cache paths for preprocessed EMG/AUX.
