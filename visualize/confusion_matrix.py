import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle


def plot_normalized_confusion_matrix(
    cm,
    save_path,
    class_labels=None,
    figsize=(4, 5),
    dpi=150,
    new_order=None
):
    """
    Plot a row-normalized confusion matrix and save it to disk.

    Parameters
    ----------
    cm : array-like, shape (N, N)
        Confusion matrix.
    save_path : str
        Path where the figure will be saved (e.g., "cm.png" or "cm.pdf").
    class_labels : list of str, optional
        Class names for axes. Defaults to C0, C1, ...
    figsize : tuple, optional
        Figure size.
    dpi : int, optional
        Figure resolution.
    new_order : list of int, optional
        Reordering of classes (applied to both rows and columns).
        Example: [0, 5, 1, 2, 3, 4]
    """

    cm = np.asarray(cm)

    # Reorder confusion matrix if requested
    if new_order is not None:
        cm = cm[np.ix_(new_order, new_order)]
        if class_labels is not None:
            class_labels = [class_labels[i] for i in new_order]

    # Normalize row-wise
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    # Color-blind safe luminance-driven colormap
    colors = [
        "#f4fbff",  # very light cyan
        "#cfe8f3",
        "#9ecae1",
        "#4fa3c7",
        "#1f6fa8",
        "#08306b"   # deep navy
    ]
    cmap = LinearSegmentedColormap.from_list("cyan_navy", colors)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1)

    # Annotations
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            val = cm_norm[i, j]
            ax.text(
                j, i, f"{val:.3f}",
                ha="center", va="center",
                fontsize=6,
                fontweight="bold" if i == j else "normal",
                color="white" if val > 0.55 else "black"
            )

    # Diagonal highlight boxes
    for i in range(cm_norm.shape[0]):
        ax.add_patch(
            Rectangle(
                (i - 0.5, i - 0.5),
                1, 1,
                fill=False,
                edgecolor="#000000",
                linewidth=1.2
            )
        )

    # Labels
    ax.set_xlabel("Predicted Class", fontsize=8)
    ax.set_ylabel("True Class", fontsize=8)

    n = cm.shape[0]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))

    if class_labels is None:
        class_labels = [f"C{i}" for i in range(n)]

    ax.set_xticklabels(class_labels, rotation=45, ha="right", fontsize=6)
    ax.set_yticklabels(class_labels, fontsize=6)

    # Cell boundaries
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Clean frame
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
