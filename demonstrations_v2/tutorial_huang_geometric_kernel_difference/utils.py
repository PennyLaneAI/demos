"""
Utility function that displays multiple PennyLane circuits side by side
"""

from io import BytesIO
import matplotlib.image as mpimg
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt


def draw_circuits_side_by_side(circuits, titles, n_qubits, figsize=(14, 5)):
    """Display multiple PennyLane circuits side by side."""

    images = []
    for circuit in circuits:
        fig, _ = qml.draw_mpl(circuit)(np.zeros(n_qubits))
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        images.append(mpimg.imread(buf))
        plt.close(fig)

    fig, axes = plt.subplots(1, len(circuits), figsize=figsize)
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.show()