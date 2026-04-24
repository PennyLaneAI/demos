r"""Loading classical data with low-depth circuits
==============================================

Encoding arbitrary classical data into quantum states usually comes at a high computational
cost, either in terms of qubits or gate count. However, real-world data typically exhibits some
inherent structure (such as image data) which can be leveraged to load them with a much smaller cost
on a quantum computer. The paper **“Typical Machine Learning Datasets as Low‑Depth Quantum
Circuits”** (2025) [#Kiwit]_ develops an efficient algorithm for finding low-depth
quantum circuits to load classical image data into quantum states.

This demo gives an introduction to the paper **“Typical Machine Learning Datasets as
Low‑Depth Quantum Circuits”** (2025). We will do the following:

1. **Define** how classical images can be encoded as quantum states.
2. **Construct** low-depth quantum circuits that efficiently generate these image states.
3. **Train and evaluate** a small variational quantum circuit (VQC) classifier on the dataset.
"""

######################################################################
# 1. Quantum image states
# ~~~~~~~~~~~~~~~~~~~~~~~
#
# Images with :math:`2^n` pixels are mapped to states of the form
# :math:`\left| \psi(\mathbf x) \right> = \frac{1}{\sqrt{2^{n}}}\sum_{j=0}^{2^{n}-1} \left| c(x_j) \right> \otimes \left| j \right>`,
# where the **address register** :math:`\left| j\right>` holds the pixel position (:math:`n` qubits),
# and additional **color qubits** :math:`\left| c(x_j)\right>` encode the pixel intensities. For
# grayscale images, we use the *flexible representation of quantum images (FRQI)*
# [#LeFlexible]_ [#LeAdvances]_ as an encoding. In this case, the data value
# :math:`{x}_j` of each pixel is just a single number corresponding to the grayscale value of that
# pixel. We can encode this information in the :math:`z`-polarization of an additional color qubit as
# :math:`\left|c({x}_j)\right> = \cos({\textstyle\frac{\pi}{2}} {x}_j) \left| 0 \right> + \sin({\textstyle\frac{\pi}{2}} {x}_j) \left| 1 \right>`,
# with the pixel value normalized to :math:`{x}_j \in [0,1]`. Thus, a grayscale image with :math:`2^n`
# pixels is encoded into a quantum state with :math:`n+1` qubits.
#
# For color images, the *multi-channel representation of quantum images (MCRQI)*
# [#SunMulti]_ [#SunRGB]_ can be used. Python implementations of the MCRQI
# encoding and decoding are provided at the end of this demo and are discussed in Ref.
# [#Kiwit]_.
#

from pennylane import numpy as np

# Grayscale encodings and decodings


def FRQI_encoding(images):
    """
    Input : (batchsize, N, N) ndarray
        A batch of square arrays representing grayscale images.
    Returns : (batchsize, 2, N**2) ndarray
        A batch of quantum states encoding the grayscale images using the FRQI.
    """
    # get image size and number of qubits
    batchsize, N, _ = images.shape
    n = 2 * int(np.log2(N))
    # reorder pixels hierarchically
    states = np.reshape(images, (batchsize, *(2,) * n))
    states = np.transpose(states,
                          [0] + [ax + 1 for q in range(n // 2) for ax in (q, q + n // 2)])
    # FRQI encoding by stacking cos and sin components
    states = np.stack([np.cos(np.pi / 2 * states), np.sin(np.pi / 2 * states)], axis=1)
    # normalize and reshape
    states = np.reshape(states, (batchsize, 2, N**2)) / N
    return states


def FRQI_decoding(states):
    """
    Input : (batchsize, 2, N**2) ndarray
        A batch of quantum states encoding grayscale images using the FRQI.
    Returns : (batchsize, N, N) ndarray
        A batch of square arrays representing the grayscale images.
    """
    # get batchsize and number of qubits
    batchsize = states.shape[0]
    states = np.reshape(states, (batchsize, 2, -1))
    n = int(np.log2(states.shape[2]))
    # invert FRQI encoding to get pixel values
    images = np.arccos(states[:, 0] ** 2 * 2**n - states[:, 1] ** 2 * 2**n) / np.pi
    # undo hierarchical ordering
    images = np.reshape(images, (batchsize, *(2,) * n))
    images = np.transpose(images, [0, *range(1, n, 2), *range(2, n + 1, 2)])
    # reshape to square image
    images = np.reshape(images, (batchsize, 2 ** (n // 2), 2 ** (n // 2)))
    return images


######################################################################
# 2. Low depth image circuits
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# In general, the complexity of preparing the resulting state *exactly* scales exponentially with the
# number of qubits. Known constructions (without auxiliary qubits) use :math:`\mathcal{O}(4^n)` gates
# [#LeFlexible]_ [#LeAdvances]_. However, encoding typical images this way leads to
# lowly entangled quantum states that are well approximated by tensor-network states such as
# matrix-product states (MPSs) [#Jobst]_ whose bond dimension :math:`\chi` does not need
# to scale with the image resolution. Thus, preparing the state *approximately* with a small error is
# possible with a number of gates that scales only as :math:`\mathcal{O}(\chi^2n)`, i.e., linearly
# with the number of qubits. While the cost of the classical preprocessing may be similar to the exact
# state preparation, the resulting quantum circuits are exponentially more efficient.
#
# The following illustration shows the quantum circuits inspired by MPSs. The left side shows a
# circuit with a staircase pattern with two layers (represented in turquoise and pink), where
# two-qubit gates are applied sequentially, corresponding to a right-canonical MPS. The right side
# shows the proposed circuit architecture corresponding to an MPS in mixed canonical form. By
# effectively shifting the gates with the dashed outlines to the right, the gates are applied
# sequentially outward starting from the center. This reduces the circuit depth while maintaining its
# expressivity.
#
# .. figure:: /_static/demonstration_assets/low_depth_circuits_mnist/circuit.png
#    :align: center
#    :width: 80 %
#    :alt: Illustration of quantum circuits inspired by MPSs
#
#    Illustration of quantum circuits inspired by MPSs


######################################################################
# Downloading the quantum image dataset
# -------------------------------------
#
# The dataset configuration sets the name as ``'low-depth-mnist'`` and constructs the dataset path as
# ``datasets/low-depth-mnist/low-depth-mnist.h5``. For dataset loading, if the file exists locally, it
# is loaded using ``qp.data.Dataset.open``. Otherwise, the dataset is downloaded from the PennyLane
# data repository via ``qp.data.load``, note that the dataset size is approximately 1 GB.
#

import os
import jax
import pennylane as qp

# JAX supports the single-precision numbers by default.
# The following line enables double-precision.
jax.config.update("jax_enable_x64", True)
# Set JAX to use CPU, simply set this to 'gpu' or 'tpu' to use those devices.
jax.config.update("jax_platform_name", "cpu")

# Here you can choose the dataset and the encoding depth, depth 4 and depth 8 are available
DATASET_NAME = "low-depth-mnist"

dataset_path = f"datasets/{DATASET_NAME}.h5"

# Load the dataset if already downloaded
if os.path.exists(dataset_path):
    dataset_params = qp.data.Dataset.open(dataset_path)
else:
    # Download the dataset (~ 1 GB)
    [dataset_params] = qp.data.load(DATASET_NAME)

######################################################################
# In the following cell, we define the ``get_circuit`` function that creates a quantum circuit based
# on the provided layout. The ``circuit_layout`` is an attribute of the dataset that specifies the
# sequence of quantum gates and their target qubits, which depends on the number of qubits and circuit
# depth. After defining the circuit function, we extract the relevant data for binary classification
# (digits 0 and 1 only) and compute the quantum states by executing the circuits with their
# corresponding parameters. These generated states will be used later for training the quantum
# classifier. You can find more information and download the datasets at
# `PennyLane Datasets: Low-Depth Image Circuits <https://pennylane.ai/datasets/collection/low-depth-image-circuits>`_.
#

TARGET_LABELS = [0, 1]


def get_circuit(circuit_layout):
    """
    Create a quantum circuit with a given layout for preparing quantum states.
    The circuit only contains RY rotation gates and CNOT gates, designed for efficient
    state preparation with low circuit depth.

    :param circuit_layout: List of tuples containing gate types ('RY' or 'CNOT')
    and their target wires.
    :return circuit: A JAX-compiled quantum circuit function that takes parameters and
    returns the quantum state.
    """
    dev = qp.device("default.qubit", wires=11)

    @jax.jit
    @qp.qnode(dev)
    def circuit(params):
        counter = 0
        for gate, wire in circuit_layout:

            if gate == "RY":
                qp.RY(params[counter], wire)
                counter += 1

            elif gate == "CNOT":
                qp.CNOT(wire)

        return qp.state()

    return circuit


# Unpack the dataset attributes, in this demo only digits 0 and 1 will be used
labels = np.asarray(dataset_params.labels)
selection = np.isin(labels, TARGET_LABELS)
labels_01 = labels[selection]
exact_state = np.asarray(dataset_params.exact_state)[selection]

circuit_layout = dataset_params.circuit_layout_d4
circuit = get_circuit(circuit_layout)
params_01 = np.asarray(dataset_params.params_d4)[selection]

states_01 = []
n = len(params_01)

for i, params in enumerate(params_01):
    states_01.append(circuit(params))
    # Print every 10%
    if (i + 1) % (n // 10) == 0:
        print(f"{(i + 1) / n * 100:.0f}% of the states computed")

states_01 = np.asarray(states_01)

fidelities_01 = np.asarray(dataset_params.fidelities_d4)[selection]

######################################################################
# Reconstructing images from quantum states
# -----------------------------------------
#
# To investigate how well the low-depth circuits reproduce the target images, we first **reconstruct**
# the pictures encoded in each quantum state. The histogram below reports the *fidelity*
# :math:`F = \left|\langle \psi_{\text{exact}} \mid \psi_{\text{circ.}} \rangle\right|^{2}`, i.e. the
# overlap between the exact FRQI state :math:`|\psi_{\text{exact}}\rangle` and its 4-layer
# center-sequential approximation :math:`|\psi_{\text{circ.}}\rangle`.
#
# - **Digit 1** samples (orange) cluster at a fidelity :math:`F` close to 1, indicating that four
#   layers already capture these images almost perfectly.
# - **Digit 0** samples (blue) display a broader, slightly lower-fidelity distribution, hinting at the
#   greater entanglement required to reproduce their curved outline.
#
# On the right we decode the states back into pixel space. In line with the histogram, the
# reconstructed “1” is virtually indistinguishable from its original, whereas the reconstructed “0”
# shows minor blurring. By selecting a deeper circuit, the quality of the reconstructed images could be
# improved by trading quality for efficiency.
#

import matplotlib.pyplot as plt

# Select images with highest fidelity
idx_0 = np.argmax(fidelities_01[labels_01 == 0])
idx_1 = np.argmax(fidelities_01[labels_01 == 1])

orig_0 = FRQI_decoding(exact_state[labels_01 == 0][idx_0][None, :])[0]
orig_1 = FRQI_decoding(exact_state[labels_01 == 1][idx_1][None, :])[0]

rec_0 = FRQI_decoding(states_01[labels_01 == 0][idx_0][None, :])[0]
rec_1 = FRQI_decoding(states_01[labels_01 == 1][idx_1][None, :])[0]

# Create a grid of figures to show both the fidelity distribution and the original
# and reconstructed images
fig = plt.figure(figsize=(9, 5))
gs = fig.add_gridspec(2, 3, width_ratios=[1.2, 1, 1], wspace=0.05)

# Histogram (spans both rows, leftmost column)
ax_hist = fig.add_subplot(gs[:, 0])
ax_hist.hist(fidelities_01[labels_01 == 0], bins=20, alpha=0.5, label="Digit 0")
ax_hist.hist(fidelities_01[labels_01 == 1], bins=20, alpha=0.5, label="Digit 1")
ax_hist.set_xlabel("Fidelity")
ax_hist.set_ylabel("Count")
ax_hist.legend(loc="upper right")

# Image axes (2 × 2 on the right)
ax00 = fig.add_subplot(gs[0, 1])
ax01 = fig.add_subplot(gs[0, 2])
ax10 = fig.add_subplot(gs[1, 1])
ax11 = fig.add_subplot(gs[1, 2])

ax00.imshow(np.abs(orig_0), cmap="gray")
ax00.set_title("Original 0")
ax01.imshow(np.abs(orig_1), cmap="gray")
ax01.set_title("Original 1")
ax10.imshow(np.abs(rec_0), cmap="gray")
ax10.set_title("Reconstructed 0")
ax11.imshow(np.abs(rec_1), cmap="gray")
ax11.set_title("Reconstructed 1")

# Remove all tick marks from image axes
for ax in [ax00, ax01, ax10, ax11]:
    ax.set_xticks([])
    ax.set_yticks([])

######################################################################
# 3. Quantum classifiers
# ~~~~~~~~~~~~~~~~~~~~~~
#
# In this demo, we train a **variational quantum circuit** as a classifier. Our datasets require
# ``N_QUBITS = 11``, therefore we use the same number of qubits for the classifier. Given a data state
# :math:`\rho(x)=\lvert\psi(x)\rangle\langle\psi(x)\rvert`, a generic **quantum classifier** evaluates
# :math:`f_{\ell}(x) = \operatorname{Tr}\bigl[ O_{\ell}(\theta)\,\rho(x) \bigr]`, with trainable
# circuit parameters :math:`\theta` that rotate a measurement operator :math:`O_\ell`. Variants
# explored in the paper [#Kiwit]_ include
#
# - **Linear VQC** — sequential two‑qubit SU(4) layers (15 parameters per gate).
# - **Non‑linear VQC** — gate parameters depend on input data *x* via auxiliary retrieval circuits.
# - **Quantum‑kernel SVMs** — replacing inner products by quantum state overlaps.
# - **Tensor‑network (MPS/MPO) classifiers** for large qubit counts.
#
# In this demo we use a small linear VQC. The circuit consists of two qubit gates corresponding to
# the ``SO(4)`` gates
#
# .. figure:: /_static/demonstration_assets/low_depth_circuits_mnist/so4.png
#    :align: center
#    :width: 20 %
#    :alt: Illustration of the SO(4) decomposition
#
# arranged in the sequential layout.
#

import optax

# Define the hyperparameters
EPOCHS = 5
BATCH = 128
VAL_FRAC = 0.2
N_QUBITS = 11
DEPTH = 4
N_CLASSES = 2
SEED = 0

# Explicitly compute the number of model parameters
N_PARAMS_FIRST_LAYER = N_QUBITS
N_PARAMS_BLOCK = 4
N_PARAMS_NETWORK = N_PARAMS_FIRST_LAYER + (N_QUBITS - 1) * DEPTH * N_PARAMS_BLOCK

key = jax.random.PRNGKey(SEED)

# Define the model and training functions
dev = qp.device("default.qubit", wires=N_QUBITS)


@jax.jit
@qp.qnode(dev, interface="jax")
def circuit(network_params, state):
    p = iter(network_params)
    qp.StatePrep(state, wires=range(N_QUBITS))

    # First two layers of local RY rotations
    for w in range(N_QUBITS):
        qp.RY(next(p), wires=w)

    # SO(4) building blocks
    for _ in range(DEPTH):
        for j in range(N_QUBITS - 1):
            qp.CNOT(wires=[j, j + 1])
            qp.RY(next(p), wires=j)
            qp.RY(next(p), wires=j + 1)
            qp.CNOT(wires=[j, j + 1])
            qp.RY(next(p), wires=j)
            qp.RY(next(p), wires=j + 1)

    # Probability of computational basis states of the last qubit
    # Can be extended to more qubits for multiclass case
    return qp.probs(N_QUBITS - 1)


model = jax.vmap(circuit, in_axes=(None, 0))


def loss_acc(params, batch_x, batch_y):
    logits = model(params, batch_x)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch_y).mean()
    acc = (logits.argmax(-1) == batch_y).mean()
    return loss, acc


# training step
@jax.jit
def train_step(params, opt_state, batch_x, batch_y):
    func = lambda p: loss_acc(p, batch_x, batch_y)
    (loss, acc), grads = jax.value_and_grad(func, has_aux=True)(
        params
    )
    updates, opt_state = opt.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, loss, acc


# data loader
def loader(X, y, batch_size, rng_key):
    idx = jax.random.permutation(rng_key, len(X))
    for i in range(0, len(X), batch_size):
        sl = idx[i : i + batch_size]
        yield X[sl], y[sl]


######################################################################
# Preparing the training / validation split
# ------------------------------------------
#
# We start by **casting** the FRQI amplitude vectors and their digit labels into JAX arrays. Next, the
# states and labels are shuffled from a pseudorandom key derived from the global ``SEED``. Then, the
# data is split into training and validation. Finally, we gather the tensors corresponding to the
# training ``(X_train, y_train)`` and validation sets ``(X_val, y_val)``.
#

from jax import numpy as jnp

# Prepare the data

X_all = jnp.asarray(
    states_01.real, dtype=jnp.float64
)  # we select the real part only
# the imaginary part is zero since we only use RY and CNOT gates
y_all = jnp.asarray(labels_01, dtype=jnp.int32)

key_split, key_perm = jax.random.split(jax.random.PRNGKey(SEED))
perm = jax.random.permutation(key_perm, len(X_all))
split_pt = int(len(X_all) * (1 - VAL_FRAC))

train_idx = perm[:split_pt]
val_idx = perm[split_pt:]

X_train, y_train = X_all[train_idx], y_all[train_idx]
X_val, y_val = X_all[val_idx], y_all[val_idx]

######################################################################
# Training setup and loop
# -----------------------
#
# We begin by **initializing** the network weights ``params`` with values drawn uniformly from
# :math:`[0, 2\pi]` and initialize the **Adam optimizer** with a learning rate of
# :math:`1 \times 10^{-2}`. The **training loop** then iterates for ``EPOCHS``:
#
# 1. For each mini-batch, ``train_step`` performs a forward pass, computes the cross-entropy loss and
#    accuracy, back-propagates gradients, and updates ``params`` through the optimizer state
#    ``opt_state``.
# 2. Using the *current* parameters, we evaluate the same metrics on the validation set **without**
#    gradient updates.
# 3. Epoch-mean loss (``tl``, ``vl``) and accuracy (``ta``, ``va``) are appended to the tracking lists
#    for later plotting.
#
# The first epoch will take longer than following epochs because of the just-in-time compilation.
#

# Define the training setup and start the training loop

# optimizer
params = 2 * jnp.pi * jax.random.uniform(key, (N_PARAMS_NETWORK,), dtype=jnp.float64)
opt = optax.adam(1e-2)
opt_state = opt.init(params)

# training loop
rng = key_split
train_loss_curve, val_loss_curve = [], []
train_acc_curve, val_acc_curve = [], []
for epoch in range(1, EPOCHS + 1):
    # train
    rng, sub = jax.random.split(rng)
    train_losses, train_accs = [], []
    for bx, by in loader(X_train, y_train, BATCH, sub):
        params, opt_state, l, a = train_step(params, opt_state, bx, by)
        train_losses.append(l)
        train_accs.append(a)

    # validation
    val_losses, val_accs = [], []
    for bx, by in loader(X_val, y_val, BATCH, rng):
        l, a = loss_acc(params, bx, by)
        val_losses.append(l)
        val_accs.append(a)

    tl = jnp.mean(jnp.stack(train_losses))
    vl = jnp.mean(jnp.stack(val_losses))
    ta = jnp.mean(jnp.stack(train_accs))
    va = jnp.mean(jnp.stack(val_accs))

    train_loss_curve.append(tl)
    val_loss_curve.append(vl)
    train_acc_curve.append(ta)
    val_acc_curve.append(va)
    print(f"Epoch {epoch:03d}/{EPOCHS} | "
          f"train_loss={tl:.4f}, val_loss={vl:.4f}, "
          f"train_acc={ta:.4f}, val_acc={va:.4f}")

# Plot the training curves
(
    fig,
    ax,
) = plt.subplots(1, 2, figsize=(12.8, 4.8))
ax[0].plot(train_loss_curve, label="Train")
ax[0].plot(val_loss_curve, label="Validation")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Loss")
ax[0].legend()
ax[1].plot(train_acc_curve)
ax[1].plot(val_acc_curve)
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Accuracy")

######################################################################
# Conclusion
# ~~~~~~~~~~
#
# In this notebook we have demonstrated the use of low-depth quantum circuits to load and
# subsequently classify (a subset of) the MNIST dataset.
# By filtering to specific target labels, constructing parametrized circuits from the provided
# layouts, and evaluating their states and fidelities, we have gained hands-on experience with
# quantum machine learning workflows on real data encodings.
#
# Explore the full set of `provided
# datasets <https://pennylane.ai/datasets/collection/low-depth-image-circuits>`__—they contain a
# variety of different datasets at varying circuit depths, parameterizations, and target classes. You
# can adapt the presented workflow to different subsets and datasets, experiment with your own models,
# and contribute back insights on how these benchmark datasets can best support the development of
# practical quantum machine learning approaches.
#
#
# References
# ~~~~~~~~~~
#
# .. [#Kiwit] 
# 
#     F.J. Kiwit, B. Jobst, A. Luckow, F. Pollmann and C.A. Riofrío. Typical Machine Learning Datasets
#     as Low-Depth Quantum Circuits. *Quantum Sci. Technol.* in press (2025). DOI:
#     https://doi.org/10.1088/2058-9565/ae0123.
#
# .. [#LeFlexible]
#
#     P.Q. Le, F. Dong and K. Hirota. A flexible representation of quantum images for polynomial
#     preparation, image compression, and processing operations. *Quantum Inf. Process* 10, 63–84 (2011).
#     DOI: https://doi.org/10.1007/s11128-010-0177-y.
#
# .. [#LeAdvances]
# 
#     P.Q. Le, A.M. Iliyasu, F. Dong, and K. Hirota. A Flexible Representation and Invertible
#     Transformations for Images on Quantum Computers. In: Ruano, A.E., Várkonyi-Kóczy, A.R. (eds) *New
#     Advances in Intelligent Signal Processing. Studies in Computational Intelligence*, vol 372.
#     Springer, Berlin, Heidelberg (2011). DOI: https://doi.org/10.1007/978-3-642-11739-8_9.
#
# .. [#SunMulti]
# 
#     B. Sun *et al.* A Multi-Channel Representation for images on quantum computers using the RGBα
#     color space, 2011 *IEEE 7th International Symposium on Intelligent Signal Processing*, Floriana,
#     Malta, pp. 1-6 (2011). DOI: https://doi.org/10.1109/WISP.2011.6051718.
#
# .. [#SunRGB]
# 
#     B. Sun, A. Iliyasu, F. Yan, F. Dong, and K. Hirota. An RGB Multi-Channel Representation for
#     Images on Quantum Computers, *J. Adv. Comput. Intell. Intell. Inform.*, Vol. 17 No. 3, pp. 404–417
#     (2013). DOI: https://doi.org/10.20965/jaciii.2013.p0404.
#
# .. [#Jobst]
# 
#     B. Jobst, K. Shen, C.A. Riofrío, E. Shishenina and F. Pollmann. Efficient MPS representations
#     and quantum circuits from the Fourier modes of classical image data. *Quantum* 8, 1544 (2024). DOI:
#     https://doi.org/10.22331/q-2024-12-03-1544.
#

######################################################################
# Appendix
# ~~~~~~~~~
#

# The CIFAR-10 and Imagenette datasets use the following MCRQI color encoding and decoding [4,5]


def MCRQI_encoding(images):
    """
    Input : (batchsize, N, N, 3) ndarray
        A batch of arrays representing square RGB images.
    Returns : (batchsize, 8, N**2) ndarray
        A batch of quantum states encoding the RGB images using the MCRQI.
    """
    # get image size and number of qubits
    batchsize, N, _, channels = images.shape
    n = 2 * int(np.log2(N))
    # reorder pixels hierarchically
    states = np.reshape(images, (batchsize, *(2,) * n, channels))
    states = np.transpose(
        states,
        [0] + [ax + 1 for q in range(n // 2) for ax in (q, q + n // 2)] + [n + 1],
    )
    # MCRQI encoding by stacking cos and sin components
    states = np.stack(
        [
            np.cos(np.pi / 2 * states[..., 0]),
            np.cos(np.pi / 2 * states[..., 1]),
            np.cos(np.pi / 2 * states[..., 2]),
            np.ones(states.shape[:-1]),
            np.sin(np.pi / 2 * states[..., 0]),
            np.sin(np.pi / 2 * states[..., 1]),
            np.sin(np.pi / 2 * states[..., 2]),
            np.zeros(states.shape[:-1]),
        ],
        axis=1,
    )
    # normalize and reshape
    states = np.reshape(states, (batchsize, 8, N**2)) / (2 * N)
    return states


def MCRQI_decoding(states):
    """
    Input : (batchsize, 8, N**2) ndarray
        A batch of quantum states encoding RGB images using the MCRQI.
    Returns : (batchsize, N, N, 3) ndarray
        A batch of arrays representing the square RGB images.
    """
    # get batchsize and number of qubits
    batchsize = states.shape[0]
    states = np.reshape(states, (batchsize, 8, -1))
    N2 = states.shape[2]
    N = int(np.sqrt(N2))
    n = int(np.log2(N2))
    # invert MCRQI encoding to get pixel values
    images = np.arccos(states[:, :3] ** 2 * 4 * N2 - states[:, 4:7] ** 2 * 4 * N2) / np.pi
    # undo hierarchical ordering
    images = np.reshape(images, (batchsize, 3, *(2,) * n))
    images = np.transpose(images, [0, *range(2, n + 1, 2), *range(3, n + 2, 2), 1])
    # reshape to square image
    images = np.reshape(images, (batchsize, N, N, 3))
    return images
