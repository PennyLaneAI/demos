r"""Enhancing quantum self-attention with strongly entangling circuit classifiers
=================================================================================

Build and train a quantum self-attention-inspired binary classifier with PennyLane and PyTorch,
following the circuit-classifier approach in [#Tran]_. Learn how its registers and readout work,
then adapt its data, depth, and training setup using the practical recipes at the end.

Quantum self-attention explores quantum-circuit alternatives to the classical self-attention
mechanism at the heart of modern deep-learning models. Several architectures were proposed recently,
including the quantum self-attention network (QSAN) [#QSAN]_, the quantum kernel self-attention
network (QKSAN) [#QKSAN]_, the quantum self-attention neural network (QSANN) [#QSANN]_ and related
circuit designs [#QSANM]_. Here we
explore a trainable circuit readout for classification rather than a fixed measurement alone.

In this demo we adapt the approach of our paper **"Efficient Circuit Classifier Design for
Enhancing Quantum Self-Attention in Vision Transformers"** [#Tran]_: we treat the quantum
self-attention network purely as a *feature extractor* and append a trainable **strongly
entangling-layers circuit classifier** [#Schuld]_ instead of measuring the attention register
directly. Concretely, we will

1. **Build** each building block of the quantum self-attention network (amplitude encoding,
   query/key/value unitaries, the barbell swap operation, and the quantum-logic-similarity module)
   with PennyLane;
2. **Load** a small, PennyLane-native benchmark (downscaled MNIST [#Bowles]_) so the whole demo is
   self-contained;
3. **Train** the quantum layer with Nesterov momentum and inspect its training and test accuracy.

This is a small illustrative experiment, not a reproduction of the paper's benchmark comparisons.
CPU simulation does not establish a computational advantage over classical attention.

Prerequisites and scope
-----------------------

You should be comfortable with qubit gates, expectation values, binary classification, and basic
PyTorch optimization. For a gentler introduction to the learning loop, start with
:doc:`demos/tutorial_variational_classifier`.

This tutorial operates on one classical feature vector at a time. It is not a complete vision
transformer: it does not construct image patches, a token-to-token attention matrix, or a softmax
attention layer. Both the feature-extraction circuit and the classifier are trained jointly;
"feature extractor" does not mean that its parameters are frozen.

Running the Python script requires PennyLane, PennyLane-Lightning, PyTorch, Matplotlib, and the
PennyLane dataset dependencies (``h5py``, ``fsspec``, and ``aiohttp``). The first data load requires
network access. The repository's demo builder installs core dependencies and the additional
dependency in ``requirements.in``. Training a state-vector simulation can be expensive; see
the reduced-budget recipe below before running the full experiment.
"""

import matplotlib.pyplot as plt
import pennylane as qml
import torch

torch.manual_seed(7)

###############################################################################
# The architecture
# ----------------
#
# Given a :math:`D`-dimensional feature vector :math:`x`, each register of the network uses
# :math:`N_r = \lceil \log_2 D \rceil` qubits with **amplitude encoding**
#
# .. math::
#     \lvert \psi(x) \rangle = \frac{1}{\lVert x \rVert_2}\sum_i x_i \lvert i \rangle,
#     \qquad \lVert x \rVert_2 > 0.
#
# The network uses four registers (query, key, value and the quantum-logic-similarity result), which
# in our case (with :math:`D = 16`) needs only :math:`4 \times 4 = 16` qubits. We begin by loading the
# same :math:`16`-dimensional features into the query, key and value registers.
# These are three preparations from known classical data, not copies of an unknown quantum state.
# Normalization removes the overall scale of each input; information encoded only in that scale
# is not retained. :class:`~pennylane.AmplitudeEmbedding` performs this normalization for us.
#
# After the swaps and value preparation, the physical wires have the following roles:
#
# .. list-table:: Register layout before the classifier
#    :header-rows: 1
#
#    * - Wires
#      - State or role
#      - Used by classifier?
#    * - 0--3
#      - Value
#      - Yes
#    * - 4--7
#      - Query
#      - Yes
#    * - 8--11
#      - Key
#      - No, but participates in QLS
#    * - 12--15
#      - QLS result
#      - No direct readout
#

###############################################################################
# 1. Amplitude encoding
# ~~~~~~~~~~~~~~~~~~~~~
#


def state_preparation(f=None):
    """Amplitude-encode the :math:`16`-dimensional input ``f`` into three registers."""
    qml.AmplitudeEmbedding(features=f, wires=[0, 1, 2, 3], normalize=True)
    qml.AmplitudeEmbedding(features=f, wires=[4, 5, 6, 7], normalize=True)
    qml.AmplitudeEmbedding(features=f, wires=[8, 9, 10, 11], normalize=True)


###############################################################################
# 2. Trainable query/key/value unitaries
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The trainable unitary :math:`U_q(\theta_q)` (and likewise ``U_k``, ``U_v``) maps the input state
# into the query (and key/value) states :math:`\lvert Q \rangle = U_q(\theta_q)\lvert\psi\rangle`,
# :math:`\lvert K \rangle = U_k(\theta_k)\lvert\psi\rangle` and
# :math:`\lvert V \rangle = U_v(\theta_v)\lvert\psi\rangle`. Each layer alternates single-qubit
# :math:`RY` rotations with nearest-neighbour :math:`CNOT` entangling gates.
# A fixed Hadamard on every wire precedes the trainable layers. Each unitary is applied on wires
# 0--3; the swaps move its output to the appropriate register before the next unitary is applied.
#


def unitary_embedding(theta_list):
    """One layer of RY rotations followed by CNOT entangling gates."""
    for i in range(len(theta_list)):
        qml.RY(theta_list[i], wires=i)
    for i in range(len(theta_list) - 1):
        qml.CNOT(wires=[i, i + 1])


def combine_unitary_embedding(number_layers, theta_list):
    """Apply Hadamard gates followed by ``number_layers`` unitary-embedding layers."""
    for i in range(len(theta_list[0])):
        qml.H(wires=i)
    for layer in range(number_layers):
        unitary_embedding(theta_list[layer])


###############################################################################
# 3. Barbell operation and quantum logic similarity
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The **barbell operation** swaps the content of two registers, here to move the query state
# :math:`\lvert Q \rangle` into the second register before both are consumed by the similarity
# module. The **quantum logic similarity (QLS) module** [#QSAN]_ is a Boolean comparison inspired by
# attention, not an implementation of the ordinary real-valued dot product :math:`Q K^T`.
# Toffoli gates write the pairwise AND of query/key computational-basis bits into the result, and
# two :math:`CNOT` gates propagate partial parity along wires 12, 13 and 14. We preserve the
# reference implementation's shortened cascade; wire 15 retains its pairwise comparison.
# For example, query bits ``1101`` and key bits ``1011`` first produce ``1001`` in an initially
# zero result register. CNOTs 12->13 and 13->14 then produce ``1111``. On superposed inputs these
# operations act coherently and can entangle the registers; the result is not a classical score.
#


def barbell_operation(wires_1, wires_2):
    """Swap the content of the two registers ``wires_1`` and ``wires_2``."""
    for i, j in zip(wires_1, wires_2):
        qml.SWAP(wires=[i, j])


def QLS_module(wires_control_1, wires_control_2, wires_result):
    """Quantum logic similarity: TOFFOLIs compare query/key pairs, CNOTs compress the result."""
    for i, j, k in zip(wires_control_1, wires_control_2, wires_result):
        qml.Toffoli(wires=[i, j, k])
    for i in range(len(wires_result) - 2):
        qml.CNOT(wires=[wires_result[i], wires_result[i + 1]])


###############################################################################
# 4. The slicing module
# ~~~~~~~~~~~~~~~~~~~~~~
#
# The **slicing operation** uses multi-controlled :math:`X` gates to compress the information
# gathered by the QLS module back into the register holding the query features. In this demo we keep
# this module around for completeness but, following the reference implementation of our paper, we
# feed the value and query registers into the circuit classifier below. The result register is not
# read out or used as a control again. Its final CNOTs therefore cannot affect the prediction,
# although the preceding Toffolis can affect the query's reduced state through entanglement.
#


def multicontrol_slice(wires_qls, wires_q):
    """Multi-controlled X gates compressing the QLS result into the query register."""
    for i in wires_q:
        qml.MultiControlledX(wires=[*wires_qls, i])


###############################################################################
# 5. Circuit classifier
# ~~~~~~~~~~~~~~~~~~~~~~
#
# The key idea of our work is to use a **strongly entangling layer** ansatz [#Schuld]_ as the
# classifier instead of measuring the output qubits directly. Each layer consists of single-qubit
# rotations on every wire followed by :math:`CNOT` gates connecting wire :math:`i` to wire
# :math:`(i+r) \bmod 8`, with a layer-dependent range :math:`r`. The classification result is read
# out as the expectation value :math:`\langle Z_0 \rangle` of the first qubit.
#

###############################################################################
# Loading the data
# ----------------
#
# The **downscaled MNIST** dataset [#Bowles]_ provides :math:`19` datasets of flat feature vectors
# of dimension :math:`d = 2, \ldots, 20`, generated by PCA-compressing the classic MNIST images
# (binary task: digit 3 vs. digit 5, labels :math:`\pm 1`). We use the
# :math:`d = 16` version, restricted to 50 training and 30 test samples for this illustration.
# The dataset is downloaded automatically by PennyLane. Runtime depends on the CPU and simulator;
# the small subset is not sufficient for a benchmark comparison.
#

[ds] = qml.data.load("other", name="downscaled-mnist")

X = torch.tensor(ds.train["16"]["inputs"], dtype=torch.float64)
Y = torch.tensor(ds.train["16"]["labels"], dtype=torch.float64).reshape(-1)
X_test = torch.tensor(ds.test["16"]["inputs"], dtype=torch.float64)
Y_test = torch.tensor(ds.test["16"]["labels"], dtype=torch.float64).reshape(-1)

# Keep a small, labelled subset for the demo.
N_TRAIN, N_TEST = 50, 30
X, Y = X[:N_TRAIN], Y[:N_TRAIN]
X_test, Y_test = X_test[:N_TEST], Y_test[:N_TEST]

assert X.shape == (N_TRAIN, 16) and Y.shape == (N_TRAIN,)
assert X_test.shape == (N_TEST, 16) and Y_test.shape == (N_TEST,)
assert torch.all((Y == -1) | (Y == 1)) and torch.all((Y_test == -1) | (Y_test == 1))
assert torch.isfinite(X).all() and torch.isfinite(X_test).all()
assert torch.all(torch.linalg.vector_norm(X, dim=1) > 0)
assert torch.all(torch.linalg.vector_norm(X_test, dim=1) > 0)

print(f"Train set: {X.shape}, labels {Y.shape}")
print(f"Test set:  {X_test.shape}, labels {Y_test.shape}")
print("Training label counts:", {label: int((Y == label).sum()) for label in (-1, 1)})
print("Test label counts:    ", {label: int((Y_test == label).sum()) for label in (-1, 1)})

###############################################################################
# The full quantum circuit
# -------------------------
#
# We can now write the variational circuit adapted from the reference implementation:
#
# 1. amplitude-encode the input into the query/key/value registers;
# 2. prepare the query :math:`\lvert Q \rangle`, swap it into the second register;
# 3. prepare the key :math:`\lvert K \rangle`, swap it into the third register;
# 4. apply the QLS comparison between the query and key registers;
# 5. prepare the value :math:`\lvert V \rangle`;
# 6. apply the strongly entangling circuit classifier on the first 8 wires and measure
#    :math:`\langle Z_0 \rangle`.
#

NUM_QUBITS = 16
dev = qml.device("lightning.qubit", wires=NUM_QUBITS)


@qml.qnode(dev, interface="torch", diff_method="adjoint")
def full_circuit(x, weights, parameters):
    """Run the full quantum self-attention classifier.

    ``weights[u][l]`` holds the four angles for unitary ``u in {Q, K, V}`` and layer
    ``l``; ``parameters`` are the weights of the strongly entangling circuit classifier.
    """
    state_preparation(f=x)
    num_layers = weights.shape[1]

    # Prepare the query state and swap it into the second register
    combine_unitary_embedding(number_layers=num_layers, theta_list=weights[0])
    barbell_operation([0, 1, 2, 3], [4, 5, 6, 7])

    # Prepare the key state and swap it into the third register
    combine_unitary_embedding(number_layers=num_layers, theta_list=weights[1])
    barbell_operation([0, 1, 2, 3], [8, 9, 10, 11])

    # Compute the quantum logic similarity between query and key
    QLS_module([4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15])

    # Prepare the value state
    combine_unitary_embedding(number_layers=num_layers, theta_list=weights[2])

    # Circuit classifier on the extracted features
    qml.StronglyEntanglingLayers(weights=parameters, wires=range(8))
    return qml.expval(qml.PauliZ(0))


###############################################################################
# Initialising the parameters
# ---------------------------
#
# We use one layer of unitary embedding per query/key/value unitary (12 angles in total) and
# :math:`15` strongly entangling layers for the classifier, matching the configuration studied in
# the paper and initialised with a fixed seed for reproducibility.
#

N_LAYERS_QK = 1  # unitary embedding layers for Q, K and V
N_LAYERS_STRONG = 15  # strongly entangling layers of the circuit classifier

strong_shape = qml.StronglyEntanglingLayers.shape(n_layers=N_LAYERS_STRONG, n_wires=8)
parameters = torch.nn.Parameter(torch.rand(strong_shape, dtype=torch.float64))
weights = torch.nn.Parameter(0.01 * torch.randn(3, N_LAYERS_QK, 4, dtype=torch.float64))

print(f"Strongly-entangling-layer parameter shape: {parameters.shape}")

###############################################################################
# Inspect one prediction
# -----------------------
#
# The Q/K/V angles have shape ``(3, N_LAYERS_QK, 4)``. The classifier has shape
# ``(N_LAYERS_STRONG, 8, 3)`` because each wire receives three rotation angles per layer.
# This configuration contains :math:`12 + 15\times 8\times 3 = 372` trainable parameters.
#
# Our analytic (shot-free) simulator returns an expectation value in :math:`[-1, 1]`, not a
# class label. We predict +1 for a nonnegative value and -1 otherwise, assigning exact zero to
# +1 consistently. The ``torch`` interface connects the result to PyTorch's gradient graph;
# adjoint differentiation computes circuit-parameter derivatives on the simulator. This is
# not a finite-shot hardware training procedure.
#


def predict_labels(expectations):
    """Convert expectation values to binary labels, assigning zero to +1."""
    return torch.where(expectations >= 0, 1.0, -1.0)


with torch.no_grad():
    initial_prediction = full_circuit(X[0], weights, parameters)
print(f"Initial expectation: {initial_prediction.item():.4f}")
print(
    f"Predicted label: {predict_labels(initial_prediction).item():+.0f}; target: {Y[0].item():+.0f}"
)

###############################################################################
# Cost function and accuracy
# ---------------------------
#
# Following the paper we minimise the squared error between the expectation value
# :math:`\langle Z_0 \rangle` and the target label :math:`y \in \{\pm 1\}` and we report the
# classification accuracy, i.e. how often the sign of the prediction matches the label.
#
# For a batch of :math:`B` samples, the mean squared error is
# :math:`L = B^{-1}\sum_b (y_b-f_\theta(x_b))^2`. Targets are a one-dimensional tensor of shape
# ``(B,)`` so that subtracting the predictions does not accidentally broadcast into a matrix.
# MSE encourages expectation values near the target signs; accuracy alone does not measure this
# margin. A lower MSE need not improve the number of correctly classified examples every epoch.
#


def accuracy(labels, predictions):
    """Fraction of exact matches between targets and predicted class labels."""
    return (labels == predictions).to(torch.float64).mean().item()


def cost(weights, parameters, X, Y):
    """Mean squared error between the network output and the labels."""
    predictions = torch.stack([full_circuit(x, weights, parameters) for x in X])
    return torch.mean((Y - predictions) ** 2)


###############################################################################
# Training with Nesterov momentum
# --------------------------------
#
# We train for 150 epochs with PyTorch's **Nesterov momentum** SGD optimizer. Its update convention
# and random initialization differ from the original PennyLane optimizer, so trajectories need not
# match the paper. Cost and accuracy are recorded on the same, updated parameters each epoch.
# The test set is used only for reporting, not for selecting hyperparameters or stopping training.
# Each update clears accumulated gradients, differentiates the batch loss, and updates both sets
# of angles. Evaluation uses ``torch.no_grad()`` because no derivatives are needed for reporting.
#

LEARNING_RATE = 0.5
EPOCHS = 150

opt = torch.optim.SGD([weights, parameters], lr=LEARNING_RATE, momentum=0.9, nesterov=True)

cost_history = []
train_acc_history = []
test_acc_history = []
epochs = []

for it in range(EPOCHS):
    opt.zero_grad()
    loss = cost(weights, parameters, X, Y)
    loss.backward()
    opt.step()

    with torch.no_grad():
        predictions = torch.stack([full_circuit(x, weights, parameters) for x in X])
        curr_cost = torch.mean((Y - predictions) ** 2).item()
        train_acc = accuracy(Y, predict_labels(predictions))
        predictions_test = torch.stack([full_circuit(x, weights, parameters) for x in X_test])
        test_acc = accuracy(Y_test, predict_labels(predictions_test))

    cost_history.append(curr_cost)
    train_acc_history.append(train_acc)
    test_acc_history.append(test_acc)
    epochs.append(it)

    if (it + 1) % 10 == 0 or it in (0, EPOCHS - 1):
        print(
            f"Iter: {it + 1:4d} | Cost: {curr_cost:0.7f} | "
            f"Train acc: {train_acc:0.3f} | Test acc: {test_acc:0.3f}"
        )

###############################################################################
# Results
# --------
#
# The learning curves below show the cost, the training accuracy and the test accuracy over the 150
# epochs. Compare training and test accuracy to assess generalisation. High training accuracy alone
# does not establish an improvement over a fixed readout or a classical baseline.
# A widening gap suggests overfitting; persistently low accuracy on both sets can indicate
# optimization difficulties or limited expressivity. Because the test set contains only 30
# examples, changing one prediction changes its accuracy by about 3.3 percentage points.
#

fig, axs = plt.subplots(3, 1, figsize=(10, 9))

axs[0].plot(epochs, cost_history)
axs[0].set_ylabel("Cost")
axs[0].set_title("Cost")

axs[1].plot(epochs, train_acc_history, color="C1")
axs[1].set_ylabel("Train accuracy")
axs[1].set_title("Training accuracy")

axs[2].plot(epochs, test_acc_history, color="C2")
axs[2].set_xlabel("Epoch")
axs[2].set_ylabel("Test accuracy")
axs[2].set_title("Test accuracy")

fig.tight_layout()

final_train_acc = train_acc_history[-1]
final_test_acc = test_acc_history[-1]
print(f"Final train accuracy: {final_train_acc:0.3f}")
print(f"Final test accuracy:  {final_test_acc:0.3f}")

###############################################################################
# How to adapt the experiment
# ---------------------------
#
# The following recipes describe changes to the sections above, rather than launching additional
# training runs. Restart from initialization after each independent experiment so that trained
# parameters and optimizer momentum are not silently carried over.
#
# Run a small smoke test
# ~~~~~~~~~~~~~~~~~~~~~~
#
# Before a full run, change ``N_TRAIN, N_TEST = 50, 30`` to ``N_TRAIN, N_TEST = 2, 2`` and
# ``EPOCHS = 150`` to ``EPOCHS = 2``. Keep the register sizes and classifier depth unchanged to
# exercise the actual architecture. This checks execution, not classification performance.
# Restore the original budgets for the reported experiment. The full loop makes 130 sample-level
# forward evaluations per epoch, in addition to the work required for derivatives.
#
# To check gradients, insert the following immediately after ``loss.backward()``:
#
# .. code-block:: python
#
#    for parameter in (weights, parameters):
#        assert parameter.grad is not None
#        assert torch.isfinite(parameter.grad).all()
#        print(parameter.grad.norm().item())
#
# A finite gradient is a numerical sanity check, not evidence that optimization will succeed.
#
# Use your own binary dataset
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Replace the dataset-loading block with tensors from your own train/test split. Inputs must have
# shape ``(number_of_samples, 16)``, finite real values, and nonzero row norms. Targets must be
# flat arrays containing -1 and +1. If your targets are 0 and 1, map them explicitly:
#
# .. code-block:: python
#
#    X = torch.as_tensor(train_features, dtype=torch.float64)
#    X_test = torch.as_tensor(test_features, dtype=torch.float64)
#    Y = 2 * torch.as_tensor(train_labels_01, dtype=torch.float64).reshape(-1) - 1
#    Y_test = 2 * torch.as_tensor(test_labels_01, dtype=torch.float64).reshape(-1) - 1
#
# Here ``train_features``, ``test_features``, ``train_labels_01``, and ``test_labels_01`` are your
# arrays, not attributes supplied by PennyLane. Keep the input checks and choose subset sizes no
# larger than the available splits. Split the data before fitting preprocessing such as PCA or
# standardization: fit on training data only, then apply the fitted transformation to other splits.
# Do not normalize a zero vector; decide whether to remove it or use a justified representation.
#
# For fewer than 16 features, explicitly zero-pad each row to 16 if that encoding suits the task.
# For more than 16, fit a reduction to 16 features on the training split, or redesign all register
# sizes, swaps, QLS wiring, and classifier wires. Changing only ``NUM_QUBITS`` is insufficient.
#
# Change circuit depth
# ~~~~~~~~~~~~~~~~~~~~
#
# Set ``N_LAYERS_STRONG`` before constructing ``strong_shape`` and ``parameters``. Set
# ``N_LAYERS_QK`` before constructing ``weights``; the circuit infers this depth from
# ``weights.shape[1]``. For example, two Q/K/V layers and five classifier layers require
# :math:`3\times 2\times 4 + 5\times 8\times 3 = 144` angles. Recreate the optimizer after
# replacing parameter tensors, otherwise it will retain references to the old tensors.
#
# More layers add capacity and simulation cost, but do not guarantee better generalisation.
# Select depths and learning rates on a separate validation split, never on the final test set.
#
# Train only the classifier
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
# To study a fixed feature extractor, set the following before constructing the optimizer:
#
# .. code-block:: python
#
#    weights.requires_grad_(False)
#    opt = torch.optim.SGD([parameters], lr=LEARNING_RATE, momentum=0.9, nesterov=True)
#
# Start from the same seeded initialization when comparing joint and classifier-only training.
# This is an ablation of trainability, not a classical-attention baseline.
#
# Use a probability-based loss
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The probability of the +1 eigenvalue of :math:`Z` is :math:`p_+=(1+f_\theta(x))/2`.
# To associate that measurement outcome with the positive class, replace the MSE return in
# ``cost`` with binary cross-entropy:
#
# .. code-block:: python
#
#    probabilities = ((predictions + 1) / 2).clamp(1e-7, 1 - 1e-7)
#    targets = (Y + 1) / 2
#    return torch.nn.functional.binary_cross_entropy(probabilities, targets)
#
# Clamping prevents logarithms of zero, but changes the objective near the endpoints. Update the
# reported cost calculation to the same loss as well; otherwise the cost plot still displays MSE.
# Passing the raw expectation directly as a probability, or using +/-1 targets for binary
# cross-entropy, is incorrect. The zero-threshold classification rule remains unchanged.
#
# Compare results responsibly
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Use the same data splits and preprocessing for each method, tune only on validation data,
# and repeat training with multiple seeds. Compare against a fixed circuit readout and a simple
# classical classifier before claiming an improvement. Report held-out performance, variability,
# parameter counts, and measured runtime; this tutorial alone does not establish quantum advantage.
#
#
# Conclusion
# ----------
#
# We have implemented a circuit inspired by quantum self-attention followed by a strongly
# entangling classifier, and trained it using classical optimization. The printed metrics describe
# this small MNIST experiment only. Reproducing the comparisons in [#Tran]_ requires matched data,
# baselines, optimizer settings and repeated runs. In particular, this implementation does not feed
# the QLS result register back through the slicing module.
#
# To explore further, you can:
#
# - increase the number of strongly entangling layers and study the interplay between entanglement
#   depth and generalisation using a separate validation set;
# - map the expectation value to class probabilities before trying a cross-entropy loss;
# - apply the same architecture to other PennyLane-native datasets.
#
#
# References
# ----------
#
# .. [#Tran]
#
#     Quang-Huyen Tran, Yu-Han Lin, Hanh T. M. Tran, Duy-Tuan Dao and Van-Linh Nguyen. Efficient
#     Circuit Classifier Design for Enhancing Quantum Self-Attention in Vision Transformers. *IEEE
#     GLOBECOM Workshops (GC Wkshps)* (2025). DOI: https://doi.org/10.1109/gcwkshps68340.2025.11591004.
#
# .. [#QSAN]
#
#     Jinye Shi, Run-Xia Zhao, Wei Wang, Shangbin Zhang and Xi Li. QSAN: A Near-Term Achievable
#     Quantum Self-Attention Network. *IEEE Trans. Neural Netw. Learn. Syst.* (2025). DOI:
#     https://doi.org/10.1109/tnnls.2024.3504828.
#
# .. [#QKSAN]
#
#     Run-Xia Zhao, Jinye Shi and Xi Li. QKSAN: A Quantum Kernel Self-Attention Network. *IEEE
#     Trans. Pattern Anal. Mach. Intell.* 46 (2024). DOI: https://doi.org/10.1109/tpami.2024.3434974.
#
# .. [#QSANN]
#
#     Guangxi Li, Xuanqiang Zhao and Xin Wang. Quantum Self-Attention Neural Networks for Text
#     Classification. *Sci. China Inf. Sci.* 67 (2024). DOI: https://doi.org/10.1007/s11432-023-3879-7.
#
# .. [#QSANM]
#
#     Jin Zheng, Qing Gao and Zibo Miao. Design of a Quantum Self-Attention Neural Network on Quantum
#     Circuits. *IEEE SMC* (2023). DOI: https://doi.org/10.1109/SMC53992.2023.10393989.
#
# .. [#Schuld]
#
#     Maria Schuld, Alex Bocharov, Krysta Svore and Nathan Wiebe. Circuit-Centric Quantum
#     Classifiers. *Phys. Rev. A* 101, 032308 (2020). DOI:
#     https://doi.org/10.1103/physreva.101.032308.
#
# .. [#Bowles]
#
#     Joseph Bowles, Shahnawaz Ahmed and Maria Schuld. PennyLane Datasets for "Better than
#     classical? The subtle art of benchmarking quantum machine learning models" (2024).
#     https://pennylane.ai/datasets/downscaled-mnist.
#
