r"""Before you train: Pre-screening quantum kernels with geometric difference
=============================================================================

Can we predict---*before* investing a ton of research hours---whether a quantum kernel has the
potential to beat a classical one across all kernel methods?

From a practitioner’s perspective, such a **pre-screening test** is invaluable: it lets us rule out
quantum kernels that don’t offer any potential quantum advantage right from the start.

Huang *et al.* [#Huang2021]_ introduced exactly this test. Their proposed
**geometric difference** :math:`g` metric is a single scalar that quantifies how differently the
geometries defined by two kernels represent your data. The formula for :math:`g` is:

.. math:: g = \sqrt{\|\sqrt{K_q} K_c^{-1} \sqrt{K_q}\|_\infty},

where :math:`K_q` and :math:`K_c` are quantum and classical Gram matrices, respectively.

The following demonstration is designed to perform a pre-screening test on several 
kernel methods—both classical and quantum—on a simple classification task.

What g tells us and why it is important
---------------------------------------

When :math:`g \approx 1`, the quantum kernel’s geometry is essentially the same as a good classical
kernel’s. The quantum kernel offers no geometric advantage, making it unlikely to outperform the 
classical kernel **in any kernel-based learning algorithm** (e.g., SVM, Gaussian Processes). Huang 
et al. proved this concept in a rigorous mathematical way in their paper. Conversely, if :math:`g >> 1`, 
the quantum geometry is genuinely different. A kernel method using the quantum kernel *might* offer an advantage.

The approach presented in this demo focuses on ruling out underperforming quantum kernels before investing in training.
From a complexity theory point of view, computing :math:`g` scales as :math:`O(n^3)` due to the
matrix inversion, and the most expensive training algorithms such as Gaussian Processes also scale
as :math:`O(n^3)` so we might think we are not saving any computational time. However, from a
practical perspective, the real savings come from avoiding wasted researcher effort.

When a quantum kernel performs poorly, researchers often spend days exploring different algorithm
hyperparameters, cross-validation strategies, and implementation debugging. If :math:`g \approx 1`,
you immediately know the quantum kernel’s geometry offers no advantage---it’s not your implementation,
not your algorithm choice, and not a hyperparameter issue. The kernel is fundamentally limited
compared to classical kernels on this specific dataset.


Background context: kernels
---------------------------

A **kernel** is a function :math:`k(x, x')` that measures similarity between data points without
explicitly computing their feature representations in high-dimensional spaces, thus lowering the
computational cost.

An example of a classical kernel is the Radial Basis Function (RBF) kernel given by 
:math:`k(x, x') = \exp(-\gamma \|x - x'\|^2)`. It implicitly computes the inner product 
:math:`\langle\phi(x), \phi(x')\rangle`. The feature map :math:`\phi(x)` projects 
to infinite dimensions, but it is never calculated directly.

Quantum kernels are similar but leverage the Hilbert space of a quantum computer. A quantum kernel is defined by 
:math:`k(x, x') = |\langle\psi(x)|\psi(x')\rangle|^2`, where :math:`|\psi(x)\rangle` is the quantum 
state encoding the classical data :math:`x`. For :math:`n` qubits, the quantum state lives in a 
:math:`2^n`-dimensional Hilbert space that is implicitly manipulated.

**Key concept**: The **kernel matrix** (Gram matrix) :math:`K` has entries :math:`K_{ij} = k(x_i, x_j)` that store
all pairwise similarities between data points.


Demonstration setup
-------------------

This demonstration uses the synthetic two-moons dataset from scikit-learn to perform a
pre-screening test on four quantum kernel variants against a classical RBF kernel. By
calculating the geometric difference and then training SVMs with each kernel, we illustrate
how this metric can filter out unpromising quantum kernels before investing in extensive
hyperparameter tuning and training. This procedure is summarized as:  

1. **Dataset**: Synthetic two-moons data generated with ``scikit-learn``.
2. **Five kernels to compare**:

   - **Classical baseline**: Gaussian-RBF kernel
   - **Quantum kernels**:

     - Separable-rotation embedding (E1)
     - IQP-style embedding (E2)
     - Projected kernel from E1 (maximizing :math:`g` as proposed by Huang et al.)
     - Projected kernel from E2

3. **Our approach**: Calculate :math:`g` values between the classical kernel and each quantum
   kernel.
"""

# We first start by generating and visualizing the artificial data
import numpy as np
import matplotlib.pyplot as plt
import scipy

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(422)

X_raw, y = make_moons(n_samples=300, noise=0.10, random_state=0)

# train/test split BEFORE any scaling (avoid data leakage) ------------
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.30, random_state=0, stratify=y
)

# standardize the data
scaler = StandardScaler().fit(X_train_raw)  # statistics from train only
X_train = scaler.transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

print(f"Train size: {X_train.shape[0]}    Test size: {X_test.shape[0]}")

# visualize it using a scatter plot
plt.figure(figsize=(4, 4))
plt.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], s=15, alpha=0.8, label="class 0")
plt.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], s=15, alpha=0.8, label="class 1")
plt.axis("equal")
plt.title("Two‑moons— training split (standardised)")
plt.legend(frameon=False)
plt.show()

######################################################################
#
# Quantum kernels: fidelity-based and projected variants
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We consider **five different kernels** derived from three sources: a classical RBF kernel and two
# quantum embedding circuits, **E1** and **E2**.
# Each kernel defines a different geometry for measuring similarity between data points.
#
# The classical kernel is the **Radial basis function kernel (RBF).**
# This classical baseline is defined as:
#
# .. math:: k_{\text{RBF}}(x, x') = \exp(-\gamma \|x - x'\|^2).
#
# This maps data into an infinite-dimensional space where closer inputs remain close, and distant
# ones become nearly orthogonal.
# It captures a **geometric**, distance-based notion of similarity in input space.
#
# The two quantum embedding circuits are:
#
# - **E1 – Separable RX rotations.**
#   Each input feature :math:`x_j` is encoded into a single qubit using an :math:`RX(x_j)` gate.
#   The circuit is fully separable (no entanglement), producing the quantum state
#   :math:`\lvert \psi_{\text{E1}}(x) \rangle`.
#
# - **E2 – IQP embedding.**
#   PennyLane’s ``qp.IQPEmbedding`` applies Hadamards, parameterized :math:`RZ(x_j)` rotations, and
#   entangling ZZ gates.
#   This creates an entangled quantum state :math:`\lvert \psi_{\text{E2}}(x) \rangle`, inspired by
#   Instantaneous Quantum Polynomial (IQP) circuits.
#
# The four quantum kernels derived from the embedding circuits are defined in the following way. 
#
# - **QK – Standard quantum kernels.**
#   For both E1 and E2, the kernel is defined by the **fidelity** between quantum states:
#
#   .. math:: k_{\text{QK-E1}}(x, x') = |\langle \psi_{\text{E1}}(x) \mid \psi_{\text{E1}}(x') \rangle|^2,
#
#   .. math:: k_{\text{QK-E2}}(x, x') = |\langle \psi_{\text{E2}}(x) \mid \psi_{\text{E2}}(x') \rangle|^2,
#
#   where :math:`\psi_{\text{E1}}(x)` and :math:`\psi_{\text{E2}}(x)` are the quantum states generated
#   by E1 and E2 respectively. These kernels reflect how aligned two quantum feature states are in
#   Hilbert space.
#
# - **PQK – Projected quantum kernels (PQK-E1 / PQK-E2).**
#   For a projected quantum kernel, instead of computing fidelity, the output quantum state
#   :math:`|\psi(x)\rangle`
#   is **measured** to extract the expectation values of Pauli operators:
#
#   .. math:: v(x) = \left[ \langle X_0 \rangle, \langle Y_0 \rangle, \langle Z_0 \rangle, \dots, \langle Z_{n-1} \rangle \right]
#
#   A classical **RBF kernel** is then applied to these real-valued vectors:
#
#   .. math:: k_{\text{PQK}}(x, x') = \exp\left( -\gamma \| v(x) - v(x') \|^2 \right).
#
#   We obtain two different projected quantum kernels from E1 and E1:
#
#   .. math:: k_{\text{PQK-E1}}(x, x') = \exp\left( -\gamma \|v_{\text{E1}}(x) - v_{\text{E1}}(x')\|^2 \right),
#
#   .. math:: k_{\text{PQK-E2}}(x, x') = \exp\left( -\gamma \|v_{\text{E2}}(x) - v_{\text{E2}}(x')\|^2 \right),
#
#   where :math:`v_{\text{E1 }}(x)` and :math:`v_{\text{E2}}(x)` are the Pauli expectation vectors from
#   E1 and E2, respectively.
#
# Let's define the embedding circuits E1 and E2, and visualize them using an auxiliary drawing function. 

import numpy as np
import pennylane as qp
import matplotlib.pyplot as plt

n_features = X_train.shape[1]
n_qubits = n_features


# -- E1: separable RX rotations ---------------------------------------------
def embedding_E1(features):
    for j, xj in enumerate(features):
        qp.RX(np.pi * xj, wires=j)


# -- E2: IQP embedding via PennyLane template --------------------------------
def embedding_E2(features):
    qp.IQPEmbedding(features, wires=range(n_features))


from utils import draw_circuits_side_by_side

draw_circuits_side_by_side(
    [embedding_E1, embedding_E2],
    ["E1 Embedding Circuit", "E2 Embedding Circuit"],n_qubits
)


######################################################################
#
# Gram matrix computation
# ~~~~~~~~~~~~~~~~~~~~~~~
#
# Using the kernels defined above, we now build the **Gram (kernel) matrices** required to compute the
# practitioner’s metric :math:`g`. For a dataset of :math:`N` samples and a kernel function :math:`k(\cdot, \cdot)`, the Gram matrix
# :math:`K \in \mathbb{R}^{N \times N}` is defined entrywise as:
#
# .. math:: K_{ij} = k(x_i, x_j).
#
# Each entry :math:`K_{ij}` measures how similar two data points are, and the full matrix :math:`K`
# provides a **global view** of the data in the kernel’s feature space. 
# We compute five such matrices, one for each kernel defined above.
#
# The Gram matrices will be used in downstream evaluations to compare kernel geometries and analyze
# expressivity and generalization metrics like :math:`g`.
#
# The following code builds all five Gram (kernel) matrices: Classical, QK-E1, QK-E2, PQK-E1, and PQK-E2.

from sklearn.metrics.pairwise import rbf_kernel

# ---------------------------------------------------------------------------#
# Classical RBF Gram matrix                                                  #
# ---------------------------------------------------------------------------#
def classical_rbf_kernel(X, gamma=1.0):
    return rbf_kernel(X, gamma=gamma)


K_classical = classical_rbf_kernel(X_train)
print(f"K_RBF shape: {K_classical.shape}")

# ---------------------------------------------------------------------------#
# Quantum fidelity-based Gram matrices                                       #
# ---------------------------------------------------------------------------#
dev = qp.device("default.qubit", wires=n_qubits, shots=None)


def overlap_prob(x, y, embed):
    """Probability of measuring |0…0⟩ after U(x) U†(y)."""

    @qp.qnode(dev)
    def circuit():
        embed(x)
        qp.adjoint(embed)(y)
        return qp.probs(wires=range(n_qubits))

    return circuit()[0]


def quantum_kernel_matrix(X, embed):
    return qp.kernels.kernel_matrix(X, X, lambda v1, v2: overlap_prob(v1, v2, embed))


print("Computing QK-E1 (fidelity)...")
K_quantum_E1 = quantum_kernel_matrix(X_train, embed=embedding_E1)

print("Computing QK-E2 (fidelity)...")
K_quantum_E2 = quantum_kernel_matrix(X_train, embed=embedding_E2)

print(f"K_QK_E1 shape: {K_quantum_E1.shape}")
print(f"K_QK_E2 shape: {K_quantum_E2.shape}")

######################################################################
#
# Next, we compute the projected quantum kernels (Pauli vectors + classical RBF).
#

# ---------------------------------------------------------------------------#
# Projected quantum kernels                                                  #
# ---------------------------------------------------------------------------#
def get_pauli_vectors(embedding_func, X):
    """Returns Pauli expectation vectors for each input using the given embedding."""
    observables = []
    for i in range(n_qubits):
        observables.extend([qp.PauliX(i), qp.PauliY(i), qp.PauliZ(i)])

    @qp.qnode(dev)
    def pauli_qnode(x):
        embedding_func(x)
        return [qp.expval(obs) for obs in observables]

    vectors = [pauli_qnode(x) for x in X]
    return np.array(vectors)


def calculate_gamma(vectors):
    """Use heuristic gamma = 1 / (d * var) for RBF kernel on Pauli space."""
    d = vectors.shape[1]
    var = np.var(vectors)
    return 1.0 / (d * var) if var > 1e-8 else 1.0


def pqk_kernel_matrix(X, embedding_func):
    """Computes PQK kernel matrix from Pauli vectors + RBF kernel."""
    pauli_vecs = get_pauli_vectors(embedding_func, X)
    gamma = calculate_gamma(pauli_vecs)
    return rbf_kernel(pauli_vecs, gamma=gamma)


print("Computing PQK-E1 (Pauli + RBF)...")
K_pqk_E1 = pqk_kernel_matrix(X_train, embedding_E1)

print("Computing PQK-E2 (Pauli + RBF)...")
K_pqk_E2 = pqk_kernel_matrix(X_train, embedding_E2)

print(f"K_PQK_E1 shape: {K_pqk_E1.shape}")
print(f"K_PQK_E2 shape: {K_pqk_E2.shape}")

######################################################################
#
# Let's now visualize the Gram Matrices. Each matrix shows how similar data points are to each other:
# brighter colors indicate higher similarity and different patterns indicate different geometries.

import matplotlib.pyplot as plt

# Visualize first 20x20 subset of each Gram matrix for clarity
subset_size = 20
matrices = [K_classical, K_quantum_E1, K_quantum_E2, K_pqk_E1, K_pqk_E2]
titles = ["Classical RBF", "QK-E1", "QK-E2", "PQK-E1", "PQK-E2"]

rows, cols = 2, 3
fig, axes = plt.subplots(rows, cols, figsize=(14, 8))

# Flatten axes array to loop over
axes = axes.flatten()

for i, (K, title) in enumerate(zip(matrices, titles)):
    im = axes[i].imshow(K[:subset_size, :subset_size], cmap="viridis", aspect="equal")
    axes[i].set_title(title)
    axes[i].set_xlabel("Sample index")
    if i % cols == 0:  # only first column gets a y-label
        axes[i].set_ylabel("Sample index")
    plt.colorbar(im, ax=axes[i], fraction=0.046)

# Hide any unused subplots (since we have 5 matrices, but a 2x3 grid = 6 spots)
for j in range(len(matrices), len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.show()

######################################################################
# 
# We then compute the practitioner’s metric :math:`g` for each quantum kernel, according to the formula used in the paper by Huang et al.

from scipy.linalg import sqrtm


def compute_g(K_classical, K_quantum, eps=1e-7):
    """
    Compute geometric difference g between classical and quantum kernels.
    Formula: g = sqrt( || sqrt(K_quantum) @ inv(K_classical) @ sqrt(K_quantum) || )
    """
    N = K_classical.shape[0]

    # Regularize and invert K_classical
    Kc_reg = K_classical + eps * np.eye(N)
    Kc_inv = np.linalg.inv(Kc_reg)

    # Compute the square root of the quantum kernel
    sqrt_Kq = sqrtm(K_quantum)
    # Construct M = sqrt(Kq) @ Kc⁻¹ @ sqrt(Kq)
    M = sqrt_Kq @ Kc_inv @ sqrt_Kq

    # g = sqrt(max eigenvalue of M)
    max_eigval = np.max(np.linalg.eigvalsh(M))
    return np.sqrt(np.maximum(max_eigval, 0.0))


# Compute g for all four quantum kernels
g_QK_E1 = compute_g(K_classical, K_quantum_E1)
g_QK_E2 = compute_g(K_classical, K_quantum_E2)
g_PQK_E1 = compute_g(K_classical, K_pqk_E1)
g_PQK_E2 = compute_g(K_classical, K_pqk_E2)

# Display results
print("\n--- Geometric Difference (g) ---")
print(f"g (RBF vs QK‑E1):    {g_QK_E1:.4f}")
print(f"g (RBF vs QK‑E2):    {g_QK_E2:.4f}")
print(f"g (RBF vs PQK‑E1):   {g_PQK_E1:.4f}")
print(f"g (RBF vs PQK‑E2):   {g_PQK_E2:.4f}")

######################################################################
# What does a high g mean?
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We can see that in terms of :math:`g`: PQK-E1 > PQK-E2 > QK-E1 > QK-E2.
# A common misconception is that a higher geometric difference :math:`g` automatically means better
# classification performance, which might lead us to believe, for example, that in terms of final
# accuracy, the ranking will also be PQK-E1 > PQK-E2 > QK-E1 > QK-E2.
#
# This intuition is understandable—after all, a larger :math:`g` suggests that the quantum kernel
# perceives the data very differently from a classical one.
# But as we’ll see, **a higher** :math:`g` **doesn’t always translate into better accuracy, it just
# means there’s higher potential for an improvement over the classical model**.
# In fact, a higher :math:`g` can sometimes correspond to worse performance on the original task.
#
# Let’s see this in action.
#

# We train SVMs using each kernel and compare test accuracy
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import rbf_kernel


def train_evaluate_svm(K_train, K_test, y_train, y_test, name):
    print(f"Training SVM with {name} kernel...")
    clf = SVC(kernel="precomputed")
    clf.fit(K_train, y_train)
    acc = accuracy_score(y_test, clf.predict(K_test))
    print(f"  Test accuracy: {acc:.4f}")
    return acc


results = {}

# Classical RBF
K_rbf_test = rbf_kernel(X_test, X_train)
results["Classical RBF"] = train_evaluate_svm(
    K_classical, K_rbf_test, y_train, y_test, "Classical RBF"
)

# Quantum Kernel E1
K_qk_e1_test = qp.kernels.kernel_matrix(
    X_test, X_train, lambda x, y: overlap_prob(x, y, embedding_E1)
)
results["QK-E1"] = train_evaluate_svm(K_quantum_E1, K_qk_e1_test, y_train, y_test, "Quantum E1")

# Quantum Kernel E2
K_qk_e2_test = qp.kernels.kernel_matrix(
    X_test, X_train, lambda x, y: overlap_prob(x, y, embedding_E2)
)
results["QK-E2"] = train_evaluate_svm(K_quantum_E2, K_qk_e2_test, y_train, y_test, "Quantum E2")

# PQK E1
pauli_test_E1 = get_pauli_vectors(embedding_E1, X_test)
gamma_E1 = calculate_gamma(np.vstack((get_pauli_vectors(embedding_E1, X_train), pauli_test_E1)))
K_pqk_e1_test = rbf_kernel(pauli_test_E1, get_pauli_vectors(embedding_E1, X_train), gamma=gamma_E1)
results["PQK-E1"] = train_evaluate_svm(K_pqk_E1, K_pqk_e1_test, y_train, y_test, "PQK E1")

# PQK E2
pauli_test_E2 = get_pauli_vectors(embedding_E2, X_test)
gamma_E2 = calculate_gamma(np.vstack((get_pauli_vectors(embedding_E2, X_train), pauli_test_E2)))
K_pqk_e2_test = rbf_kernel(pauli_test_E2, get_pauli_vectors(embedding_E2, X_train), gamma=gamma_E2)
results["PQK-E2"] = train_evaluate_svm(K_pqk_E2, K_pqk_e2_test, y_train, y_test, "PQK E2")

# Summary
print("\n--- Accuracy Comparison ---")
for model, acc in results.items():
    print(f"{model:>15}: {acc:.4f}")

# Accuracy Comparison

import matplotlib.pyplot as plt

# Extract model names and accuracies
model_names = list(results.keys())
accuracies = [results[name] for name in model_names]

# Create bar chart
plt.figure(figsize=(10, 4))
bars = plt.bar(model_names, accuracies)
plt.ylim(0, 1.10)
plt.ylabel("Test Accuracy")
plt.title("SVM Accuracy by Kernel")
plt.xticks(rotation=15)
plt.grid(axis="y", linestyle="--", alpha=0.4)

# Annotate values
for bar, acc in zip(bars, accuracies):
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2, yval + 0.015, f"{acc:.2f}", ha="center", va="bottom"
    )

plt.show()

######################################################################
#
# Our test results reveal an important subtlety:
# **A higher geometric difference** :math:`g` **does not guarantee better classification accuracy.**
# For instance, **PQK‑E2** achieved perfect test accuracy (:math:`100\%`), despite having a lower
# :math:`g` than PQK‑E1.
#
# This highlights a key message from the paper: The role of :math:`g` is *not* to predict which kernel 
# will perform best on a given task, but rather to obtain a collection of kernels that have the *potential* 
# to offer an advantage.
#
# Here, PQK-E1 and PQK-E2 both had the potential for an advantage over classical, but PQK-E2 is the
# only one that actually achieved the advantage. As a simple practical rule, if :math:`g` is low, then
# we can immediately discard a quantum kernel, whereas if :math:`g` is high, we keep the kernel as a
# potential solution because it offers a potential for an improvement on our classification problem.
# This way, we have an important diagnostic tool to filter out bad quantum kernels for our data.
#
# Conclusion
# ----------
#
# In this notebook, we explored a fundamental question in quantum machine learning: Can we anticipate, 
# before training, whether a quantum kernel might outperform a classical one?
#
# To address this, we used the geometric difference :math:`g`, a pre-training metric introduced by
# Huang et al. that quantifies how *differently* a quantum kernel organizes the data compared to a
# classical kernel. The main takeaways from this demonstration are: 
#
# - :math:`g` **is a diagnostic, not a performance predictor.**
#   A large :math:`g` indicates that the quantum kernel induces a very different geometry from the
#   classical one—a *necessary*, but not *sufficient*, condition for quantum advantage.
#
# - **Higher** :math:`g` **does not imply higher accuracy.**
#   In our results, **PQK‑E2** had a high :math:`g` and achieved perfect accuracy, but **PQK‑E1**,
#   with a higher :math:`g`, obtained a lower accuracy on the original task. This confirms that
#   :math:`g` measures *potential*, not realized performance.
#
# - :math:`g`\ ’s **value is in ruling out unpromising kernels.**
#   Kernels with very small :math:`g` are unlikely to offer any meaningful advantage over classical
#   methods—the quantum kernel introduces no genuinely new distinctions beyond what a
#   classical RBF can produce. By contrast, a high :math:`g` only tells us that *some advantage may
#   be possible*, not that it will be realized.
#
# What if we took labels into account? The authors in [#Huang2021]_ proposed a method to artificially construct new labels 
# that align with a quantum kernel’s geometry. This is a toy construction, but pretty fun to play around with.
# Feel free to do so!
#
# References
# ----------
#
# .. [#Huang2021]
#
#     Huang, Hsin-Yuan, Michael Broughton, Masoud Mohseni, Ryan Babbush, Sergio Boixo, Hartmut Neven, and Jarrod R. McClean. 
#     "Power of data in quantum machine learning." 
#     `Nature Communications 12, 2631 (2021) <https://www.nature.com/articles/s41467-021-22539-9>`__.
#
