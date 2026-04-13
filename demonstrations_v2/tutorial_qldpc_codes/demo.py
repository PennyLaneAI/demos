r"""Quantum low-density parity-check (qLDPC) codes for quantum error correction
================================================================================

Quantum computers are envisioned to be incredibly powerful devices. While many of the
machines available today boast hundreds of qubits, their performance remains limited
by noise, which manifests as computational errors and impacts their utility.
Thus, for a fault-tolerant quantum computer to exist and be capable of running
indefinitely with minimal permissible errors, we need quantum error correction (QEC).

For this purpose, QEC codes encode :math:`k` logical qubits into :math:`n` physical qubits.
To allow for fault-tolerant computation with these :math:`k` logical qubits, the QEC codes
must serve as both reliable memory and a substrate for logic. This motivates the need for
the following four properties:

1. A high encoding rate :math:`R = k / n`.
2. Local and low-weight parity checks (= measurements for error detection).
3. As many gates from the universal gate set as possible should be implementable transversally,
   i.e., applied independently to each physical qubit without coupling qubits within the same
   code block.
4. Linear time classical decoding and corresponding error correction.

Unfortunately, these requirements are not all mutually compatible. For example, widely used
topological codes, such as surface codes, use local, nearest-neighbour connections, but usually
have poor encoding rates. It remains unclear which combination of these trade-offs will offer
the best long-term solution. However, because solving real-world problems requires scaling up to
thousands of logical qubits, moving beyond strict nearest-neighbour constraints has become crucial.

Quantum low-density parity-check (qLDPC) codes are particularly well-suited for this purpose,
as they leverage high connectivity between qubits to drastically reduce qubit overheads, making
them the codes of choice for the hardware platforms that support such qubit connectivity.
As their name implies, they enforce a low-degree constraint, meaning each physical qubit
participates in only a few parity checks, and each check measures only a few qubits.
While this constraint is less essential for platforms with reconfigurable all-to-all connectivity,
their highly efficient decoding remains a major asset. In this demo, we will cover the basics
of qLDPC codes, including their construction and decoding. For readers unfamiliar with the
fundamentals of QEC, we recommend reading our tutorials on the
:doc:`Surface Code <demos/tutorial_game_of_surface_codes>`,
:doc:`Stabilizer Codes <demos/tutorial_stabilizer_codes>`, and
:doc:`Lattice Surgery <demos/tutorial_lattice_surgery>` that cover them in detail.

.. figure::
    ../_static/demo_thumbnails/opengraph_demo_thumbnails/pennylane-demo-quantum-lowdensity-paritycheck-open-graph.png
    :align: center
    :width: 50%
    :target: javascript:void(0)


Classical LDPC Codes
--------------------

To understand quantum LDPC codes, we begin by looking at their classical counterparts, which
have revolutionized modern telecommunications (powering Wi-Fi and 5G networks) by approaching
the absolute theoretical limits of data transmission, known as the `Shannon limit
<https://en.wikipedia.org/wiki/Shannon_limit>`_. Classical LDPC codes achieve this with highly
efficient, linear-time decoding that exploits the sparse structure of their parity checks.

A classical LDPC code :math:`C[n,k,d]` protects :math:`k` logical bits by encoding them into
:math:`n` physical bits. Here :math:`d` is the minimum distance of the code, which dictates the
number of errors the code can correct. The encoding rules are defined by an :math:`m\times n`
parity-check matrix (:math:`H`), where :math:`k = n - m` [#qldpc1]_, where :math:`m` is the
number of parity checks. The "low-density" part of their name comes from this matrix being
overwhelmingly sparse, i.e., filled mostly with zeros, and more specifically, the column/row
weights (number of 1s in each column/row) are strictly bounded constants independent of :math:`n`.
For a practical code, both must be bounded, because they carry distinct physical meaning: (i) a
low column weight limits how many parity checks each bit participates in, while (ii) a low row
weight limits how many bits each parity check involves, keeping decoding local and efficient.

Mathematically, these codes are visualized as `Tanner graphs
<https://en.wikipedia.org/wiki/Tanner_graph>`_, which are bipartite graphs with edges
representing connections between the variable nodes (:math:`n` physical bits)
and the check nodes (:math:`m` parity constraints). For example, the following
is a Tanner graph for a simple :math:`[5, 2, 3]` LDPC code:
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Define the parity-check matrix H
H = np.array([[1, 0, 1, 0, 1], [0, 1, 1, 0, 0], [1, 1, 0, 1, 0]])

# Construct the Tanner Graph with variable nodes and check nodes.
G = nx.Graph()
num_checks, num_vars = H.shape
var_nodes = [f"v{i}" for i in range(num_vars)]
check_nodes = [f"c{j}" for j in range(num_checks)]
G.add_nodes_from(var_nodes, bipartite=0)
G.add_nodes_from(check_nodes, bipartite=1)
for (i, j) in zip(*np.nonzero(H)):
    G.add_edge(f"c{i}", f"v{j}")

# Plot the Bipartite Graph
plt.figure(figsize=(6, 3))
pos = nx.bipartite_layout(G, var_nodes)
colors = ["#70CEFF"] * num_vars + ["#C756B2"] * num_checks
nx.draw(G, pos, with_labels=True, node_color=colors, node_size=500)
text_options = {"verticalalignment": "center", "fontsize": 12}
plt.text(-0.75, 0.0, "Variable Nodes", rotation=90, color=colors[0], **text_options)
plt.text(1.1, 0.0, "Check Nodes", rotation=270, color=colors[-1], **text_options)
plt.show()

######################################################################
# When noise corrupts our bits, it creates an error vector :math:`\vec{e}`, where a ``1``
# at position :math:`i` means bit :math:`i` was flipped. The system identifies this
# error by computing the syndrome :math:`s = H\vec{e} \pmod{2}`, which flags the
# parity checks that are violated. We can see this with an error on bit ``0``:
#

error_vec = np.array([1, 0, 0, 0, 0]) # bit flip on bit 0
syndrome = (H @ error_vec) % 2

print(f"Syndrome: {syndrome}")

######################################################################
# Here, checks ``0`` and ``2`` fire because bit ``0`` participates in exactly those two rows of
# :math:`H`, visible as edges in the Tanner graph above. Crucially, as each check node only connects
# to a handful of data nodes (and vice versa), these codes achieve linear-time decoding complexity.
# This efficiency is possible because errors can be decoded using local message-passing algorithms
# sharing probabilistic information along the edges of the Tanner graph until all parity constraints
# are satisfied. This will come in handy when we learn about decoding, but before that, let's first
# return to constructing qLDPC codes.
#
# Calderbank-Shor-Steane (CSS) construction
# ------------------------------------------
#
# The key goal of qLDPC codes is to replicate this sparsity (and unlock linear-time decoding) in
# the quantum realm, but this is a non-trivial task. Due to quantum no-go theorems, we cannot
# directly read the state to perform standard parity checks. Furthermore, qubits suffer from phase
# flips alongside standard bit flips—an error type with no classical analog. Therefore, to detect
# errors, we must rely on measuring commuting multi-qubit Pauli operators that can identify both
# types of flips.
#
# The most elegant and widely used solution to build a series of such operators is the
# Calderbank-Shor-Steane (CSS) code construction [#CSS]_. CSS codes belong to the broader class of
# stabilizer codes, where the codespace is defined as the ``+1`` eigenspace of a set of commuting
# multi-qubit Pauli operators called *stabilizers*. A CSS code introduces these stabilizers as two
# separate sets of parity checks: one containing only Pauli-Z operators to catch bit flips, and
# another containing only Pauli-X operators to catch phase flips.
#
# For a CSS code on :math:`n` qubits with :math:`m_1` X-checks and :math:`m_2` Z-checks, each
# set is naturally represented as a classical parity-check matrix, :math:`H_X` and :math:`H_Z`
# respectively. Each defines a bipartite Tanner graph where check nodes connect to the qubit
# nodes they act on. Since both graphs share the same :math:`n` qubit nodes, they can be combined
# into a single unified hypergraph. This hypergraph is captured by its incidence matrix, where
# :math:`H^{ij}_P = 1` for :math:`P \in \{X, Z\}`, if the :math:`i^{th}` :math:`P`-type
# check has support on the :math:`j^{th}` qubit, and :math:`0` otherwise. Both incidence
# matrices pack naturally into a single :math:`(m_1 + m_2) \times 2n` block matrix
# :math:`H = [0, H_Z;\, H_X, 0]` in the symplectic form, where each row directly
# corresponds to one stabilizer generator. As an example, consider the following
# CSS code known as the `Steane code <https://errorcorrectionzoo.org/c/steane>`_
# :math:`[[7,1,3]]`, constructed from the two :math:`m=3` Hamming codes:
#

def hamming_code(rank: int) -> np.ndarray:
    """Returns a Hamming code parity check matrix of a given rank."""
    bit_masks = np.arange(1, 2**rank)[:, None] & (1 << np.arange(rank)[::-1])
    return (bit_masks > 0).astype(np.uint8).T

h1, h2 = hamming_code(3), hamming_code(3)
(m1, n1), (m2, n2) = h1.shape, h2.shape

css_code = np.hstack((
        np.vstack([np.zeros((m1, n1), dtype=np.uint8), h1]),
        np.vstack([h2, np.zeros((m2, n2), dtype=np.uint8)])
))

print(f"Shape of the CSS code: {css_code.shape}")

######################################################################
# For these codes, all stabilizers must commute, which is ensured by having each of the
# :math:`X - Z` stabilizer pairs overlap on an even number of qubits. Mathematically, this is
# equivalent to the symplectic orthogonality condition :math:`H_X(H_Z)^T = 0 \pmod{2}`,
# which can easily be verified like so:
#

hx, hz = css_code[m1:, :n1], css_code[:m1, n1:] # Extract individual components.

print(f"Does H_X * H_Z^T = 0? {np.allclose((hx @ hz.T) % 2, 0)}")
print(f"Does H_Z * H_X^T = 0? {np.allclose((hz @ hx.T) % 2, 0)}\n")

######################################################################
# Finally, we can also confirm that our constructed matrix encodes exactly one logical qubit
# by computing the code dimension (:math:`k`), obtained by subtracting the linearly independent
# stabilizer constraints from the total number of physical qubits.
#

def binary_matrix_rank(binary_matrix: np.ndarray) -> int:
    r"""Returns the rank of a binary matrix over :math:`\mathbb{Z}_2`."""
    rank, matrix = 0, np.asarray(binary_matrix, dtype=bool).copy()
    while len(matrix):
        matrix, pivot = matrix[:-1], matrix[-1]
        if not pivot.any():
            continue
        rank += 1 # New pivot found
        rows_with_bit = matrix[:, np.flatnonzero(pivot)[-1]]
        matrix[rows_with_bit] ^= pivot
    return rank

code_dim = hx.shape[1] - binary_matrix_rank(hx) - binary_matrix_rank(hz)
print(f"Code dimension (k): {code_dim}\n")

######################################################################
# Hypergraph Product Codes
# ------------------------
#
# Finding a single sparse matrix for a classical code is straightforward. However,
# for a quantum CSS code, we additionally need two sparse matrices that also commute
# with each other. While this is still doable, coming up with the matrices that
# simultaneously yield a constant encoding rate :math:`k = \Theta(n)` and a growing
# distance :math:`d = \Theta(n^\alpha)` for some :math:`0 < \alpha \leq 1`, where
# :math:`n` is the number of physical qubits, is notoriously difficult. This severely
# limits the design space compared to classical codes, which is why it took decades
# for researchers to discover families of good qLDPC codes. A foundational step in
# this direction was the Hypergraph Product (HGP) construction [#HGP]_, which takes
# two classical LDPC codes, :math:`C_1` (:math:`[n_1, k_1, d_1]`)
# and :math:`C_2` (:math:`[n_2, k_2, d_2]`), with parity-check matrices :math:`H_1`
# (:math:`m_1 \times n_1`) and :math:`H_2` (:math:`m_2 \times n_2`), respectively,
# and produces a CSS code with the following parity-check matrices:
#
# .. math::
#
#     H_X = (H_1 \otimes I_{n_2} | I_{m_1} \otimes H_2^T),\\
#     H_Z = (I_{n_1} \otimes H_2 | H_1^T \otimes I_{m_2}).
#
# Here, the algebraic properties of the tensor product ensure that :math:`H_X` and :math:`H_Z`
# satisfy the symplectic orthogonality condition. Furthermore, the transposed matrix :math:`H_i^T`
# defines the transpose code, which has its own parameters :math:`[m_i, k_i^t, d_i^t]`, where
# the superscript :math:`t` simply labels the dimension and distance of this new code.
# For example, look at the following HGP code constructed from two :math:`d_1=3` and :math:`d_2=3`
# repetition codes, which is equivalent to a toric code :math:`[[13, 1, 3]]`:
#

from pennylane.qchem.tapering import _kernel as binary_matrix_kernel
from pennylane.math import binary_finite_reduced_row_echelon

def rep_code(distance: int) -> np.ndarray:
    """Construct repetition code parity check matrix for specified distance."""
    return np.eye(distance - 1, distance, k=0, dtype=np.uint8) + np.eye(
        distance - 1, distance, k=1, dtype=np.uint8
    )

def hgp_code(h1: np.ndarray, h2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Construct HGP code parity check matrices."""
    (m1, n1), (m2, n2) = h1.shape, h2.shape
    hx = np.hstack((np.kron(h1, np.identity(n2)), np.kron(np.identity(m1), h2.T)))
    hz = np.hstack((np.kron(np.identity(n1), h2), np.kron(h1.T, np.identity(m2))))
    return hx.astype(np.int8), hz.astype(np.int8)

h1, h2 = rep_code(3), rep_code(3)
hx, hz = hgp_code(h1, h2)
print(f"Does H_X * H_Z^T = 0? {np.allclose((hx @ hz.T) % 2, 0)}")

######################################################################
# With the commutativity condition satisfied, we can look at the structure of these parity-check
# matrices directly. For each matrix, each non-zero entry means that data qubit (column)
# participates in the stabilizer. For geometrically local codes, non-zero entries would cluster
# near the diagonal, but as we see below, the above HGP construction scatters them across the
# entire qubit range:
#

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 2.5), sharey=True)

ax1.imshow(hx, cmap="Reds")
ax2.imshow(hz, cmap="Blues")
ax1.set_xlabel(r"H$_X$ | Data-qubit index", fontsize=11)
ax2.set_xlabel(r"H$_Z$ | Data-qubit index", fontsize=11)
ax1.set_ylabel(r"Stabilizer index", fontsize=11)

plt.tight_layout()
plt.show()


######################################################################
# This represents the non-local connectivity, which is the defining characteristic of
# qLDPC codes. Next, we determine the code dimension, :math:`k`, that follows directly
# from the ranks of the seed matrices:
#

(m1, n1), (m2, n2) = h1.shape, h2.shape
r1, r2 = binary_matrix_rank(h1), binary_matrix_rank(h2)
k1, k2 = n1 - r1, n2 - r2
k1t, k2t = m1 - r1, m2 - r2
print(f"Code dimension (k) of the HGP code: {k1 * k2 + k1t * k2t}\n")

######################################################################
# Finally, we determine the distance :math:`d` of our HGP code. We can compute
# it by finding the classical distances of the constituent codes and their
# transposes, and taking the minimum across them.
#

def compute_distance(parity_matrix: np.ndarray) -> int:
    """Compute the classical distance of the code based on the parity-check matrices."""
    kernel_matrix = binary_matrix_kernel(
        binary_finite_reduced_row_echelon(parity_matrix)
    )  # compute the kernel of the parity-check matrix
    if (k := kernel_matrix.shape[0]) == 0:
        return np.inf  # the code distance is not defined

    # Compute every single codeword simultaneously and compute Hamming weight
    ints = np.arange(1 << k, dtype=np.uint32)[:, None]
    shifts = np.arange(k, dtype=np.uint32)
    coeffs = ((ints >> shifts) & 1).astype(np.uint8)
    codewords = (coeffs @ kernel_matrix) % 2
    weights = codewords[1:].sum(axis=1)
    return int(np.min(weights))

d1, d2 = compute_distance(h1), compute_distance(h2)
d1t, d2t = compute_distance(h1.T), compute_distance(h2.T)
print(f"Distance (d) of the HGP code: {(dist := min(d1, d2, d1t, d2t))}")
print(f"Physical qubits (n) of the HGP code: {n1*n2 + m1*m2} == {2*dist*(dist-1) + 1}")

######################################################################
# As shown above, the resulting quantum code :math:`Q[[n,k,d]]`, encodes
# :math:`k = k_1k_2 + k_1^t k_2^t` logical qubits into :math:`n = n_1n_2 + m_1m_2`
# physical qubits with distance :math:`d=\min(d_1,d_2,d_1^t,d_2^t)`. This means that
# the HGP codes achieve a constant encoding rate :math:`R=\Theta(1)`, but their distance
# grows only as :math:`d=\mathcal{O}(\sqrt{n})`, matching the surface code scaling. Note
# that the distance computed here is the classical distance, which is not the same as
# the quantum distance. The latter is more complex to compute as it requires finding
# the minimum weight of an error that goes undetected by the checks but is not
# a stabilizer. For HGP codes specifically, the classical distance serves as an
# upper bound on the quantum distance and thus remains a reliable proxy in practice,
# though the two can diverge significantly for other code families.
#
# Modern qLDPC Codes
# -------------------
#
# To build truly scalable quantum computation devices, we need to at least achieve a linear
# distance scaling, i.e., :math:`d=\Theta(n)`. In recent years, there has been some progress
# towards this goal, primarily through a series of breakthroughs, some of which are:
#
# 1. **Lifted Product (LP) Codes:** To overcome the :math:`\mathcal{O}(\sqrt{n})` distance barrier
#    of standard HGP codes, LP codes replace the binary scalar entries of a classical seed matrix
#    with elements of a cyclic group algebra, i.e., polynomials in :math:`\mathbb{F}_2[x]/(x^N-1)`,
#    representing cyclic shifts [#LPCodes]_. By taking the hypergraph product over this polynomial
#    space and *lifting* the result back into a sparse binary matrix, they inject powerful
#    algebraic constraints. This first boosted the minimum distance to :math:`d=\Theta(n/\log n)`,
#    and was later improved to fully linear :math:`d = \Theta(n)`, while maintaining a constant
#    encoding rate :math:`R = \Theta(1)`.
#
# 2. **Quantum Tanner (QT) Codes:** Seeking to maximize both storage density and error-correcting
#    power, these codes are constructed by placing qubits on the squares of a left-right Cayley
#    complex, i.e., a 2D structure built from two Cayley graphs of a finite group with high
#    spectral expansion [#QTCodes]_. By enforcing local classical Tanner code constraints at every
#    vertex of this complex, the expander geometry prevents small errors from forming undetectable
#    logical operators. This achieves constant encoding rate with strictly linear distance
#    :math:`d = \Theta(n)`, meeting the quantum Gilbert-Varshamov bound.
#
# 3. **Bivariate Bicycle (BB) Codes:** These codes bridge the gap between the abstract algebra of
#    expander graphs that require highly non-local hardware wiring, and the physical reality of
#    quantum processors [#BBCodes]_. They are built using pairs of low-degree polynomials :math:`A(x,y)`
#    and :math:`B(x,y)` over the ring :math:`\mathbb{F}_2[x,y]/(x^\ell - 1, y^m - 1)`, where
#    :math:`x` and :math:`y` generate cyclic shifts along the two axes of an :math:`\ell \times m`
#    torus. This ensures that physical qubits can be laid out in a quasi-2D architecture with
#    strictly bounded, short-range connections, making them highly viable for current hardware.
#
# To see why these modern constructions are so powerful, let us look at a simplified construction
# of the QT codes. The function ``tanner_code`` below takes the base parity-check matrices of
# an HGP code and iteratively applies a row-and-column elimination procedure to enforce the
# local-code structure of the Tanner graph, which removes the inter-block redundancies in the
# HGP parity-checks.
#

def tanner_code(h1: np.ndarray, h2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Construct Tanner code parity check matrices [arXiv:2309.11719]."""
    itr, idx = h1.shape[0] * h2.shape[0], h1.shape[1] * h2.shape[1]
    def apply_gaussian_pivot(mat: np.ndarray) -> np.ndarray:
        col = mat[:, idx]
        if not col.any():
            return mat
        pivot = int(col.argmax())
        mask = col.astype(bool)
        mask[pivot] = False
        mat[mask] ^= mat[pivot]
        return np.delete(mat, pivot, axis=0).astype(np.int8)

    hx, hz = hgp_code(h1, h2)
    for ix in range(itr):
        hx = np.delete(apply_gaussian_pivot(hx) if ix % 2 == 0 else hx, idx, axis=1)
        hz = np.delete(apply_gaussian_pivot(hz) if ix % 2 == 1 else hz, idx, axis=1)

    return hx, hz

ns_hgp, ns_tan = [], []
for dist in (distances := range(2, 20)):
    h1, h2 = rep_code(dist), rep_code(dist)
    ns_hgp.append(h1.shape[0] * h2.shape[0] + h1.shape[1] * h2.shape[1])
    hx, hz = tanner_code(h1, h2)
    ns_tan.append(max(hx.shape[1], hz.shape[1]))

plt.figure(figsize=(6, 3))
plt.plot(distances, ns_hgp, '-o', label="HGP Code")
plt.plot(distances, ns_tan, '-*', label="Quantum Tanner Code")
plt.grid(True, which="both", ls="--", c="lightgray", alpha=0.7)
plt.ylabel("# Physical qubits")
plt.xlabel("Rep. code distance")
plt.legend()
plt.tight_layout()
plt.show()

######################################################################
# Decoding qLDPC Codes
# ----------------------
#
# As mentioned earlier, Tanner graphs constructed using the parity-check matrix of the code
# can be used for decoding errors efficiently using an iterative message-passing algorithm
# like `Belief Propagation (BP) <https://pennylane.ai/qml/demos/tutorial_bp_catalyst>`_ [#BProp]_.
# This decoding process can be thought of as a collaborative exercise, where the variable nodes
# (qubits) and check nodes (parity rules) act like detectives passing *messages* back and forth.
# A variable node sends a confidence level message, *"I am 84% sure that I have an error"*. The
# check node looks at the notes from all connected qubits, applies the parity rule, and replies
# *"Based on the group's evidence, adjust your confidence to 96%"*. Mathematically, these messages
# are Log-Likelihood Ratios (LLRs), which are updated iteratively until all parity rules are
# satisfied (consensus) or a fixed number of iterations is reached, making the whole process
# executable in polynomial time.
#
# For classical codes, BP is near-optimal and runs in :math:`\mathcal{O}(\log n)` iterations.
# However, quantum codes suffer from degeneracy, where multiple different error patterns trigger
# the exact same syndrome. This leads to ambiguity in the BP decoder, causing it to endlessly
# flip-flop without reaching a consensus. When BP fails to converge, we use Ordered Statistics
# Decoding (OSD) with order-0 as a fallback [#OSD0]_. One can think of OSD-0 as a tie-breaker,
# which takes the final, unresolved LLRs from BP and ranks the qubits from most to least confident.
# It locks in the most confident qubits as absolute truth, and then uses Gaussian elimination to
# mathematically force a valid parity solution for the remaining uncertain qubits. Let us define
# a decoder class that implements this, where the BP is implemented using the ``tanh`` product rule
# and the OSD-0 uses :func:`~.pennylane.math.binary_finite_reduced_row_echelon` to perform
# `Gaussian elimination <https://en.wikipedia.org/wiki/Gaussian_elimination>`_.
#

class BPOSDDecoder:
    """A lightweight Belief Propagation + OSD-0 decoder.

    Args:
        H (np.ndarray): Parity-check matrix for the codeword (m x n).
        error_rate (float): Prior probability that any single bit is flipped.
        max_iter (int): Maximum BP iterations before falling back to OSD.
    """

    def __init__(self, H, error_rate=0.05, max_iter=50):
        self.H = np.asarray(H, dtype=int)
        self.m, self.n = self.H.shape
        self.max_iter = max_iter
        self.channel_llr = np.log((1 - error_rate) / error_rate)

    def decode(self, syndrome: np.ndarray) -> tuple[bool, np.ndarray, str]:
        """Decode a length-m syndrome vector and return the estimated error."""
        # Initialize messages from check to variable nodes and the total belief
        parity_matrix, target_syndrome = self.H, np.asarray(syndrome, dtype=int)
        c2v_messages = np.zeros((self.m, self.n))
        prior_llr = self.channel_llr # baseline likelihood - prior belief
        posterior_llr = np.full(self.n, prior_llr)

        for _ in range(self.max_iter): # BP loop
            # Variable-to-Check Update (Extrinsic Information)
            var_to_check_msgs = parity_matrix * (posterior_llr[None, :] - c2v_messages)
            # Check-to-Variable Update (Tanh Product Rule)
            c2v_messages = self.update_checks(var_to_check_msgs, target_syndrome)
            # Update Total Beliefs (Posterior LLR)
            posterior_llr = prior_llr + (c2v_messages * parity_matrix).sum(axis=0)
            # Make a Hard Decision
            estimated_error = (posterior_llr < 0).astype(int)  
            # Verify the Syndrome
            if np.all((parity_matrix @ estimated_error) % 2 == target_syndrome):
                return (True, estimated_error, "BP")

        # OSD Fallback and verify the syndrome
        estimated_error = self.osd0(target_syndrome, posterior_llr)  
        is_success = bool(np.all((parity_matrix @ estimated_error) % 2 == target_syndrome))
        return (is_success, estimated_error, "OSD-0")

    def update_checks(self, var_messages: np.ndarray, syndrome: np.ndarray) -> np.ndarray:
        r"""Check-to-variable update via the :math:`\tanh` product rule.

        .. math::
            M_{cv}[i,j] = (-1)^{s_i}\cdot 2 \arctanh{\prod_{j'\neq j}\tanh(M_{vc}[i,j']/2)}
        """
        parity_matrix, delta = self.H, 1e-15

        # Compute the tanh (for incoming messages) and check for edge-existence
        tanh_msgs = np.tanh(var_messages / 2.0)
        tanh_msgs = np.where((parity_matrix == 1) & (tanh_msgs == 0.0), delta, tanh_msgs)

        # Compute the "extrinsic" product for each edge and apply the syndrome
        masked_tanh_msgs = np.where(parity_matrix == 1, tanh_msgs, 1.0)
        check_node_prods = np.prod(masked_tanh_msgs, axis=1, keepdims=True)
        extrinsic_tanh = np.clip(check_node_prods/masked_tanh_msgs, -1 + delta, 1 - delta)

        # Apply the syndrome constraint and convert back to LLR space
        syndrome_sign = (1 - 2 * syndrome)[:, None] # (-1)^s_i
        check_to_var_msgs = parity_matrix * syndrome_sign * 2 * np.arctanh(extrinsic_tanh)
        return check_to_var_msgs

    def osd0(self, syndrome: np.ndarray, llr: np.ndarray) -> np.ndarray:
        """Implements the ordered statistics decoding with order-0 fallback"""
        # Permute using the LLR magnitude and augment it with the syndrome
        reliability_order = np.argsort(-np.abs(llr))
        H_permuted = self.H[:, reliability_order]
        augmented_matrix = np.hstack([H_permuted, syndrome.reshape(-1, 1)])

        # Perform Gaussian elimination over F(2) and extract results
        rref_matrix = binary_finite_reduced_row_echelon(augmented_matrix)
        H_reduced, updated_syndrome = rref_matrix[:, :self.n], rref_matrix[:, -1]

        # Set all non-pivot variables to 0 and permute the errors back
        has_pivot, pivot_cols = H_reduced.any(axis=1), H_reduced.argmax(axis=1)
        final_error, permuted_error = np.zeros((2, self.n), dtype=int)
        permuted_error[pivot_cols[has_pivot]] = updated_syndrome[has_pivot]
        final_error[reliability_order] = permuted_error
        return final_error

######################################################################
# Let us test our decoder on the HGP code constructed from the repetition codes
# with distance :math:`3`. We will intentionally inject a specific weight-:math:`2`
# bit-flip error, compute its syndrome, and ask the decoder to find a correction.
#

h1, h2 = rep_code(3), rep_code(3)
hx, hz = hgp_code(h1, h2)

# Inject a weight-2 X-error (bit-flip)
x_error = np.zeros(hx.shape[1], dtype=int)
x_error[1], x_error[5] = 1, 1
print(f"X-error: {x_error}")

# Z-stabilizers detect X-errors
z_syndrome = (hz @ x_error) % 2
print(f"Z-syndrome: {z_syndrome}")

dec_z = BPOSDDecoder(hz, error_rate=0.05, max_iter=50)
res = dec_z.decode(z_syndrome)[1]
print(f"Decoded error: {res}")

######################################################################
# As multiple physical error patterns map to the exact same syndrome, the decoder
# might find an alternative, equally valid path. When we apply its guessed
# correction to our system, we are essentially creating a *residual* error,
# :math:`E_{residual} = (E_{injected} + E_{decoded}) \pmod{2}`. This will be a null
# vector, if the guessed correction is exact. Alternatively, it can happen to be
# a valid :math:`X`-stabilizer, which means that the combined effect of the noise and
# our guessed correction simply applied a stabilizer to the code block. Since stabilizers
# inherently leave the logical codespace perfectly untouched, our quantum information is
# successfully preserved, even though the decoder guessed a completely different physical
# path! We can see this by adding it as a new row to the :math:`H_X` parity-check matrix
# and checking if it increases its :math:`\mathbb{Z}_2` rank, as shown below.
#

residual = (res + x_error) % 2
hx_w_res = np.vstack([hx, residual])

if np.allclose(residual, 0):
    print("Result: Exact correction")
elif binary_matrix_rank(hx_w_res) == binary_matrix_rank(hx):
    print("Result: Corrected up to stabilizer.")
else:
    print("Result: Logical error.")

######################################################################
# In particular, even though the decoder did not perfectly undo the error,
# the logical state of the qubit is perfectly preserved because any residual
# that is a stabilizer acts as the identity on the codespace. It is worth noting that
# the decoder above assumes a **code-level noise model**, where errors act directly on
# data qubits and syndromes are read perfectly. In practice, both stabilizer measurements
# and syndrome readout introduce faults, requiring a circuit-level noise model and a
# corresponding extension of the decoder to handle imperfect syndrome extraction.
#
# Logical Gates for qLDPC Codes
# ------------------------------
#
# Beyond just storing information safely, a practical quantum computer must also perform logic
# operations on encoded qubits. This is done using logical operators, which are the specific
# combinations of physical single-qubit gates that act on the encoded logical qubits. The most
# fundamental of these are the logical Pauli operators. To be valid, a logical operator must
# commute with all of the code's stabilizers, ensuring it does not trigger an error syndrome.
# However, it must not be a member of the stabilizer group itself; instead, it must act
# nontrivially on the encoded logical state rather than as the identity.
#
# For general CSS codes, including qLDPC code families, we can systematically construct a
# canonical basis of :math:`k` logical operator pairs :math:`\{(L_X^{(i)}, L_Z^{(i)})\}_{i=1}^{k}`
# using linear algebra over :math:`\mathbb{F}_2`. The algorithm requires two sequential passes
# of `Gaussian elimination <https://en.wikipedia.org/wiki/Gaussian_elimination>`_, first on
# :math:`H_X`, then on the remaining free columns of :math:`H_Z`, to identify the logical
# sector. By doing this, we natively guarantee the canonical anticommutation condition
# :math:`L_X^{(i)} \cdot L_Z^{(j)} = \delta_{ij} \pmod{2}`, where :math:`\delta_{ij}` is the
# Kronecker delta. For example, below we construct logical operators for a simple toric code.
#

def compute_logical_ops(hx: np.ndarray, hz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the canonical logical operators for stabilizer codes."""
    # Reduced row echelon forms to get X- and Z-stabilizer pivot columns
    n = hx.shape[1]
    hx_rref = binary_finite_reduced_row_echelon(hx)
    hx_rref = hx_rref[np.any(hx_rref, axis=1)]
    sx_cols = hx_rref.argmax(axis=1)
    non_sx_cols = np.sort(list(set(range(n)) - set(sx_cols)))
    hz_rref = binary_finite_reduced_row_echelon(hz[:, non_sx_cols])
    hz_rref = hz_rref[np.any(hz_rref, axis=1)]
    sz_cols = non_sx_cols[hz_rref.argmax(axis=1)]

    # Get the logical columns that are free in both Sx and Sz
    l_cols = np.sort(list(set(non_sx_cols) - set(sz_cols)))
    if (k := len(l_cols)) == 0:
        return np.empty((0, n), dtype=int), np.empty((0, n), dtype=int)
    l_r_cols = np.searchsorted(non_sx_cols, l_cols)

    # Construct the logical operators Lx and Lz
    lx, lz = np.zeros((k, n), dtype=int), np.zeros((k, n), dtype=int)
    lx[np.arange(k), l_cols], lz[np.arange(k), l_cols] = 1, 1
    lx[:, sz_cols], lz[:, sx_cols] = hz_rref[:, l_r_cols].T, hx_rref[:, l_cols].T
    return lx, lz

# 2-bit repetition code on a ring
h1, h2 = np.ones((2, 2)), np.ones((2, 2))
hx, hz = hgp_code(h1, h2)  # toric code
lx, lz = compute_logical_ops(hx, hz)

print(f"Lx: {lx}")
print(f"Lz: {lz}")

######################################################################
# We can now verify the obtained logical operators by checking the following
# anticommutation relations:
#

print("\nDoes Lx commute with all Z-stabilizers? ", np.allclose((hz @ lx.T) % 2, 0))
print("Does Lz commute with all X-stabilizers? ", np.allclose((hx @ lz.T) % 2, 0))
print("Do Lx and Lz anticommute? ", np.allclose(lx @ lz.T, np.eye(lx.shape[0])))

######################################################################
# Transversal Gates for qLDPC Codes
# ----------------------------------
#
# Transversal gates are logical operations realized by applying gates independently
# to corresponding qubits across code blocks, with no interactions within a single
# code block. For a single logical qubit, this means independent single-qubit gates
# applied in parallel across all physical qubits. Because each physical qubit
# is acted on by at most one gate, errors cannot spread within the block, making them
# inherently fault-tolerant. For example, a transversal :math:`T^\dagger` gate in the
# :math:`[[15, 1, 3]]` quantum `Reed-Muller code <https://errorcorrectionzoo.org/c/stab_15_1_3>`_
# corresponds to applying the :class:`~.pennylane.T` gate on all physical qubits, i.e.,
# :math:`T^\dagger_L = T^{\otimes 15}`.
#
# However, the `Eastin-Knill theorem <https://en.wikipedia.org/wiki/Eastin%E2%80%93Knill_theorem>`_
# restricts the set of logical unitary product operators that can be applied transversally for
# any nontrivial local-error-detecting quantum code to be non-universal. For most stabilizer codes,
# the transversal gate set is limited to the Clifford group. The non-Clifford gates such as
# :class:`~.pennylane.T` must instead be realized indirectly, for example via `magic state injection
# <https://pennylane.ai/qml/glossary/what-are-magic-states>`__ [#Transversal]_.
#
# A notable property of certain qLDPC code families is their native support for transversal
# non-Clifford gates, such as the :class:`~.pennylane.CCZ` gate. While this property
# depends on the specific algebraic structure of the code rather than being a consequence of the
# low-density parity check construction alone, it substantially reduces the hardware overhead
# needed for universal quantum computing. We can test if any given operation is transversal
# for a given code by testing if (i) it preserves its *codespace*, i.e., the subspace
# stabilized by all stabilizer generators, and (ii) maps logical operators to valid
# logical operators. For example, we can check whether the :class:`~.pennylane.SWAP` gate is
# transversal for the previously constructed toric code by first verifying condition (i).
#

from itertools import product
import stim

def compute_stabilizer_group(hx: np.ndarray, hz: np.ndarray) -> tuple[list, set]:
    """Generates the independent Pauli checks and the full stabilizer group."""
    # Create PauliStrings for X-checks and Z-checks
    generators = [
        stim.PauliString(["".join(["I", "X"][bit]) for bit in row])
        for row in hx if np.any(row)
    ] + [
        stim.PauliString(["".join(["I", "Z"][bit]) for bit in row])
        for row in hz if np.any(row)
    ]

    n_qubits, full_group = hx.shape[1], set()
    for bits in product([0, 1], repeat=len(generators)):
        current_pauli = stim.PauliString(n_qubits)
        for bit, gen in zip(bits, generators):
            if bit:
                current_pauli *= gen
        full_group.add(str(current_pauli))
    return generators, full_group

def codespace_preservation(operations: list, generators: list, stabilizers: set) -> bool:
    """Verify if the given generators preserve the codespace."""
    tableau, result = stim.Tableau(num_qubits=len(generators[0])), True
    for operation in operations:
        tableau.append(*operation)

    for generator in generators:
        if str(evolved := tableau(generator)) in stabilizers:
            print(f"{generator}  -->  {evolved}  (Valid!)")
        else:
            print(f"{generator}  -->  {evolved}  (Invalid!)")
            result = False
            break
    return result

swap = stim.Tableau.from_named_gate("SWAP")
ops = [[swap, (0, 1)], [swap, (2, 3)], [swap, (4, 5)], [swap, (6, 7)]]
generators, stabilizers = compute_stabilizer_group(hx, hz)
preserved = codespace_preservation(ops, generators, stabilizers)
print(f"Result: The codespace is preserved: {preserved}")

######################################################################
# Next, we verify condition (ii), which requires the mapped logical operators
# :math:`L_X` and :math:`L_Z` to be consistent, i.e., they commute with all stabilizers
# but do not collapse into one themselves. A gate that passes only the first condition
# preserves the codespace, but it may act trivially or incoherently on the encoded qubit.
# Consequently, the second condition serves as the definite test of logical correctness
# as we will see below.
#

def logical_operators_consistency(operations: list, logical_ops: tuple) -> bool:
    """Verify if the given logical operators are consistent."""
    tableau, result = stim.Tableau(num_qubits=len(generators[0])), True
    for operation in operations:
        tableau.append(*operation)

    evolved_lxs, evolved_lzs = [], []
    result = True
    for itr, (lx_row, lz_row) in enumerate(zip(*logical_ops)):
        lx_pauli = stim.PauliString(["".join(["I", "X"][bit]) for bit in lx_row])
        lz_pauli = stim.PauliString(["".join(["I", "Z"][bit]) for bit in lz_row])
        evolved_lx, evolved_lz = tableau(lx_pauli), tableau(lz_pauli)
        evolved_lxs.append(evolved_lx)
        evolved_lzs.append(evolved_lz)

        for label, evolved in [("Lx", evolved_lx), ("Lz", evolved_lz)]:
            commutes = all(evolved.commutes(g) for g in generators)
            ntrivial = str(evolved) not in stabilizers
            valid = commutes and ntrivial
            print(f"{label}[{itr}] --> {evolved} | "\
                    f"commutes={commutes}, nontrivial={ntrivial}")
            if not valid:
                result = False
                break

    for ix, elx in enumerate(evolved_lxs):
        for iz, elz in enumerate(evolved_lzs):
            if (not elx.commutes(elz)) != (ix == iz):
                print(f"{{Lx[{ix}], Lz[{iz}]}} anticommutation mismatch!")
                result = False
    return result

consistent = logical_operators_consistency(ops, (lx, lz))
print(f"\nResult: The logical operators are consistent: {consistent}")
print(f"Result: The gate operation is transversal: {preserved and consistent}")

######################################################################
# In addition to the gate operations being transversal, there's active work being done to develop
# efficient frameworks to perform logical Pauli measurements [#LMHM]_, which is another
# critical requirement for the practical utility of these codes.
#
# Conclusion
# ----------
#
# The journey to fault-tolerant quantum computing hinges on managing errors without requiring an
# astronomical number of physical qubits. By relaxing the strict nearest-neighbor constraints of
# topological codes, quantum low-density parity-check (qLDPC) codes offer a profound paradigm
# shift: they trade massive qubit overheads for more complex hardware connectivity. While this
# adds challenges in superconducting qubit platforms, it is a natural fit for photonic platforms.
# In this demo, we explored the fundamentals of these codes, including their construction,
# decoding mechanisms, and key properties like their transversal gate sets.
#
# While recent breakthroughs in qLDPC codes, such as lifted product and quantum Tanner codes,
# have achieved linear distance scaling, their physical construction remains quite involved in
# practice. Furthermore, other promising candidates for
# low-overhead quantum memory, like bivariate bicycle codes, still exhibit asymptotic badness,
# as their distance scales sub-linearly as the number of physical qubits increases. Beyond these
# structural challenges, more work is needed to design transversal gates that allow for complex
# logical operations. We also need to improve general-purpose BP-OSD decoders so they can better
# exploit the specific structure of the code for real-time error correction.
#
# Fortunately, advancements in dynamically reconfigurable and modular architectures are beginning
# to turn these highly connected codes into a physical reality. While many engineering hurdles
# remain, particularly in designing universal transversal gate sets and executing efficient logical
# measurements, things look promising. Supported by fast, linear-time decoding algorithms and
# increasingly flexible hardware, qLDPC codes have evolved past elegant mathematical formalism.
# They are rapidly progressing toward practicality, offering a compelling, high-performance
# alternative to traditional surface or toric codes.
#
# References
# ----------
#
# .. [#qldpc1]
#
#     N. P. Breuckmann, J. N. Eberhardt,
#     "Quantum Low-Density Parity-Check Codes",
#     `PRX Quantum 2, 040101 <https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.040101>`__, 2021.
#
# .. [#CSS]
#
#     T. Rakovszky, V. Khemani,
#     "The Physics of (good) LDPC Codes I. Gauging and dualities",
#     `arXiv:2310.16032 <https://arxiv.org/abs/2310.16032>`__, 2023.
#
# .. [#HGP]
#
#     J.-P. Tillich, G. Zémor,
#     "Quantum LDPC Codes With Positive Rate and Minimum Distance Proportional to the Square Root of the Blocklength",
#     `IEEE Transactions on Information Theory 60(1), 119–136 <https://ieeexplore.ieee.org/document/6671468>`__, 2014.
#
# .. [#LPCodes]
#
#     F. G. Jeronimo, T. Mittal, R. O'Donnell, P. Paredes, M. Tulsiani,
#     "Explicit Abelian Lifts and Quantum LDPC Codes",
#     `arXiv:2112.01647 <https://arxiv.org/abs/2112.01647>`__, 2021.
#
# .. [#QTCodes]
#
#     A. Leverrier, G. Zémor,
#     "Quantum Tanner codes",
#     `arXiv:2202.13641 <https://arxiv.org/abs/2202.13641>`__, 2022.
#
# .. [#BBCodes]
#
#     S. Bravyi, A. W. Cross, J. M. Gambetta, D. Maslov, P. Rall, T. J. Yoder,
#     "High-threshold and low-overhead fault-tolerant quantum memory",
#     `Nature <https://www.nature.com/articles/s41586-024-07107-7>`__, 2024.
#
# .. [#BProp]
#
#     J. Old, M. Rispler,
#     "Generalized Belief Propagation Algorithms for Decoding of Surface Codes",
#     `Quantum 7, 1037 <https://quantum-journal.org/papers/q-2023-06-07-1037/>`__, 2023.
#
# .. [#OSD0]
#
#     J. Valls, F. Garcia-Herrero, N. Raveendran, B. Vasic,
#     "Syndrome-Based Min-Sum vs OSD-0 Decoders: FPGA Implementation and Analysis for Quantum LDPC Codes",
#     `IEEE Access <https://ieeexplore.ieee.org/document/9562513>`__, 2021.
#
# .. [#Transversal]
#
#     H. Leitch, A. Kay,
#     "Transversal Gates for Highly Asymmetric qLDPC Codes",
#     `arXiv:2506.15905 <https://arxiv.org/abs/2506.15905>`__, 2025.
#
# .. [#LMHM]
#
#     B. Ide, M. G. Gowda, P. J. Nadkarni, G. Dauphinais,
#     "Fault-tolerant logical measurements via homological measurement",
#     `Phys. Rev. X 15, 021088 <https://journals.aps.org/prx/abstract/10.1103/PhysRevX.15.021088>`__, 2024.
#