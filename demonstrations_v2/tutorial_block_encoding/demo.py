r"""

Block encoding with matrix access oracles
=========================================


Prominent quantum algorithms such as quantum phase estimation and quantum singular value
transformation sometimes need to use **non-unitary** matrices inside quantum circuits. This is problematic
because quantum computers can only perform unitary evolutions 🔥. Block encoding is a technique
that solves this problem by embedding a non-unitary operator as a sub-block of a larger unitary 
matrix 🧯.

In previous demos we have discussed methods for `simulator-friendly <https://pennylane.ai/qml/demos/tutorial_intro_qsvt#transforming-matrices-encoded-in-matrices>`_
encodings and block encodings using `linear combination of unitaries <https://pennylane.ai/qml/demos/tutorial_lcu_blockencoding>`_
(LCU) decompositions. In this tutorial we explore another general block encoding framework that can be
very efficient for sparse and structured matrices: block encoding with matrix access oracles.

.. figure:: ../_static/demonstration_assets/block_encoding/thumbnail_Block_Encodings_Matrix_Oracle.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

A general circuit for block encoding an arbitrary matrix :math:`A \in \mathbb{C}^{N \times N}` with
:math:`N = 2^{n}` can be constructed as shown in the figure below, if we have access to the oracles 
:math:`U_A` and :math:`U_B:`

.. figure:: ../_static/demonstration_assets/block_encoding/general_circuit.png
    :width: 50%
    :align: center

Where the :math:`H^{\otimes n}` operation is a Hadamard transformation on :math:`n` qubits. The
:math:`U_A` operation is an oracle which encodes the matrix element :math:`A_{i,j}` into the the 
amplitude of the ancilla qubit. The :math:`U_B` oracle ensures that we iterate over every 
combination of :math:`(i,j).`

Finding an optimal quantum gate decomposition that implements :math:`U_A` and :math:`U_B` is not 
always possible. We now explore two different approaches for constructing these oracles that can be 
very efficient for matrices with specific structure or sparsity.

Block encoding structured matrices 
-----------------------------------
In order to better understand the oracle access framework, let us first define :math:`U_A` and :math:`U_B`
for the exact block-encoding of :math:`A.` The :math:`U_A` oracle is responsible for encoding the 
matrix entries of :math:`A` into the amplitude of an auxilary qubit :math:`|0\rangle_{\text{anc}}:`

.. math::

    U_A |0\rangle_{\text{anc}} |i\rangle |j\rangle = |A_{i,j}\rangle_{\text{anc}} |i\rangle |j\rangle,

where

.. math::

    |A_{i,j}\rangle_{\text{anc}} \equiv A_{i,j}|0\rangle_{\text{anc}} + \sqrt{1 - |A_{i,j}|^2}|1\rangle_{\text{anc}}.

The :math:`U_B` oracle is responsible for ensuring proper indexing of each entry in :math:`A` 
and for this algorithm, it simplifies to be the :class:`~.pennylane.SWAP` gate:

.. math:: U_B |i\rangle|j\rangle \ = \ |j\rangle |i\rangle

The naive approach is to construct :math:`U_A` using a sequence of multi-controlled :math:`Ry(\alpha)` 
rotation gates with rotation angles computed as :math:`\alpha = \text{arccos}(A_{i,j}).` It turns out
that this requires :math:`O(N^{4})` gates and is very inefficient. A more efficient approach is the 
Fast Approximate BLock Encodings (FABLE) technique [#fable]_, which
uses the oracle access framework and some clever approximations 🧠. The level of approximation in FABLE 
can be adjusted to simplify the resulting circuit. For matrices with specific structures, FABLE provides an 
efficient circuit *without* reducing accuracy.

The FABLE circuit is constructed from a set of single :math:`Ry` rotation and C-NOT gates. We can remove
the need for multi-controlled rotations using a special transformation of the angles (see [#fable]_ for details).
The rotation angles, :math:`(\theta_1, ..., \theta_n),` are obtained from a transformation of the elements 
of the block-encoded matrix.

.. math:: \begin{pmatrix} \theta_1 \\ \cdots \\ \theta_n \end{pmatrix} =
          M \begin{pmatrix} \alpha_1 \\ \cdots \\ \alpha_n \end{pmatrix}.

The angles :math:`\alpha` are obtained from the matrix elements of the matrix :math:`A` as
:math:`\alpha_1 = \text{arccos}(A_{00}), ...,` and :math:`M` is the transformation matrix that can
be obtained with the :func:`~.pennylane.templates.state_preparations.mottonen.compute_theta`
function of PennyLane.

The :class:`~.pennylane.FABLE` circuit is implemented in PennyLane and
can be easily used to block encode matrices of any given shape. Here, we manually construct the
circuit for a structured :math:`4 \times 4` matrix.
"""

import pennylane as qml
from pennylane.templates.state_preparations.mottonen import compute_theta, gray_code
import numpy as np
import matplotlib.pyplot as plt

A = np.array([[-0.51192128, -0.51192128,  0.6237114 ,  0.6237114 ],
              [ 0.97041007,  0.97041007,  0.99999329,  0.99999329],
              [ 0.82429855,  0.82429855,  0.98175843,  0.98175843],
              [ 0.99675093,  0.99675093,  0.83514837,  0.83514837]])

##############################################################################
# We also compute the rotation angles.

alphas = np.arccos(A).flatten()
thetas = compute_theta(alphas)

##############################################################################
# The next step is to identify and prepare the qubit registers used in the oracle access framework.
# There are three registers :code:`"ancilla"`, :code:`"wires_i"`, :code:`"wires_j"`. The
# :code:`"ancilla"` register will always contain a single qubit, this is the auxilary qubit where we
# apply the rotation gates mentioned above. The :code:`"wires_i"` and :code:`"wires_j"` registers are
# the same size for this algorithm and need to be able to encode :math:`A` itself, so they will both
# have :math:`2` qubits for our matrix.

ancilla_wires = ["ancilla"]

s = int(np.log2(A.shape[0]))
wires_i = [f"i{index}" for index in range(s)]
wires_j = [f"j{index}" for index in range(s)]

##############################################################################
# Finally, we obtain the control wires for the C-NOT gates and a wire map that we later use to
# translate the control wires into the wire registers we prepared.

code = gray_code(int(2 * np.log2(len(A))))
control_wires = np.log2(code ^ np.roll(code, -1)).astype(int)

wire_map = {control_index : wire for control_index, wire in enumerate(wires_j + wires_i)}

##############################################################################
# We now construct the :math:`U_A` and :math:`U_B` oracles as well as the operator representing the
# tensor product of Hadamard gates.

def UA(thetas, control_wires, ancilla):
    for theta, control_index in zip(thetas, control_wires):
        qml.RY(2 * theta, wires=ancilla)
        qml.CNOT(wires=[wire_map[control_index]] + ancilla)


def UB(wires_i, wires_j):
    for w_i, w_j in zip(wires_i, wires_j):
        qml.SWAP(wires=[w_i, w_j])


def HN(input_wires):
    for w in input_wires:
        qml.Hadamard(wires=w)

##############################################################################
# We construct the circuit using these oracles and draw it.

dev = qml.device('default.qubit', wires=ancilla_wires + wires_i + wires_j)
@qml.qnode(dev)
def circuit():
    HN(wires_i)
    qml.Barrier()  # to separate the sections in the circuit
    UA(thetas, control_wires, ancilla_wires)
    qml.Barrier()
    UB(wires_i, wires_j)
    qml.Barrier()
    HN(wires_i)
    return qml.probs(wires=ancilla_wires + wires_i)

qml.draw_mpl(circuit, style='pennylane')()
plt.show()

##############################################################################
# Finally, we compute the matrix representation of the circuit and print its top-left block to
# compare it with the original matrix.

print(f"Original matrix:\n{A}", "\n")
wire_order = ancilla_wires + wires_i[::-1] + wires_j[::-1]
M = len(A) * qml.matrix(circuit, wire_order=wire_order)().real[0:len(A),0:len(A)]
print(f"Block-encoded matrix:\n{M}", "\n")

##############################################################################
# You can easily confirm that the circuit block encodes the original matrix defined above. Note that
# the dimension of :math:`A` should be :math:`2^n` where :math:`n` is an integer. For matrices with
# an arbitrary size, we can add zeros to reach the correct dimension. The padding will be
# automatically applied in :class:`~.pennylane.FABLE` implemented in
# PennyLane.
#
# The interesting point about the FABLE method is that we can eliminate those rotation gates that
# have an angle smaller than a defined threshold. This leaves a sequence of C-NOT gates that in
# most cases cancel each other out. You can confirm that two C-NOT gates applied to the same wires
# cancel each other. Let's now remove all the rotation gates that have an angle smaller than
# :math:`0.01` and draw the circuit.

tolerance= 0.01

def UA(thetas, control_wires, ancilla):
    for theta, control_index in zip(thetas, control_wires):
        if abs(2 * theta)>tolerance:
            qml.RY(2 * theta, wires=ancilla)
        qml.CNOT(wires=[wire_map[control_index]] + ancilla)

qml.draw_mpl(circuit, style='pennylane')()
plt.show()

##############################################################################
# Compressing the circuit by removing some of the rotations is an approximation. We can now remove
# the C-NOT gates that cancel each other out and see how good this approximation is in the case
# of our example. We will make a small modification to :math:`U_A` so that it removes those
# C-NOT gates that cancel each other out.

def UA(thetas, control_wires, ancilla):
    nots=[]
    for theta, control_index in zip(thetas, control_wires):
        if abs(2 * theta) > tolerance:
            for c_wire in nots:
                qml.CNOT(wires=[c_wire] + ancilla)
            qml.RY(2 * theta,wires=ancilla)
            nots=[]
        if (cw := wire_map[control_index]) in nots:
            del(nots[nots.index(cw)])
        else:
            nots.append(wire_map[control_index])
    for c_wire in nots:
        qml.CNOT([c_wire] + ancilla)

qml.draw_mpl(circuit, style='pennylane')()
plt.show()

print(f"Original matrix:\n{A}", "\n")
wire_order = ancilla_wires + wires_i[::-1] + wires_j[::-1] 
M = len(A) * qml.matrix(circuit,wire_order=wire_order)().real[0:len(A),0:len(A)]
print(f"Block-encoded matrix:\n{M}", "\n")

##############################################################################
# You can see that the compressed circuit is equivalent to the original circuit. This happens
# because our original matrix is highly structured and many of the rotation angles are zero.
# However, this is not always true for an arbitrary matrix. Can you construct another matrix that
# allows a significant compression of the block encoding circuit without affecting the accuracy?
#
# Block encoding sparse matrices
# ------------------------------
# The quantum circuit for the oracle :math:`U_A,` presented above, accesses every entry of
# :math:`A` and thus requires :math:`~ O(N^2)` gates to implement the oracle [#fable]_. In the
# special cases where :math:`A` is structured and sparse, we can generate a more efficient quantum
# circuit representation for :math:`U_A` and :math:`U_B` [#sparse]_ by only keeping track of the
# non-zero entries of the matrix.
#
# Let :math:`b(i,j)` be a function such that it takes a column index :math:`j` and returns the
# row index for the :math:`i^{th}` non-zero entry in that column of :math:`A.` Note, in this
# formulation, the :math:`|i\rangle` qubit register now refers to the number of non-zero entries
# in :math:`A.` For sparse matrices, this can be much smaller than :math:`N,` thus saving us many
# qubits. We use this to define :math:`U_A` and :math:`U_B.`
#
# Like in the structured approach, the :math:`U_A` oracle is responsible for encoding the matrix
# entries of :math:`A` into the amplitude of the ancilla qubit. However, we now use :math:`b(i,j)`
# to access the row index of the non-zero entries:
#
# .. math::
#
#     U_A |0\rangle_{\text{anc}} |i\rangle |j\rangle = |A_{b(i,j),j}\rangle_{\text{anc}} |i\rangle |j\rangle,
#
# In this case the :math:`U_B` oracle is responsible for implementing the :math:`b(i,j)` function
# and taking us from the column index to the row index in the qubit register:
#
# .. math:: U_B |i\rangle|j\rangle \ = \ |i\rangle |b(i,j)\rangle
#
# Let's work through an example. Consider the sparse matrix given by:
#
# .. math:: A = \begin{bmatrix}
#       \alpha & \gamma & 0 & \dots & \beta\\
#       \beta & \alpha & \gamma & \ddots & 0 \\
#       0 & \beta & \alpha & \gamma \ddots & 0\\
#       0 & \ddots & \beta & \alpha & \gamma\\
#       \gamma & 0 & \dots & \beta & \alpha \\
#       \end{bmatrix},
#
# where :math:`\alpha,` :math:`\beta` and :math:`\gamma` are real numbers. The following code block
# prepares the matrix representation of :math:`A` for an :math:`8 \times 8` sparse matrix.

alpha, beta, gamma  = 0.1, 0.6, 0.3

A = np.array([[alpha, gamma,     0,     0,     0,     0,     0,  beta],
              [ beta, alpha, gamma,     0,     0,     0,     0,     0],
              [    0,  beta, alpha, gamma,     0,     0,     0,     0],
              [    0,     0,  beta, alpha, gamma,     0,     0,     0],
              [    0,     0,     0,  beta, alpha, gamma,     0,     0],
              [    0,     0,     0,     0,  beta, alpha, gamma,     0],
              [    0,     0,     0,     0,     0,  beta, alpha, gamma],
              [gamma,     0,     0,     0,     0,     0,  beta, alpha]])

print(f"Original A:\n{A}", "\n")

##############################################################################
# Once again we identify and prepare the qubit registers used in the oracle access framework.
#
# The :code:`"ancilla"` register will still contain a single qubit, the target for the
# controlled rotation gates. The :code:`"wires_i"` register needs to be large enough to binary
# encode the maximum number of non-zero entries in any column or row. Given the structure of
# :math:`A` defined above, we have at most 3 non-zero entries, thus this register will have 2
# qubits. Finally, the :code:`"wires_j"` register will be used to encode :math:`A` itself, so it
# will have 3 qubits. We prepare the wires below:

s = int(np.log2(A.shape[0]))  # number of qubits needed to encode A

ancilla_wires = ["ancilla"]   # always 1 qubit for controlled rotations
wires_i = ["i0", "i1"]        # depends on the sparse structure of A
wires_j = [f"j{index}" for index in range(s)]  # depends on the size of A 

##############################################################################
# The :math:`U_A` oracle for this matrix is constructed from controlled rotation gates, similar to
# the FABLE circuit.

def UA(theta, wire_i, ancilla):
    qml.ctrl(qml.RY, control=wire_i, control_values=[0, 0])(theta[0], wires=ancilla)
    qml.ctrl(qml.RY, control=wire_i, control_values=[1, 0])(theta[1], wires=ancilla)
    qml.ctrl(qml.RY, control=wire_i, control_values=[0, 1])(theta[2], wires=ancilla)

##############################################################################
# The :math:`U_B` oracle is defined in terms of the so-called ``Left`` and ``Right`` shift operators.
# They correspond to the modular arithmetic operations :math:`+1` or :math:`-1` respectively [#sparse]_.

def shift_op(s_wires, shift="Left"):        
    for index in range(len(s_wires)-1, 0, -1):
        control_values = [1] * index if shift == "Left" else [0] * index
        qml.ctrl(qml.PauliX, control=s_wires[:index], control_values=control_values)(wires=s_wires[index])
    qml.PauliX(s_wires[0])


def UB(wires_i, wires_j):
    qml.ctrl(shift_op, control=wires_i[0])(wires_j, shift="Left")
    qml.ctrl(shift_op, control=wires_i[1])(wires_j, shift="Right")

##############################################################################
# We now construct our circuit to block encode the sparse matrix and draw it.

dev = qml.device("default.qubit", wires=(ancilla_wires + wires_i + wires_j))

@qml.qnode(dev)
def complete_circuit(thetas):
    HN(wires_i)
    qml.Barrier() # to separate the sections in the circuit
    UA(thetas, wires_i, ancilla_wires)
    qml.Barrier()
    UB(wires_i, wires_j)
    qml.Barrier()
    HN(wires_i)
    return qml.probs(wires=ancilla_wires + wires_i)

s = 4  # normalization constant
thetas = 2 * np.arccos(np.array([alpha - 1, beta, gamma]))

qml.draw_mpl(complete_circuit, style='pennylane')(thetas)
plt.show()

##############################################################################
# Finally, we compute the matrix representation of the circuit and print its top-left block to
# compare it with the original matrix.

print("\nBlockEncoded Mat:")
wire_order = ancilla_wires + wires_i[::-1] + wires_j[::-1] 
mat = qml.matrix(complete_circuit, wire_order=wire_order)(thetas).real[:len(A), :len(A)] * s
print(mat, "\n")

##############################################################################
# You can confirm that the circuit block encodes the original sparse matrix defined above.
# Note that if we wanted to increase the dimension of :math:`A` (for example :math:`16 \times 16`),
# then we need only to add more wires to the ``j`` register in the device and in :code:`UB`.
#
# Conclusion
# -----------------------
# Block encoding is a powerful technique in quantum computing that allows us to implement a
# non-unitary operation in a quantum circuit by embedding the operation in a larger unitary gate.
# In this tutorial, we reviewed two important block encoding methods with code examples using
# PennyLane. This allows you to use PennyLane to explore and benchmark several block encoding
# approaches for a desired problem. The efficiency of the block encoding methods typically depends
# on the sparsity and structure of the original matrix. We hope that you can use these tips and
# tricks to find a more efficient block encoding for your matrix!
#
# References
# ----------
#
# .. [#fable]
#
#     Daan Camps, Roel Van Beeumen,
#     "FABLE: fast approximate quantum circuits for block-encodings",
#     `arXiv:2205.00081 <https://arxiv.org/abs/2205.00081>`__, 2022
#
#
# .. [#sparse]
#
#     Daan Camps, Lin Lin, Roel Van Beeumen, Chao Yang,
#     "Explicit quantum circuits for block encodings of certain sparse matrices",
#     `arXiv:2203.10236 <https://arxiv.org/abs/2203.10236>`__, 2022
#
#
