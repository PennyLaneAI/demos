r"""Nonlinear amplitude transformation
==========================


The macroscopic world is inherently nonlinear. From the complex dynamics of financial markets to the activation functions in neural networks, nonlinear functions are the backbone of engineering, optimization, and machine learning. In contrast, quantum mechanics is fundamentally linear: the evolution of a closed system is always governed by unitary operators. A central challenge in quantum algorithm design is bridging this gap to implement nonlinear transformations on a quantum computer.

"""

######################################################################
# As also described in previous demos, :doc:`block
# encoding <demos/tutorial_block_encoding>`
# [#blockencoding]_ and :doc:`quantum singular value
# transformation <demos/tutorial_intro_qsvt>`
# [`2 <https://arxiv.org/abs/1806.01838>`__] have become the “gold standard” for implementing matrix
# functions. However, these techniques primarily transform the singular values (or eigenvalues) of an
# operator. In many quantum machine learning settings - especially amplitude encoding - the data isn’t
# stored in an operator at all. Instead, it lives directly in the amplitudes of a quantum state.
# 
# To transform these amplitudes nonlinearly, we need a generalized approach. The Nonlinear Amplitude
# Transformation framework [`3 <https://arxiv.org/abs/2107.10764>`__,
# `4 <https://arxiv.org/abs/2309.09839>`__] enable us to map an input state
# :math:`|\psi\rangle = \sum x_i |i\rangle` to a target state
# :math:`|\phi\rangle \propto \sum f(x_i) |i\rangle`, using only unitary operations, ancillas, and
# (typically) postselection. The key conceptual move is to convert “amplitudes-as-data” into a form
# that QSVT can act on, by building a block-encoding whose relevant spectrum contains the amplitude
# values we care about.
# 
# In this demo, we will: 
# 
# - construct a block-encoding of amplitude data starting from a state-preparation unitary, 
# - use QSVT to apply a polynomial approximation of a nonlinear function (e.g., a smooth activation) to those amplitudes, 
# - validate the transformation numerically via an application to a canonical quantum machine learning task of binary classification on downscaled MNIST-style images.
# 
# .. figure:: ../_static/demonstration_assets/nonlinear-amplitude-transformation/pennylane-demo-nonlinear-transformation-qsvt-method.png
#   :alt: Schematic of the nonlinear amplitude transformation with QSVT
#   :width: 95%
#   :align: center
# 
#   Figure 1: *A schematic of the nonlinear transformation transformation with QSVT*
# 

######################################################################
# Diagonal block encoding of amplitudes
# -------------------------------------
# 
# The introduction highlighted a basic mismatch: 
# 
# - QSVT applies a polynomial to the singular values or eigenvalues of an operator. 
# - In amplitude encoding, the data live directly in the amplitudes of a quantum state. 
# 
# Before QSVT becomes useful, the amplitudes must be re-expressed as spectral data of
# an operator that can be block-encoded.
# 
# From amplitude encoding to an operator QSVT can transform
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# Recall the notion of a block encoding [`1 <https://arxiv.org/abs/1804.01973>`__]. A unitary
# :math:`U_A` block-encodes an operator :math:`A` if its top-left block equals :math:`A`:
# 
# .. math::
# 
# 
#    U_A=\begin{pmatrix}
#    A & \cdot \\
#    \cdot & \cdot
#    \end{pmatrix}.
# 
# Once :math:`A` is available in this form, QSVT can implement polynomial transformations of the
# encoded operator, informally :math:`A \mapsto P(A)`, by acting on its spectrum.
# 
# Now consider a state prepared by a unitary :math:`U`:
# 
# .. math::
# 
# 
#    |\psi\rangle=U|0\rangle=\sum_{i=1}^N \psi_i|i\rangle,
# 
# where :math:`{|i\rangle}` are computational basis states. For now, assume :math:`\psi_i \in [-1,1]`
# are real (the complex case follows the same template by handling real and imaginary parts
# separately). Each amplitude is a matrix element of :math:`U`:
# 
# .. math::
# 
# 
#    \psi_i = \langle i|U|0\rangle.
# 
# So the amplitude vector :math:`{\psi_i}` appears as the first column of :math:`U`. The obstacle is
# that QSVT does not act on a column of a unitary; it acts on the spectrum of an operator accessed
# through a block encoding.
# 
# The nonlinear amplitude transformation approach resolves this by constructing, from :math:`U` and
# controlled uses of :math:`U^\dagger`, a new unitary :math:`U_\Psi` whose encoded block is the
# diagonal operator
# 
# .. math::
# 
# 
#    \Psi = \mathrm{diag}(\psi_1,\dots,\psi_N)
#    =
#    \begin{pmatrix}
#    \psi_1 & 0 & \cdots & 0 \\
#    0 & \psi_2 & \cdots & 0 \\
#    \vdots & \vdots & \ddots & \vdots \\
#    0 & 0 & \cdots & \psi_N
#    \end{pmatrix}.
# 
# so that
# 
# .. math::
# 
# 
#    U_{\Psi}=
#    \begin{pmatrix}
#    \Psi & \cdot \\
#    \cdot & \cdot
#    \end{pmatrix}.
# 
# This is in sense equivalent to “encoding the first column into a diagonal,” but the key point is
# subtler: :math:`U` is not modified. Instead, an auxiliary unitary :math:`U_\Psi` is engineered so
# that the amplitudes :math:`\psi_i` appear as the diagonal entries of the encoded operator. In the
# constructions of [`3 <https://arxiv.org/abs/2107.10764>`__,
# `4 <https://arxiv.org/abs/2309.09839>`__], this requires only a constant number of controlled
# invocations of :math:`U` and :math:`U^\dagger`. For the purposes of this demo, we treat
# :math:`U_\Psi` as a primitive and focus on what it enables. The construction idea is intuitionally
# similar to building a quantum walk operator, and interested readers are encouraged to read original
# papers for details.
# 
# With :math:`\Psi` block-encoded, QSVT can be used to implement :math:`P(\Psi)` for a chosen
# polynomial :math:`P`. Since :math:`\Psi` is diagonal, this corresponds to applying
# 
# .. math::
# 
# 
#    \psi_i \mapsto P(\psi_i)
# 
# to all amplitudes in parallel within the postselected branch of the circuit. The next section builds
# the QSVT polynomial approximation for a smooth nonlinearity (e.g., :math:`\tanh`) and validates the
# resulting amplitude transformation numerically.
# 

######################################################################
# A concrete block-encoding circuit (toy size)
# --------------------------------------------
# 
# The previous section introduced the key primitive: a unitary :math:`U_\Psi` whose top-left block
# encodes the diagonal operator
# 
# .. math::
# 
# 
#    \Psi = \mathrm{diag}(\psi_0,\ldots,\psi_{N-1}),
# 
# where :math:`\psi_k` are the amplitudes of an input state
# :math:`|\psi\rangle = \sum_k \psi_k |k\rangle` prepared by a state-preparation unitary :math:`U`.
# 
# This block-encoding is the bridge that makes QSVT applicable: once the amplitudes appear as a
# spectrum (here, as the diagonal entries of :math:`\Psi`), a polynomial transform :math:`P(\Psi)`
# corresponds to applying :math:`\psi_k \mapsto P(\psi_k)` in parallel (up to postselection).
# 
# Here, we build :math:`U_\Psi` explicitly for a small system (:math:`n=2`, so :math:`N=4`) to make
# the construction tangible. The code below spells out the walk-style ingredients used in Guo et
# al. (2024): a reflection :math:`R`, controlled applications of the state-preparation unitary and its
# adjoint, and a pair of composite steps :math:`W` and :math:`G` that together produce the desired
# block structure. A phase toggle :math:`p \in \{0,1\}` switches between encoding the real part
# (:math:`p=0`) and the imaginary part (:math:`p=1`); here we focus on the real case.
# 
# Sanity check: after building the circuit, we inspect its matrix representation and look at the
# top-left :math:`N\times N` block. For a correct block-encoding, this block should behave like
# :math:`\Psi` (up to known normalization conventions), meaning its diagonal entries should match the
# input amplitudes :math:`\{\psi_k\}`. This is the smallest-scale verification that the circuit is
# implementing the intended “amplitudes :math:`\rightarrow` diagonal operator” transformation before
# we move on to applying QSVT polynomials.
# 

import pennylane as qp
from pyqsp.poly import PolyTaylorSeries
import matplotlib.pyplot as plt
from pennylane import numpy as pnp
import jax
from jax import numpy as jnp
import optax

pnp.random.seed(42)

# Circuit setup
main_qubits = 2
dim = 2**main_qubits
rot_wire = [0]
ancilla_wires = list(range(1, main_qubits + 3))
main_wires = list(range(main_qubits + 3, 2 * main_qubits + 3))
all_wires = list(range(2 * main_qubits + 3))
dev = qp.device("lightning.qubit", wires = all_wires)



# -----------------------------------------------------------------------------
#  Implementation of the block‑encoding for real or imaginary
#  parts of amplitudes.

# Controlled-Z on multiple controls.  control_values specify which bit value
# selects the gate; default is all zeros.
def MultiControlledZ(wires, control_values=None):
    if control_values is None:
        control_values = [0] * (len(wires) - 1)
    qp.ctrl(qp.Z(wires=wires[-1]),
             control=wires[:-1],
             control_values=control_values)

# R_gate implements the reflection R used in the block construction.
def R(wires):
    assert len(wires) % 2 == 1
    n = len(wires)//2
    qp.PauliX(wires=wires[0])
    MultiControlledZ(wires=wires[1:n+1]+[wires[0]])
    qp.PauliX(wires=wires[0])

# Apply U on the data register conditioned on ancilla B=0.  U can be a callable
# or an Operator.  Additional arguments are passed through via *args, **kwargs.
def Uc(base, wires, *args, **kwargs):
    assert len(wires) % 2 == 1
    n = len(wires)//2
    if isinstance(base, qp.typing.TensorLike):
        qp.ControlledQubitUnitary(base,
                                   control_wires=wires[n],
                                   wires=wires[:n],
                                   control_values=[0],
                                   unitary_check=True)
    elif isinstance(base, qp.operation.Operator) or callable(base):
        qp.ctrl(base, control=wires[n],
                 control_values=[0])(wires=wires[:n], *args, **kwargs)

# Adjoint of U on the data register controlled on ancilla B=0.
def Uc_adj(base, wires, *args, **kwargs):
    assert len(wires) % 2 == 1
    n = len(wires)//2
    if isinstance(base, qp.typing.TensorLike):
        qp.adjoint(qp.ControlledQubitUnitary)(base,
                                                control_wires=wires[n],
                                                wires=wires[:n],
                                                control_values=[0],
                                                unitary_check=True)
    elif isinstance(base, qp.operation.Operator) or callable(base):
        qp.ctrl(qp.adjoint(base),
                 control=wires[n],
                 control_values=[0])(wires=wires[:n], *args, **kwargs)

# Copy the ancilla B qubit into the address register (controlled Toffoli chain).
# This coherently adds or subtracts the basis state |k> to the prepared state.
def C(wires):
    assert len(wires) % 2 == 1
    n = len(wires)//2
    for i in range(n):
        qp.Toffoli(wires=[wires[n], wires[n+i+1], wires[i]])

# The adjoint of C_to_data, reversing the coherent copy.
def C_adj(wires):
    assert len(wires) % 2 == 1
    n = len(wires)//2
    for i in range(n-1, -1, -1):
        qp.Toffoli(wires=[wires[n], wires[n+i+1], wires[i]])

# One step of the W operator.  If p_flag=1 an S gate is applied to the ancilla B
# to pick up a phase for the imaginary part.
def W(base, wires, p, *args, **kwargs):
    assert len(wires) % 2 == 1
    n = len(wires)//2
    qp.Hadamard(wires[n])
    Uc(base, wires, *args, **kwargs)
    C(wires)
    if bool(p):
        qp.S(wires[n])
    qp.Hadamard(wires[n])


# Adjoint of W_block.
def W_adj(base, wires, p, *args, **kwargs):
    assert len(wires) % 2 == 1
    n = len(wires)//2
    qp.Hadamard(wires[n])
    if bool(p):
        qp.adjoint(qp.S)(wires[n])
    C_adj(wires)
    Uc_adj(base, wires, *args, **kwargs)
    qp.Hadamard(wires[n])


# G_block implements the operator G = W S0 W^† Z_B.  Its adjoint is defined
# similarly.  See Eq. (9) of the Guo *et al.* (2024).
def G(base, wires, p, *args, **kwargs):
    assert len(wires) % 2 == 1
    n = len(wires)//2
    qp.PauliZ(wires[n])
    W_adj(base, wires, p, *args, **kwargs)
    R(wires)
    W(base, wires, p, *args, **kwargs)

# Adjoint of G_block.
def G_adj(base, wires, p, *args, **kwargs):
    assert len(wires) % 2 == 1
    n = len(wires)//2
    W_adj(base, wires, p, *args, **kwargs)
    R(wires)
    W(base, wires, p, *args, **kwargs)
    qp.PauliZ(wires[n])

# RealDiagonalBlockEncoding wraps the above primitives to encode the real part
# of the amplitudes. p=1 switches to the imaginary part.
def RealDiagonalBlockEncoding(U, wires, ancilla_wires, p=0, *args, **kwargs):
    assert len(ancilla_wires) == len(wires) + 2
    qp.Hadamard(wires=ancilla_wires[0])
    W(base=U,
        wires=ancilla_wires[1:]+wires,
        p=p, *args, **kwargs)
    qp.ctrl(G, control=ancilla_wires[0],
                control_values=[0])(base=U, wires=ancilla_wires[1:]+wires,
                                    p=p, **kwargs)
    qp.ctrl(G_adj, control=ancilla_wires[0],
                control_values=[1])(base=U, wires=ancilla_wires[1:]+wires,
                                    p=p, *args, **kwargs)
    qp.Hadamard(wires=ancilla_wires[0])
    W_adj(base=U, wires=ancilla_wires[1:]+wires, p=p, *args, **kwargs)
    qp.PauliX(wires=ancilla_wires[0])
    qp.PauliZ(wires=ancilla_wires[0])
    qp.PauliX(wires=ancilla_wires[0])

######################################################################
# Below we create a simple block‑encoding for :math:`n=2` and inspect its matrix to confirm that its
# diagonal corresponds to the input amplitudes.
# 

feature_vector = [1, 2, 3, 4]
feature_vector = feature_vector/pnp.linalg.norm(feature_vector)
block_encoding = qp.prod(RealDiagonalBlockEncoding)(
    qp.AmplitudeEmbedding, wires=main_wires,
    ancilla_wires=ancilla_wires,
    features=feature_vector,
    normalize=True)

@qp.qnode(dev)
def be_circuit(feature_vector, main_wires, ancilla_wires):
    RealDiagonalBlockEncoding(
    qp.AmplitudeEmbedding, wires=main_wires,
    ancilla_wires=ancilla_wires,
    features=feature_vector,
    normalize=True)
    return qp.probs(ancilla_wires)

######################################################################
# We now compute the matrix of the full unitary and extract its top-left :math:`4\times 4` block,
# which should be approximately diagonal with diagonal entries equal to the normalized feature
# amplitudes.
# 

qp.matrix(be_circuit)(feature_vector, main_wires, ancilla_wires)[:4,:4]

######################################################################
# We draw the block encoding circuit in its entirety.
# 

qp.draw_mpl(be_circuit)(feature_vector, main_wires, ancilla_wires)

######################################################################
# Nonlinear amplitude transformation
# ----------------------------------
# 
# With the diagonal block encoding :math:`U_{\Psi}` in place, QSVT
# [`2 <https://arxiv.org/abs/1806.01838>`__] provides a systematic way to apply a polynomial map to
# the encoded amplitudes. Concretely, for a polynomial :math:`P_d` of degree :math:`d`, QSVT
# constructs a new unitary whose top-left block encodes :math:`P_d(\Psi)`:
# 
# .. math::
# 
# 
#    U_{\Psi}\;\longrightarrow\;
#    U_{P_d(\Psi)}=
#    \begin{pmatrix}
#    P_d(\Psi) & \cdot \\
#    \cdot & \cdot
#    \end{pmatrix},
#    \qquad
#    P_d(\Psi)=\mathrm{diag}\!\big(P_d(\psi_1),\ldots,P_d(\psi_N)\big).
# 
# In practice, the target nonlinearity :math:`f` is typically not a polynomial, so we choose
# :math:`P_d` to approximate :math:`f` on :math:`[-1,1]` up to a desired error tolerance. The QSVT
# cost scales linearly in the degree: implementing :math:`U_{P_d(\Psi)}` uses :math:`O(d)`
# applications of the block encoding :math:`U_{\Psi}`, and therefore :math:`O(d)` calls to the
# underlying state-preparation routine used to build :math:`U_{\Psi}`.
# 
# The constructed :math:`U_{P_d(\Psi)}` is then applied to the reference state and post-selection or
# amplitude amplification [`5 <https://arxiv.org/abs/quant-ph/0005055>`__] is used to obtain the final
# transformed state. The choice of the reference state significantly impacts the algorithm’s
# efficiency. A direct way to “read out” the diagonal action is to start from a uniform superposition
# :math:`\frac{1}{\sqrt{N}}\sum_i |i\rangle`, which applies :math:`P_d(\psi_i)` to every basis
# component. However, this may introduce a dependency on the dimension :math:`N`, which can be
# prohibitively expensive for large systems. Another method, as outlined by Rattew and Rebentrost
# [`4 <https://arxiv.org/abs/2309.09839>`__], is to use the equivalent of importance sampling in this
# context and to start from the prepared state itself,
# 
# .. math::
# 
# 
#    |\psi\rangle=\sum_i \psi_i |i\rangle,
# 
# and implement the modified function
# 
# .. math::
# 
# 
#    g(x)=\frac{f(x)}{x}.
# 
# Applying QSVT to :math:`g(\Psi)` and acting on :math:`|\psi\rangle` yields amplitudes
# 
# .. math::
# 
# 
#    g(\psi_i)\,\psi_i = \frac{f(\psi_i)}{\psi_i}\,\psi_i = f(\psi_i),
# 
# In some cases, this can effectively “recover” the target function :math:`f(x)` without the overhead
# of the system dimension :math:`N`, as we showcase in the implementation of the :math:`\tanh`
# function below.
# 
# QSVT in PennyLane
# ~~~~~~~~~~~~~~~~~
# 
# PennyLane provides tools to implement QSVT once a block encoding is available. The function
# ``qp.poly_to_angles`` computes QSVT phase angles from the polynomial coefficients (ordered from
# lowest to highest power). The resulting angles can be used to build the projector phases and apply
# the transformation via ``qp.QSVT``. See the PennyLane API docs
# [`6 <https://docs.pennylane.ai/en/stable/code/api/pennylane.qsvt.html?utm_source=chatgpt.com>`__,\ `7 <https://docs.pennylane.ai/en/stable/code/api/pennylane.poly_to_angles.html?utm_source=chatgpt.com>`__]
# for ``qp.poly_to_angles`` and ``qp.qsvt`` for details.
# 

######################################################################
# Applying a nonlinearity with QSVT
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# With the diagonal block encoding in hand, the remaining step is to apply a polynomial approximation
# of a target nonlinearity using QSVT. The key ingredients are:
# 
# 1. a polynomial :math:`P_d` that approximates the target function on :math:`[-1,1]`,
# 2. the corresponding QSVT phase angles :math:`\{\phi_j\}`, and
# 3. a sequence of projector-controlled phase shifts that implement the QSVT “signal processing” loop.
# 

def ProjCtrlPhaseShift(control_wires, target_wire, phi):
    qp.MultiControlledX(wires=control_wires + target_wire,
                         control_values=[0] * len(control_wires))
    qp.RZ(phi = 2 * phi, wires=target_wire)
    qp.MultiControlledX(wires=control_wires + target_wire,
                         control_values=[0] * len(control_wires))

def generate_poly(deg, func, odd):
    poly = PolyTaylorSeries().taylor_series(
        func=func, degree=deg, max_scale=0.9,
        chebyshev_basis=True, cheb_samples=2*deg)
    pcoefs = poly.coef
    if odd:
        pcoefs[0::2] = 0
    else:
        pcoefs[1::2] = 0
    return pcoefs

######################################################################
# QSVT imposes a parity structure on the implemented polynomial: depending on the construction, the
# polynomial must be either purely odd or purely even. We therefore generate Chebyshev/Taylor-based
# polynomial approximations and then explicitly zero out the unwanted parity coefficients:
# 
# - :math:`P_d(x) \approx \tanh(x)` as an odd polynomial,
# - :math:`G_d(x) \approx \tanh(x)/x` as an even polynomial.
# 
# The second choice is used for a dimension-friendly “importance” variant: when :math:`f(0)=0`,
# applying :math:`G_d(\Psi)` to the original state :math:`|\psi\rangle=\sum_i \psi_i|i\rangle`
# produces amplitudes proportional to :math:`G_d(\psi_i)\psi_i \approx \tanh(\psi_i)`, avoiding the
# need to start from a uniform superposition.
# 
# Two ways to run the transformation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
# We compare two initializations:
# 
# - Uniform initialization: start from :math:`\frac{1}{\sqrt{N}}\sum_i |i\rangle` and apply
#   :math:`P_d(\Psi)`, yielding amplitudes proportional to :math:`P_d(\psi_i)` (up to postselection).
# - Importance initialization: start from :math:`|\psi\rangle` and apply :math:`G_d(\Psi)`, yielding
#   amplitudes proportional to :math:`G_d(\psi_i)\psi_i \approx \tanh(\psi_i)`.
# 

deg = 4
tanh_poly = generate_poly(deg, pnp.tanh, odd=True)
tanh_div_x_poly = generate_poly(deg, lambda x: pnp.tanh(x)/x, odd=False)
tanh_angles = qp.poly_to_angles(tanh_poly, "QSVT", angle_solver="root-finding")
tanh_div_x_angles = qp.poly_to_angles(tanh_div_x_poly, "QSVT", angle_solver="root-finding")

tanh_projectors = [qp.prod(ProjCtrlPhaseShift)(ancilla_wires, rot_wire, tanh_angles[i]) for i in range(len(tanh_angles))]
tanh_div_x_projectors = [qp.prod(ProjCtrlPhaseShift)(ancilla_wires, rot_wire, tanh_div_x_angles[i]) for i in range(len(tanh_div_x_angles))]

@qp.qnode(dev)
def circuit(block_encoding, projectors, main_wires, ancilla_wires, rot_wire, importance=False):
    if importance:
        qp.AmplitudeEmbedding(feature_vector, main_wires, normalize=True)
    else:
        for wire in main_wires:
            qp.Hadamard(wire)
    qp.Hadamard(rot_wire)
    qp.QSVT(block_encoding, projectors)
    qp.Hadamard(rot_wire)
    return qp.state(), qp.probs(rot_wire + ancilla_wires)

state, probs = circuit(block_encoding, tanh_projectors, main_wires, ancilla_wires, rot_wire)
uniform_state = state[:dim]/pnp.sqrt(probs[0])

state, probs = circuit(block_encoding, tanh_div_x_projectors, main_wires, ancilla_wires, rot_wire, importance=True)
important_state = state[:dim]/pnp.sqrt(probs[0])

normed_vector = feature_vector/pnp.linalg.norm(feature_vector, 2)
goal = pnp.tanh(normed_vector)
goal = goal/pnp.linalg.norm(goal, 2)

######################################################################
# Results: comparing the transformed amplitudes
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# The goal of this experiment is to verify that the QSVT pipeline performs an elementwise nonlinear
# map on the amplitude-encoded data. For each computational basis state :math:`|k\rangle`, we compare
# the postselected (and renormalized) output amplitude against the ideal target value
# :math:`\tanh(\psi_k)` computed classically.
# 
# The figure below plots the amplitude on each basis state index :math:`k`:
# 
# - Uniform: QSVT applies the odd polynomial approximation :math:`P_d(\Psi)\approx\tanh(\Psi)`
#   starting from a uniform superposition over :math:`|k\rangle`.
# - Importance: QSVT applies the even polynomial approximation
#   :math:`G_d(\Psi)\approx\tanh(\Psi)/\Psi` starting from the prepared state :math:`|\psi\rangle`, so
#   that :math:`G_d(\psi_k)\psi_k\approx\tanh(\psi_k)`.
# - True: the ideal target vector :math:`\tanh(\psi)` (normalized).
# 
# Any visible deviation from the True curve is primarily due to the finite polynomial degree :math:`d`
# (here :math:`d=4`) and the conservative scaling used to keep the polynomial within the QSVT-valid
# regime.
# 

x = pnp.arange(dim)

plt.figure()
plt.plot(x, pnp.real(uniform_state), marker="o", label="Uniform (QSVT on P_d)")
plt.plot(x, pnp.real(important_state), marker="o", label="Importance (QSVT on G_d)")
plt.plot(x, pnp.real(goal), marker="o", label="True tanh(ψ)")

plt.xlabel("Basis state index k")
plt.ylabel("Output amplitude on |k⟩")

# show k as bitstrings for readability
bit_labels = [format(i, f"0{main_qubits}b") for i in range(dim)]
plt.xticks(x, bit_labels)

plt.legend()
plt.show()


######################################################################
# Application: A small Quantum Multi-Layer Perceptron (QMLP)
# ----------------------------------------------------------
# 
# So far, the demo has focused on the primitive itself: using diagonal block encodings and QSVT to
# implement an elementwise nonlinear map on amplitude-encoded data. Finally, we showcase the nonlinear
# transformation of complex amplitude (NTCA) method as a genuine non-linear activation layer within a
# trainable quantum model. We build a small quantum analogue of a two-layer MLP: two trainable linear
# layers (implemented as parameterized unitaries) separated by a :math:`\tanh` activation implemented
# via NTCA:
# 
# .. math::
# 
# 
#    \textstyle
#    x
#    \;\longrightarrow\;
#    U\lvert x\rangle
#    \;\xrightarrow{\text{NTCA}}\;
#    \sum_i \tanh\left(\sum_jU_{ij} x_j\right)\lvert i\rangle
#    \;\longrightarrow\;
#    W \left(\sum_i \tanh\left(\sum_jU_{ij} x_j\right)\lvert i\rangle\right)
#    \;\longrightarrow\;
#    \text{measurement}.
# 
# Here, :math:`|x\rangle` denotes an amplitude encoding of the input vector :math:`x` (after
# normalization). The unitary :math:`U` plays the role of the first linear layer by mixing amplitudes.
# The NTCA layer then applies :math:`\tanh(\cdot)` approximately and elementwise to the resulting
# amplitudes, producing a genuinely nonlinear feature map in the computational basis. Finally, the
# second unitary :math:`W` mixes these activated features before a measurement layer produces a
# prediction.
# 
# We train the QMLP on a down-scaled version of MNIST, where each image is mapped to a low-dimensional
# feature vector compatible with amplitude encoding. The objective is not state-of-the-art accuracy;
# rather, it is to demonstrate that the NTCA layer can be inserted into an end-to-end differentiable
# pipeline and used as an activation function inside a trainable quantum model.
# 
# As a broader perspective, the same “linear mixing + elementwise nonlinearity” motif underpins more
# advanced architectures. Recent work has explored the feasibility of quantum implementations of
# transformer-style inference under various access models and resource assumptions
# [`6 <https://arxiv.org/abs/2402.16714>`__]. The QMLP here should be viewed as a minimal instance of
# that design pattern, focused on making the role of a nonlinear activation layer explicit.
# 

[ds] = qp.data.load("other", name="downscaled-mnist")

data = pnp.array(ds.train['4']['inputs'][:1000])
labels = (pnp.array(ds.train['4']['labels'][:1000])+1)/2
dev = qp.device("default.qubit", wires = all_wires)

def embedding(weights, features, wires):
    qp.AmplitudeEmbedding(features, wires, normalize=True)
    qp.BasicEntanglerLayers(weights, wires)

@qp.qnode(dev,interface="jax")
def qnn(weights, features, angles, main_wires, ancilla_wires, rot_wire):
    embedding(weights[:,:,0], features, main_wires)
    qp.Hadamard(rot_wire)
    ProjCtrlPhaseShift(control_wires=ancilla_wires,
                       target_wire=rot_wire,
                       phi=angles[-1])
    for i in range(1, deg):
        RealDiagonalBlockEncoding(
            embedding, wires=main_wires,
            ancilla_wires=ancilla_wires,
            features=features,
            weights=weights[:,:,0])
        ProjCtrlPhaseShift(control_wires=ancilla_wires,
                           target_wire=rot_wire,
                           phi=angles[-i-1])
    qp.Hadamard(rot_wire)
    qp.StronglyEntanglingLayers(weights[:,:,1:], main_wires)
    return qp.state(), qp.probs(rot_wire + ancilla_wires)

@jax.jit
def bce_loss(weights, features, targets):
    state, probs = qnn(weights, features, tanh_div_x_angles, main_wires, ancilla_wires, rot_wire)
    post_sel_state = state[:dim]/jnp.sqrt(probs[0])
    out = jnp.sum(jnp.abs(post_sel_state[:dim//2])**2)
    return - targets * jnp.log(out) - (1 - targets) * jnp.log(1-out)

@jax.jit
def loss_fn(weights, features, targets):
    # We define the loss function to feed our optimizer
    mse_pred = jax.vmap(bce_loss, in_axes=[None, 0, 0])(weights, features, targets)
    loss = jnp.mean(mse_pred)
    return loss

opt = optax.adam(learning_rate=0.1)
max_steps = 100

@jax.jit
def update_step_jit(i, args):
    weights, features, targets, opt_state, print_training = args
    loss_val, grads = jax.value_and_grad(loss_fn)(weights, features, targets)
    updates, opt_state = opt.update(grads, opt_state)
    weights = optax.apply_updates(weights, updates)

    def print_fn():
        jax.debug.print("Step: {i}  Loss: {loss_val}", i=i, loss_val=loss_val)
    # if print_training=True, print the loss every 10 steps
    jax.lax.cond((jnp.mod(i, 10) == 0) & print_training, print_fn, lambda: None)
    return (weights, features, targets, opt_state, print_training)

@jax.jit
def optimization_jit(weights, features, targets, print_training=False):
    opt_state = opt.init(weights)
    args = (weights, features, targets, opt_state, print_training)
    # We loop over update_step_jit max_steps iterations to optimize the parameters
    (weights, _, _, _, _) = jax.lax.fori_loop(0, max_steps+1, update_step_jit, args)
    return weights

weights = pnp.random.default_rng().random(size=(3, main_qubits, 4))
best_weight = optimization_jit(weights, data, labels, print_training=True)

def predict(weights, features):
    state, probs = qnn(weights, features, tanh_div_x_angles, main_wires, ancilla_wires, rot_wire)
    post_sel_state = state[:dim]/jnp.sqrt(probs[0])
    out = jnp.sum(jnp.abs(post_sel_state[:dim//2])**2)
    preds = jnp.where(out>=0.5, 1, 0)
    return preds

def accuracy(weights, features, targets):
    pred = jax.vmap(predict, in_axes=[None, 0])(weights, features)
    diff = jnp.count_nonzero(pred - targets)
    acc = 1-diff/pred.size
    return acc

data = pnp.array(ds.test['4']['inputs'][:200])
labels = (pnp.array(ds.test['4']['labels'][:200])+1)/2

accuracy(best_weight, data, labels)

######################################################################
# Conclusion
# ----------
# 
# Nonlinear functions are difficult to implement in quantum algorithms because quantum dynamics are
# linear: a closed system evolves unitarily. When quantum algorithms exhibit “nonlinear-looking”
# behavior, it typically comes from measurement and conditioning. NTCA makes this mechanism
# systematic: it converts amplitude data into spectral data (via a block encoding), applies a
# polynomial approximation using QSVT, and extracts the transformed amplitudes through postselection.
# The result is a principled way to implement elementwise nonlinear maps on amplitudes without
# violating linearity.
# 
# In this demo, we have implemented the nonlinear amplitude transformation described in Guo et
# al. (2024) and Rattew and Rebentrost (2024) [`3 <https://arxiv.org/abs/2107.10764>`__,
# `4 <https://arxiv.org/abs/2309.09839>`__]. We verified the diagonal amplitude block encoding on a
# toy example, applied a :math:`\tanh` nonlinearity via QSVT, and integrated the activation as a layer
# inside a small quantum classifier trained on downscaled MNIST.
# 
# Key Takeaways: 
# 
# - A systematic bridge from amplitudes to nonlinearity: NTCA enables elementwise maps :math:`\psi_i \mapsto f(\psi_i)` by turning amplitudes into an operator spectrum that QSVT can transform. 
# - Clear resource story: the block-encoding construction uses a constant number of calls to the state-preparation routine, while the main accuracy–cost knob is the polynomial degree :math:`d` (QSVT uses :math:`O(d)` applications of the block encoding). 
# - Broad applicability: while we demonstrated :math:`\tanh`, the same workflow applies to many bounded functions that admit good polynomial approximations on :math:`[-1,1]`. 
# - QML integration: NTCA can be used as an activation layer between trainable “linear” quantum layers, enabling MLP-style architectures in amplitude-based quantum pipelines.
# 