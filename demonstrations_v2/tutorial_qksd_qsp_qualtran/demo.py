r"""Using PennyLane and Qualtran to analyze how QSP can improve measurements of molecular properties
====================================================================================================

Want to efficiently measure molecular properties using quantum computers? In this demo, we
outline a powerful workflow for measuring molecular properties on future 
fault-tolerant quantum computers. We demonstrate how to construct the
Quantum Krylov Subspace Diagonalization (QKSD) ground state of the 
water molecule, represented as a sum of Chebyshev polynomials, using a Quantum Signal Processing
(QSP) circuit in PennyLane.
This direct state preparation approach is highly efficient, enabling the measurement of 
properties like reduced density matrices with a constant number of circuit executions and 
thereby avoiding the quadratic scaling costs of other methods.
We then use the integration between PennyLane and Qualtran to perform a detailed resource analysis.

.. figure:: ../_static/demo_thumbnails/opengraph_demo_thumbnails/pennylane-demo-qualtran-covestro-krylov-subspace-paper-open-graph.png
    :align: center
    :width: 70%
    :target: javascript:void(0)

This demo is based on some of the techniques and results of the paper titled
`Molecular Properties from Quantum Krylov Subspace Diagonalization <https://arxiv.org/abs/2501.05286>`_
[#Oumarou]_. In the following sections we will follow this paper and:

* Introduce QKSD briefly.
* Build a circuit that uses QSP to prepare the QKSD ground-state.
* Simulate circuits that estimate the one-particle reduced density matrix of a molecular system from the QKSD ground-state.
* Use the `PennyLane-Qualtran integration <https://pennylane.ai/qml/demos/tutorial_how_to_use_qualtran_with_pennylane>`_ to count relevant circuit resources and demonstrate linear resource scaling with respect to the Krylov dimension, :math:`D`.
"""

######################################################################
# Quantum Krylov Subspace Diagonalization
# ---------------------------------------
#
# QKSD is a notable candidate algorithm for fault-tolerant quantum computing that approximates the
# low-lying eigenvalues and eigenstates of a large Hamiltonian by focusing on a smaller subspace of the same
# Hamiltonian [#QKSD]_. The general steps to perform QKSD are:
#
# * Obtain the Hamiltonian, :math:`\hat{H}`, describing a system of interest, e.g., a molecule.
# * Define a `Krylov subspace <https://en.wikipedia.org/wiki/Krylov_subspace>`_ :math:`\mathcal{K}`
#   defined by the span of quantum states :math:`| \psi_{k = 0, \dots, D-1} \rangle`, where :math:`D` is the Krylov 
#   dimension and :math:`| \psi_k \rangle` can be efficiently prepared on a quantum computer.
# * Project :math:`\hat{H}` into the Krylov subspace with a quantum computer, resulting in a new,
#   "smaller" Hamiltonian :math:`\tilde{H}` whose eigenvalues and eigenstates approximate those of
#   :math:`\hat{H}`.
# * Calculate the overlap matrix, :math:`\tilde{S}`, defined by the inner-products of each element
#   of the Krylov subspace.
# * Solve the generalized eigenvalue problem:
#   :math:`\tilde{H}c^m = E_m \tilde{S}c^m` on a classical computer,
#   where :math:`c^m` are the eigenstates of :math:`\tilde{H}`.
#
# The result of this generalized eigenvalue problem gives approximations of the low-lying
# eigenenergies and eigenstates of the Hamiltonian [#QKSD]_, including the Krylov ground-state,
# :math:`|\Psi_0\rangle = \sum_k c^0_k | \psi_k \rangle`.
# Such an eigenstate is a linear combination of the states spanning the Krylov subspace and can
# be prepared with `QSP <https://pennylane.ai/qml/demos/function_fitting_qsp>`_, as we will see
# shortly.
#
# Let's now apply the above formalism to a familiar example: the :math:`H_2O` molecule. We will use
# the Jordan-Wigner mapping of the :math:`H_2O` Hamiltonian with an active space of 4 electrons in 4
# molecular orbitals in the cc-pVDZ basis. To save time on this computationally-expensive basis, we
# provide pre-calculated coefficients and Pauli words for this Hamiltonian below. It is also possible
# to obtain these values for other molecules and basis sets using either
# `PennyLane Datasets <https://pennylane.ai/datasets/h2o-molecule>`_ or the :mod:`pennylane.qchem`
# module. We use the pre-calculated results below to create a :class:`~pennylane.ops.Hamiltonian` object.

import pennylane as qml
qml.capture.enable()
import numpy as np

coeffs = [-2.6055398817649027, 0.4725512342017669, 0.06908378485045799, 0.06908378485045799, 0.42465877221739423, 0.040785962774025165, 0.02016371425557293, 0.33650704944175736, 0.059002396986463035, 0.059002396986463035, 0.2401948108687125, 0.2696430914655511, 0.04971997220167805, 0.04971997220167805, 0.2751875205710399, 0.30033122257656397, 0.22551650339251875]
paulis = [qml.GlobalPhase(0, 0), qml.Z(0), qml.Y(0) @ qml.Z(1) @ qml.Y(2), qml.X(0) @ qml.Z(1) @ qml.X(2), qml.Z(1), qml.Z(2), qml.Z(3), qml.Z(0) @ qml.Z(1), qml.Y(0) @ qml.Y(2), qml.X(0) @ qml.X(2), qml.Z(0) @ qml.Z(2), qml.Z(0) @ qml.Z(3), qml.Y(0) @ qml.Z(1) @ qml.Y(2) @ qml.Z(3), qml.X(0) @ qml.Z(1) @ qml.X(2) @ qml.Z(3), qml.Z(1) @ qml.Z(2), qml.Z(1) @ qml.Z(3), qml.Z(2) @ qml.Z(3)]
hamiltonian = qml.Hamiltonian(coeffs, paulis)

######################################################################
# Next, we will define the Krylov subspace, :math:`\mathcal{K}`.
# We first define the states that span :math:`\mathcal{K}`: 
#
# .. math:: \ket{\psi_k} = T_k(H)\ket{\psi_0},
#
# where :math:`T_k` is the :math:`k`-th Chebyshev polynomial and :math:`|\psi_0\rangle` is the
# Hartree-Fock state. The Chebyshev polynomials are defined by:
#
# .. math:: T_0(\hat{H}) = \mathbb{1}
# .. math:: T_1(\hat{H}) = \hat{H}
# .. math:: T_{n+1}(\hat{H}) = 2 \hat{H} T_n(\hat{H}) - T_{n-1}(\hat{H})
#
# Other subspace definitions are possible [#QKSD]_, but we choose
# `Chebyshev polynomials <https://en.wikipedia.org/wiki/Chebyshev_polynomials>`_
# for convenience. The reason for this is that we plan to
# prepare the Krylov ground-state using QSP, which directly implements Chebyshev polynomials.
# Other types of functions or polynomials would first need to be converted into
# Chebyshev polynomials, adding more complexity to our program.
# 
# With the states defined, the Krylov subspace of dimension :math:`D` is then: 
#
# .. math:: \mathcal{K} = \text{span}(\{\ket{\psi_k}\}_{k=0}^{D-1}).
#
# After projecting :math:`\hat{H}` into :math:`\mathcal{K}` and solving the generalized eigenvalue
# problem described above, we obtain the Krylov ground-state [#QKSD]_,
# 
# .. math:: \ket{\Psi_0} = \sum_{k=0}^{D-1} c^0_k \ket{\psi_k} = \sum_{k=0}^{D-1} c_k^0 T_k(H) \ket{\psi_0},
#
# where :math:`c_k^m` is the :math:`k`-th coefficient of the :math:`m`-th eigenstate of
# :math:`\tilde{H}`. We pre-calculate these coefficients and use them to obtain the
# QSP rotation angles that prepare the QKSD ground-state in the section below.
#
# Using QSP to directly create the QKSD ground-state
# --------------------------------------------------
#
# Only using QKSD to calculate one-particle or two-particle reduced density matrices
# results in a quadratic scaling of the number of expectation
# values that need to be measured with respect to the Krylov dimension, :math:`D` [#Oumarou]_.
# Instead, reference [#Oumarou]_ shows that it is possible to reduce this to constant scaling by preparing
# :math:`|\Psi_0\rangle` via QSP and measuring individual terms of the Hamiltonian.
# The QSP circuit we will use, :math:`U_\text{QSP}`, is defined as a series of alternating
# block-encoding, :math:`U_\text{BE}`, and rotation operators, :math:`S`, such that
# [#qspref]_ [#Oumarou]_ :
# 
# .. math:: U_\text{QSP} = S(\phi_0)\prod_k^{d-1}{U_\text{BE}(\hat{H})S(\phi_k)}.
# 
# This QSP circuit prepares a Chebyshev polynomial of the block-encoded Hamiltonian:
#
# .. math:: U_\text{QSP} \ket{\psi_0} \ket{0}_a = \sum_{i=0}^{D-1} c_i T_i(H) \ket{\psi_0} \ket{0}_a + \beta\ket{\perp}.
#
# With specific rotation angles, :math:`\phi`, :math:`U_\text{QSP}` implements the 
# Chebyshev polynomial corresponding to the Krylov ground-state:
#
# .. math:: U_\text{QSP}|\psi_0\rangle|0\rangle = \ket{\Psi_0}\ket{0}_a + \beta\ket{\perp}
#
# For more details about implementing polynomials of block-encoded Hamiltonians, block-encoding
# operators, and rotation operators, see the
# `Intro to QSVT demo <https://pennylane.ai/qml/demos/tutorial_intro_qsvt>`_.
#
#
# Measuring the reduced density matrices
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The elements of the one-particle reduced density matrix, :math:`\gamma_{pq}`, are obtained by
# measuring the expectation value of the fermionic one-particle excitation operators acting on
# the Krylov ground-state [#Oumarou]_. That is,
#
# .. math:: \gamma_{pq} = \langle\Psi_0 | a^{\dagger}_p a_q | \Psi_0\rangle,
#
# where :math:`a^{\dagger}` and :math:`a` are the creation and annihilation operators, respectively.
# The indices :math:`p` and :math:`q` determine in which orbital particles are added or removed. 
#
# We can obtain these expectation values by measuring the right observables and post-processing
# the result according to reference [#Oumarou]_. Specifically, we use the Jordan-Wigner mapping of
# the fermionic one-particle excitation operators and measure the resulting Pauli word observables.
# By measuring each Pauli word, :math:`P` and a product of the Pauli word and a reflection around 0,
# :math:`R_0P`, we obtain:  
#
# .. math:: o_1 = \bra{\psi_0}\bra{0}_aU_\text{QSP}^{*}P U_\text{QSP}\ket{0}_a\ket{\psi_0}
# 
# .. math:: o_2 = \bra{\psi_0}\bra{0}_aU_\text{QSP}^{*}R_0 PU_\text{QSP}\ket{0}_a\ket{\psi_0}
#
# These measurements can then be combined according to [#Oumarou]_
#
# .. math:: 2\langle \Psi_0 |P|\Psi_0\rangle = \eta^2(o_1 + o_2),
#
# giving the value of :math:`\gamma_{pq}` for the Hamiltonian in the Jordan-Wigner mapping. Note 
# that it is also possible to obtain the two-particle reduced density matrix,
# :math:`\Gamma_{pqrs}`, by following the same protocol for fermionic two-particle excitation
# operators, :math:`a^{\dagger}_p a^{\dagger}_q a_r a_s`.
#
# Now let's see how to build the QSP circuit and measurements in PennyLane.
#
# Defining the QSP circuit
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# To define the QSP circuit, we first create a function to implement the rotations, :math:`S`. The
# block-encoding, :math:`U_\text{BE}`, will be implemented via the :class:`~pennylane.PrepSelPrep`
# class. We then create a QSP template to alternate between :math:`U_\text{BE}` and :math:`S`.
# Alternating between these operations
# will prepare a Chebyshev polynomial of the input block-encoded matrix.

def rotation_about_reflection_axis(angle, wires):
    """Rotation operation, S(phi_k)"""
    qml.ctrl(qml.PauliX(wires[0]), wires[1:], (0,) * len(wires[1:]))
    qml.RZ(angle, wires[0])
    qml.ctrl(qml.PauliX(wires[0]), wires[1:], (0,) * len(wires[1:]))

def qsp(lcu, angles, rot_wires, prep_wires):
    """Applies QSP by alternating between S(phi_k) and U_{QSP}"""
    for angle in angles[::-1][:-1]:
        rotation_about_reflection_axis(angle, rot_wires)
        qml.PrepSelPrep(lcu, control=prep_wires)
    rotation_about_reflection_axis(angles[0], rot_wires)

######################################################################
# Since QSP can only produce fixed-parity real Chebyshev polynomials [#qspref]_ and our QKSD ground-state has
# complex polynomials, we also create a template that combines the real and imaginary parts of the
# polynomial.

def qsp_poly_complex(lcu, angles_real, angles_imag, ctrl_wire, rot_wires, prep_wires):
    qml.H(ctrl_wire)
    qml.ctrl(qsp, ctrl_wire, 0)(lcu, angles_real, rot_wires, prep_wires)
    qml.ctrl(qsp, ctrl_wire, 1)(lcu, angles_imag, rot_wires, prep_wires)
    qml.H(ctrl_wire)

######################################################################
# Additionally, we require a template to perform the :math:`R_0` reflection for our measurements.

def reflection(wire, ctrl_wires):
    """Reflection operator, R_0"""
    qml.ctrl(qml.X(wire), ctrl_wires, [0] * len(ctrl_wires))
    qml.Z(wire)
    qml.ctrl(qml.X(wire), ctrl_wires, [0] * len(ctrl_wires))

######################################################################
# We will also need to split even- and odd-parity polynomial terms, which is captured in the
# final circuit below. Since we have split the polynomial and implemented each part via a separate QSP
# circuit, we will also need to split the rotation angles that implement each of these polynomials.
# The final circuit below accepts a separate argument for odd/even and real/imaginary angles:
# ``odd_real``, ``odd_imag``, ``even_real``, and ``even_imag``.
#
# We also define an argument for an input observable, ``obs``, and an argument defining whether to
# measure the reflection of the observable, ``measure_reflection``. 
#
# With these building blocks in place, we can now define the overarching QNode
# that will apply the QSP circuit to the reference state :math:`|\psi_0\rangle`
# and return measurements of the elements of
# the one-particle reduced density matrix:

dev = qml.device("lightning.qubit")
ref_state = qml.qchem.hf_state(electrons=4, orbitals=4) 

@qml.qnode(dev)
def krylov_qsp(lcu, even_real, even_imag, odd_real, odd_imag, obs, measure_reflection=False):
    """Prepares the Krylov ground-state by applying QSP with the input angles.
    Then measures the expectation value of the desired observable. 
    """
    num_aux = int(np.log(len(lcu.operands)) / np.log(2)) + 1

    start_wire = hamiltonian.wires[-1] + 1
    rot_wires = list(range(start_wire, start_wire + num_aux + 1))
    prep_wires = rot_wires[1:]
    ctrl_wires = [prep_wires[-1] + 1, prep_wires[-1] + 2]
    rdm_ctrl_wire = ctrl_wires[-1] + 1

    #prepare the reference state
    for i in hamiltonian.wires:
        qml.X(i)

    if measure_reflection: # preprocessing for reflection measurement
        qml.X(rdm_ctrl_wire)

    qml.H(ctrl_wires[0])
    qml.ctrl(qsp_poly_complex, ctrl_wires[0], 0)(lcu, even_real, even_imag, ctrl_wires[1],
                                                 rot_wires, prep_wires)
    qml.ctrl(qsp_poly_complex, ctrl_wires[0], 1)(lcu, odd_real, odd_imag, ctrl_wires[1],
                                                 rot_wires, prep_wires)
    qml.H(ctrl_wires[0])

    # measurements
    if measure_reflection:
        return qml.expval(qml.prod(reflection)(rdm_ctrl_wire, set(ctrl_wires+prep_wires))@obs)
    return qml.expval(obs)


######################################################################
# We can now use this circuit to build one-particle and two- particle reduced density matrices.
# We can obtain the Jordan-Wigner mapping of the fermionic operators via PennyLane using the
# :func:`~pennylane.fermi.from_string` and :func:`~pennylane.jordan_wigner` functions as follows:

Epq = qml.fermi.from_string('0+ 0-')
obs = qml.jordan_wigner(Epq)

######################################################################
# For this demo, we used a Krylov subspace dimension of :math:`D=15` and pre-computed QSP angles
# that implement the corresponding sum of Chebyshev polynomials :math:`\sum_{k=0}^{D-1}c_k^0 T_k(\hat{H})`.
# For a given polynomial it is possible to obtain the QSP angles using :func:`~pennylane.poly_to_angles`.
#
# The angles below will produce the QKSD ground-state :math:`|\Psi_0\rangle` via QSP. Since QSP
# can only produce fixed-parity real Chebyshev polynomials [#qspref]_ and our QKSD ground-state has
# mixed-parity complex polynomials, we split them and apply separately.

even_real = np.array([3.11277458, 2.99152757, 3.15307452, 3.40611024, 3.00166196, 3.03597059, 3.25931224, 3.04073693, 3.25931224, 3.03597059, 3.00166196, 3.40611024, 3.15307452, 2.99152757, -40.86952257])
even_imag = np.array([3.17041073, 3.29165774, 3.13011078, 2.87707507, 3.28152334, 3.24721472, 3.02387307, 3.24244837, 3.02387307, 3.24721472, 3.28152334, 2.87707507, 3.13011078, 3.29165774, -47.09507173])
odd_real = np.array([3.26938242, 3.43658284, 3.17041296, 3.10158929, 3.22189574, 2.93731798, 3.25959312, 3.25959312, 2.93731798, 3.22189574, 3.10158929, 3.17041296, 3.43658284, -37.57132208])
odd_imag = np.array([3.01380289, 2.84660247, 3.11277234, 3.18159601, 3.06128956, 3.34586733, 3.02359219, 3.02359219, 3.34586733, 3.06128956, 3.18159601, 3.11277234, 2.84660247, -44.11008691])
######################################################################
# We then measure the QSP circuit using these angles and post-process according to
# the equations above:

P = krylov_qsp(hamiltonian, even_real, even_imag, odd_real, odd_imag, obs=obs)
RP = krylov_qsp(hamiltonian, even_real, even_imag, odd_real, odd_imag, obs=obs,
                measure_reflection=True)

print("P:", P)
print("RP:", RP)

lambda_lcu = np.sum(np.abs(coeffs))
coherent_result = 2*lambda_lcu*(P+RP)
print("Coherent result:", coherent_result)

######################################################################
# Analyzing with Qualtran
# -----------------------
# 
# We can analyze the resources and flow of this program by using the Qualtran call graph.
# We first convert the PennyLane circuit to a Qualtran bloq using :func:`~pennylane.to_bloq` and
# then use the call graph to count the required gates:

bloq = qml.to_bloq(krylov_qsp, map_ops=False,
    even_real=even_real, even_imag=even_imag,
    odd_real=odd_real, odd_imag=odd_imag,
    lcu=hamiltonian, obs=obs)

######################################################################
# We can then use Qualtran tools to analyze and process the gate counts of the circuit. Below,
# we use the ``call_graph`` to obtain a breakdown of the gates used, applying the
# ``generalize_rotation_angle`` generalizer to neatly group all rotations for clearer viewing:

from qualtran.resource_counting.generalizers import generalize_rotation_angle

graph, sigma = bloq.call_graph(generalizer=generalize_rotation_angle)
print("--- Gate counts ---")
for gate, count in sigma.items():
    print(f"{gate}: {count}")

######################################################################
# As explained in [#Oumarou]_,
# increasing :math:`D` improves the accuracy of the Krylov minimal energy
# compared to the true ground state energy. This extra accuracy is paid for by requiring additional gates.
# Let's see how the number of gates increases with increasing Krylov subspace dimension. We can
# increase the Krylov subspace dimension by increasing the number of terms in our Chebyshev polynomial,
# captured in this demo via the angles variables. Let's try :math:`D=20`
# by setting the number of terms in these angles to 20. As the resource estimation is independent of
# the exact angle values, we are able to set them randomly instead of recomputing the formally:

even_real = even_imag = odd_real = odd_imag = np.random.random(20)

######################################################################
# We then repeat the gate counts and see they have increased:

bloq = qml.to_bloq(krylov_qsp, map_ops=False, even_real=even_real,
                   even_imag=even_imag, odd_real=odd_real,
                   odd_imag=odd_imag, lcu=hamiltonian, obs=obs)

graph, sigma = bloq.call_graph(generalizer=generalize_rotation_angle)
print("--- Gate counts ---")
for gate, count in sigma.items():
    print(f"{gate}: {count}")

######################################################################
# We can plot how the number of gates increases with the Krylov dimension to see if it is linear
# as described in reference [#Oumarou]_. Below we plot how the ``Toffoli``, ``CNOT``, and ``X`` 
# gate count increase with the Krylov dimension:

import matplotlib.pyplot as plt
from qualtran.bloqs.basic_gates import Toffoli, CNOT, XGate

def count_cnots(krylov_dimension):
    even_real = even_imag = odd_real = odd_imag = np.random.random(krylov_dimension)
    bloq = qml.to_bloq(krylov_qsp, map_ops=False, even_real=even_real,
                       even_imag=even_imag, odd_real=odd_real,
                       odd_imag=odd_imag, lcu=hamiltonian, obs=obs)
    _, sigma = bloq.call_graph(generalizer=generalize_rotation_angle)
    return sigma

Ds = [10, 20, 30, 40, 50]
sigmas = [count_cnots(D) for D in Ds]

plt.style.use("pennylane.drawer.plot")
plt.scatter(Ds, [sigma[CNOT()] for sigma in sigmas], label='CNOTs')
plt.scatter(Ds, [sigma[Toffoli()] for sigma in sigmas], label = 'Toffolis')
plt.scatter(Ds, [sigma[XGate()] for sigma in sigmas], label = 'X Gates')
plt.xlabel('Krylov Subspace Dimension, D')
plt.ylabel('Number of Gates')
plt.legend()
plt.show()

######################################################################
# As expected, the gate counts increase linearly with the Krylov subspace dimension.
# 
# We can also show the call graph of the circuit, a diagrammatic representation of how each operation
# in the circuit calls other operations. This can be useful to understand how a circuit works or 
# which sections are resource-intensive.
#
# The first layer of this circuit, before decomposing the operations, contains many
# ``RZ`` gates controlled on two qubits with different angles. As we saw above, we can group similar
# operations together and simplify the view. To this end, we first write a custom Qualtran generalizer
# that will combine controlled-``RZ`` gates: 

import attrs
import sympy

from qualtran.resource_counting.generalizers import _ignore_wrapper

PHI = sympy.Symbol(r'\phi')

def generalize_ccrz(b):
    from qualtran.bloqs.basic_gates import Rz
    from qualtran.bloqs.mcmt.controlled_via_and import ControlledViaAnd

    if isinstance(b, ControlledViaAnd) and isinstance(b.subbloq, Rz):
        return attrs.evolve(b, subbloq = Rz(angle=PHI))
    
    return _ignore_wrapper(generalize_ccrz, b)

######################################################################
# We can then use this generalizer to draw our call graph using Qualtran's ``show_call_graph``
#
# .. code-block:: python
#
#     from qualtran.drawing import show_call_graph
#
#     graph, sigma = bloq.call_graph(generalizer=generalize_ccrz)
#     show_call_graph(qpe_bloq, max_depth=1)
#
# .. figure:: ../_static/demonstration_assets/qksd_qsp_qualtran/generalize_ccrz_call_graph.svg
#     :align: center
#     :width: 50%

######################################################################
# Conclusion
# ----------
# This demo demonstrated how to use PennyLane to measure the one-particle
# reduced density matrix of the water molecule with a linearly-scaling
# number of operations and how to integrate with Qualtran to demonstrate these resource requirements.
# This was done by using Quantum Krylov Subspace Diagonalization techniques to compress a
# complicated molecular Hamiltonian, finding its ground-state classically, and then using Quantum Signal
# Processing to efficiently measure its one-particle reduced density matrix. 
#
# We then used the integration between PennyLane and Qualtran to perform a detailed resource analysis.
# By converting our PennyLane ``QNode`` into a Qualtran ``Bloq``, we precisely 
# counted the required quantum gates and confirmed that the number of gates scales linearly with the 
# Krylov subspace dimension, a crucial result from the source paper [#Oumarou]_. This favorable scaling 
# highlights the potential of QSP-based techniques for tackling meaningful quantum chemistry 
# problems and provides a practical framework for bridging the gap between theoretical algorithms 
# and their implementation on future quantum hardware.

######################################################################
# References
# -----------
#
# .. [#qspref]
#
#     Guang Hao Low, Isaac L. Chuang
#     "Hamiltonian Simulation by Qubitization",
#     `arXiv:1610.06546 <https://arxiv.org/abs/1610.06546v3>`__, 2019.
# 
# .. [#Oumarou]
#
#     Oumarou Oumarou, Pauline J. Ollitrault, Cristian L. Cortes, Maximilian Scheurer, Robert M. Parrish, Christian Gogolin
#     "Molecular Properties from Quantum Krylov Subspace Diagonalization"
#     `arXiv:2501.05286 <https://arxiv.org/abs/2501.05286>`__, 2025.
#
# .. [#QKSD]
#
#     Nobuyuki Yoshioka, Mirko Amico, William Kirby, Petar Jurcevic, Arkopal Dutt, Bryce Fuller, Shelly Garion,
#     Holger Haas, Ikko Hamamura, Alexander Ivrii, Ritajit Majumdar, Zlatko Minev, Mario Motta, Bibek Pokharel,
#     Pedro Rivero, Kunal Sharma, Christopher J. Wood, Ali Javadi-Abhari, and Antonio Mezzacapo
#     "Diagonalization of large many-body Hamiltonians on a quantum processor",
#     `arXiv:2407.14431 <https://arxiv.org/abs/2407.14431>`__, 2024.
#
