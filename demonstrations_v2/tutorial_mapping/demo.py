r"""

Mapping fermionic operators to qubit operators
==============================================

Simulating quantum systems stands as one of the most anticipated applications of quantum
chemistry with the potential to transform our understanding of chemical and physical systems. These
simulations typically require mapping schemes that transform fermionic representations into qubit
representations. There are a variety of mapping schemes used in quantum computing but the
conventional ones are the Jordan-Wigner, Parity, and Bravyi-Kitaev transformations [#Tranter]_. In
this demo, you will learn about these mapping schemes and their implementation in PennyLane. You
will also learn how to use these mappings in the context of computing the ground state energy
of a molecular system.

.. figure:: ../_static/demonstration_assets/mapping/long_image.png
    :align: center
    :width: 80%
    :target: javascript:void(0)

Jordan-Wigner Mapping
---------------------
The state of a quantum system in the `second quantized <https://en.wikipedia.org/wiki/Second_quantization>`__
formalism is typically represented in the occupation-number basis. For fermions, the occupation
number is either :math:`0` or :math:`1` as a result of the Pauli exclusion principle. The
occupation-number basis states can be represented by a vector that is constructed by
applying the fermionic creation operators, :math:`a^\dagger,` to a vacuum state:

.. math::

    a^\dagger | 0 \rangle = | 1 \rangle.

Similarly, electrons can be removed from a state by applying the fermionic annihilation
operators :math:`a:`

.. math::

    a | 1 \rangle = | 0 \rangle.

An intuitive way to represent a fermionic system in the qubit basis is to store the fermionic
occupation numbers in qubit states. This requires constructing qubit creation and annihilation
operators that can be applied to an initial state to provide the desired occupation number state.
These operators are defined as

.. math::

    Q^{\dagger}_j = \frac{1}{2}(X_j - iY_j),

and

.. math::

    Q_j = \frac{1}{2}(X_j + iY_j),

where :math:`X` and :math:`Y` are Pauli operators. However, an important property of fermionic
creation and annihilation operators is the anti-commutation relations between them, which is not
preserved by directly using the analogous qubit operators.

.. math::

    [a^{\dagger}_j, a^{\dagger}_k]_+ = 0, \:\:\:\:\:\:\: [a_j, a_k]_+=0, \:\:\:\:\:\:\: [a_j, a^{\dagger}_k]_+ = \delta_{jk} I,

where :math:`I` is the identity operator. These relations are essential for capturing the Pauli
exclusion principle which requires the fermionic wave function to be antisymmetric. The
anti-commutation relations between fermionic operators can be incorporated by adding a sequence of
Pauli :math:`Z` operators to the qubit operators (can you verify that?). According to these,
the fermionic creation and annihilation operators can be represented in the basis of Pauli operators:

.. math::

    a_{j}^{\dagger} = \frac{1}{2}(X_j - iY_j) \otimes_{k<j} Z_{k},

and

.. math::

    a_{j} =  \frac{1}{2}(X_j + iY_j) \otimes_{k<j} Z_{k}.

This representation is called the **Jordan-Wigner** mapping. Let's now look at an example
using PennyLane. We map a simple fermionic operator to a qubit operator
using the Jordan-Wigner mapping. First, we define our
`fermionic operator <https://pennylane.ai/qml/demos/tutorial_fermionic_operators>`__,
:math:`a_{5}^{\dagger},` which creates an electron in the fifth orbital. One
way to do this in PennyLane is to use :func:`~.pennylane.fermi.from_string`. We
then map the operator using :func:`~.pennylane.fermi.jordan_wigner`.
"""

import pennylane as qml
qml.capture.enable()
from pennylane.fermi import from_string, jordan_wigner

qubits = 10
fermi_op = from_string("5+")
pauli_jw = jordan_wigner(fermi_op, ps=True)
pauli_jw

###############################################################################
# The long sequence of the :math:`Z` operations can significantly increase the
# resources needed to implement the operator on quantum hardware, as it may require using entangling
# operations across multiple qubits, which can be challenging to implement efficiently. One way to
# avoid having such long tails of :math:`Z` operations is to work in the parity basis where the
# fermionic state stores the parity instead of the occupation number.
#
# Parity Mapping
# --------------
# In the Parity representation, the state of a fermionic system is represented with a binary
# vector that corresponds to the parity of the orbitals. Note that parity for a set of orbitals
# is defined as the sum (mod 2) of the occupation numbers. Let's look at an example using the
# PennyLane function :func:`~.pennylane.qchem.hf_state` to obtain the state of a system with
# :math:`10` orbitals and :math:`5` electrons in both the occupation-number and parity bases.

orbitals = 10
electrons = 5
state_number = qml.qchem.hf_state(electrons, orbitals)
state_parity = qml.qchem.hf_state(electrons, orbitals, basis="parity")

print("State in occupation number basis:\n", state_number)
print("State in parity basis:\n", state_parity)

##############################################################################
# Note that Parity mapping solves the non-locality problem of the parity information by storing
# the parity of orbital :math:`j` in qubit :math:`j` while the occupation information for the
# orbital is stored non-locally. In the parity basis, we cannot represent the creation or
# annihilation of a particle in orbital
# :math:`j` by simply operating with qubit creation or annihilation operators. In fact, the state of
# the :math:`(j − 1)`-th qubit provides information about the occupation  state of qubit :math:`j`
# and whether we need to act with a creation or annihilation operator. Similarly, the creation or
# annihilation of a particle in qubit :math:`j` changes the parity of all qubits following it.
# As a result, the operator that is equivalent to creation and annihilation operators in
# the parity basis is a two-qubit operator acting on qubits :math:`j` and :math:`j − 1,` and
# an update operator which updates the parity of all qubits with index larger than j:
#
# .. math::
#
#     a_{j}^{\dagger} = \frac{1}{2}(Z_{j-1} \otimes X_j - iY_j) \otimes_{k>j} X_{k}
#
# and
#
# .. math::
#
#     a_{j} = \frac{1}{2}(Z_{j-1} \otimes X_j + iY_j) \otimes_{k>j} X_{k}
#
# Let's now look at an example where we map our fermionic operator :math:`a_{5}^{\dagger}` in a
# :math:`10` qubit system with
# Parity mapping using :func:`~.pennylane.fermi.parity_transform` in PennyLane.

qubits = 10
pauli_pr = qml.parity_transform(fermi_op, qubits, ps=True)
pauli_pr

##############################################################################
# It is evident from this example that the Parity mapping doesn't improve upon the
# the Jordan-Wigner mapping as the long :math:`Z` strings are now replaced by :math:`X`
# strings. However, a very important advantage of using parity mapping is the ability to taper two
# qubits by leveraging symmetries of molecular Hamiltonians. You can find
# more information about this in our
# `qubit tapering <https://pennylane.ai/qml/demos/tutorial_qubit_tapering>`__ demo.
# Let's look at an example.

generators = [qml.prod(*[qml.Z(i) for i in range(qubits-1)]), qml.Z(qubits-1)]
paulixops = qml.paulix_ops(generators, qubits)
paulix_sector = [1, 1]
taper_op = qml.taper(pauli_pr, generators, paulixops, paulix_sector)
qml.simplify(taper_op)

###############################################################################
# Note that the tapered operator doesn't include qubit :math:`8` and :math:`9.`
#
# Bravyi-Kitaev Mapping
# ---------------------
# The Bravyi-Kitaev mapping aims to improve the linear scaling of the Jordan-Wigner and Parity
# mappings, in the number of qubits,
# by storing both the occupation number and the parity non-locally. In this formalism, even-labelled
# qubits store the occupation number of orbitals and odd-labelled qubits store parity
# through partial sums of occupation numbers. The corresponding creation and annihilation operators
# are defined `here <https://docs.pennylane.ai/en/stable/code/api/pennylane.fermi.bravyi_kitaev.html>`__.
# Let's use the :func:`~.pennylane.fermi.bravyi_kitaev` function to map our :math:`a_{5}^{\dagger}`
# operator.

pauli_bk = qml.bravyi_kitaev(fermi_op, qubits, ps=True)
pauli_bk

##############################################################################
# It is clear that the local nature of the transformation in the Bravyi-Kitaev mapping helps to
# improve the number of qubits. This advantage becomes even more clear if you
# work with a larger qubit
# system. We now use the Bravyi-Kitaev mapping to construct a qubit Hamiltonian and
# compute its ground state energy with the `VQE <https://pennylane.ai/qml/demos/tutorial_vqe/>`__
# method.
#
# Energy Calculation
# ------------------
# To perform a VQE calculation for a desired Hamiltonian, we need an initial state typically set to
# a Hartree-Fock state, and a set of excitation operators to build an ansatz that allows us to
# obtain the ground state and then compute the expectation value of the Hamiltonian. It is important
# to note that the initial state and the excitation operators should be consistent with the
# mapping scheme used for obtaining the qubit Hamiltonian. Let's now build these three components
# for :math:`H_2` and compute its ground state energy. For this example, we will use the
# Bravyi-Kitaev transformation but similar calculations can be run with the other mappings.
#
# Molecular Hamiltonian
# ^^^^^^^^^^^^^^^^^^^^^
# First, let's build the molecular Hamiltonian. This requires defining the atomic symbols and
# coordinates.

from pennylane import qchem
from jax import numpy as jnp
import jax

jax.config.update("jax_enable_x64", True)

symbols  = ['H', 'H']
geometry = jnp.array([[0.0, 0.0, -0.69434785],
                     [0.0, 0.0,  0.69434785]])

mol = qchem.Molecule(symbols, geometry)

##############################################################################
# We then use the :func:`~pennylane.qchem.fermionic_hamiltonian` function to build
# the fermionic Hamiltonian for our molecule.

h_fermi = qchem.fermionic_hamiltonian(mol)()

##############################################################################
# We now use :func:`~pennylane.fermi.bravyi_kitaev` to transform the fermionic Hamiltonian to its
# qubit representation.

electrons = 2
qubits = len(h_fermi.wires)
h_pauli = qml.bravyi_kitaev(h_fermi, qubits, tol=1e-16)

##############################################################################
# Initial state
# ^^^^^^^^^^^^^
# We now need the initial state that has the correct number of electrons. We use the Hartree-Fock
# state which can be obtained in a user-defined basis by using :func:`~.pennylane.qchem.hf_state` in
# PennyLane. For that, we need to specify the number of electrons, the number of orbitals and the
# desired mapping.

hf_state = qchem.hf_state(electrons, qubits, basis="bravyi_kitaev")

##############################################################################
# Excitation operators
# ^^^^^^^^^^^^^^^^^^^^
# We now build our quantum circuit with the :class:`~.pennylane.UCCSD` ansatz. This ansatz is
# constructed with a set of single and double excitation operators. In PennyLane, we have
# :class:`~.pennylane.SingleExcitation` and :class:`~.pennylane.DoubleExcitation` operators which
# are very efficient, but they are only compatible with the Jordan-Wigner mapping. Here we construct
# the excitation operators manually. We start from the fermionic single and double excitation
# operators defined as [#Yordanov]_
#
# .. math::
#
#     T_i^k(\theta) = \theta(a_k^{\dagger}a_i - a_i^{\dagger}a_k)
#
# and
#
# .. math::
#
#     T_{ij}^{kl}(\theta) = \theta(a_k^{\dagger}a_l^{\dagger}a_i a_j -
#     a_i^{\dagger}a_j^{\dagger}a_k a_l),
#
# where :math:`\theta` is an adjustable parameter. We can easily construct these fermionic
# excitation operators in PennyLane and then map them to the qubit basis with
# :func:`~pennylane.fermi.bravyi_kitaev`, similar to the way we transformed the fermionic
# Hamiltonian.

from pennylane.fermi import from_string

singles, doubles = qchem.excitations(electrons, qubits)

singles_fermi = []
for ex in singles:
    singles_fermi.append(from_string(f"{ex[1]}+ {ex[0]}-")
                       - from_string(f"{ex[0]}+ {ex[1]}-"))

doubles_fermi = []
for ex in doubles:
    doubles_fermi.append(from_string(f"{ex[3]}+ {ex[2]}+ {ex[1]}- {ex[0]}-")
                       - from_string(f"{ex[0]}+ {ex[1]}+ {ex[2]}- {ex[3]}-"))

##############################################################################
# The fermionic operators are now mapped to qubit operators.

singles_pauli = []
for op in singles_fermi:
    singles_pauli.append(qml.bravyi_kitaev(op, qubits, ps=True))

doubles_pauli = []
for op in doubles_fermi:
    doubles_pauli.append(qml.bravyi_kitaev(op, qubits, ps=True))

##############################################################################
# Note that we need to exponentiate these operators to be able to use them in the circuit
# [#Yordanov]_. We also use a set of pre-defined parameters to construct the excitation gates.

params = jnp.array([0.22347661, 0.0, 0.0])

dev = qml.device("default.qubit", wires=qubits)

@qml.qnode(dev)
def circuit(params):
    qml.BasisState(hf_state, wires=range(qubits))

    for i, excitation in enumerate(doubles_pauli):
        qml.exp((excitation * params[i] / 2).operation()), range(qubits)

    for j, excitation in enumerate(singles_pauli):
        qml.exp((excitation * params[i + j + 1] / 2).operation()), range(qubits)

    return qml.expval(h_pauli)

print('Energy =', circuit(params))

##############################################################################
# Using the above circuit, we produce the ground state energy of :math:`H_2` molecule.
#
# Conclusion
# ---------------
# In this demo, we learned about various mapping schemes available in PennyLane and how
# they can be used to convert fermionic operators to qubits operators. We also learned
# the pros and cons associated with each scheme. The Jordan-Wigner mapping provides an intuitive
# approach while parity mapping allows tapering qubits in molecular systems. However, these two
# methods usually give qubit operators with a long chain of Pauli operators, which makes them
# challenging to implement on quantum hardware. The Bravyi-Kitaev mapping, on the other hand,
# emphasizes locality and resource efficiency, making it an attractive option for certain
# applications. Through this demonstration, we recognize the importance of choosing an appropriate
# mapping scheme tailored to the specific problem at hand and the available quantum
# resources. Lastly, we showed how a user can employ these different mappings in ground state energy
# calculations through an example. We would like to encourage the interested readers to run
# calculations for different molecular systems and observe how the scaling in the number of qubits
# is influenced by the selected mapping techniques.
#
# References
# ----------
#
# .. [#Tranter]
#
#      A. Tranter, S. Sofia, *et al.*, "The Bravyi–Kitaev Transformation:
#      Properties and Applications". `International Journal of Quantum Chemistry 115.19 (2015).
#      <https://onlinelibrary.wiley.com/doi/10.1002/qua.24969>`__
#
# .. [#Yordanov]
#
#      Y. S. Yordanov, *et al.*, "Efficient quantum circuits for quantum computational chemistry".
#      `Physical Review A 102.6 (2020).
#      <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.102.062612>`__
#
