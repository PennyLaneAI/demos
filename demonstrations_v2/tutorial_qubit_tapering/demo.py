r"""

Qubit tapering
==============

.. meta::
    :property="og:description": Learn how to taper off qubits
    :property="og:image": https://pennylane.ai/qml/_static/demonstration_assets/qubit_tapering.png

.. related::
    tutorial_quantum_chemistry Building molecular Hamiltonians
    tutorial_vqe A brief overview of VQE
    tutorial_givens_rotations Givens rotations for quantum chemistry
    tutorial_adaptive_circuits Adaptive circuits for quantum chemistry
    tutorial_differentiable_HF Differentiable Hartree-Fock

The performance of variational quantum algorithms is considerably limited by the number of qubits
required to represent wave functions. In the context of quantum chemistry, this
limitation hinders the treatment of large molecules with algorithms such as the :doc:`variational quantum
eigensolver (VQE) <demos/tutorial_vqe>`. Several approaches have been developed to reduce the qubit requirements for
quantum chemistry calculations. In this tutorial, we demonstrate the symmetry-based qubit
tapering approach which allows reducing the number of qubits required to perform molecular quantum
simulations based on the :math:`\mathbb{Z}_2` symmetries present in molecular Hamiltonians
[#bravyi2017]_ [#setia2019]_.

A molecular Hamiltonian in the qubit basis can be expressed as a linear combination of Pauli words
as

.. math:: H = \sum_{i=1}^r h_i P_i,

where :math:`h_i` is a real coefficient and :math:`P_i` is a tensor product of Pauli and
identity operators acting on :math:`M` qubits

.. math:: P_i \in \pm \left \{ I, X, Y, Z \right \} ^ {\bigotimes M}.

The main idea in the symmetry-based qubit tapering approach is to find a unitary operator :math:`U`
that transforms :math:`H` to a new Hamiltonian :math:`H'` which has the same eigenvalues as
:math:`H`

.. math:: H' = U^{\dagger} H U = \sum_{i=1}^r c_i \mu_i,

such that each :math:`\mu_i` term in the new Hamiltonian always acts trivially, e.g., with an
identity or a Pauli operator, on a set of qubits. This allows tapering-off those qubits from the
Hamiltonian.

For instance, consider the following Hamiltonian

.. math:: H' = Z_0 X_1 - X_1 + Y_0 X_1,

where all terms in the Hamiltonian act on the second qubit with the :math:`X` operator. It is
straightforward to show that each term in the Hamiltonian commutes with :math:`I_0 X_1` and the
ground-state eigenvector of :math:`H'` is also an eigenvector of :math:`I_0 X_1` with eigenvalues
:math:`\pm 1.` We can also rewrite the Hamiltonian as

.. math:: H' = (Z_0 I_1 - I_0 I_1 + Y_0 I_1) I_0 X_1,

which gives us

.. math:: H'|\psi \rangle = \pm1 (Z_0 I_1 - I_0 I_1 + Y_0 I_1)|\psi \rangle,

where :math:`|\psi \rangle` is an eigenvector of :math:`H'.` This means that the Hamiltonian
:math:`H` can be simplified as

.. math:: H_{tapered} = \pm1 (Z_0 - I_0 + Y_0).

The tapered Hamiltonian :math:`H_{tapered}` has the eigenvalues

.. math:: [-2.41421, 0.41421],

and

.. math::  [2.41421, -0.41421],

depending on the value of the :math:`\pm 1` prefactor. The eigenvalues of the original Hamiltonian
:math:`H` are

.. math:: [2.41421, -2.41421,  0.41421, -0.41421],

which are thus reproduced by the tapered Hamiltonian.

More generally, we can construct the unitary :math:`U` such that each :math:`\mu_i` term acts with a
Pauli-X operator on a set of qubits
:math:`\left \{ j \right \}, j \in \left \{ l, ..., k \right \}` where :math:`j` is the qubit label.
This guarantees that each term of the transformed Hamiltonian commutes with each of the Pauli-X
operators applied to the :math:`j`-th qubit:

.. math:: [H', X^j] = 0,

and the eigenvectors of the transformed Hamiltonian :math:`H'` are also eigenvectors of each of the
:math:`X^{j}` operators. Then we can factor out all of the :math:`X^{j}` operators from the
transformed Hamiltonian and replace them with their eigenvalues :math:`\pm 1.` This gives us a
set of tapered Hamiltonians depending on which eigenvalue :math:`\pm 1` we chose for each of the
:math:`X^{j}` operators. For  instance, in the case of two tapered qubits, we have four eigenvalue
sectors: :math:`[+1, +1]`, :math:`[-1, +1]`, :math:`[+1, -1],` :math:`[-1, -1].` In these tapered
Hamiltonians, the set of :math:`\left \{ j \right \}, j \in \left \{ l, ..., k \right \}` qubits
are eliminated. For tapered molecular Hamiltonians, it is possible to determine the optimal sector
of the eigenvalues that corresponds to the ground state. This is explained in more detail in the
following sections.

The unitary operator :math:`U` can be constructed as a
`Clifford <https://en.wikipedia.org/wiki/Clifford_gates>`__ operator [#bravyi2017]_

.. math:: U = \Pi_j \left [\frac{1}{\sqrt{2}} \left (X^{q(j)} + \tau_j \right) \right],

where :math:`\tau` denotes the generators of the symmetry group of :math:`H` and
:math:`X^{q}` operators act on those qubits that will be ultimately tapered off from
the Hamiltonian. The symmetry group of the Hamiltonian is defined as an Abelian group of Pauli words that commute
with each term in the Hamiltonian (excluding :math:`−I`). The
`generators <https://en.wikipedia.org/wiki/Generating_set_of_a_group>`__ of the symmetry group are
those elements of the group that can be combined, along with their inverses, to create any other
member of the group.

Let's use the qubit tapering method and obtain the ground state energy of the `Helium hydride
cation <https://en.wikipedia.org/wiki/Helium_hydride_ion>`__ :math:`\textrm{HeH}^+.`

Tapering the molecular Hamiltonian
----------------------------------

In PennyLane, a :doc:`molecular Hamiltonian <demos/tutorial_quantum_chemistry>` can be created by specifying the atomic symbols and
coordinates.
"""
import pennylane as qml
qml.capture.enable()
from jax import numpy as jnp
import jax

jax.config.update("jax_enable_x64", True)
symbols = ["He", "H"]
geometry = jnp.array([[0.00000000, 0.00000000, -0.87818361],
                     [0.00000000, 0.00000000,  0.87818362]])

molecule = qml.qchem.Molecule(symbols, geometry, charge=1)
H, qubits = qml.qchem.molecular_hamiltonian(molecule)
H

##############################################################################
# This Hamiltonian contains 27 terms where each term acts on up to four qubits.
#
# We can now obtain the symmetry generators and the :math:`X^{j}` operators that are
# used to construct the unitary :math:`U` operator that transforms the :math:`\textrm{HeH}^+`
# Hamiltonian. In PennyLane, these are constructed by using the
# :func:`~.pennylane.symmetry_generators` and :func:`~.pennylane.paulix_ops` functions.

generators = qml.symmetry_generators(H)
paulixops = qml.paulix_ops(generators, qubits)

for idx, generator in enumerate(generators):
    print(f"generator {idx+1}: {generator}, paulix_op: {paulixops[idx]}")

##############################################################################
# Once the operator :math:`U` is applied, each of the Hamiltonian terms will act on the qubits
# :math:`q_2, q_3` either with the identity or with a Pauli-X operator. For each of these qubits,
# we can simply replace the Pauli-X operator with one of its eigenvalues :math:`+1` or :math:`-1.`
# This results in a total number of :math:`2^k` Hamiltonians, where :math:`k` is the number of
# tapered-off qubits and each Hamiltonian corresponds to one eigenvalue sector. The optimal sector
# corresponding to the ground-state energy of the molecule can be obtained by using the
# :func:`~.pennylane.qchem.optimal_sector` function.


n_electrons = 2
paulix_sector = qml.qchem.optimal_sector(H, generators, n_electrons)
print(paulix_sector)

##############################################################################
# The optimal eigenvalues are :math:`-1, -1` for qubits :math:`q_2, q_3,` respectively. We can now
# build the tapered Hamiltonian with the :func:`~.pennylane.taper` function which
# constructs the operator :math:`U,` applies it to the Hamiltonian and finally tapers off the
# qubits :math:`q_2, q_3` by replacing the Pauli-X operators acting on those qubits with the optimal
# eigenvalues.

H_tapered = qml.taper(H, generators, paulixops, paulix_sector)
H_tapered_coeffs, H_tapered_ops = H_tapered.terms()
H_tapered = qml.Hamiltonian(jnp.real(jnp.array(H_tapered_coeffs)), H_tapered_ops)
print(H_tapered)

##############################################################################
# The new Hamiltonian has only 9 non-zero terms acting on only 2 qubits! We can verify that the
# original and the tapered Hamiltonian both give the correct ground state energy of the
# :math:`\textrm{HeH}^+` cation, which is :math:`-2.862595242378` Ha computed with the full
# configuration interaction (FCI) method. In PennyLane, it's possible to build a sparse matrix
# representation of Hamiltonians. This allows us to directly diagonalize them to obtain exact values
# of the ground-state energies.

H_sparse = qml.SparseHamiltonian(H.sparse_matrix(), wires=H.wires)
H_tapered_sparse = qml.SparseHamiltonian(H_tapered.sparse_matrix(), wires=H_tapered.wires)

print("Eigenvalues of H:\n", qml.eigvals(H_sparse, k=16))
print("\nEigenvalues of H_tapered:\n", qml.eigvals(H_tapered_sparse, k=4))

##############################################################################
# Note that a second-quantized Hamiltonian is independent of the number of electrons and its
# eigenspectrum contains the energies of the neutral and charged molecules. Therefore, the
# smallest eigenvalue returned by :func:`~.pennylane.eigvals` for a molecular Hamiltonian
# might correspond to the neutral or charged molecule. While in the case of :math:`\textrm{HeH}^+,`
# qubit tapering allows specifying the optimal sector of the eigenvectors corresponding only to the
# correct number of electrons, it is generally guaranteed that the optimal sector covers all
# eigenvectors with the correct number of electrons, but may contain additional eigenvectors of
# different charge. Therefore, the ground-state energy of the :math:`\textrm{HeH}^+` cation is
# the smallest eigenvalue of the tapered Hamiltonian.
#
# Tapering the reference state
# ----------------------------
# The ground state Hartree-Fock energy of :math:`\textrm{HeH}^+` can be computed by directly
# applying the Hamiltonians to the Hartree-Fock state. For the tapered Hamiltonian, this requires
# transforming the Hartree-Fock state with the same symmetries obtained for the original
# Hamiltonian. This reduces the number of qubits in the Hartree-Fock state to match that of the
# tapered Hamiltonian. It can be done with the :func:`~.pennylane.qchem.taper_hf` function.

state_tapered = qml.qchem.taper_hf(generators, paulixops, paulix_sector,
                                   num_electrons=n_electrons, num_wires=len(H.wires))
print(state_tapered)

##############################################################################
# Recall that the original Hartree-Fock state for the :math:`\textrm{HeH}^+` cation is
# :math:`[1 1 0 0].` We can now generate the qubit representation of these states and compute the
# Hartree-Fock energies for each Hamiltonian.

dev = qml.device("default.qubit", wires=H.wires)
@qml.qnode(dev, interface="jax")
def circuit():
    qml.BasisState(jnp.array([1, 1, 0, 0]), wires=H.wires)
    return qml.state()

qubit_state = circuit()
HF_energy = qubit_state.T @ H.sparse_matrix().toarray() @ qubit_state
print(f"HF energy: {jnp.real(HF_energy):.8f} Ha")

dev = qml.device("lightning.qubit", wires=H_tapered.wires)
@qml.qnode(dev, interface="jax")
def circuit():
    qml.BasisState(jnp.array([1, 1]), wires=H_tapered.wires)
    return qml.state()

qubit_state = circuit()
HF_energy = qubit_state.T @ H_tapered.sparse_matrix().toarray() @ qubit_state
print(f"HF energy (tapered): {jnp.real(HF_energy):.8f} Ha")

##############################################################################
# These values are identical to the reference Hartree-Fock energy :math:`-2.8543686493` Ha.
#
# VQE simulation
# --------------
# Finally, we can use the tapered Hamiltonian and the tapered reference state to perform a VQE
# simulation and compute the ground-state energy of the :math:`\textrm{HeH}^+` cation. We build a
# tapered variational ansatz :doc:`[3] <demos/tutorial_givens_rotations>`
# that prepares an entangled state by evolving the tapered Hartree-Fock state using the tapered
# particle-conserving gates, i.e., the :func:`~.pennylane.SingleExcitation` and
# :func:`~.pennylane.DoubleExcitation` operations tapered using 
# :func:`~.pennylane.qchem.taper_operation`.

singles, doubles = qml.qchem.excitations(n_electrons, len(H.wires))
tapered_doubles = [
    qml.taper_operation(qml.DoubleExcitation, generators, paulixops, paulix_sector,
                        wire_order=H.wires, op_wires=double) for double in doubles
]
tapered_singles = [
    qml.taper_operation(qml.SingleExcitation, generators, paulixops, paulix_sector,
                        wire_order=H.wires, op_wires=single) for single in singles
]

dev = qml.device("lightning.qubit", wires=H_tapered.wires)

@qml.qnode(dev, interface="jax")
def tapered_circuit(params):
    qml.BasisState(state_tapered, wires=H_tapered.wires)
    for idx, tapered_op in enumerate(tapered_doubles + tapered_singles):
        tapered_op(params[idx])
    return qml.expval(H_tapered)

##############################################################################
# We define an optimizer and the initial values of the circuit parameters and optimize the circuit
# parameters with respect to the ground state energy.

import optax
import catalyst

opt = optax.sgd(learning_rate=0.8) # sgd stands for StochasticGradientDescent
init_params = jnp.zeros(len(doubles) + len(singles))

def update_step(i, params, opt_state):
    """Perform a single gradient update step"""
    grads = catalyst.grad(tapered_circuit)(params)
    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return (params, opt_state)

loss_history = []

opt_state = opt.init(init_params)
params = init_params

for i in range(1, 41):
    params, opt_state = update_step(i, params, opt_state)
    energy = tapered_circuit(params)
    if not i % 5:
        print(f"n: {i}, E: {energy:.8f} Ha, Params: {params}")

##############################################################################
# The computed energy matches the FCI energy, :math:`-2.862595242378` Ha, while the number of qubits
# and the number of Hamiltonian terms are significantly reduced with respect to their original
# values.
#
# Conclusions
# -----------
# Molecular Hamiltonians possess symmetries that can be leveraged to reduce the number of qubits
# required in quantum computing simulations. This tutorial introduces a PennyLane functionality that
# can be used for qubit tapering based on :math:`\mathbb{Z}_2` symmetries. The procedure includes
# obtaining tapered Hamiltonians and tapered reference states that can be used in variational
# quantum algorithms such as VQE.
#
# References
# ----------
#
# .. [#bravyi2017]
#
#     Sergey Bravyi, Jay M. Gambetta, Antonio Mezzacapo, Kristan Temme, "Tapering off qubits to
#     simulate fermionic Hamiltonians". `arXiv:1701.08213 <https://arxiv.org/abs/1701.08213>`__
#
# .. [#setia2019]
#
#     Kanav Setia, Richard Chen, Julia E. Rice, Antonio Mezzacapo, Marco Pistoia, James Whitfield,
#     "Reducing qubit requirements for quantum simulation using molecular point group symmetries".
#     `arXiv:1910.14644 <https://arxiv.org/abs/1910.14644>`__
#
