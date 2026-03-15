r"""How to build compressed double-factorized Hamiltonians
==========================================================

Compressed double factorization offers a powerful approach to overcome key limitations in
quantum chemistry simulations. Specifically, it tackles the runtime's dependency on the
Hamiltonian's one-norm and the shot requirements linked to the number of terms [#cdf]_.
In this tutorial, we will learn how to construct the electronic Hamiltonian in the compressed
double-factorized form using tensor contractions. We will also show how this technique allows
having a linear combination of unitaries (LCU) representation suitable for error-corrected
algorithms, which facilitates efficient simulations via linear-depth circuits with
`Givens rotations <https://arxiv.org/abs/1711.04789>`_.

.. figure:: ../_static/demo_thumbnails/opengraph_demo_thumbnails/OGthumbnail_how_to_build_cdf_hamiltonians.png
    :align: center
    :width: 70%
    :target: javascript:void(0)

Constructing the electronic Hamiltonian
----------------------------------------

The Hamiltonian of a molecular system in the second-quantized form can be expressed as a
sum of the one-body and two-body terms as follows:

.. math::  H = \mu + \sum_{\sigma, pq} h_{pq} a^\dagger_{\sigma, p} a_{\sigma, q} + \frac{1}{2} \sum_{\sigma \tau, pqrs} g_{pqrs} a^\dagger_{\sigma, p} a^\dagger_{\tau, q} a_{\tau, r} a_{\sigma, s},

where the tensors :math:`h_{pq}` and :math:`g_{pqrs}` are the one- and two-body integrals,
:math:`a^\dagger` and :math:`a` are the creation and annihilation operators, :math:`\mu` is the
nuclear repulsion energy constant, :math:`\sigma \in {\uparrow, \downarrow}` represents the spin,
and :math:`p, q, r, s` are the orbital indices. In PennyLane, we can obtain :math:`\mu`,
:math:`h_{pq}` and :math:`g_{pqrs}` using the :func:`~pennylane.qchem.electron_integrals` function:
"""

import pennylane as qml
qml.capture.enable()

symbols = ["H", "H", "H", "H"]
geometry = qml.math.array([[0., 0., -0.2], [0., 0., -0.1], [0., 0., 0.1], [0., 0., 0.2]])

mol = qml.qchem.Molecule(symbols, geometry)
nuc_core, one_body, two_body = qml.qchem.electron_integrals(mol)()

print(f"One-body and two-body tensor shapes: {one_body.shape}, {two_body.shape}")

######################################################################
# In the above expression of :math:`H`, the two-body tensor :math:`g_{pqrs}`
# can be rearranged to define :math:`V_{pqrs}` in the `chemist notation
# <http://vergil.chemistry.gatech.edu/notes/permsymm/permsymm.pdf>`_,
# which leads to a one-body offset term :math:`\sum_{s} V_{pssq}`. This
# allows us to rewrite the Hamiltonian as:
#
# .. math::  H_{\text{C}} = \mu + \sum_{\sigma \in {\uparrow, \downarrow}} \sum_{pq} T_{pq} a^\dagger_{\sigma, p} a_{\sigma, q} + \sum_{\sigma, \tau \in {\uparrow, \downarrow}} \sum_{pqrs} V_{pqrs} a^\dagger_{\sigma, p} a_{\sigma, q} a^\dagger_{\tau, r} a_{\tau, s},
#
# with the transformed one-body terms :math:`T_{pq} = h_{pq} - 0.5 \sum_{s} g_{pqss}`.
# We can obtain the :math:`V_{pqrs}` and :math:`T_{pq}` tensors as:
#

two_chem = 0.5 * qml.math.swapaxes(two_body, 1, 3)  # V_pqrs
one_chem = one_body - 0.5 * qml.math.einsum("pqss", two_body)  # T_pq

######################################################################
# A key feature of this representation is that the modified two-body terms can be factorized
# into a sum of low-rank terms, which can be used to efficiently simulate the Hamiltonian [#cdf2]_.
# We will see how to do this with the double-factorization methods in the next section.
#
# Double factorizing the Hamiltonian
# -----------------------------------
#
# The double factorization of a Hamiltonian can be described as a Hamiltonian manipulation
# technique based on decomposing :math:`V_{pqrs}` into symmetric tensors :math:`L^{(t)}`
# called *factors*, such that :math:`V_{pqrs} = \sum_t^T L_{pq}^{(t)} L_{rs}^{(t) {\dagger}}`
# and the rank :math:`T \leq N^2`, where :math:`N` is the number of orbitals. We can do this
# by performing an eigenvalue or a pivoted Cholesky decomposition of the modified two-body
# tensor. Moreover, each of the :math:`L^{(t)}` can be further eigendecomposed as
# :math:`L^{(t)}_{pq} = \sum_{i} U_{pi}^{(t)} W_i^{(t)} U_{qi}^{(t)}` to perform a second
# tensor factorization. This enables us to express the two-body tensor :math:`V_{pqrs}` in
# the following double-factorized form in terms of orthonormal core tensors :math:`Z^{(t)}`
# and symmetric leaf tensors :math:`U^{(t)}` [#cdf2]_:
#
# .. math::  V_{pqrs} \approx \sum_t^T \sum_{ij} U_{pi}^{(t)} U_{qi}^{(t)} Z_{ij}^{(t)} U_{rj}^{(t)} U_{sj}^{(t)},
#
# where :math:`Z_{ij}^{(t)} = W_i^{(t)} W_j^{(t)}`. This decomposition is referred
# to as the *explicit* double factorization (XDF) and decreases the number of terms
# in the qubit basis from :math:`O(N^4)` to :math:`O(N^3)`, assuming the rank of the
# second tensor factorization to be :math:`O(N)`. In PennyLane, this can be done using the
# :func:`~pennylane.qchem.factorize` function, where one can choose the decomposition method for
# the first tensor factorization (``cholesky``), truncate the resulting factors by discarding
# the ones with individual contributions below a specified threshold (``tol_factor``), and
# optionally control the ranks of their second factorization (``tol_eigval``) as shown below:
#

factors, _, _ = qml.qchem.factorize(two_chem, cholesky=True, tol_factor=1e-5)
print("Shape of the factors: ", factors.shape)

approx_two_chem = qml.math.tensordot(factors, factors, axes=([0], [0]))
assert qml.math.allclose(approx_two_chem, two_chem, atol=1e-5)

######################################################################
# Performing block-invariant symmetry shift
# ------------------------------------------
#
# We can further improve the double-factorization by employing the block-invariant
# symmetry shift (BLISS) technique, which modifies the Hamiltonian's action on the
# undesired electron-number subspace [#bliss]_. It helps decrease the one-norm and
# the spectral range of the Hamiltonian. In PennyLane, this symmetry shift can be
# done using the :func:`~pennylane.qchem.symmetry_shift` function, which returns
# the symmetry-shifted integral tensors and core constant:
#

core_shift, one_shift, two_shift = qml.qchem.symmetry_shift(
    nuc_core, one_chem, two_chem, n_elec = mol.n_electrons
) # symmetry-shifted terms of the Hamiltonian

######################################################################
# Then we can use these shifted terms to obtain a double-factorized representation of
# the Hamiltonian that has a lower one-norm than the original one. For instance, we can
# compare the improvement in the one-norm of the shifted Hamiltonian over the original one
# by accessing the :class:`~.pennylane.estimator.qpe_resources.DoubleFactorization`'s ``lamb`` attribute:
#

from pennylane.estimator.qpe_resources import DoubleFactorization as DF

DF_chem_norm = DF(one_chem, two_chem, chemist_notation=True).lamb
DF_shift_norm =  DF(one_shift, two_shift, chemist_notation=True).lamb
print(f"Decrease in one-norm: {DF_chem_norm - DF_shift_norm}")

######################################################################
# Compressing the double factorization
# -------------------------------------
#
# In many practical scenarios, the double factorization method can be further optimized by
# performing a numerical tensor-fitting of the decomposed two-body terms to obtain :math:`V^\prime`
# such that the approximation error :math:`||V - V^\prime||` remains below a desired threshold
# [#cdf]_. This is referred to as the *compressed* double factorization (CDF) as it reduces the
# number of terms in the factorization of the two-body term from :math:`O(N^3)` to :math:`O(N)`
# and achieves lower approximation errors than the truncated XDF. Compression can be done by
# beginning with :math:`O(N)` random core and leaf tensors and optimizing them based on the
# following cost function :math:`\mathcal{L}` in a greedy manner:
#
# .. math::  \mathcal{L}(U, Z) = \frac{1}{2} \bigg|V_{pqrs} - \sum_t^T \sum_{ij} U_{pi}^{(t)} U_{pj}^{(t)} Z_{ij}^{(t)} U_{qk}^{(t)} U_{ql}^{(t)}\bigg|_{\text{F}} + \rho \sum_t^T \sum_{ij} \bigg|Z_{ij}^{(t)}\bigg|^{\gamma},
#
# where :math:`|\cdot|_{\text{F}}` denotes the Frobenius norm, :math:`\rho` is a constant
# scaling factor, and :math:`|\cdot|^\gamma` specifies the optional L1 and L2 `regularization
# <https://en.wikipedia.org/wiki/Regularization_(mathematics)#L1_and_L2_Regularization>`_
# that improves the energy variance of the resulting representation. In PennyLane, this
# compression can be done by using the ``compressed=True`` keyword argument in the
# :func:`~pennylane.qchem.factorize` function. The regularization term will be included
# if the ``regularization`` keyword argument is set to either ``"L1"`` or ``"L2"``:
#

_, two_body_cores, two_body_leaves = qml.qchem.factorize(
    two_shift, tol_factor=1e-2, cholesky=True, compressed=True, regularization="L2"
) # compressed double-factorized shifted two-body terms with "L2" regularization
print(f"Two-body tensors' shape: {two_body_cores.shape, two_body_leaves.shape}")

approx_two_shift = qml.math.einsum(
    "tpk,tqk,tkl,trl,tsl->pqrs",
    two_body_leaves, two_body_leaves, two_body_cores, two_body_leaves, two_body_leaves
) # computing V^\prime and comparing it with V below
assert qml.math.allclose(approx_two_shift, two_shift, atol=1e-2)

######################################################################
# While the previous shape output for the factors ``(10, 4, 4)`` meant we had :math:`10` two-body
# terms in our factorization, the current shape output ``(6, 4, 4)`` indicates that we have
# :math:`6` terms. This means that the number of terms in the factorization has decreased almost
# by half, which is quite significant!
#
# Constructing the double-factorized Hamiltonian
# -----------------------------------------------
#
# We can eigendecompose the one-body tensor to obtain similar orthonormal :math:`U^{(0)}` and
# symmetric :math:`Z^{(0)}` tensors for the one-body term and use the compressed factorization
# of the two-body term described in the previous section to express the Hamiltonian in the
# double-factorized form as:
#
# .. math:: H_{\text{CDF}} = \mu + \sum_{\sigma \in {\uparrow, \downarrow}} U^{(0)}_{\sigma} \left( \sum_{p} Z^{(0)}_{p} a^\dagger_{\sigma, p} a_{\sigma, p} \right) U_{\sigma}^{(0)\ \dagger} + \sum_t^T \sum_{\sigma, \tau \in {\uparrow, \downarrow}} U_{\sigma, \tau}^{(t)} \left( \sum_{pq} Z_{pq}^{(t)} a^\dagger_{\sigma, p} a_{\sigma, p} a^\dagger_{\tau, q} a_{\tau, q} \right) U_{\sigma, \tau}^{(t)\ \dagger}.
#
# This Hamiltonian can be easily mapped to the qubit basis via `Jordan-Wigner
# transformation <https://pennylane.ai/qml/demos/tutorial_mapping>`_ (JWT) using
# :math:`a_p^\dagger a_p = n_p \mapsto 0.5 * (1 - z_p)`, where :math:`n_p` is the number
# operator and :math:`z_p` is the Pauli-Z operation acting on the qubit corresponding to
# orbital :math:`p`. The mapped form naturally gives rise to a measurement grouping, where
# the terms within the basis transformation :math:`U^{(i)}` can be measured simultaneously.
# These can be obtained with the :func:`~pennylane.qchem.basis_rotation` function, which
# performs the double-factorization and JWT automatically.
#
# Another advantage of the double-factorized form is the efficient simulation of the Hamiltonian
# evolution. Before discussing it in the next section, we note that mapping a two-body term to
# the qubit basis will result in two additional one-qubit Pauli-Z terms. We can simplify their
# simulation circuits by accounting for these additional terms directly in the one-body tensor
# using a correction ``one_body_extra``. We can then decompose the corrected one-body terms into
# the orthonormal :math:`U^{\prime(0)}` and symmetric :math:`Z^{\prime(0)}` tensors instead:
#

two_core_prime = (qml.math.eye(mol.n_orbitals) * two_body_cores.sum(axis=-1)[:, None, :])
one_body_extra = qml.math.einsum(
    'tpk,tkk,tqk->pq', two_body_leaves, two_core_prime, two_body_leaves
) # one-body correction

# factorize the corrected one-body tensor to obtain the core and leaf tensors
one_body_eigvals, one_body_eigvecs = qml.math.linalg.eigh(one_shift + one_body_extra)
one_body_cores = qml.math.expand_dims(qml.math.diag(one_body_eigvals), axis=0)
one_body_leaves = qml.math.expand_dims(one_body_eigvecs, axis=0)

print(f"One-body tensors' shape: {one_body_cores.shape, one_body_leaves.shape}")

######################################################################
# We can now specify the Hamiltonian programmatically in the (compressed)
# double-factorized form as a ``dict`` object with the following three keys:
# ``nuc_constant`` (:math:`\mu`),
# ``core_tensors`` (:math:`\left[ Z^{\prime(0)}, Z^{(t_1)}, \ldots, Z^{(t_T)} \right]`), and
# ``leaf_tensors`` (:math:`\left[ U^{\prime(0)}, U^{(t_1)}, \ldots, U^{(t_T)} \right]`):
#

cdf_hamiltonian = {
    "nuc_constant": core_shift[0],
    "core_tensors": qml.math.concatenate((one_body_cores, two_body_cores), axis=0),
    "leaf_tensors": qml.math.concatenate((one_body_leaves, two_body_leaves), axis=0),
} # CDF Hamiltonian

######################################################################
# Simulating the double-factorized Hamiltonian
# ---------------------------------------------
#
# To simulate the time evolution of the CDF Hamiltonian,
# we will first need to learn how to apply the unitary operations
# represented by the exponentiated leaf and core tensors. The former can be done using the
# :class:`~.pennylane.BasisRotation` operation, which implements the unitary transformation
# :math:`\exp \left( \sum_{pq}[\log U]_{pq} (a^\dagger_p a_q - a^\dagger_q a_p) \right)`
# using the `Givens rotation networks
# <https://docs.pennylane.ai/en/stable/code/api/pennylane.qchem.givens_decomposition.html>`_
# that can be efficiently implemented on quantum hardware. The ``leaf_unitary_rotation``
# function below does this for a leaf tensor:
#

def leaf_unitary_rotation(leaf, wires):
    """Applies the basis rotation transformation corresponding to the leaf tensor."""
    basis_mat = qml.math.kron(leaf, qml.math.eye(2)) # account for spin
    return qml.BasisRotation(unitary_matrix=basis_mat, wires=wires)

######################################################################
# Similarly, the unitary transformation for the core tensors can be applied efficiently
# via the ``core_unitary_rotation`` function defined below. The function uses the
# :class:`~.pennylane.RZ` and :class:`~.pennylane.IsingZZ` gates for implementing
# the diagonal and entangling phase rotations for the one- and two-body core tensors,
# respectively, and :class:`~.pennylane.GlobalPhase` for the corresponding global phases:
#

import itertools as it

def core_unitary_rotation(core, body_type, wires):
    """Applies the unitary transformation corresponding to the core tensor."""
    ops = []
    if body_type == "one_body":  # implements one-body term
        for wire, cval in enumerate(qml.math.diag(core)):
            for sigma in [0, 1]:
                ops.append(qml.RZ(-cval, wires=2 * wire + sigma))
        ops.append(qml.GlobalPhase(qml.math.sum(core), wires=wires))

    if body_type == "two_body":  # implements two-body term
        for odx1, odx2 in it.product(range(len(wires) // 2), repeat=2):
            cval = core[odx1, odx2] / 4.0
            for sigma, tau in it.product(range(2), repeat=2):
                if odx1 != odx2 or sigma != tau:
                    ops.append(qml.IsingZZ(cval, wires=[2*odx1+sigma, 2*odx2+tau]))
        gphase = 0.5 * qml.math.sum(core) - 0.25 * qml.math.trace(core)
        ops.append(qml.GlobalPhase(-gphase, wires=wires))
    return ops

######################################################################
# We can now use these functions to approximate the evolution operator :math:`e^{-iHt}` for
# a time :math:`t` with the Suzuki-Trotter product formula, which uses symmetrized products
# :math:`S_m` defined for an order :math:`m \in [1, 2, 4, \ldots, 2k \in \mathbb{N}]`
# and repeated multiple times [#trotter]_. In general, this can be easily implemented for
# standard non-factorized Hamiltonians using the :class:`~.pennylane.TrotterProduct` operation,
# which defines these products recursively, leading to an exponential scaling in its complexity
# with the number of terms in the Hamiltonian and making it inefficient for larger system sizes.
#
# The exponential scaling can be improved to a great extent by working with the compressed
# double-factorized form of the Hamiltonian as it allows reducing the number of terms in the
# Hamiltonian from :math:`O(N^4)` to :math:`O(N)`. While doing this is not directly supported
# in PennyLane in the form of a template, we can still implement the first-order Trotter
# step using the following :func:`CDFTrotterStep` function that uses the CDF Hamiltonian
# with the ``leaf_unitary_rotation`` and ``core_unitary_rotation`` functions defined earlier.
# We can then use the :func:`~.pennylane.trotterize` function to implement any higher-order
# Suzuki-Trotter products.
#

def CDFTrotterStep(time, cdf_ham, wires):
    """Implements a first-order Trotter step for a CDF Hamiltonian.

    Args:
        time (float): time-step for a Trotter step.
        cdf_ham (dict): dictionary describing the CDF Hamiltonian.
        wires (list): list of integers representing the qubits.
    """
    cores, leaves = cdf_ham["core_tensors"], cdf_ham["leaf_tensors"]
    for bidx, (core, leaf) in enumerate(zip(cores, leaves)):
        # Note: only the first term is one-body, others are two-body
        body_type = "two_body" if bidx else "one_body"
        qml.prod(
            # revert the change-of-basis for leaf tensor
            leaf_unitary_rotation(leaf.conjugate().T, wires),
            # apply the rotation for core tensor scaled by the time-step
            *core_unitary_rotation(time * core, body_type, wires),
            # apply the basis rotation for leaf tensor
            leaf_unitary_rotation(leaf, wires),
        ) # Note: prod applies operations in the reverse order (right-to-left).

######################################################################
# We now use this function to simulate the evolution of the :math:`H_4` Hamiltonian
# described in the compressed double-factorized form for a given number of
# steps ``n_steps`` and starting from a Hartree-Fock state ``hf_state``
# with the following circuit that applies a second-order Trotter product:
#

time, circ_wires = 1.0, range(2 * mol.n_orbitals)
hf_state = qml.qchem.hf_state(electrons=mol.n_electrons, orbitals=len(circ_wires))

@qml.qnode(qml.device("lightning.qubit", wires=circ_wires))
def cdf_circuit(n_steps, order):
    qml.BasisState(hf_state, wires=circ_wires)
    qml.trotterize(CDFTrotterStep, n_steps, order)(time, cdf_hamiltonian, circ_wires)
    qml.GlobalPhase(cdf_hamiltonian["nuc_constant"], wires=circ_wires)
    return qml.state()

circuit_state = cdf_circuit(n_steps=10, order=2)

######################################################################
# We now test the accuracy of the Hamiltonian simulation by evolving the
# Hartree-Fock state analytically ourselves and testing the fidelity
# of the ``evolved_state`` with the ``circuit_state``:
#

from pennylane.math import fidelity_statevector
from scipy.linalg import expm

# Evolve the state vector |0...0> to the |HF> state of the system
init_state = qml.math.array([1] + [0] * (2**len(circ_wires) - 1))
hf_state_vec = qml.matrix(qml.BasisState(hf_state, wires=circ_wires)) @ init_state

H = qml.qchem.molecular_hamiltonian(mol)[0] # original Hamiltonian
evolved_state = expm(-1j * qml.matrix(H) * time) @ hf_state_vec # e^{-iHt} @ |HF>

print(f"Fidelity of two states: {fidelity_statevector(circuit_state, evolved_state)}")

######################################################################
# As we can see, the fidelity of the evolved state from the circuit is close to
# :math:`1.0`, which indicates that the evolution of the CDF Hamiltonian sufficiently
# matches that of the original one.
#
# Conclusion
# -----------
#
# Compressed double-factorized representation for the Hamiltonians serves three key purposes.
# First, it provides for a more compact representation of the Hamiltonian that can be stored
# and manipulated easier. Second, it allows more efficient simulations of the Hamiltonian
# time evolution because the number of terms is reduced quadratically. Third, the compact
# representation can be further manipulated to reduce the one-norm of the Hamiltonian, which
# helps reduce the simulation cost when using block encoding or qubitization techniques.
# Overall, employing CDF-based Hamiltonians for quantum chemistry problems provides a
# relatively promising path to reducing the complexity of fault-tolerant quantum algorithms.
#
# References
# -----------
#
# .. [#cdf]
#
#     Oumarou Oumarou, Maximilian Scheurer, Robert M. Parrish, Edward G. Hohenstein, and Christian Gogolin,
#     "Accelerating Quantum Computations of Chemistry Through Regularized Compressed Double Factorization",
#     `Quantum 8, 1371 <https://doi.org/10.22331/q-2024-06-13-1371>`__, 2024.
#
# .. [#cdf2]
#
#     Jeffrey Cohn, Mario Motta, and Robert M. Parrish,
#     "Quantum Filter Diagonalization with Compressed Double-Factorized Hamiltonians",
#     `PRX Quantum 2, 040352 <https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.040352>`__, 2021.
#
# .. [#bliss]
#
#     Ignacio Loaiza, and Artur F. Izmaylov,
#     "Block-Invariant Symmetry Shift: Preprocessing technique for second-quantized Hamiltonians to improve their decompositions to Linear Combination of Unitaries",
#     `arXiv:2304.13772 <https://arxiv.org/abs/2304.13772>`__, 2023.
#
# .. [#trotter]
#
#     Sergiy Zhuk, Niall Robertson, and Sergey Bravyi,
#     "Trotter error bounds and dynamic multi-product formulas for Hamiltonian simulation",
#     `arXiv:2306.12569 <https://arxiv.org/abs/2306.12569>`__, 2023.
#
