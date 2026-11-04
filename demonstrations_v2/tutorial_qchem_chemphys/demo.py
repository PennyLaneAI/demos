r"""

Molecular Hamiltonian Representations
=====================================

.. meta::
    :property="og:description": Learn how to construct a Hamiltonian using use chemist, physicist and quantum conventions
    :property="og:image": https://pennylane.ai/qml/_static/demonstration_assets/thumbnail_tutorial_external_libs.png


.. related::
    tutorial_quantum_chemistry Quantum chemistry with PennyLane
    tutorial_fermionic_operators Fermionic operators
    tutorial_mapping Mapping fermionic operators to qubit operators

Molecular Hamiltonians in second quantization can be constructed in different ways depending on the
arrangement of the two-electron integral tensor. Here, we show you how to construct a fermionic
molecular Hamiltonian from two-electron integral tensors represented in different conventions. The
integrals computed with any of these conventions can be easily converted to the others. This allows
constructing any desired representation of the molecular Hamiltonian without re-calculating the
integrals. We start by a detailed step-by-step construction of the integrals, and the corresponding
Hamiltonians, and then provide compact helper functions that automate the conversion of the
integrals and the construction of the Hamiltonians.
"""

##############################################################################
# Integral Notations
# ------------------
# We use three common notations for two-electron integrals. First, we look at a notation that is
# commonly used in the quantum computing community and quantum software libraries, which we refer to
# as the ``Quantum`` notation. Then we review two other notations that are typically referred to as
# the ``Chemist's`` and ``Physicist's`` notations.
#
# Quantum notation
# ~~~~~~~~~~~~~~~~
# The two-electron integrals in this notation are defined as
#
# .. math::
#
#     \langle \langle pq | rs \rangle \rangle = \int dr_1 dr_2 \phi_p^*(r_1) \phi_q^*(r_2) \frac{1}{r_{12}} \phi_r(r_2) \phi_s(r_1),
#
# where :math:`\phi` is a spatial molecular orbital. We have used the double bracket
# :math:`\langle \langle \cdot  \rangle \rangle` to distinguish this notation from the others. The
# corresponding Hamiltonian in second quantization is defined in terms of the fermionic creation and
# annihilation operators, :math:`a^{\dagger}` and :math:`a`, as
#
# .. math::
#
#     H = \sum_{pq} h_{pq} a_p^{\dagger} a_q + \frac{1}{2} \sum_{pqrs} h_{pqrs} a_p^{\dagger} a_q^{\dagger} a_r a_s.
#
# where :math:`h_{pq}` denotes a one-electron integral. We have skipped the spin indices and
# the core constant for brevity. Note that the order of the creation and annihilation operator
# indices matches the order of the coefficient indices for both :math:`h_{pq}` and :math:`h_{pqrs}`
# terms.
#
# We now construct the Hamiltonian using PennyLane. PennyLane employs the quantum computing
# convention for the two-electron integral tensor, which can be efficiently computed via the
# built-in ``electron_integrals`` function. We use the water molecule as an example.

import numpy as np
import pennylane as qml
from pennylane.fermi import FermiWord, FermiSentence, from_string

symbols = ["H", "O", "H"]
geometry = np.array([[-0.0399, -0.0038, 0.0000],
                     [1.5780, 0.8540, 0.0000],
                     [2.7909, -0.5159, 0.0000]])

mol = qml.qchem.Molecule(symbols, geometry)

core_constant, one_mo, two_mo = qml.qchem.electron_integrals(mol)()

##############################################################################
# PennyLane uses the restricted `Hartree-Fock <https://en.wikipedia.org/wiki/Hartree%E2%80%93Fock_method>`__
# method by default which returns the integrals in the
# basis of spatial molecular orbitals. That means, for the water molecule, we have computed the
# integrals over the :math:`1s`, :math:`2s`, and the :math:`2p_x, 2p_y, 2p_z` orbitals without
# accounting for spin. To construct the full Hamiltonian, the integrals objects need to be expanded
# to include spin orbitals, i.e., :math:`1s_{\alpha}`, :math:`1s_{\beta}` etc. Assuming an
# interleaved convention for the order of spin orbitals, i.e., :math:`|\alpha, \beta, \alpha, \beta, ...>`,
# the following functions give the expanded one-electron and two-electron objects. Note using the
# `unrestricted Hartree-Fock <https://en.wikipedia.org/wiki/Unrestricted_Hartree%E2%80%93Fock>`__
# method provides the full integrals objects in the basis of spin orbitals and the expansion is not
# needed.

def transform_one(h_mo: np.ndarray) -> np.ndarray:
    """Converts a one-electron integral matrix from the molecular orbital (MO) basis to the
     spin-orbital (SO) basis.

    Args:
        h_mo (array): The one-electron matrix with shape (n, n) where n is the number of
            spatial orbitals.

    Returns:
        The one-electron matrix with shape (2n, 2n) where n is the number of
            spatial orbitals.
    """

    n_so = 2 * h_mo.shape[0]
    h_so = np.zeros((n_so, n_so))

    alpha, beta = slice(0, n_so, 2), slice(1, n_so, 2)

    h_so[alpha, alpha] = h_mo
    h_so[beta, beta] = h_mo

    return h_so


def transform_two(g_mo):
    """Converts a two-electron integral tensor from the molecular orbital (MO) basis to the
     spin-orbital (SO) basis.

    Args:
        g_mo (array): The two-electron tensor with shape (n, n, n, n) where n is the number
            of spatial orbitals.

    Returns:
        The two-electron matrix with shape (2n, 2n, 2n, 2n) where n is the number of
            spatial orbitals.
    """

    n_so = 2 * g_mo.shape[0]

    g_so = np.zeros((n_so, n_so, n_so, n_so))

    alpha = slice(0, n_so, 2)
    beta = slice(1, n_so, 2)

    g_so[alpha, alpha, alpha, alpha] = g_mo
    g_so[alpha, beta, beta, alpha] = g_mo
    g_so[beta, alpha, alpha, beta] = g_mo
    g_so[beta, beta, beta, beta] = g_mo

    return g_so

##############################################################################
# The transformation to the spin orbital basis must respect the spin-selection rules. For instance,
# the following integral over spin orbitals :math:`\chi` must be zero if the spins :math:`\sigma` of
# the initial and final spin orbitals are different.
#
# .. math::
#
#     \langle \chi_p | \hat{h} | \chi_q \rangle = \langle \phi_i | \hat{h} | \phi_j \rangle \cdot \langle \sigma_p | \sigma_q \rangle,
#
# since the one-electron operator :math:`\hat{h}` is spin-independent. For the one-electron
# integrals, only the :math:`\alpha \alpha` and :math:`\beta \beta` combinations have a non-zero
# value. Similarly, the two-electron operator :math:`1/r_{12}` is spin-independent and if we use the
# compact notation ``1221`` to represent the order in the two-electron integral, the non-zero
# combinations are: :math:`\alpha \alpha \alpha \alpha`, :math:`\alpha \beta \beta \alpha`, :math:`\beta \alpha \alpha\beta` and :math:`\beta \beta \beta \beta`.
# These combination rules are used in our functions.
#
# We now obtain the full electron integrals in the basis of spin orbitals.

one_so = transform_one(one_mo)
two_so = transform_two(two_mo)

##############################################################################
# Having the electron integrals objects, computing the Hamiltonian is straightforward. We simply
# loop over the elements of the tensors and multiply them by the corresponding fermionic operators.
# For better performance, we can skip the negligible integral components and just obtain the
# operator indices for the non-zero integrals.

one_operators = qml.math.argwhere(abs(one_so) >= 1e-12)
two_operators = qml.math.argwhere(abs(two_so) >= 1e-12)

##############################################################################
# Now we construct the Hamiltonian as a ``FermiSentence`` object.

sentence = FermiSentence({FermiWord({}): core_constant[0]})

for p, q in one_operators:
    sentence.update({from_string(f'{p}+ {q}-'): one_so[p, q]})

for p, q, r, s in two_operators:
    sentence.update({from_string(f'{p}+ {q}+ {r}- {s}-'): two_so[p, q, r, s] / 2})

sentence.simplify()

##############################################################################
# Note that the order of indices for the fermionic creation and annihilation operators matches
# the order of indices in the integral objects, i.e., :math:`pq` and :math:`pqrs`. We finally map
# the fermionic Hamiltonian to the qubit basis and compute its ground-state energy, which should
# match the reference value :math:`-75.01562736 \ \text{H}`.

h = qml.jordan_wigner(sentence)
qml.eigvals(qml.SparseHamiltonian(h.sparse_matrix(), wires=h.wires))

##############################################################################
# We can further verify this value by using the PennyLane function ``molecular_hamiltonian`` which
# automatically constructs the Hamiltonian.

h = qml.qchem.molecular_hamiltonian(mol)[0]
qml.eigvals(qml.SparseHamiltonian(h.sparse_matrix(), wires=h.wires))

##############################################################################
# Chemist's notation
# ~~~~~~~~~~~~~~~~~~
# This notation is commonly used by quantum chemistry software libraries such as ``PySCF``. The
# two-electron integral tensor in this notation is defined as
#
# .. math::
#
#     (pq | rs) = \int dr_1 dr_2 \phi_p^*(r_1) \phi_q(r_1) \frac{1}{r_{12}} \phi_r^*(r_2) \phi_s(r_2).
#
# The corresponding Hamiltonian in second quantization is then defined as
#
# .. math::
#
#     H = \sum_{pq} (h_{pq} - \frac{1}{2} \sum_s h_{pssq}) a_p^{\dagger} a_q + \frac{1}{2} \sum_{pqrs} h_{pqrs} a_p^{\dagger} a_q a_r^{\dagger} a_s.
#
# Note that a one-body correction term should be included. Also note that the order of indices of
# the creation and annihilation operators in the second term matches the order of the two-electron
# integral coefficients. See the appendix for a full derivation of the Hamiltonian.
#
# Let's now build the Hamiltonian using electron integrals computed by ``PySCF`` which adopts the
# Chemist's notation. Note that ``PySCF`` one-body integral does not include the correction term
# mentioned above for the Hamiltonian and we need to add it manually.

from pyscf import gto, ao2mo, scf

mol_pyscf = gto.M(atom='''H -0.02111417 -0.00201087  0.;
                          O  0.83504162  0.45191733  0.;
                          H  1.47688065 -0.27300252  0.''')
rhf = scf.RHF(mol_pyscf)
energy = rhf.kernel()

one_ao = mol_pyscf.intor_symmetric('int1e_kin') + mol_pyscf.intor_symmetric('int1e_nuc')
two_ao = mol_pyscf.intor('int2e_sph')

one_mo = np.einsum('pi,pq,qj->ij', rhf.mo_coeff, one_ao, rhf.mo_coeff)
two_mo = ao2mo.incore.full(two_ao, rhf.mo_coeff)

core_constant = np.array([rhf.energy_nuc()])

##############################################################################
# We used the restricted Hartree-Fock method and need to expand the integrals to account
# for spin orbitals. To do that, we need to slightly upgrade our ``transform_two`` function
# because the allowed spin combinations in the Chemist's notation, denoted by ``1122``, are
# :math:`\alpha \alpha \alpha \alpha`, :math:`\alpha \alpha \beta \beta`,
# :math:`\beta \beta \alpha \alpha` and :math:`\beta \beta \beta \beta`.

def transform_two(g_mo, notation):
    """Converts a two-electron integral tensor from the molecular orbital (MO) basis to the
     spin-orbital (SO) basis.

    Args:
        g_mo (array): The two-electron tensor with shape (n, n, n, n) where n is the number
            of spatial orbitals.
        notation (str): The notation used to compute the two-electron integrals tensor.

    Returns:
        The two-electron matrix with shape (2n, 2n, 2n, 2n) where n is the number of
            spatial orbitals.
    """

    n = g_mo.shape[0]
    n_so = 2 * n

    g_so = np.zeros((n_so, n_so, n_so, n_so))

    alpha = slice(0, n_so, 2)
    beta = slice(1, n_so, 2)

    g_so[alpha, alpha, alpha, alpha] = g_mo
    g_so[beta, beta, beta, beta] = g_mo

    if notation == 'quantum':  # '1221'
        g_so[alpha, beta, beta, alpha] = g_mo
        g_so[beta, alpha, alpha, beta] = g_mo
        return g_so

    if notation == 'chemist':  # '1122'
        g_so[alpha, alpha, beta, beta] = g_mo
        g_so[beta, beta, alpha, alpha] = g_mo
        return g_so


##############################################################################
# We now compute the integrals in the spin-orbitals basis and add the one-body correction term.

one_so = transform_one(one_mo)
two_so = transform_two(two_mo, 'chemist')

one_so_corrected = one_so - 0.5 * np.einsum('pssq->pq', two_so)

##############################################################################
# Constructing the Hamiltonian is now straightforward.

one_operators = qml.math.argwhere(abs(one_so_corrected) >= 1e-12)
two_operators = qml.math.argwhere(abs(two_so) >= 1e-12)

sentence = FermiSentence({FermiWord({}): core_constant[0]})

for p, q in one_operators:
    sentence.update({from_string(f'{p}+ {q}-'): one_so_corrected[p, q]})

for p, q, r, s in two_operators:
    sentence.update({from_string(f'{p}+ {q}- {r}+ {s}-'): two_so[p, q, r, s]/2})

sentence.simplify()

##############################################################################
# Note that the order of the indices for the fermionic operators match those of the electron
# integral coefficients. Also note that the two-body operator has the order
# :math:`a^{\dagger} a a^{\dagger} a`. Let's now validate the Hamiltonian by computing the
# ground-state energy.

h = qml.jordan_wigner(sentence)
qml.eigvals(qml.SparseHamiltonian(h.sparse_matrix(), wires=h.wires))

##############################################################################
# Physicist's notation
# ~~~~~~~~~~~~~~~~~~~~
# The two-electron integral tensor in this notation is defined as
#
# .. math::
#
#     \langle pq | rs \rangle = \int dr_1 dr_2 \phi_p^*(r_1) \phi_q^*(r_2) \frac{1}{r_{12}} \phi_r(r_1) \phi_s(r_2).
#
# The corresponding Hamiltonian in second quantization is then constructed as
#
# .. math::
#
#     H = \sum_{pq} h_{pq} a_p^{\dagger} a_q + \frac{1}{2} \sum_{pqrs} h_{pqrs} a_p^{\dagger} a_q^{\dagger} a_s a_r,
#
# It is important to note the order of the creation and annihilation operators in the second term,
# which is :math:`a_p^{\dagger} a_q^{\dagger} a_s a_r`. This order must be preserved when the
# Hamiltonian is constructed with the two-electron integral tensor represented in the Physicist's
# notation.
#
# Let's now build this Hamiltonian step-by-step. There is not a commonly-used software library
# to compute integrals in this notation. However, we can easily convert the integrals we already
# computed in other notations to the Physicist's notation. For instance, we use ``PySCF`` integrals
# and convert them to the Physicist's notation.
#
# The conversion can be done by the transformation :math:`(pq | rs) \to (pr | qs)` where we have
# swapped the :math:`1,2` indices. This transformation can be done with

two_mo = two_mo.transpose(0, 2, 1, 3)

##############################################################################
# Now we need to expand the integrals to the spin orbital basis. To do that, we need to again
# slightly upgrade our ``transform_two`` function to include the allowed spin combination in the
# Physicist's notation, denoted by ``1212``, as :math:`\alpha \alpha \alpha \alpha`,
# :math:`\alpha \beta \alpha \beta`, :math:`\beta \alpha \beta \alpha`
# and :math:`\beta \beta \beta \beta`.

def transform_two(g_mo, notation):
    """Converts a two-electron integral tensor from the molecular orbital (MO) basis to the
     spin-orbital (SO) basis.

    Args:
        g_mo (array): The two-electron tensor with shape (n, n, n, n) where n is the number
            of spatial orbitals.
        notation (str): The notation used to compute the two-electron integrals tensor.

    Returns:
        The two-electron matrix with shape (2n, 2n, 2n, 2n) where n is the number of
            spatial orbitals.
    """

    n = g_mo.shape[0]
    n_so = 2 * n

    g_so = np.zeros((n_so, n_so, n_so, n_so))

    alpha = slice(0, n_so, 2)
    beta = slice(1, n_so, 2)

    g_so[alpha, alpha, alpha, alpha] = g_mo
    g_so[beta, beta, beta, beta] = g_mo

    if notation == 'quantum':  # '1221'
        g_so[alpha, beta, beta, alpha] = g_mo
        g_so[beta, alpha, alpha, beta] = g_mo
        return g_so

    if notation == 'chemist':  # '1122'
        g_so[alpha, alpha, beta, beta] = g_mo
        g_so[beta, beta, alpha, alpha] = g_mo
        return g_so

    if notation == 'physicist':  # '1212'
        g_so[alpha, beta, alpha, beta] = g_mo
        g_so[beta, alpha, beta, alpha] = g_mo
        return g_so

##############################################################################
# The new tensor can then be used to construct the Hamiltonian.

one_so = transform_one(one_mo)
two_so = transform_two(two_mo, 'physicist')

one_operators = qml.math.argwhere(abs(one_so) >= 1e-12)
two_operators = qml.math.argwhere(abs(two_so) >= 1e-12)

sentence = FermiSentence({FermiWord({}): core_constant[0]})

for p, q in one_operators:
    sentence.update({from_string(f'{p}+ {q}-'): one_so[p, q]})

for p, q, r, s in two_operators:
    sentence.update({from_string(f'{p}+ {q}+ {s}- {r}-'): two_so[p, q, r, s]/2})

sentence.simplify()

##############################################################################
# Note the order of the creation and annihilation operator indices, which is
# :math:`a_p^{\dagger} a_q^{\dagger} a_s a_r`. Now we compute the ground state energy to validate
# the Hamiltonian.

h = qml.jordan_wigner(sentence)
qml.eigvals(qml.SparseHamiltonian(h.sparse_matrix(), wires=h.wires))

##############################################################################
# Integral conversion
# -------------------
# The two-electron integrals computed with one convention can be easily converted to the other
# conventions, as we already did to convert the Chemist's notation to the Physicist's one. Such
# conversions allow constructing different representations of the molecular Hamiltonian without
# re-calculating the integrals. The following function applies the conversion rules for all three
# conventions.

def convert_integrals(two_body, in_notation, out_notation):
    """Converts a two-electron integral tensor between different conventions.

    Args:
        two_body (array): The two-electron tensor with shape (2n, 2n, 2n, 2n) where n is
            the number of spatial orbitals.
        in_notation (str): The notation used to compute the two-electron integrals tensor.
        out_notation (str): The desired notation to represent the two-electron
            integrals tensor.

    Returns:
        The two-electron matrix with shape (2n, 2n, 2n, 2n) where n is the number of
            spatial orbitals.
    """
    if in_notation == out_notation:
        return two_body

    if in_notation == "chemist" and out_notation == "physicist":
        return two_body.transpose(0, 2, 1, 3)

    if in_notation == "chemist" and out_notation == "quantum":
        return two_body.transpose(0, 2, 3, 1)

    if in_notation == "quantum" and out_notation == "chemist":
        return two_body.transpose(0, 3, 1, 2)

    if in_notation == "quantum" and out_notation == "physicist":
        return two_body.transpose(0, 1, 3, 2)

    if in_notation == "physicist" and out_notation == "chemist":
        return two_body.transpose(0, 2, 1, 3)

    if in_notation == "physicist" and out_notation == "quantum":
        return two_body.transpose(0, 1, 3, 2)


##############################################################################
# We can also create a versatile function that computes the Hamiltonian for each convention.

def fermionic_observable(core_constant, one_body, two_body, notation, cutoff=1e-12):
    """Converts a two-electron integral tensor between different conventions.

    Args:
        core_constant (float): The core constant containing the contribution of the
            core orbitals and nuclei
        one_body (array): The one-electron tensor with shape (2n, 2n) where n is
            the number of spatial orbitals.
        two_body (array): The two-electron tensor with shape (2n, 2n, 2n, 2n) where
            n is the number of spatial orbitals.
        notation (str): The notation used to compute the two-electron integrals tensor.
        cutoff (float). The tolerance for neglecting an integral. Defaults is 1e-12.

    Returns:
        The molecular Hamiltonian in the Pauli basis.
    """
    if notation == "chemist":
        one_body = one_body - 0.5 * np.einsum('pssq->pq', two_body)

    op_one = qml.math.argwhere(abs(one_body) >= cutoff)
    op_two = qml.math.argwhere(abs(two_body) >= cutoff)

    sentence = FermiSentence({FermiWord({}): core_constant})

    for p, q in op_one:
        sentence.update({from_string(f'{p}+ {q}-'): one_body[p, q]})

    if notation == "quantum":
        for p, q, r, s in op_two:
            sentence.update({from_string(f'{p}+ {q}+ {r}- {s}-'): two_body[p, q, r, s] / 2})

    if notation == "chemist":
        for p, q, r, s in op_two:
            sentence.update({from_string(f'{p}+ {q}- {r}+ {s}-'): two_body[p, q, r, s] / 2})

    if notation == "physicist":
        for p, q, r, s in op_two:
            sentence.update({from_string(f'{p}+ {q}+ {s}- {r}-'): two_body[p, q, r, s] / 2})

    sentence.simplify()

    return qml.jordan_wigner(sentence)


##############################################################################
# We now have all the necessary tools in our arsenal to convert the integrals to a desired
# notation and construct the corresponding Hamiltonian automatically. Let's look at a few examples.
#
# First we convert the integrals obtained with PennyLane in the Quantum notation to the Chemist's
# notation and compute the corresponding Hamiltonian. This conversion can be done by the
# transformation :math:`\langle pq | rs \rangle \ \to \ \langle pq | sr \rangle \ \to \ \langle ps | qr \rangle`.
# We have first swapped the :math:`2,3` indices to go from the Quantum notation to the Physicist's
# notation and then swap :math:`1,2` indices to get the integrals in the Chemist's notation. This
# transformation can be done with :math:`transpose(0, 3, 1, 2)`. Let's now use our helper functions
# for the full workflow.

core_constant, one_mo, two_mo = qml.qchem.electron_integrals(mol)()
one_so = transform_one(one_mo)
two_so = transform_two(two_mo, 'quantum')

two_so_converted = convert_integrals(two_so, 'quantum', 'physicist')

h = fermionic_observable(core_constant[0], one_so, two_so_converted, 'physicist')
qml.eigvals(qml.SparseHamiltonian(h.sparse_matrix(), wires=h.wires))

##############################################################################
# You can verify that the Hamiltonian and its ground state energy are correct.
#
# We now convert the integrals to the Physicist's notation.

two_so_converted = convert_integrals(two_so, 'quantum', 'chemist')

h = fermionic_observable(core_constant[0], one_so, two_so_converted, 'chemist')
qml.eigvals(qml.SparseHamiltonian(h.sparse_matrix(), wires=h.wires))

##############################################################################
# As another example, let's convert the Chemist's notation to the Quantum notation.

one_mo = np.einsum('pi,pq,qj->ij', rhf.mo_coeff, one_ao, rhf.mo_coeff)
two_mo = ao2mo.incore.full(two_ao, rhf.mo_coeff)
core_constant = np.array([rhf.energy_nuc()])

one_so = transform_one(one_mo)
two_so = transform_two(two_mo, 'chemist')

two_so_converted = convert_integrals(two_so, 'chemist', 'quantum')

h = fermionic_observable(core_constant[0], one_so, two_so_converted, 'quantum')
qml.eigvals(qml.SparseHamiltonian(h.sparse_matrix(), wires=h.wires))

##############################################################################
# Similarly, we can go from Chemist's notation to the Physicist's notation.

two_so_converted = convert_integrals(two_so, 'chemist', 'physicist')

h = fermionic_observable(core_constant[0], one_so, two_so_converted, 'physicist')
qml.eigvals(qml.SparseHamiltonian(h.sparse_matrix(), wires=h.wires))

##############################################################################
# The other possible conversions follow a similar logic.
#
# Active space
# ------------
# The computations performed so far explicitly correlate all electrons. A common approximation
# treats the core electrons as a static mean-field potential restricting the explicit correlation
# treatment to the valence orbitals. When we define such an active space, we need to include the
# effect of the core orbitals as corrections to the core constant term and the one-electron
# integrals. These corrections also depend on the original conventions we used to compute the
# integrals objects. Here we list the corrections that need to be included for each convention. Note
# that after applying the corrections, the Hamiltonians are constructed as before, except that the
# sums go over the active orbitals only.
#
# For the Quantum notation, the corrections to the core constant and the one-electron integrals are
#
# .. math::
#
#     E_{core} = E_0 + \sum_{i \in C} 2h_{ii}  + \sum_{ij \in C} 2 h_{ijji} - h_{ijij},
#
# where :math:`E_0` is the original nuclear repulsion constant and  :math:`C` denotes core
# molecular orbitals. Note that the sums are over molecular orbitals but can be simply extended to
# be on spin orbitals. Similarly, the one-electron integrals can be corrected with
#
# .. math::
#
#     h^{eff}_{pq} = h_{pq} + \sum_{i \in C} 2 h_{piiq} - h_{piqi}.
#
# For the Chemist notation, we can write similar corrections terms as
#
# .. math::
#
#     E_{core} = E_0 + \sum_{i \in C} 2h_{ii}  + \sum_{ij \in C} 2 h_{iijj} - h_{ijji},
#
# and
#
# .. math::
#
#     h^{eff}_{pq} = h_{pq} + \sum_{i \in C} 2 h_{pqii} - h_{piiq}.
#
# The corrections for the Physicist notation can be written similarly as
#
# .. math::
#
#     E_{core} = E_0 + \sum_{i \in C} 2h_{ii}  + \sum_{ij \in C} 2 h_{ijij} - h_{ijji},
#
# and
#
# .. math::
#
#     h^{eff}_{pq} = h_{pq} + \sum_{i \in C} 2 h_{piqi} - h_{piiq}.
#
# Now we define a function that implements the correction for each convention.

def active_space(core_constant, one, two, core, active, notation):

    if notation == 'quantum':
        for i in core:
            core_constant = core_constant + 2 * one[i][i]
            for j in core:
                core_constant = core_constant + 2 * two[i][j][j][i] - two[i][j][i][j]

        for p in active:
            for q in active:
                for i in core:
                    one[p, q] += 2 * two[p][i][i][q] - two[p][i][q][i]

    if notation == 'chemist':
        for i in core:
            core_constant = core_constant + 2 * one[i][i]
            for j in core:
                core_constant = core_constant + 2 * two[i][i][j][j] - two[i][j][j][i]

        for p in active:
            for q in active:
                for i in core:
                    one[p, q] += 2 * two[p][q][i][i] - two[p][i][i][q]

    if notation == 'physicist':
        for i in core:
            core_constant = core_constant + 2 * one[i][i]
            for j in core:
                core_constant = core_constant + 2 * two[i][j][i][j] - two[i][j][j][i]

        for p in active:
            for q in active:
                for i in core:
                    one[p, q] += 2 * two[p][i][q][i] - two[p][i][i][q]

    one = one[qml.math.ix_(active, active)]
    two = two[qml.math.ix_(active, active, active, active)]

    return core_constant, one, two

##############################################################################
# We can now use this function to perform computations in an active space where the first two
# orbitals are frozen.

core_constant, one, two = qml.qchem.electron_integrals(mol)()

one_chem = one.copy()
two_chem = convert_integrals(two, 'quantum', 'chemist')

core, active = [0, 1], [2, 3, 4, 5, 6]

core_constant, one, two = active_space(core_constant, one_chem, two_chem, core, active, 'chemist')

one_so = transform_one(one)
two_so = transform_two(two, 'chemist')

h = fermionic_observable(core_constant[0], one_so, two_so, 'chemist')
qml.eigvals(qml.SparseHamiltonian(h.sparse_matrix(), wires=h.wires))

##############################################################################
# The resulting ground state energy :math:`-74.99989932` is slightly higher because of the frozen
# orbital approximation we used.
#
# Appendix
# --------
# We use the molecular Hamiltonian corresponding to Physicist's convention to derive the Hamiltonian
# corresponding to the Chemist's convention. Recall the following anti-commutation rules for the
# fermionic creation and annihilation operators
#
# .. math::
#
#     [a^{\dagger}_i, a^{\dagger}_j]_+ = 0, \:\:\:\:\:\:\: [a_i, a_j]_+=0, \:\:\:\:\:\:\: [a_i, a^{\dagger}_j]_+ = \delta_{ij} I,
#
# where :math:`\delta_{ij}` and :math:`I` are the Kronecker delta and the identity operator,
# respectively. In the Hamiltonian represented by the Physicist's convention, we use these
# anti-commutation rules to move the :math:`a_r` operator from right to left. We first swap the
# operator with :math:`a_s` and then swap it again with :math:`a_q^{\dagger}`. This gives us the
# two-body term :math:`a_p^{\dagger} a_r a_q^{\dagger} a_s` and the one-body operator
# :math:`a_p^{\dagger} a_s`. Re-arranging the indices gives the Hamiltonian in the Chemist's
# convention.
#
# Conclusions
# -----------
# We recommend the following upgrades to PennyLane.
#
# 1. Add functions to convert one- and two-electron integral tensor from the molecular orbital (MO)
#    basis to the spin-orbital (SO) basis.
#
# 2. Add a function that converts a two-electron integral tensor between different conventions.
#
# 3. Upgrade the ``qchem.fermionic_observable`` function to support the Chemist's and Physicist's
#    conventions.
#
# References
# ----------
#
# .. [#szabo1996]
#
#     Attila Szabo, Neil S. Ostlund, "Modern Quantum Chemistry: Introduction to Advanced Electronic
#     Structure Theory". Dover Publications, 1996.
