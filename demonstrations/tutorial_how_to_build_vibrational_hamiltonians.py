r"""How to build vibrational Hamiltonians
=========================================
Vibrational motions are crucial for describing the quantum properties of molecules and
materials. Molecular vibrations can affect the outcome of chemical reactions and there are several
vibrational spectroscopy techniques that provide valuable insight into understanding chemical
systems and processes [#loaiza]_ [#motlagh]_. Classical quantum
computations have been routinely implemented to describe vibrational motions of molecules. However,
for challenging vibrational systems, classical methods typically have fundamental theoretical
limitations that prevent their practical implementation. This makes quantum algorithms an ideal
choice where classical methods are not efficient or accurate.

Quantum algorithms require a precise description of the system Hamiltonian to compute vibrational
properties. In this demo, we learn how to use PennyLane features to construct and manipulate
different representations of vibrational Hamiltonians. We also briefly discuss the implementation of
the Hamiltonian in an interesting quantum algorithm for computing the vibrational dynamics of a
molecule.

.. figure:: ../_static/demo_thumbnails/opengraph_demo_thumbnails/OGthumbnail_vibrational_hamiltonian.png
    :align: center
    :width: 70%
    :target: javascript:void(0)
"""

######################################################################
# Vibrational Hamiltonian
# -----------------------
# A molecular vibrational Hamiltonian can be defined in terms of the kinetic energy operator of the
# nuclei, :math:`T`, and the potential energy operator, :math:`V`, which describes the interactions
# between the nuclei as:
#
# .. math::
#
#     H = T + V.
#
# The kinetic and potential energy operators can be written in terms of momentum and position
# operators, respectively. There are several ways to construct the potential energy operator which
# lead to different representations of the vibrational Hamiltonian. Here we explain some of these
# representations and provide PennyLane codes for constructing them.
#
# Christiansen representation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The Christiansen representation of the vibrational Hamiltonian relies on the n-mode expansion of
# the potential energy surface :math:`V` over vibrational normal coordinates :math:`Q`.
#
# .. math::
#
#     V({Q}) = \sum_i V_1(Q_i) + \sum_{ij} V_2(Q_i,Q_j) + ...,
#
# This provides a general representation of the potential energy surface where the terms :math:`V_n`
# depend on :math:`n` vibrational modes at most.
#
# The Christiansen Hamiltonian is then constructed in second-quantization based on this potential
# energy surface in terms of bosonic creation :math:`b^{\dagger}` and annihilation :math:`b`
# operations:
#
# .. math::
#
#     H = \sum_{i}^M \sum_{k_i, l_i}^{N_i} C_{k_i, l_i}^{(i)} b_{k_i}^{\dagger} b_{l_i} +
#         \sum_{i<j}^{M} \sum_{k_i,l_i}^{N_i} \sum_{k_j,l_j}^{N_j} C_{k_i k_j, l_i l_j}^{(i,j)}
#         b_{k_i}^{\dagger} b_{k_j}^{\dagger} b_{l_i} b_{l_j},
#
# where :math:`M` represents the number of normal modes and :math:`N` is the number of modals.
# Recall that a modal is a one-mode vibrational wave function defined as a function of a normal
# coordinate. The coefficients :math:`C` represent n-mode integrals which depend on the
# :math:`n`-mode contribution of the potential energy, :math:`V_n`, defined above.
#
# PennyLane provides a set of functions to construct the Christiansen Hamiltonian, either directly
# in one step or from its building blocks step by step. An important step in both methods is to
# construct the potential energy operator which is done based on
# single-point energy calculations along the normal modes of the molecule. The
# :func:`~.pennylane.qchem.vibrational_pes` function in PennyLane provides a convenient way to
# perform the potential energy calculations for a given molecule. The calculations are typically
# done by performing single-point energy calculations at small distances along the
# normal mode coordinates. Computing the energies for each mode separately provides the term
# :math:`V_1` while displacing atoms along two different modes simultaneously gives the term
# :math:`V_2` and so on.
#
# Let's look at an example for the HF molecule.

import numpy as np
import pennylane as qml

symbols  = ['H', 'F']
geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.79]])
mol = qml.qchem.Molecule(symbols, geometry)
pes = qml.qchem.vibrational_pes(mol, optimize=False)

######################################################################
# The :func:`~.pennylane.qchem.vibrational_pes` function creates a
# :class:`~.pennylane.qchem.VibrationalPES` object that stores the potential energy surface
# and vibrational information. This object is the input for several functions that we learn about
# here. For instance, the :func:`~.pennylane.qchem.christiansen_integrals` function accepts this
# object to compute the integrals needed to construct the Christiansen
# Hamiltonian defined above.

integrals = qml.qchem.vibrational.christiansen_integrals(pes, n_states=4)
h_bosonic = qml.qchem.christiansen_bosonic(integrals[0])
print(h_bosonic)

######################################################################
# The bosonic Hamiltonian constructed with :func:`~.pennylane.qchem.christiansen_bosonic` can be
# mapped to its qubit form by using the :func:`~.pennylane.qchem.christiansen_mapping` function.

h_qubit = qml.bose.christiansen_mapping(h_bosonic)
h_qubit

######################################################################
# This provides a vibrational Hamiltonian in the qubit basis that can be used in any desired quantum
# algorithm in PennyLane.
#
# Note that PennyLane also provides the :func:`~.pennylane.qchem.christiansen_hamiltonian` function
# that uses the :class:`~.pennylane.qchem.VibrationalPES` object directly and builds the 
# Christiansen Hamiltonian in qubit representation.

h_christiansen = qml.qchem.vibrational.christiansen_hamiltonian(pes,n_states=4)

######################################################################
# You can verify that the two Hamiltonians are identical.
#
# Taylor representation
# ^^^^^^^^^^^^^^^^^^^^^
# The Taylor representation of the vibrational Hamiltonian relies on a Taylor expansion of
# the potential energy surface :math:`V` in terms of the vibrational mass-weighted normal
# coordinate operators :math:`q`.
#
# .. math::
#
#     V = V_0 + \sum_i F_i q_i + \sum_{ij} F_{ij} q_i,q_j + ....
#
# Note that the force constants :math:`F` are derivatives of the potential energy surface.
#
# The Taylor Hamiltonian can then be constructed by defining the kinetic and potential energy
# components in terms of the momentum and position operators.
#
# .. math::
#
#     H = \sum_{i \ge j} K_{ij} p_i p_j + \sum_{i\ge j} \Phi_{ij}^{(2)} q_i q_j +
#          \sum_{i \ge j \ge k} \Phi_{ijk}^{(3)} q_i q_j q_k + ...,
#
# where the coefficients :math:`\Phi` are obtained from the force constants :math:`F` after
# rearranging the terms in the potential energy operator according to the number of different modes.
#
# The :func:`~.pennylane.qchem.taylor_coeffs` function computes the coefficients :math:`\Phi` from
# a :class:`~.pennylane.qchem.VibrationalPES` object.

one, two = qml.qchem.taylor_coeffs(pes, min_deg=2, max_deg=4)

######################################################################
# We can then use these coefficients to construct a bosonic form of the Taylor Hamiltonian.

h_bosonic = qml.qchem.taylor_bosonic(coeffs=[one, two], freqs=pes.freqs, uloc=pes.uloc)
print(h_bosonic)

######################################################################
# This bosonic Hamiltonian can be mapped to the qubit basis using mapping schemes of
# :func:`~.pennylane.bose.binary_mapping` or :func:`~.pennylane.bose.unary_mapping` functions.

h_qubit = qml.binary_mapping(h_bosonic, n_states=4)
h_qubit

######################################################################
# This Hamiltonian can be used in any desired quantum algorithm in PennyLane.
#
# Note that PennyLane also provides the :func:`~.pennylane.qchem.taylor_hamiltonian` function
# that uses the :class:`~.pennylane.qchem.VibrationalPES` object directly and builds the qubit
# Taylor Hamiltonian.

h_taylor = qml.qchem.vibrational.taylor_hamiltonian(pes, n_states=4)

######################################################################
# You can verify that the two Hamiltonians are identical.
#
# Conclusion
# ----------
# The `qchem module <https://docs.pennylane.ai/en/latest/code/qml_qchem.html>`__ in PennyLane
# provides a set of tools that can be used to construct several representations of vibrational
# Hamiltonians. Here we learned how to use these tools to build vibrational Hamiltonians
# step-by-step and also use the PennyLane built-in functions to easily construct the Hamiltonians in
# one step. The qchem module provides
# `features <https://docs.pennylane.ai/en/latest/code/api/pennylane.qchem.vibrational_pes.html>`__
# for constructing a vibrational potential energy surface efficiently using parallel executor
# options. The modular design of the vibrational features also facilitate further extension of the
# tools to support building other Hamiltonian types and representations and develop novel mapping
# methods intuitively.
#
# References
# ----------
#
# .. [#loaiza]
#
#     I. Loaiza *et al.*,
#     "Simulating near-infrared spectroscopy on a quantum computer for enhanced chemical detection",
#     arXiv:2504.10602, 2025.
#
# .. [#motlagh]
#
#     D. Motlagh *et al.*,
#     "Quantum Algorithm for Vibronic Dynamics: Case Study on Singlet Fission Solar Cell Design",
#     	arXiv:2411.13669, 2024.
#
# About the authors
# -----------------
#
