r"""

Resource estimation for quantum chemistry
=========================================

.. meta::
    :property="og:description": Learn how to estimate the number of qubits and gates needed to
     implement quantum algorithms
    :property="og:image": https://pennylane.ai/qml/_static/demonstration_assets/resource_estimation.jpeg

.. related::
    tutorial_quantum_chemistry Quantum chemistry with PennyLane
    tutorial_vqe A brief overview of VQE


Quantum algorithms such as
`quantum phase estimation <https://docs.pennylane.ai/en/stable/code/api/pennylane.QuantumPhaseEstimation.html>`_
(QPE) and the `variational quantum eigensolver <https://pennylane.ai/qml/demos/tutorial_vqe>`_ (VQE)
are widely studied in quantum chemistry as potential avenues to tackle problems that are intractable
for conventional computers. However, we currently do not have quantum computers or simulators
capable of implementing large-scale
versions of these algorithms. This makes it difficult to properly explore their accuracy and
efficiency for problem sizes where the actual advantage of quantum algorithms can potentially occur.
Despite these difficulties, it is still possible to perform **resource estimation**
to assess what we need to implement such quantum algorithms.

In this demo, we describe how to use PennyLane's resource estimation functionality to estimate the total number of logical qubits and gates required to
implement the QPE algorithm for simulating molecular Hamiltonians represented in first and second
quantization. We focus on `non-Clifford gates <https://en.wikipedia.org/wiki/Clifford_gates>`_, which
are the most expensive to implement in a fault-tolerant setting. We also explain how to estimate the
total number of measurements needed to compute expectation values using algorithms such as VQE.

Quantum Phase Estimation
------------------------
The QPE algorithm can be used to compute the phase associated with an eigenstate of a unitary
operator. For the purpose of quantum simulation, the unitary operator :math:`U` can be chosen to
share eigenvectors with a molecular Hamiltonian :math:`H,` for example by setting
:math:`U = e^{-iH}.` Estimating the phase of such a unitary then permits recovering the
corresponding eigenvalue of the Hamiltonian. A conceptual QPE circuit diagram is shown below.

.. figure:: /_static/demonstration_assets/resource_estimation/qpe.png
    :width: 60%
    :align: center

    Circuit representing the quantum phase estimation algorithm.

For most cases of interest, this algorithm requires more qubits and longer circuit depths than what
can be implemented on existing hardware. The PennyLane functionality in the
:mod:`qml.estimator <pennylane.estimator>` module allows us to estimate the number of logical qubits
and the number of non-Clifford gates that are needed to implement the algorithm. We can estimate
these resources by simply defining system specifications and a target error for estimation. Let's
see how!

QPE cost for simulating molecules
*********************************
We study the double low-rank Hamiltonian factorization algorithm of [#vonburg2021]_ and use its cost
equations as provided in APPENDIX C of [#lee2021]_.
This algorithm requires the one- and two-electron
`integrals <https://pennylane.ai/qml/demos/tutorial_differentiable_HF#the-hartree-fock-method>`_
as input. These integrals can be obtained in different ways and here we use PennyLane to compute
them. We first need to define the atomic symbols and coordinates for the given molecule. Let's use
the water molecule at its equilibrium geometry with the
`6-31g basis set <https://en.wikipedia.org/wiki/Basis_set_(chemistry)>`_ as an example.
"""
import pennylane as qml
qml.capture.enable()
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

symbols = ['O', 'H', 'H']
geometry = np.array([[0.00000000,  0.00000000,  0.28377432],
                     [0.00000000,  1.45278171, -1.00662237],
                     [0.00000000, -1.45278171, -1.00662237]])

##############################################################################
# Then we construct a molecule object and compute the one- and two-electron
# integrals in the molecular orbital basis.

mol = qml.qchem.Molecule(symbols, geometry, basis_name='6-31g')
core, one, two = qml.qchem.electron_integrals(mol)()

##############################################################################
# We now create an instance of the :class:`~.pennylane.estimator.qpe_resources.DoubleFactorization` class

algo = qml.estimator.DoubleFactorization(one, two)

##############################################################################
# and obtain the estimated number of non-Clifford gates and logical qubits.

print(f'Estimated gates : {algo.gates:.2e} \nEstimated qubits: {algo.qubits}')

##############################################################################
# This estimation is for a target error that is set to the chemical accuracy, 0.0016
# :math:`\text{Ha},` by default. We can change the target error to a larger value which leads to a
# smaller number of non-Clifford gates and logical qubits.

chemical_accuracy = 0.0016
error = chemical_accuracy * 10
algo = qml.estimator.DoubleFactorization(one, two, error=error)
print(f'Estimated gates : {algo.gates:.2e} \nEstimated qubits: {algo.qubits}')

##############################################################################
# We can also estimate the number of non-Clifford gates with respect to the threshold error values
# for discarding the negligible factors in the factorized Hamiltonian [#vonburg2021]_ and plot the
# estimated numbers.

threshold = [10**-n for n in range(10)]
n_gates = []
n_qubits = []

for tol in threshold:
    algo_ = qml.estimator.DoubleFactorization(one, two, tol_factor=tol, tol_eigval=tol)
    n_gates.append(algo_.gates)
    n_qubits.append(algo_.qubits)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(threshold, n_gates, ':o', markerfacecolor='none', color='teal')

ax.set_ylabel('n gates')
ax.set_xlabel('threshold')
ax.set_xscale('log')
fig.tight_layout()

##############################################################################
# QPE cost for simulating periodic materials
# ******************************************
# For periodic materials, we estimate the cost of implementing the QPE algorithm of [#zini2023]_
# using Hamiltonians represented in first quantization and in a plane wave basis. We first need to
# define the number of plane waves, the number of electrons, and the lattice vectors that construct
# the unit cell of the periodic material. Let's use dilithium iron silicate
# :math:`\text{Li}_2\text{FeSiO}_4` as an example taken from [#delgado2022]_. For this material, the
# unit cell contains 156 electrons and has dimensions :math:`9.49 \times 10.20 \times 11.83` in
# `atomic units <https://en.wikipedia.org/wiki/Bohr_radius>`_. We also use :math:`10^5` plane waves.

planewaves = 100000
electrons = 156
vectors = np.array([[9.49,  0.00,  0.00],
                    [0.00, 10.20,  0.00],
                    [0.00,  0.00, 11.83]])

##############################################################################
# We now create an instance of the :class:`~.pennylane.estimator.qpe_resources.FirstQuantization` class
algo = qml.estimator.FirstQuantization(planewaves, electrons, vectors=vectors)

##############################################################################
# and obtain the estimated number of non-Clifford gates and logical qubits.
print(f'Estimated gates : {algo.gates:.2e} \nEstimated qubits: {algo.qubits}')

##############################################################################
# We can also plot the estimated numbers as a function of the number of plane waves for different
# target errors

error = [0.1, 0.01, 0.001]  # in atomic units
planewaves = [10 ** n for n in range(1, 10)]
n_gates = []
n_qubits = []

for er in error:
    n_gates_ = []
    n_qubits_ = []

    for pw in planewaves:
        algo_ = qml.estimator.FirstQuantization(pw, electrons, vectors=vectors, error=er)
        n_gates_.append(algo_.gates)
        n_qubits_.append(algo_.qubits)
    n_gates.append(n_gates_)
    n_qubits.append(n_qubits_)

fig, ax = plt.subplots(2, 1)

for i in range(len(n_gates)):
    ax[0].plot(planewaves, n_gates[i], ':o', markerfacecolor='none', label=error[i])
ax[1].plot(planewaves, n_qubits[i], ':o', markerfacecolor='none', label=error[-1])

ax[0].set_ylabel('n gates')
ax[1].set_ylabel('n qubits')

for i in [0, 1]:
    ax[i].set_xlabel('n planewaves')
    ax[i].tick_params(axis='x')
    ax[0].set_yscale('log')
    ax[i].set_xscale('log')
    ax[i].legend(title='error [Ha]')

fig.tight_layout()

##############################################################################
# The algorithm uses a decomposition of the Hamiltonian as a linear combination of unitaries,
#
# .. math:: H=\sum_{i} c_i U_i.
#
# The parameter :math:`\lambda=\sum_i c_i,` which can be interpreted as the 1-norm of the
# Hamiltonian, plays an important role in determining the cost of implementing the QPE
# algorithm [#delgado2022]_. In PennyLane, the 1-norm of the Hamiltonian can be obtained with

print(f'1-norm of the Hamiltonian: {algo.lamb}')

##############################################################################
# PennyLane allows you to get more detailed information about the cost of the algorithms as
# explained in the documentation of :class:`~.pennylane.estimator.qpe_resources.FirstQuantization`
# and :class:`~.pennylane.estimator.qpe_resources.DoubleFactorization` classes.
#
# Variational quantum eigensolver
# ------------------------------------------
# In variational quantum algorithms such as VQE, the expectation value of an observable is
# typically computed by decomposing the observable into a linear combination of Pauli words,
# which are tensor products of Pauli and Identity operators. The expectation values are calculated
# through linearity by measuring the expectation value for each of these terms and combining the
# results. The number of qubits required for the measurement is trivially determined by
# the number of qubits the observable acts on. The number of gates required to implement the
# variational algorithm is determined by a circuit ansatz that is also known a priori. However,
# estimating the number of circuit evaluations, i.e. the number of shots, required to achieve a
# certain error in computing the expectation value is not as straightforward. Let's now use
# PennyLane to estimate the number of shots needed to compute the expectation value of the water
# Hamiltonian.
#
# First, we construct the molecular Hamiltonian.

molecule = qml.qchem.Molecule(symbols, geometry)
H = qml.qchem.molecular_hamiltonian(molecule)[0]
H_coeffs, H_ops = H.terms()

##############################################################################
# The number of measurements needed to compute :math:`\left \langle H \right \rangle` can be
# obtained with the :func:`~.pennylane.estimator.measurement.estimate_shots` function, which requires the
# Hamiltonian coefficients as input. The number of measurements required to compute
# :math:`\left \langle H \right \rangle` with a target error set to the chemical accuracy, 0.0016
# :math:`\text{Ha},` is obtained as follows.

m = qml.estimator.estimate_shots(H_coeffs)
print(f'Shots : {m:.2e}')

##############################################################################
# This number corresponds to the measurement process where each term in the Hamiltonian is measured
# independently. The number can be reduced by using
# :func:`~.pennylane.pauli.group_observables()`, which partitions the Pauli words into
# groups of commuting terms that can be measured simultaneously.

ops, coeffs = qml.pauli.group_observables(H_ops, H_coeffs)
coeffs = [np.array(c) for c in coeffs] # cast as numpy array

m = qml.estimator.estimate_shots(coeffs)
print(f'Shots : {m:.2e}')

##############################################################################
# It is also interesting to illustrate how the number of shots depends on the target error.

error = np.array([0.02, 0.015, 0.01, 0.005, 0.001])
m = [qml.estimator.estimate_shots(H_coeffs, error=er) for er in error]

e_ = np.linspace(error[0], error[-1], num=50)
m_ = 1.4e4 / np.linspace(error[0], error[-1], num=50)**2

fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(error, m, 'o', markerfacecolor='none', color='teal', label='estimated')
ax.plot(e_, m_, ':', markerfacecolor='none', color='olive', label='$ 1.4e4 * 1/\epsilon^2 $')

ax.set_ylabel('shots')
ax.set_xlabel('error [Ha]')
ax.set_yscale('log')
ax.tick_params(axis='x', labelrotation = 90)
ax.legend()
fig.tight_layout()

##############################################################################
# We have added a line showing the dependency of the shots to the error, as
# :math:`\text{shots} = 1.4\text{e}4 \times 1/\epsilon^2,` for comparison. Can you draw any
# interesting information form the plot?
#
# Conclusions
# -----------
# This tutorial shows how to use the resource estimation functionality in PennyLane to compute the
# total number of non-Clifford gates and logical qubits required to simulate a Hamiltonian with
# quantum phase estimation algorithms. The estimation can be performed for second-quantized
# molecular Hamiltonians obtained with a double low-rank factorization algorithm,
# and first-quantized Hamiltonians of periodic materials in the plane wave basis. We also discussed
# the estimation of the total number of shots required to obtain the expectation value of an
# observable using the variational quantum eigensolver algorithm. The functionality allows one to
# obtain interesting results about the cost of implementing important quantum algorithms. For
# instance, we estimated the costs with respect to factors such as the target error in obtaining
# energies and the number of basis functions used to simulate a system. Can you think of other
# interesting information that can be obtained using this PennyLane functionality?
#
# References
# ----------
#
# .. [#vonburg2021]
#
#     Vera von Burg, Guang Hao Low, Thomas Haner, Damian S. Steiger, *et al.*,
#     "Quantum computing enhanced computational catalysis".
#     `Phys. Rev. Research 3, 033055 (2021)
#     <https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.3.033055>`__
#
# .. [#lee2021]
#
#     Joonho Lee, Dominic W. Berry, Craig Gidney, William J. Huggins, *et al.*,
#     "Even More Efficient Quantum Computations of Chemistry Through Tensor Hypercontraction".
#     `PRX Quantum 2, 030305 (2021)
#     <https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.030305>`__
#
# .. [#zini2023]
#
#     Modjtaba Shokrian Zini, Alain Delgado, Roberto dos Reis, Pablo A. M. Casares,
#     Jonathan E. Mueller, Arne-Christian Voigt, Juan Miguel Arrazola,
#     "Quantum simulation of battery materials using ionic pseudopotentials".
#     `Quantum 7, 1049 (2023) <https://quantum-journal.org/papers/q-2023-07-10-1049>`__
#
# .. [#delgado2022]
#
#     Alain Delgado, Pablo A. M. Casares, Roberto dos Reis, Modjtaba Shokrian Zini, *et al.*,
#     "Simulating key properties of lithium-ion batteries with a fault-tolerant quantum computer".
#     `Phys. Rev. A 106, 032428 (2022)
#     <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.106.032428>`__
#
