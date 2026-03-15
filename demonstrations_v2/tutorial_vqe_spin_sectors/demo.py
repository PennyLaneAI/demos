r"""
VQE in different spin sectors
=============================

.. meta::
    :property="og:description": Find the lowest-energy states of a Hamiltonian in different spin sectors
    :property="og:image": https://pennylane.ai/qml/_static/demonstration_assets/thumbnail_spectra_h2.png

.. related::
   tutorial_vqe A brief overview of VQE
   vqe_parallel VQE with parallel QPUs with Rigetti

The Variational Quantum Eigensolver (VQE) algorithm is an approach for finding the
lowest-energy state of a molecule using a quantum computer [#peruzzo2014]_.

In the absence of `spin-orbit coupling <https://en.wikipedia.org/wiki/Spin-orbit_interaction>`_,
the eigenstates of the molecular Hamiltonian can be calculated for specific values of the
spin quantum numbers. This allows us to compute quantities such as the energy of the electronic
states in different sectors of the total-spin projection :math:`S_z.` This is illustrated in the
figure below for the energy spectrum of the hydrogen molecule. In this case, the ground state has
total spin :math:`S=0` while the lowest-lying excited states, with total spin :math:`S=1,` form
a triplet related to the spin components :math:`S_z=-1, 0, 1.`

|

.. figure:: /_static/demonstration_assets/vqe_spin_sectors/energy_spectra_h2_sto3g.png
    :width: 70%
    :align: center

|

In this tutorial we demonstrate how to run VQE simulations to find the lowest-energy states
of the hydrogen molecule in different spin sectors. First, we show how to
build the electronic Hamiltonian and the total spin operator :math:`\hat{S}^2.` Next, we use
excitation operations, implemented in PennyLane as Givens rotations [#qchemcircuits]_, to prepare
the trial states of the molecule. In order to probe the molecular states with different total spin,
:math:`S=0` and :math:`S=1,` we apply excitation operations which preserve and modify, respectively,
the total-spin projection of the initial state encoded in the qubit register. Finally, we run
the VQE algorithm to compute the energy of the states.

Let's get started!

Building the Hamiltonian and the total spin operator :math:`\hat{S}^2`
----------------------------------------------------------------------
First, we need to specify the structure of the molecule. This is done by providing a list
with the symbols of the constituent atoms and a one-dimensional array with the corresponding
nuclear coordinates in `atomic units <https://en.wikipedia.org/wiki/Hartree_atomic_units>`_.
"""

import numpy as np

symbols = ["H", "H"]
coordinates = np.array([0.0, 0.0, -0.6614, 0.0, 0.0, 0.6614])

##############################################################################
# The geometry of the molecule can also be imported from an external file using
# the :func:`~.pennylane.qchem.read_structure` function.
#
# Now, we can build the electronic Hamiltonian. We use a minimal `basis set
# <https://en.wikipedia.org/wiki/Basis_set_(chemistry)>`_ approximation to represent
# the `molecular orbitals <https://en.wikipedia.org/wiki/Molecular_orbital>`_. Then,
# the qubit Hamiltonian of the molecule is built using the
# :func:`~.pennylane.qchem.molecular_hamiltonian` function.

import pennylane as qml
qml.capture.enable()

H, qubits = qml.qchem.molecular_hamiltonian(symbols, coordinates)

print("Number of qubits = ", qubits)
print("The Hamiltonian is ", H)

##############################################################################
# The outputs of the function are the Hamiltonian and the number of qubits
# required for the quantum simulations. For the :math:`\mathrm{H}_2` molecule in a minimal
# basis, we have four molecular **spin**-orbitals, which defines the number of qubits.
#
# The :func:`~.pennylane.qchem.molecular_hamiltonian` function allows us to define
# additional keyword arguments to simulate more complicated molecules. For more details
# take a look at the tutorial :doc:`demos/tutorial_quantum_chemistry`.
#
# We also want to build the total spin operator :math:`\hat{S}^2,`
#
# .. math::
#
#     \hat{S}^2 = \frac{3}{4} N + \sum_{\alpha, \beta, \gamma, \delta}
#     \langle \alpha, \beta \vert \hat{s}_1 \cdot \hat{s}_2
#     \vert \gamma, \delta \rangle ~ \hat{c}_\alpha^\dagger \hat{c}_\beta^\dagger
#     \hat{c}_\gamma \hat{c}_\delta.
#
# In the equation above, :math:`N` is the number of electrons,
# :math:`\hat{c}` and :math:`\hat{c}^\dagger` are respectively the electron annihilation and
# creation operators, and
# :math:`\langle \alpha, \beta \vert \hat{s}_1 \cdot \hat{s}_2 \vert \gamma, \delta \rangle`
# is the `matrix element of the two-body spin operator
# <https://pennylane.readthedocs.io/en/stable/code/api/pennylane.qchem.obs.spin2.html>`_
# :math:`\hat{s}_1 \cdot \hat{s}_2` in the basis of spin orbitals.
#
# We use the :func:`~.pennylane.qchem.obs.spin2` function to build the
# :math:`\hat{S}^2` observable.

electrons = 2
S2 = qml.qchem.spin2(electrons, qubits)
print(S2)

##############################################################################
# Building the quantum circuit to find the ground state
# -----------------------------------------------------
# We build the variational circuit for the ground state of the hydrogen molecule using the
# :class:`~.pennylane.SingleExcitation` and :class:`~.pennylane.DoubleExcitation` operations
# which act as Givens rotations on subspaces of two and four qubits, respectively
# [#qchemcircuits]_. The total number of excitation operations is given by
# all possible single and double excitations of the Hartree-Fock reference state. The indices
# of the qubits they act on correspond to the spin-orbitals involved in each excitation.
# For more details on how to use the excitation operations see the
# tutorial :doc:`demos/tutorial_givens_rotations`.
#
# First, we use the :func:`~.pennylane.qchem.hf_state`
# function to generate the vector representing the Hartree-Fock state
# :math:`\vert 1100 \rangle` of the :math:`\mathrm{H}_2` molecule.

hf = qml.qchem.hf_state(electrons, qubits)
print(hf)

##############################################################################
# Next, we use the :func:`~.pennylane.qchem.excitations`
# function to generate all single and double excitations of the Hartree-Fock state.
# This function allows us to define the keyword argument ``delta_sz``
# to specify the total-spin projection of the excitations with respect to the reference
# state. This is illustrated in the figure below.
#
# |
#
# .. figure:: /_static/demonstration_assets/vqe_spin_sectors/fig_excitations.png
#     :width: 100%
#     :align: center
#
# |
#
# Therefore, for the ground state of the :math:`\mathrm{H}_2` molecule we choose
# ``delta_sz = 0``.

singles, doubles = qml.qchem.excitations(electrons, qubits, delta_sz=0)
print(singles)
print(doubles)

##############################################################################
# The output lists ``singles`` and ``doubles`` contain the qubit indices involved in the
# single and double excitations. Even and odd indices correspond, respectively, to spin-up
# and spin-down orbitals. For ``delta_sz = 0`` we have two single excitations, one from qubit
# 0 to 2 and the other from qubit 1 to 3, and one double excitation from qubits 0, 1 to 2, 3.
# In order to build the variational circuit, we apply the corresponding
# excitation operations. This can be done straightforwardly using the
# :class:`~.pennylane.templates.subroutines.AllSinglesDoubles` template.


def circuit(params, wires):
    qml.AllSinglesDoubles(params, wires, hf, singles, doubles)


##############################################################################
# The circuit above prepares trial states for the :math:`\mathrm{H}_2` molecule of the form:
#
# .. math::
#
#     \vert \Psi(\theta) \rangle =
#     c_\mathrm{HF}(\theta) \vert 1100 \rangle
#     + c_{0123}(\theta) \vert 0011 \rangle
#     + c_{02}(\theta) \vert 0110 \rangle
#     + c_{13}(\theta) \vert 1001 \rangle,
#
# where the coefficients :math:`c` are functions of the variational parameters
# :math:`\theta = (\theta_1, \theta_2, \theta_3)` to be optimized by the VQE algorithm.
# Note that the prepared state :math:`\vert \Psi(\theta) \rangle` is a superposition of the
# Hartree-Fock (HF) state with all possible single and double excitations that preserve
# the spin projection of the HF reference state.

##############################################################################
# Running the VQE simulation
# --------------------------
# Now we proceed to optimize the variational parameters. First, we define the device,
# in this case a qubit simulator:

dev = qml.device("lightning.qubit", wires=qubits)

##############################################################################
# Next, we define the cost function as the following QNode, where we make use of
# the :func:`~.pennylane.expval` function to compute the expectation value of the hamiltonian.
# This requires specifying the circuit, the target Hamiltonian, and the device. It returns
# a cost function that can be evaluated with the circuit parameters:


@qml.qnode(dev, interface="jax")
def cost_fn(params):
    circuit(params, wires=range(qubits))
    return qml.expval(H)

##############################################################################
# As a reminder, we also built the total spin operator :math:`\hat{S}^2` for which
# we can now define a function to compute its expectation value:


@qml.qnode(dev, interface="jax")
def S2_exp_value(params):
    circuit(params, wires=range(qubits))
    return qml.expval(S2)


##############################################################################
# The total spin :math:`S` of the trial state can be obtained from the
# expectation value :math:`\langle \hat{S}^2 \rangle` as
#
# .. math::
#
#     S = -\frac{1}{2} + \sqrt{\frac{1}{4} + \langle \hat{S}^2 \rangle}.
#
# We define a function to compute the total spin.


def total_spin(params):
    return -0.5 + np.sqrt(1 / 4 + S2_exp_value(params))


##############################################################################
# Now, we proceed to minimize the cost function to find the ground state. We define
# the classical optimizer and initialize the circuit parameters.

from jax import random
import optax

opt = optax.sgd(learning_rate=0.8) # sgd stands for StochasticGradientDescent
key = random.PRNGKey(0)
init_params = random.normal(key, shape=(len(singles) + len(doubles),)) * np.pi

##############################################################################
# We carry out the optimization over a maximum of 100 steps, aiming to reach a
# convergence tolerance of :math:`10^{-6}` for the value of the cost function.

import catalyst
max_iterations = 100

prev_energy = 0.0
@qml.qjit
def update_step(i, params, opt_state):
    """Perform a single gradient update step"""
    grads = qml.grad(cost_fn)(params)
    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return (params, opt_state)

loss_history = []

opt_state = opt.init(init_params)
params = init_params

for i in range(max_iterations):
    params, opt_state = update_step(i, params, opt_state)
    energy = cost_fn(params)

print("\n" f"Final value of the ground-state energy = {energy:.8f} Ha")
print("\n" f"Optimal value of the circuit parameters = {params}")

##############################################################################
# As a result, we have estimated the lowest-energy state of the hydrogen molecule
# with total spin :math:`S = 0` which corresponds to the ground state.
#
# Finding the lowest-lying excited state with :math:`S=1`
# -------------------------------------------------------
# In the last part of the tutorial, we will use VQE to find the lowest-lying
# excited state of the hydrogen molecule with total spin :math:`S=1.`
# In this case, we use the :func:`~.pennylane.qchem.excitations` function to generate
# excitations whose total-spin projection differs by the quantity ``delta_sz=1``
# with respect to the Hartree-Fock state.

singles, doubles = qml.qchem.excitations(electrons, qubits, delta_sz=1)
print(singles)
print(doubles)

##############################################################################
# For the :math:`\mathrm{H}_2` molecule in a minimal basis set there are no
# double excitations, but only a spin-flip single excitation from qubit 1 to 2.
# In this case, the circuit will contain only one :class:`~.pennylane.SingleExcitation`
# operation. Additionally, as we want to probe the excited state of the hydrogen molecule,
# we initialize the qubit register to the state :math:`\vert 0011 \rangle.`


def circuit(params, wires):
    qml.AllSinglesDoubles(params, wires, np.flip(hf), singles, doubles)


##############################################################################
# This allows us to prepare trial states of the form
#
# .. math::
#
#     \vert \Psi(\theta) \rangle = c_{03}(\theta) \vert 0101 \rangle
#     + c_{0123}(\theta) \vert 0011 \rangle,
#
# where the first term :math:`\vert 0101 \rangle` encodes a spin-flip excitation with
# :math:`S_z=-1` and the second term is a double excitation with :math:`S_z=0.`
# Since an eigenstate of the electronic Hamiltonian cannot contain a superposition of
# states with different total-spin projections, the double excitation coefficient
# should vanish as the VQE algorithm minimizes the cost function. The optimized state will
# correspond to the lowest-energy state with spin quantum numbers :math:`S=1, S_z=-1.`
#
# Now, we define the new functions to compute the expectation values of the Hamiltonian
# and the total spin operator for the new circuit.


@qml.qnode(dev, interface="jax")
def cost_fn(params):
    circuit(params, wires=range(qubits))
    return qml.expval(H)


@qml.qnode(dev, interface="jax")
def S2_exp_value(params):
    circuit(params, wires=range(qubits))
    return qml.expval(S2)


##############################################################################
# Finally, we generate the new set of initial parameters, and proceed with the VQE algorithm to
# optimize the variational circuit.

opt = optax.sgd(learning_rate=0.8) # sgd stands for StochasticGradientDescent
key = random.PRNGKey(0)
init_params = random.normal(key, shape=(len(singles) + len(doubles),)) * np.pi

max_iterations = 100

@qml.qjit
def update_step(i, params, opt_state):
    """Perform a single gradient update step"""
    grads = qml.grad(cost_fn)(params)
    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return (params, opt_state)

loss_history = []

opt_state = opt.init(init_params)
params = init_params

for i in range(max_iterations):
    params, opt_state = update_step(i, params, opt_state)
    energy = cost_fn(params)

print("\n" f"Final value of the energy = {energy:.8f} Ha")
print("\n" f"Optimal value of the circuit parameters = {params}")

##############################################################################
# As expected, the VQE algorithms has found the lowest-energy state with total spin
# :math:`S=1` which is an excited state of the hydrogen molecule.
#
# Conclusion
# ----------
# In this tutorial we have used the standard VQE algorithm to find the ground and the
# lowest-lying excited states of the hydrogen molecule. We used the single and
# double excitation operations, implemented as Givens rotations,
# to prepare the trial states of a molecule. We also showed how to build
# the total-spin operator :math:`\hat{S}^2` and used it to compute the total spin
# of the optimized states. By choosing the total-spin projection of the
# generated excitations, we were able to probe the lowest-energy eigenstates of the
# molecular Hamiltonian in different sectors of the spin quantum numbers.
#
# References
# ----------
#
# .. [#peruzzo2014]
#
#     A. Peruzzo, J. McClean *et al.*, "A variational eigenvalue solver on a photonic
#     quantum processor". `Nature Communications 5, 4213 (2014).
#     <https://www.nature.com/articles/ncomms5213?origin=ppub>`__
#
# .. [#qchemcircuits]
#
#     J.M. Arrazola, O. Di Matteo, N. Quesada, S. Jahangiri, A. Delgado, N. Killoran.
#     "Universal quantum circuits for quantum chemistry". `arXiv:2106.13839, (2021)
#     <https://arxiv.org/abs/2106.13839>`__
#
