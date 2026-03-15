r"""
Measurement optimization
========================

.. meta::
    :property="og:description": Optimize and reduce the number of measurements required to evaluate a variational algorithm cost function.
    :property="og:image": https://pennylane.ai/qml/_static/demonstration_assets/grouping.png

.. related::

   tutorial_vqe A brief overview of VQE
   tutorial_quantum_chemistry Building molecular Hamiltonians
   tutorial_qaoa_intro Intro to QAOA

The variational quantum eigensolver (VQE) is the OG variational quantum algorithm. Harnessing
near-term quantum hardware to solve for the electronic structure of molecules, VQE is *the*
algorithm that sparked the variational circuit craze of the last 5 years, and holds great
promise for showcasing a quantum advantage on near-term quantum hardware. It has also inspired
other quantum algorithms such as the :doc:`Quantum Approximate Optimization Algorithm (QAOA)
<demos/tutorial_qaoa_intro>`.

To scale VQE beyond the regime of classical computation, however, we need to solve for the
ground state of increasingly larger molecules. A consequence is that the number of
measurements we need to make on the quantum hardware also grows polynomially—a huge bottleneck,
especially when quantum hardware access is limited and expensive.

To mitigate this 'measurement problem', a plethora of recent research dropped over the course of
2019 and 2020 [#yen2020]_ [#izmaylov2019]_ [#huggins2019]_ [#gokhale2020]_ [#verteletskyi2020]_ ,
exploring potential strategies to minimize the number of measurements required. In fact, by grouping
commuting terms of the Hamiltonian, we can significantly reduce the number of
measurements needed—in some cases, reducing the number of measurements by up to 90%!

.. figure:: /_static/demonstration_assets/measurement_optimize/grouping.png
    :width: 90%
    :align: center

In this demonstration, we revisit the VQE algorithm, see first-hand how the required number of
measurements scales as molecule size increases, and finally use these measurement optimization
strategies to minimize the number of measurements we need to make. These techniques are valuable
beyond just VQE, allowing you to add measurement optimization to your toolkit of techniques to
perform variational algorithms more efficiently.

Revisiting VQE
--------------

The study of :doc:`variational quantum algorithms </glossary/variational_circuit>` was spearheaded
by the introduction of the :doc:`variational quantum eigensolver <demos/tutorial_vqe>` (VQE) algorithm in
2014 [#peruzzo2014]_. While classical variational techniques have been known for decades to estimate
the ground state energy of a molecule, VQE allowed this variational technique to be applied using
quantum computers. Since then, the field of variational quantum algorithms has evolved
significantly, with larger and more complex models being proposed (such as
:doc:`quantum neural networks <demos/quantum_neural_net>`, :doc:`QGANs <demos/tutorial_QGAN>`, and
:doc:`variational classifiers <demos/tutorial_variational_classifier>`). However, quantum chemistry
remains one of the flagship use-cases for variational quantum algorithms, and VQE the standard-bearer.

Part of the appeal of VQE lies within its simplicity. A circuit ansatz :math:`U(\theta)` is chosen
(typically the Unitary Coupled-Cluster Singles and Doubles
(:func:`~pennylane.templates.subroutines.UCCSD`) ansatz), and the qubit representation of the
molecular Hamiltonian is computed:

.. math:: H = \sum_i c_i h_i,

where :math:`h_i` are the terms of the Hamiltonian written as a tensor product of Pauli operators or the identity
acting on wire :math:`n,` :math:`P_n \in \{I, \sigma_x, \sigma_y, \sigma_z\}:`

.. math:: h_i = \bigotimes_{n=0}^{N-1} P_n.

(The :math:`h_i` product of Pauli terms is often referred to as a 'Pauli word' in the literature.) The cost
function of the VQE is then simply the expectation value of this Hamiltonian on the state obtained
after running the variational quantum circuit:

.. math:: \text{cost}(\theta) = \langle 0 | U(\theta)^\dagger H U(\theta) | 0 \rangle.

By using a classical optimizer to *minimize* this quantity, we can estimate
the ground state energy of the Hamiltonian :math:`H:`

.. math:: H U(\theta_{min}) |0\rangle = E_{min} U(\theta_{min}) |0\rangle.

In practice, when we are using quantum hardware to compute these expectation values we expand out
the Hamiltonian as its summation, resulting in separate expectation values that need to be calculated for each term:

.. math::

    \text{cost}(\theta) = \langle 0 | U(\theta)^\dagger \left(\sum_i c_i h_i\right) U(\theta) | 0 \rangle
                        = \sum_i c_i \langle 0 | U(\theta)^\dagger h_i U(\theta) | 0 \rangle.

.. note::

    How do we compute the qubit representation of the molecular Hamiltonian? This is a more
    complicated story that involves applying a self-consistent field method (such as Hartree-Fock),
    and then performing a fermionic-to-qubit mapping such as the Jordan-Wigner or Bravyi-Kitaev
    transformations.

    For more details on this process, check out the :doc:`demos/tutorial_quantum_chemistry`
    tutorial.

The measurement problem
-----------------------

For small molecules, the VQE algorithm scales and performs exceedingly well. For example, for the
Hydrogen molecule :math:`\text{H}_2,` the final Hamiltonian in its qubit representation
has 15 terms that need to be measured. Let's obtain the Hamiltonian from
`PennyLane's dataset library <https://pennylane.ai/datasets/qchem/h2-molecule>`__
to verify the number of terms. In this tutorial, we use the :func:`~.pennylane.data.load`
function to download the dataset of the molecule.

"""

import functools
import warnings
import jax
from jax import numpy as jnp
import pennylane as qml
qml.capture.enable()

jax.config.update("jax_enable_x64", True)

dataset = qml.data.load("qchem", molname="H2", bondlength=0.7)[0]
H, num_qubits = dataset.hamiltonian, len(dataset.hamiltonian.wires)

print("Required number of qubits:", num_qubits)
print(H)

###############################################################################
# Here, we can see that the Hamiltonian involves 15 terms, so we expect to compute 15 expectation values
# on hardware. Let's generate the cost function to check this.

# Create a 4 qubit simulator
dev = qml.device("default.qubit", seed=904932)

# number of electrons
electrons = 2

# Define the Hartree-Fock initial state for our variational circuit
initial_state = qml.qchem.hf_state(electrons, num_qubits)

# Construct the UCCSD ansatz
singles, doubles = qml.qchem.excitations(electrons, num_qubits)
s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)
ansatz = functools.partial(
    qml.UCCSD, init_state=initial_state, s_wires=s_wires, d_wires=d_wires
)

# generate the cost function
@qml.set_shots(1000)
@qml.qnode(dev, interface="jax")
def cost_circuit(params):
    ansatz(params, wires=range(num_qubits))
    return qml.expval(H)

##############################################################################
# If we evaluate this cost function, we can see that it corresponds to 15 different
# executions under the hood—one per expectation value:

from jax import random as random
key, scale = random.PRNGKey(0), jnp.pi
params = random.normal(key, shape=(len(singles) + len(doubles),)) * scale
with qml.Tracker(dev) as tracker:  # track the number of executions
    print("Cost function value:", cost_circuit(params))

print("Number of quantum evaluations:", tracker.totals['executions'])

##############################################################################
# How about a larger molecule? Let's try the
# `water molecule <https://pennylane.ai/datasets/qchem/h2o-molecule>`__:

dataset = qml.data.load('qchem', molname="H2O")[0]
H, num_qubits = dataset.hamiltonian, len(dataset.hamiltonian.wires)

print("Required number of qubits:", num_qubits)
print("Number of Hamiltonian terms/required measurements:", len(H.terms()[0]))

print("\n", H)


##############################################################################
# Simply going from two atoms in :math:`\text{H}_2` to three in :math:`\text{H}_2 \text{O}`
# resulted in over triple the number of qubits required and 1086 measurements that must be made!
#
# We can see that as the size of our molecule increases, we run into a problem: larger molecules
# result in Hamiltonians that not only require a larger number of qubits :math:`N` in their
# representation, but the number of terms in the Hamiltonian scales like
# :math:`\mathcal{O}(N^4)!` 😱😱😱
#
# We can mitigate this somewhat by choosing smaller `basis sets
# <https://en.wikipedia.org/wiki/Basis_set_(chemistry)>`__ to represent the electronic structure
# wavefunction, however this would be done at the cost of solution accuracy, and doesn't reduce the number of
# measurements significantly enough to allow us to scale to classically intractable problems.
#
# .. figure:: /_static/demonstration_assets/measurement_optimize/n4.png
#     :width: 70%
#     :align: center
#
#     The number of qubit Hamiltonian terms required to represent various molecules in the specified
#     basis sets (adapted from `Minimizing State Preparations for VQE by Gokhale et al.
#     <https://pranavgokhale.com/static/Minimizing_State_Preparations_for_VQE.pdf>`__)


##############################################################################
# Simultaneously measuring observables
# ------------------------------------
#
# One of the assumptions we made above was that every term in the Hamiltonian must be measured independently.
# However, this might not be the case. From the `Heisenberg uncertainty relationship
# <https://en.wikipedia.org/wiki/Uncertainty_principle>`__ for two
# observables :math:`\hat{A}` and :math:`\hat{B},` we know that
#
# .. math:: \sigma_A^2 \sigma_B^2 \geq \frac{1}{2}\left|\left\langle [\hat{A}, \hat{B}] \right\rangle\right|,
#
# where :math:`\sigma^2_A` and :math:`\sigma^2_B` are the variances of measuring the expectation value of the
# associated observables, and
#
# .. math:: [\hat{A}, \hat{B}] = \hat{A}\hat{B}-\hat{B}\hat{A}
#
# is the commutator. Therefore,
#
# - If the two observables :math:`\hat{A}` and :math:`\hat{B}` do not commute (:math:`[\hat{A},
#   \hat{B}] \neq 0`), then :math:`\sigma_A^2
#   \sigma_B^2 > 0` and we cannot simultaneously measure the expectation values of the two
#   observables.
#
# ..
#
# - If :math:`\hat{A}` and :math:`\hat{B}` **do** commute (:math:`[\hat{A},
#   \hat{B}] = 0`), then :math:`\sigma_A^2
#   \sigma_B^2 \geq 0` and there exists a measurement basis where we can **simultaneously measure** the
#   expectation value of both observables on the same state.
#
# To explore why commutativity and simultaneous measurement are related, let's assume that there
# is a complete, orthonormal eigenbasis :math:`|\phi_n\rangle` that *simultaneously
# diagonalizes* both :math:`\hat{A}` and :math:`\hat{B}:`
#
# .. math::
#
#     ① ~~ \hat{A} |\phi_n\rangle &= \lambda_{A,n} |\phi_n\rangle,\\
#     ② ~~ \hat{B} |\phi_n\rangle &= \lambda_{B,n} |\phi_n\rangle.
#
# where :math:`\lambda_{A,n}` and :math:`\lambda_{B,n}` are the corresponding eigenvalues.
# If we pre-multiply the first equation by :math:`\hat{B},` and the second by :math:`\hat{A}`
# (both denoted in blue):
#
# .. math::
#
#     \color{blue}{\hat{B}}\hat{A} |\phi_n\rangle &= \lambda_{A,n} \color{blue}{\hat{B}}
#       |\phi_n\rangle = \lambda_{A,n} \color{blue}{\lambda_{B,n}} |\phi_n\rangle,\\
#     \color{blue}{\hat{A}}\hat{B} |\phi_n\rangle &= \lambda_{B,n} \color{blue}{\hat{A}}
#       |\phi_n\rangle = \lambda_{A,n} \color{blue}{\lambda_{B,n}} |\phi_n\rangle.
#
# We can see that assuming a simultaneous eigenbasis requires that
# :math:`\hat{A}\hat{B}|\phi_n\rangle = \hat{B}\hat{A}|\phi_n\rangle.` Or, rearranging,
#
# .. math:: (\hat{A}\hat{B} - \hat{B}\hat{A}) |\phi_n\rangle = [\hat{A}, \hat{B}]|\phi_n\rangle = 0.
#
# Our assumption that :math:`|\phi_n\rangle` simultaneously diagonalizes both :math:`\hat{A}` and
# :math:`\hat{B}` only holds true if the two observables commute.
#
# So far, this seems awfully theoretical. What does this mean in practice?
#
# In the realm of variational circuits, we typically want to compute expectation values of an
# observable on a given state :math:`|\psi\rangle.` If we have two commuting observables, we now know that
# they share a simultaneous eigenbasis:
#
# .. math::
#
#     \hat{A} &= \sum_n \lambda_{A, n} |\phi_n\rangle\langle \phi_n|,\\
#     \hat{B} &= \sum_n \lambda_{B, n} |\phi_n\rangle\langle \phi_n|.
#
# Substituting this into the expression for the expectation values:
#
# .. math::
#
#     \langle\hat{A}\rangle &= \langle \psi | \hat{A} | \psi \rangle = \langle \psi | \left( \sum_n
#         \lambda_{A, n} |\phi_n\rangle\langle \phi_n| \right) | \psi \rangle = \sum_n \lambda_{A,n}
#         |\langle \phi_n|\psi\rangle|^2,\\
#     \langle\hat{B}\rangle &= \langle \psi | \hat{B} | \psi \rangle = \langle \psi | \left( \sum_n
#         \lambda_{B, n} |\phi_n\rangle\langle \phi_n| \right) | \psi \rangle = \sum_n \lambda_{B,n}
#         |\langle \phi_n|\psi\rangle|^2.
#
# So, assuming we know the eigenvalues of the commuting observables in advance, if we perform a
# measurement in their shared eigenbasis (the :math:`|\phi_n\rangle`), we only need to perform a **single measurement** of the
# probabilities :math:`|\langle \phi_n|\psi\rangle|^2` in order to recover both expectation values! 😍
#
# Fantastic! But, can we use this to reduce the number of measurements we need to perform in the VQE algorithm?
# To do so, we must find the answer to two questions:
#
# 1. How do we determine which terms of the cost Hamiltonian commute?
#
# 2. How do we rotate the circuit into the shared eigenbasis prior to measurement?
#
# The answers to these questions aren't necessarily easy nor straightforward. Thankfully, there are
# some recent techniques we can harness to address both.

##############################################################################
# Qubit-wise commuting Pauli terms
# --------------------------------
#
# Back when we summarized the VQE algorithm, we saw that each term of the Hamiltonian is generally represented
# as a tensor product of Pauli operators:
#
# .. math:: h_i = \bigotimes_{n=0}^{N-1} P_n.
#
# Luckily, this tensor product structure allows us to take a bit of a shortcut. Rather than consider
# **full commutativity**, we can consider a more strict condition known as **qubit-wise
# commutativity** (QWC).
#
# To start with, let's consider single-qubit Pauli operators and the identity. We know that the Pauli operators
# commute with themselves as well as the identity, but they do *not* commute with
# each other:
#
# .. math::
#
#     [\sigma_i, I] = 0, ~~~ [\sigma_i, \sigma_i] = 0, ~~~ [\sigma_i, \sigma_j] = c \sigma_k \delta_{ij}.
#
# Now consider two tensor products of Pauli terms, for example :math:`X\otimes Y \otimes I` and
# :math:`X\otimes I \otimes Z.` We say that these two terms are qubit-wise commuting, since, if
# we compare each subsystem in the tensor product, we see that every one commutes:
#
# .. math::
#
#     \begin{array}{ | *1c | *1c | *1c | *1c | *1c |}
#       X &\otimes &Y &\otimes &I\\
#       X &\otimes &I &\otimes &Z
#     \end{array} ~~~~ \Rightarrow ~~~~ [X, X] = 0, ~~~ [Y, I] = 0, ~~~ [I, Z] = 0.
#
# As a consequence, both terms must commute:
#
# .. math:: [X\otimes Y \otimes I, X\otimes I \otimes Z] = 0.
#
# .. important::
#
#     Qubit-wise commutativity is a **sufficient** but not **necessary** condition
#     for full commutativity. For example, the two Pauli terms :math:`Y\otimes Y` and
#     :math:`X\otimes X` are not qubit-wise commuting, but do commute (have a go verifying this!).
#
# Once we have identified a qubit-wise commuting pair of Pauli terms, it is also straightforward to
# find the gates to rotate the circuit into the shared eigenbasis. To do so, we simply rotate
# each qubit one-by-one depending on the Pauli operator we are measuring on that wire:
#
# .. rst-class:: docstable
#
#     +------------------+-------------------------------+
#     |    Observable    | Rotation gate                 |
#     +==================+===============================+
#     | :math:`X`        | :math:`RY(-\pi/2) = H`        |
#     +------------------+-------------------------------+
#     | :math:`Y`        | :math:`RX(\pi/2)=HS^{-1}=HSZ` |
#     +------------------+-------------------------------+
#     | :math:`Z`        | :math:`I`                     |
#     +------------------+-------------------------------+
#     | :math:`I`        | :math:`I`                     |
#     +------------------+-------------------------------+
#
# Therefore, in this particular example:
#
# * Wire 0: we are measuring both terms in the :math:`X` basis, apply the Hadamard gate
# * Wire 1: we are measuring both terms in the :math:`Y` basis, apply a :math:`RX(\pi/2)` gate
# * Wire 2: we are measuring both terms in the :math:`Z` basis (the computational basis), no gate needs to be applied.
#
# Let's use PennyLane to verify this.


obs = [
    qml.PauliX(0) @ qml.PauliY(1),
    qml.PauliX(0) @ qml.PauliZ(2)
]


##############################################################################
# First, let's naively use two separate circuit evaluations to measure
# the two QWC terms.


dev = qml.device("default.qubit", wires=3)

@qml.qnode(dev, interface="jax")
def circuit1(weights):
    qml.StronglyEntanglingLayers(weights, wires=range(3))
    return qml.expval(obs[0])


@qml.qnode(dev, interface="jax")
def circuit2(weights):
    qml.StronglyEntanglingLayers(weights, wires=range(3))
    return qml.expval(obs[1])

param_shape = qml.templates.StronglyEntanglingLayers.shape(n_layers=3, n_wires=3)
key, scale = random.PRNGKey(192933), 0.1
weights = scale * random.normal(key, shape=param_shape)

print("Expectation value of XYI = ", circuit1(weights))
print("Expectation value of XIZ = ", circuit2(weights))

##############################################################################
# Now, let's use our QWC approach to reduce this down to a *single* measurement
# of the probabilities in the shared eigenbasis of both QWC observables:

@qml.qnode(dev, interface="jax")
def circuit_qwc(weights):
    qml.StronglyEntanglingLayers(weights, wires=range(3))

    # rotate wire 0 into the shared eigenbasis
    qml.RY(-jnp.pi / 2, wires=0)

    # rotate wire 1 into the shared eigenbasis
    qml.RX(jnp.pi / 2, wires=1)

    # wire 2 does not require a rotation

    # measure probabilities in the computational basis
    return qml.probs(wires=range(3))


rotated_probs = circuit_qwc(weights)
print(rotated_probs)

##############################################################################
# We're not quite there yet; we have only calculated the probabilities of the variational circuit
# rotated into the shared eigenbasis—the :math:`|\langle \phi_n |\psi\rangle|^2.` To recover the
# *expectation values* of the two QWC observables from the probabilities, recall that we need one
# final piece of information: their eigenvalues :math:`\lambda_{A, n}` and :math:`\lambda_{B, n}.`
#
# We know that the single-qubit Pauli operators each have eigenvalues :math:`(1, -1),` while the identity
# operator has eigenvalues :math:`(1, 1).` We can make use of ``np.kron`` to quickly
# generate the eigenvalues of the full Pauli terms, making sure that the order
# of the eigenvalues in the Kronecker product corresponds to the tensor product.

eigenvalues_XYI = jnp.kron(jnp.kron(jnp.array([1, -1]), jnp.array([1, -1])), jnp.array([1, 1]))
eigenvalues_XIZ = jnp.kron(jnp.kron(jnp.array([1, -1]), jnp.array([1, 1])), jnp.array([1, -1]))

# Taking the linear combination of the eigenvalues and the probabilities
print("Expectation value of XYI = ", jnp.dot(eigenvalues_XYI, rotated_probs))
print("Expectation value of XIZ = ", jnp.dot(eigenvalues_XIZ, rotated_probs))


##############################################################################
# Compare this to the result when we used two circuit evaluations. We have successfully used a
# single circuit evaluation to recover both expectation values!
#
# Luckily, PennyLane automatically performs this QWC grouping under the hood. We simply
# return the two QWC Pauli terms from the QNode:

@qml.qnode(dev, interface="jax")
def circuit(weights):
    qml.StronglyEntanglingLayers(weights, wires=range(3))
    return [
        qml.expval(qml.PauliX(0) @ qml.PauliY(1)),
        qml.expval(qml.PauliX(0) @ qml.PauliZ(2))
    ]


print(circuit(weights))


##############################################################################
# Behind the scenes, PennyLane is making use of our built-in
# :mod:`qml.pauli <pennylane.pauli>` module, which contains functions for diagonalizing QWC
# terms:

rotations, new_obs = qml.pauli.diagonalize_qwc_pauli_words(obs)

print(rotations)
print(new_obs)


##############################################################################
# Here, the first line corresponds to the basis rotations that were discussed above, written in
# terms of ``RX`` and ``RY`` rotations. Check out the :mod:`qml.pauli <pennylane.pauli>`
# documentation for more details on its provided functionality and how it works.
#
# Given a Hamiltonian containing a large number of Pauli terms,
# there is a high likelihood of there being a significant number of terms that qubit-wise commute. Can
# we somehow partition the terms into **fewest** number of QWC groups to minimize the number of measurements
# we need to take?

##############################################################################
# Grouping QWC terms
# ------------------
#
# A nice example is provided in [#verteletskyi2020]_ showing how we might tackle this. Say we have
# the following Hamiltonian defined over four qubits:
#
# .. math:: H = Z_0 + Z_0 Z_1 + Z_0 Z_1 Z_2 + Z_0 Z_1 Z_2 Z_3 + X_2 X_3 + Y_0 X_2 X_3 + Y_0 Y_1 X_2 X_3,
#
# where we are using the shorthand :math:`P_0 P_2 = P\otimes I \otimes P \otimes I` for brevity.
# If we go through and work out which Pauli terms are qubit-wise commuting, we can represent
# this in a neat way using a graph:
#
# .. figure:: /_static/demonstration_assets/measurement_optimize/graph1.png
#     :width: 70%
#     :align: center
#
# In the above graph, every node represents an individual Pauli term of the Hamiltonian, with
# edges connecting terms that are qubit-wise commuting. Groups of qubit-wise commuting terms are
# represented as **complete subgraphs**. Straight away, we can make an observation:
# there is no unique solution for partitioning the Hamiltonian into groups of qubit-wise commuting
# terms! In fact, there are several solutions:
#
# .. figure:: /_static/demonstration_assets/measurement_optimize/graph2.png
#     :width: 90%
#     :align: center
#
# Of course, of the potential solutions above, there is one that is more optimal than the others ---
# on the bottom left, we have partitioned the graph into *two* complete subgraphs, as opposed to the
# other solutions that require three complete subgraphs. If we were to go with this solution,
# we would be able to measure the expectation value of the Hamiltonian using two circuit evaluations.
#
# This problem—finding the minimum number of complete subgraphs of a graph—is actually quite well
# known in graph theory, where it is referred to as the `minimum clique cover problem
# <https://en.wikipedia.org/wiki/Clique_cover>`__ (with 'clique' being another term for a complete subgraph).
#
# Unfortunately, that's where our good fortune ends—the minimum clique cover problem is known to
# be `NP-hard <https://en.wikipedia.org/wiki/NP-hardness>`__, meaning there is no known (classical)
# solution to finding the optimum/minimum clique cover in polynomial time.
#
# Thankfully, there is a silver lining: we know of polynomial-time algorithms for finding
# *approximate* solutions to the minimum clique cover problem. These heuristic approaches, while
# not guaranteed to find the optimum solution, scale quadratically with the number of nodes in the
# graph/terms in the Hamiltonian [#yen2020]_, so work reasonably well in practice.
#
# Many of these heuristic approaches have roots in another graph problem known as `graph
# colouring <https://en.wikipedia.org/wiki/Graph_coloring>`__; the assignment of colours to
# the graph's vertices such that no adjacent vertices have the same colour. How is this related
# to the minimum clique cover problem, though? If we take our QWC graph above, and generate the
# `complement graph <https://en.wikipedia.org/wiki/Complement_graph>`__ by drawing edges
# between all *non*-adjacent nodes,
#
# .. figure:: /_static/demonstration_assets/measurement_optimize/graph3.png
#     :width: 100%
#     :align: center
#
# we see that solving the minimum clique cover problem on the QWC graph is equivalent to solving the
# graph colouring problem on the complement graph using the minimum possible number of colours.
# While there are various different heuristic algorithms, a common one is `greedy colouring
# <https://en.wikipedia.org/wiki/Graph_coloring#Greedy_coloring>`__; in fact, the open-source graph
# package `NetworkX even provides a function for greedy colouring
# <https://networkx.org/documentation/stable//reference/algorithms/generated/networkx.algorithms.coloring.greedy_color.html#networkx.algorithms.coloring.greedy_color>`__,
# ``nx.greedy_color``.
#
# Let's give this a go, using NetworkX to solve the minimum clique problem for observable grouping.
# First, we'll need to generate the QWC graph (with each node corresponding to a Hamiltonian
# term, and edges indicating two terms that are QWC).

import networkx as nx
from matplotlib import pyplot as plt

terms = [
    qml.PauliZ(0),
    qml.PauliZ(0) @ qml.PauliZ(1),
    qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2),
    qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliZ(3),
    qml.PauliX(2) @ qml.PauliX(3),
    qml.PauliY(0) @ qml.PauliX(2) @ qml.PauliX(3),
    qml.PauliY(0) @ qml.PauliY(1) @ qml.PauliX(2) @ qml.PauliX(3)
]

def format_pauli_word(term):
    """Convenience function that nicely formats a PennyLane
    tensor observable as a Pauli word"""
    if isinstance(term, qml.ops.Prod):
        return " ".join([format_pauli_word(t) for t in term])

    return f"{term.name[-1]}{term.wires.tolist()[0]}"

G = nx.Graph()

with warnings.catch_warnings():
    # Muting irrelevant warnings
    warnings.filterwarnings(
        "ignore",
        message="The behaviour of operator ",
        category=UserWarning,
    )

    # add the terms to the graph
    G.add_nodes_from(terms)

    # add QWC edges
    G.add_edges_from([
        [terms[0], terms[1]],  # Z0 <--> Z0 Z1
        [terms[0], terms[2]],  # Z0 <--> Z0 Z1 Z2
        [terms[0], terms[3]],  # Z0 <--> Z0 Z1 Z2 Z3
        [terms[1], terms[2]],  # Z0 Z1 <--> Z0 Z1 Z2
        [terms[2], terms[3]],  # Z0 Z1 Z2 <--> Z0 Z1 Z2 Z3
        [terms[1], terms[3]],  # Z0 Z1 <--> Z0 Z1 Z2 Z3
        [terms[0], terms[4]],  # Z0 <--> X2 X3
        [terms[1], terms[4]],  # Z0 Z1 <--> X2 X3
        [terms[4], terms[5]],  # X2 X3 <--> Y0 X2 X3
        [terms[4], terms[6]],  # X2 X3 <--> Y0 Y1 X2 X3
        [terms[5], terms[6]],  # Y0 X2 X3 <--> Y0 Y1 X2 X3
    ])

    plt.margins(x=0.1)
    coords = nx.spring_layout(G, seed=1)
    nx.draw(
        G,
        coords,
        labels={node: format_pauli_word(node) for node in terms},
        with_labels=True,
        node_size=500,
        font_size=8,
        node_color="#9eded1",
        edge_color="#c1c1c1",
    )

    ##############################################################################
    # We can now generate the complement graph (compare this to our handdrawn
    # version above!):

    C = nx.complement(G)
    coords = nx.spring_layout(C, seed=1)

    nx.draw(
        C,
        coords,
        labels={node: format_pauli_word(node) for node in terms},
        with_labels=True,
        node_size=500,
        font_size=8,
        node_color="#9eded1",
        edge_color="#c1c1c1"
    )

    ##############################################################################
    # Now that we have the complement graph, we can perform a greedy coloring to
    # determine the minimum number of QWC groups:

    groups = nx.coloring.greedy_color(C, strategy="largest_first")

    # plot the complement graph with the greedy colouring
    nx.draw(
        C,
        coords,
        labels={node: format_pauli_word(node) for node in terms},
        with_labels=True,
        node_size=500,
        font_size=8,
        node_color=[("#9eded1", "#aad4f0")[groups[node]] for node in C],
        edge_color="#c1c1c1"
    )


num_groups = len(set(groups.values()))
print("Minimum number of QWC groupings found:", num_groups)


for i in range(num_groups):
    print(f"\nGroup {i}:")

    for term, group_id in groups.items():
        if group_id == i:
            print(format_pauli_word(term))

##############################################################################
# Putting it all together
# -----------------------
#
# So, we now have a strategy for minimizing the number of measurements we need to perform
# for our VQE problem:
#
# 1. Determine which terms of the Hamiltonian are qubit-wise commuting, and use
#    this to construct a graph representing the QWC relationship.
#
# 2. Construct the complement QWC graph.
#
# 3. Use a graph colouring heuristic algorithm to determine a graph colouring for the complement graph
#    with a minimum number of colours. Each coloured vertex set corresponds to a
#    qubit-wise commuting group of Hamiltonian terms.
#
# 4. Generate and evaluate the circuit ansatz (with additional rotations) per
#    QWC grouping, extracting probability distributions.
#
# 5. Finally, post-process the probability distributions with the observable eigenvalues
#    to recover the Hamiltonian expectation value.
#
# Luckily, the PennyLane ``pauli`` module makes this relatively easy. Let's walk through
# the entire process using the provided grouping functions.
#
# Steps 1-3 (finding and grouping QWC terms in the Hamiltonian) can be done via the
# :func:`qml.pauli.group_observables <pennylane.pauli.group_observables>` function:

obs_groupings = qml.pauli.group_observables(terms, grouping_type='qwc', method='rlf')


##############################################################################
# The ``grouping_type`` argument allows us to choose how the commuting terms
# are determined (more on that later!) whereas ``method`` determines the colouring
# heuristic (in this case, ``"rlf"`` refers to Recursive Largest First, a variant of largest first colouring heuristic).
#
# If we want to see what the required rotations and measurements are, we can use the
# :func:`qml.pauli.diagonalize_qwc_groupings <pennylane.pauli.diagonalize_qwc_groupings>`
# function:

rotations, measurements = qml.pauli.diagonalize_qwc_groupings(obs_groupings)

##############################################################################
# However, this isn't strictly necessary—recall previously that the QNode
# has the capability to *automatically* measure qubit-wise commuting observables!

dev = qml.device("lightning.qubit", wires=4)

@qml.qnode(dev, interface="jax")
def circuit(weights, group=None, **kwargs):
    qml.StronglyEntanglingLayers(weights, wires=range(4))
    return [qml.expval(o) for o in group]

param_shape = qml.templates.StronglyEntanglingLayers.shape(n_layers=3, n_wires=4)
key = random.PRNGKey(1)
weights = random.normal(key, shape=param_shape) * 0.1
result = [jnp.array(circuit(weights, group=g)) for g in obs_groupings]

print("Term expectation values:")
for group, expvals in enumerate(result):
    print(f"Group {group} expectation values:", expvals)

# Since all the coefficients of the Hamiltonian are unity,
# we can simply sum the expectation values.
print("<H> = ", jnp.sum(jnp.hstack(result)))


##############################################################################
# Finally, we don't need to go through this process manually every time; if our cost function can be
# written in the form of an expectation value of a Hamiltonian (as is the case for most VQE and QAOA
# problems), we can use the option ``grouping_type="qwc"`` in :class:`~.pennylane.Hamiltonian` to
# automatically optimize the measurements.

H = qml.Hamiltonian(coeffs=jnp.ones(len(terms)), observables=terms, grouping_type="qwc")
_, H_ops = H.terms()
@qml.qnode(dev, interface="jax")
def cost_fn(weights):
    qml.StronglyEntanglingLayers(weights, wires=range(4))
    return qml.expval(H)
print(cost_fn(weights))

##############################################################################
# Beyond VQE
# ----------
#
# Wait, hang on. We dove so deeply into measurement grouping and optimization, we forgot to check
# how this affects the number of measurements required to perform the VQE on :math:`\text{H}_2 \text{O}!`
# Let's use our new-found knowledge to see what happens.

dataset = qml.data.load('qchem', molname="H2O")[0]
H, num_qubits = dataset.hamiltonian, len(dataset.hamiltonian.wires)
print("Number of Hamiltonian terms/required measurements:", len(H_ops))

# grouping
groups = qml.pauli.group_observables(H_ops, grouping_type='qwc', method='rlf')
print("Number of required measurements after optimization:", len(groups))

##############################################################################
# We went from 1086 required measurements/circuit evaluations to 320 (just over *one thousand*
# down to *three hundred* 😱😱😱).
#
# As impressive as this is, however, this is just the beginning of the optimization.
#
# While finding qubit-wise commutating terms is relatively straightforward, with a little
# extra computation we can push this number down even further. Recent work has explored
# the savings that can be made by considering *full* commutativity [#yen2020]_, unitary
# partitioning [#izmaylov2019]_, and Fermionic basis rotation grouping [#huggins2019]_.
# Work has also been performed to reduce the classical overhead associated with measurement
# optimization, allowing the classical measurement grouping to be performed in linear time
# [#gokhale2020]_. For example, recall that qubit-wise commutativity is only a subset of
# full commutativity; if we consider full commutativity instead, we can further reduce the
# number of groups required.
#
# Finally, it is worth pointing out that, as the field of variational quantum algorithms grows, this
# problem of measurement optimization no longer just applies to the VQE algorithm (the algorithm it
# was born from). Instead, there are a multitude of algorithms that could benefit from these
# measurement optimization techniques (QAOA being a prime example).
#
# So the next time you are working on a variational quantum algorithm and the number
# of measurements required begins to explode—stop, take a deep breath 😤, and consider grouping
# and optimizing your measurements.
#
# .. note::
#
#     Qubit-wise commuting group information for a wide variety of molecules has been
#     pre-computed, and is available for download in
#     in the `PennyLane Datasets library <https://pennylane.ai/datasets>`__.

##############################################################################
# References
# ----------
#
# .. [#peruzzo2014]
#
#     Alberto Peruzzo, Jarrod McClean *et al.*, "A variational eigenvalue solver on a photonic
#     quantum processor". `Nature Communications 5, 4213 (2014).
#     <https://www.nature.com/articles/ncomms5213?origin=ppub>`__
#
# .. [#yen2020]
#
#     Tzu-Ching Yen, Vladyslav Verteletskyi, and Artur F. Izmaylov. "Measuring all compatible
#     operators in one series of single-qubit measurements using unitary transformations." `Journal of
#     Chemical Theory and Computation 16.4 (2020): 2400-2409.
#     <https://pubs.acs.org/doi/abs/10.1021/acs.jctc.0c00008>`__
#
# .. [#izmaylov2019]
#
#    Artur F. Izmaylov, *et al.* "Unitary partitioning approach to the measurement problem in the
#    variational quantum eigensolver method." `Journal of Chemical Theory and Computation 16.1 (2019):
#    190-195. <https://pubs.acs.org/doi/abs/10.1021/acs.jctc.9b00791>`__
#
# .. [#huggins2019]
#
#     William J. Huggins, *et al.* "Efficient and noise resilient measurements for quantum chemistry
#     on near-term quantum computers." `arXiv preprint arXiv:1907.13117 (2019).
#     <https://arxiv.org/abs/1907.13117>`__
#
# .. [#gokhale2020]
#
#    Pranav Gokhale, *et al.* "Minimizing state preparations in variational quantum eigensolver by
#    partitioning into commuting families." `arXiv preprint arXiv:1907.13623 (2019).
#    <https://arxiv.org/abs/1907.13623>`__
#
# .. [#verteletskyi2020]
#
#     Vladyslav Verteletskyi, Tzu-Ching Yen, and Artur F. Izmaylov. "Measurement optimization in the
#     variational quantum eigensolver using a minimum clique cover." `The Journal of Chemical Physics
#     152.12 (2020): 124114. <https://aip.scitation.org/doi/10.1063/1.5141458>`__
#
