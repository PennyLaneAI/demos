r"""QAOA for MaxCut
===================

In this tutorial we implement the quantum approximate optimization algorithm (QAOA) for the MaxCut
problem as proposed by `Farhi, Goldstone, and Gutmann (2014) <https://arxiv.org/abs/1411.4028>`__. First, we
give an overview of the MaxCut problem using a simple example, a graph with 4 vertices and 4 edges. We then
show how to find the maximum cut by running the QAOA algorithm using PennyLane.

Background
----------

The MaxCut problem
~~~~~~~~~~~~~~~~~~
The aim of MaxCut is to maximize the number of edges (yellow lines) in a graph that are "cut" by
a given partition of the vertices (blue circles) into two sets (see figure below).

.. figure:: ../_static/demonstration_assets/qaoa_maxcut/qaoa_maxcut_partition.png
   :align: center
   :scale: 65%
   :alt: qaoa_operators

|

Consider a graph with :math:`m` edges and :math:`n` vertices. We seek the partition
:math:`z` of the vertices into two sets
:math:`A` and :math:`B` which maximizes

.. math::
  C(z) = \sum_{\alpha=1}^{m}C_\alpha(z),

where :math:`C` counts the number of edges cut. :math:`C_\alpha(z)=1` if :math:`z` places one vertex from the
:math:`\alpha^\text{th}` edge in set :math:`A` and the other in set :math:`B,` and :math:`C_\alpha(z)=0` otherwise.
Finding a cut which yields the maximum possible value of :math:`C` is an NP-complete problem, so our best hope for a
polynomial-time algorithm lies in an approximate optimization.
In the case of MaxCut, this means finding a partition :math:`z` which
yields a value for :math:`C(z)` that is close to the maximum possible value.

We can represent the assignment of vertices to set :math:`A` or :math:`B` using a bitstring,
:math:`z=z_1...z_n` where :math:`z_i=0` if the :math:`i^\text{th}` vertex is in :math:`A` and
:math:`z_i = 1` if it is in :math:`B.` For instance,
in the situation depicted in the figure above the bitstring representation is :math:`z=0101\text{,}`
indicating that the :math:`0^{\text{th}}` and :math:`2^{\text{nd}}` vertices are in :math:`A`
while the :math:`1^{\text{st}}` and :math:`3^{\text{rd}}` are in
:math:`B.` This assignment yields a value for the objective function (the number of yellow lines cut)
:math:`C=4,` which turns out to be the maximum cut. In the following sections,
we will represent partitions using computational basis states and use PennyLane to
rediscover this maximum cut.

.. note:: In the graph above, :math:`z=1010` could equally well serve as the maximum cut.

A circuit for QAOA
~~~~~~~~~~~~~~~~~~~~
This section describes implementing a circuit for QAOA using basic unitary gates to find approximate
solutions to the MaxCut problem.
Firstly, denoting the partitions using computational basis states :math:`|z\rangle,` we can represent the terms in the
objective function as operators acting on these states

.. math::
  C_\alpha = \frac{1}{2}\left(1-\sigma_{z}^j\sigma_{z}^k\right),

where the :math:`\alpha\text{th}` edge is between vertices :math:`(j,k).`
:math:`C_\alpha` has eigenvalue 1 if and only if the :math:`j\text{th}` and :math:`k\text{th}`
qubits have different z-axis measurement values, representing separate partitions.
The objective function :math:`C` can be considered a diagonal operator with integer eigenvalues.

QAOA starts with a uniform superposition over the :math:`n` bitstring basis states,

.. math::
  |+_{n}\rangle = \frac{1}{\sqrt{2^n}}\sum_{z\in \{0,1\}^n} |z\rangle.


We aim to explore the space of bitstring states for a superposition which is likely to yield a
large value for the :math:`C` operator upon performing a measurement in the computational basis.
Using the :math:`2p` angle parameters
:math:`\boldsymbol{\gamma} = \gamma_1\gamma_2...\gamma_p,` :math:`\boldsymbol{\beta} = \beta_1\beta_2...\beta_p`
we perform a sequence of operations on our initial state:

.. math::
  |\boldsymbol{\gamma},\boldsymbol{\beta}\rangle = U_{B_p}U_{C_p}U_{B_{p-1}}U_{C_{p-1}}...U_{B_1}U_{C_1}|+_n\rangle

where the operators have the explicit forms

.. math::
  U_{B_l} &= e^{-i\beta_lB} = \prod_{j=1}^n e^{-i\beta_l\sigma_x^j}, \\
  U_{C_l} &= e^{-i\gamma_lC} = \prod_{\text{edge (j,k)}} e^{-i\gamma_l(1-\sigma_z^j\sigma_z^k)/2}.

In other words, we make :math:`p` layers of parametrized :math:`U_bU_C` gates.
These can be implemented on a quantum circuit using the gates depicted below, up to an irrelevant constant
that gets absorbed into the parameters.

.. figure:: ../_static/demonstration_assets/qaoa_maxcut/qaoa_operators.png
   :align: center
   :scale: 100%
   :alt: qaoa_operators

|

.. note::
    An alternative implementation of :math:`U_{C_l}` would be :math:`ZZ(\gamma_l)`, available
    via :class:`~.pennylane.IsingZZ` in PennyLane.

Let :math:`\langle \boldsymbol{\gamma},
\boldsymbol{\beta} | C | \boldsymbol{\gamma},\boldsymbol{\beta} \rangle` be the expectation of the objective operator.
In the next section, we will use PennyLane to perform classical optimization
over the circuit parameters :math:`(\boldsymbol{\gamma}, \boldsymbol{\beta}).`
This will specify a state :math:`|\boldsymbol{\gamma},\boldsymbol{\beta}\rangle` which is
likely to yield an approximately optimal partition :math:`|z\rangle` upon performing a measurement in the
computational basis.
In the case of the graph shown above, we want to measure either 0101 or 1010 from our state since these correspond to
the optimal partitions.

.. figure:: ../_static/demonstration_assets/qaoa_maxcut/qaoa_optimal_state.png
  :align: center
  :scale: 60%
  :alt: optimal_state

|

Qualitatively, QAOA tries to evolve the initial state into the plane of the
:math:`|0101\rangle,` :math:`|1010\rangle` basis states (see figure above).


Implementing QAOA in PennyLane
------------------------------

Imports and setup
~~~~~~~~~~~~~~~~~

To get started, we import PennyLane along with the PennyLane-provided
version of NumPy.
"""

import pennylane as qml
qml.capture.enable()
from pennylane import numpy as np

np.random.seed(42)

##############################################################################
# Operators
# ~~~~~~~~~
# We specify the number of qubits (vertices) with ``n_wires`` and
# compose the unitary operators using the definitions
# above. :math:`U_B` operators act on individual wires, while :math:`U_C`
# operators act on wires whose corresponding vertices are joined by an edge in
# the graph. We also define the graph using
# the list ``graph``, which contains the tuples of vertices defining
# each edge in the graph.

n_wires = 4
graph = [(0, 1), (0, 3), (1, 2), (2, 3)]


# unitary operator U_B with parameter beta
def U_B(beta):
    for wire in range(n_wires):
        qml.RX(2 * beta, wires=wire)


# unitary operator U_C with parameter gamma
def U_C(gamma):
    for edge in graph:
        qml.CNOT(wires=edge)
        qml.RZ(gamma, wires=edge[1])
        qml.CNOT(wires=edge)
        # Could also do
        # IsingZZ(gamma, wires=edge)


##############################################################################
# We will need a way to convert a bitstring, representing a sample of multiple qubits
# in the computational basis, to integer or base-10 form.


def bitstring_to_int(bit_string_sample):
    return int(2 ** np.arange(len(bit_string_sample)) @ bit_string_sample[::-1])


##############################################################################
# Circuit
# ~~~~~~~
# Next, we create a quantum device with 4 qubits.

dev = qml.device("lightning.qubit", wires=n_wires)

##############################################################################
# We also require a quantum node which will apply the operators according to the angle parameters,
# and return the expectation value of :math:`\sum_{\text{edge} (j,k)}\sigma_z^{j}\sigma_z^k`
# for the cost Hamiltonian :math:`C.`
# We set up this node to take the parameters ``gammas`` and ``betas`` as inputs, which determine
# the number of layers (repeated applications of :math:`U_BU_C`) of the circuit via their length.
# We also give the node a keyword argument ``return_samples``. If set to ``False`` (default), the
# expectation value of the cost Hamiltonian is returned.
# Once optimized, the same quantum node can be used for sampling an approximately optimal bitstring
# by setting ``return_samples=True``.


@qml.set_shots(20)
@qml.qnode(dev)
def circuit(gammas, betas, return_samples=False):
    # apply Hadamards to get the n qubit |+> state
    for wire in range(n_wires):
        qml.Hadamard(wires=wire)
    # p instances of unitary operators
    for gamma, beta in zip(gammas, betas):
        U_C(gamma)
        U_B(beta)

    if return_samples:
        # sample bitstrings to obtain cuts
        return qml.sample()
    # during the optimization phase we are evaluating the objective using expval
    C = qml.sum(*(qml.Z(w1) @ qml.Z(w2) for w1, w2 in graph))
    return qml.expval(C)


def objective(params):
    """Minimize the negative of the objective function C by postprocessing the QNnode output."""
    return -0.5 * (len(graph) - circuit(*params))


##############################################################################
# Optimization
# ~~~~~~~~~~~~
# Finally, we optimize the objective over the
# angle parameters :math:`\boldsymbol{\gamma}` (``params[0]``) and :math:`\boldsymbol{\beta}`
# (``params[1]``) and then sample the optimized
# circuit multiple times to yield a distribution of bitstrings. The optimal partitions
# (:math:`z=0101` or :math:`z=1010`) should be the most frequently sampled bitstrings.
# We perform a maximization of :math:`C` by minimizing :math:`-C,` following the convention
# that optimizations are cast as minimizations in PennyLane.


def qaoa_maxcut(n_layers=1):
    print(f"\np={n_layers:d}")

    # initialize the parameters near zero
    init_params = 0.01 * np.random.rand(2, n_layers, requires_grad=True)

    # initialize optimizer: Adagrad works well empirically
    opt = qml.AdagradOptimizer(stepsize=0.5)

    # optimize parameters in objective
    params = init_params.copy()
    steps = 30
    for i in range(steps):
        params = opt.step(objective, params)
        if (i + 1) % 5 == 0:
            print(f"Objective after step {i+1:3d}: {-objective(params): .7f}")

    # sample 100 bitstrings by setting return_samples=True and the QNode shot count to 100
    bitstrings = qml.set_shots(circuit, shots=100)(*params, return_samples=True)
    # convert the samples bitstrings to integers
    sampled_ints = [bitstring_to_int(string) for string in bitstrings]

    # print optimal parameters and most frequently sampled bitstring
    counts = np.bincount(np.array(sampled_ints))
    most_freq_bit_string = np.argmax(counts)
    print(f"Optimized parameter vectors:\ngamma: {params[0]}\nbeta:  {params[1]}")
    print(f"Most frequently sampled bit string is: {most_freq_bit_string:04b}")

    return -objective(params), sampled_ints


# perform QAOA on our graph with p=1,2 and keep the lists of sampled integers
int_samples1 = qaoa_maxcut(n_layers=1)[1]
int_samples2 = qaoa_maxcut(n_layers=2)[1]

##############################################################################
# For ``n_layers=1``, we find an objective function value of around :math:`C=3.`
# In the case where we set ``n_layers=2``, we recover the optimal
# objective function :math:`C=4.`
#
# Plotting the results
# --------------------
# We can plot the distribution of measurements obtained from the optimized circuits. As
# expected for this graph, the partitions 0101 and 1010 are measured with the highest frequencies,
# and in the case where we set ``n_layers=2`` we obtain one of the optimal partitions with
# 100% certainty.

import matplotlib.pyplot as plt

xticks = range(0, 16)
xtick_labels = list(map(lambda x: format(x, "04b"), xticks))
bins = np.arange(0, 17) - 0.5

fig, _ = plt.subplots(1, 2, figsize=(8, 4))
for i, samples in enumerate([int_samples1, int_samples2], start=1):
    plt.subplot(1, 2, i)
    plt.title(f"n_layers={i}")
    plt.xlabel("bitstrings")
    plt.ylabel("freq.")
    plt.xticks(xticks, xtick_labels, rotation="vertical")
    plt.hist(samples, bins=bins)
plt.tight_layout()
plt.show()

