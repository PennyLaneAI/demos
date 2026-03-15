r"""Learning quantum dynamics incoherently: Variational learning using classical shadows
========================================================================================

How can we recreate and simulate an unknown quantum process with a quantum circuit? One approach is
to learn the dynamics of this process incoherently, as done by Jerbi et al. [#Jerbi]_. 
Here, we'll reproduce the numerical simulations of [#Jerbi]_ using the authors' data, as provided
in the
`Learning Dynamics Incoherently PennyLane Dataset <https://pennylane.ai/datasets/other/learning-dynamics-incoherently>`__. 

This approach differs from learning the quantum process *coherently* [#Huang]_ because it does not
require the model circuit to be connected to the target quantum process. That is, the model circuit
does not receive quantum information from the target process directly. Instead, we train the model
circuit using classical information from the classical shadow measurements. This works well for
low-entangling processes but can require an exponential number of classical shadow measurements,
depending on the unknown quantum process [#Jerbi]_. This is useful because
it's not always possible to port the quantum output of a system directly to hardware without
first measuring it.

.. figure:: ../_static/demo_thumbnails/opengraph_demo_thumbnails/OGthumbnail_large_LearningDynamicsIncoherently.png
    :align: center
    :width: 80%

In simple terms, learning dynamics incoherently consists of two steps. First, we measure the output
of the unknown process for many different inputs. In this tutorial, we do this by measuring
:doc:`classical shadows <demos/tutorial_classical_shadows>` of the target process output.

Then, we adjust a variational quantum circuit
until it produces the same input-output combinations as the unknown process.
Here, we will
simulate the model circuit output and use the classical shadow measurements to estimate the
overlap between the model output states and the unknown process output states.
"""

######################################################################
# 1. Creating an unknown target quantum process
# ----------------------------------------------
#
# For our unknown quantum process, we will use the
# time evolution of a Hamiltonian:
#
# .. math:: U(H, t) = e^{-i H t / \hbar} .
#
# For the Hamiltonian, :math:`H,` we choose a transverse-field Ising Hamiltonian (as in
# [#Jerbi]_):
#
# .. math:: H = \sum_{i=0}^{n-1} Z_iZ_{i+1} + \sum_{i=0}^{n}\alpha_iX_i,
#
# where :math:`n` is the number of qubits and :math:`\alpha` are randomly generated weights.
# 
# More specifically, we will approximate :math:`U(H, t)` via
# `Trotterization <https://en.wikipedia.org/wiki/Hamiltonian_simulation#Product_formulas>`_.
# We first create the Hamiltonian and Trotterize later with :class:`~pennylane.TrotterProduct`.
#

import pennylane as qml
qml.capture.enable()
from pennylane import numpy as pnp
import numpy as np
import matplotlib.pyplot as plt

# number of qubits for the Hamiltonian
n_qubits = 2

# set random seed for reproducibility
pnp.random.seed(7)
np.random.seed(7)

# generate random weights
alphas = np.random.normal(0, 0.5, size=n_qubits)

# create the Hamiltonian
hamiltonian = qml.sum(
    *[qml.PauliZ(wires=i) @ qml.PauliZ(wires=i + 1) for i in range(n_qubits - 1)]
) + qml.dot(alphas, [qml.PauliX(wires=i) for i in range(n_qubits)])

######################################################################
# 2. Creating random initial states
# -----------------------------------
#
# The next step is to prepare a set of initial states. We will then apply the
# unknown quantum process to each of these states and measure the output to create input-output
# pairs. Later, we will train a model circuit to generate the same input-output pairs, thus
# reproducing the unknown quantum process.
#
# Ideally, our input states should be uniformly distributed over the state space. If they are all
# clustered together, our model circuit will not learn to approximate the unknown quantum process
# behavior for states that are very different from our training set.
#
# For quantum systems, this means we want to sample
# :doc:`Haar-random states <demos/tutorial_haar_measure>`, as done below.
#

from scipy.stats import unitary_group

n_random_states = 100

# Generate several random unitaries
random_unitaries = unitary_group.rvs(2**n_qubits, n_random_states)
# Take the first column of each unitary as a random state
random_states = [random_unitary[:, 0] for random_unitary in random_unitaries]

######################################################################
# .. note ::
#
#    On a personal computer, this method becomes slow (>1 second) around 10 qubits.
#


######################################################################
# 3. Time evolution and classical shadow measurements
# ----------------------------------------------------
#
# Now we can evolve the initial states using a Trotterized version of the
# `Hamiltonian above <#creating-an-unknown-target-quantum-process>`_. This
# will approximate the time evolution of the transverse-field Ising system.
#

dev = qml.device("default.qubit")

@qml.qnode(dev)
def target_circuit(input_state):
    # prepare training state
    qml.StatePrep(input_state, wires=range(n_qubits))
    
    # evolve the Hamiltonian for time=2 in n=1 steps with the order 1 formula
    qml.TrotterProduct(hamiltonian, time=2, n=1, order=1)
    return qml.classical_shadow(wires=range(n_qubits))


qml.draw_mpl(target_circuit)(random_states[0])
plt.show()

######################################################################
#
# Since ``target_circuit`` returns :func:`~pennylane.classical_shadow`, running the circuit with a
# ``shot`` value gives the desired number of classical shadow measurements.
# We use this to create a set of shadows for each initial state.
#

n_measurements = 10000

shadows = []
for random_state in random_states:
    bits, recipes = qml.set_shots(target_circuit, shots=n_measurements)(random_state)
    shadow = qml.ClassicalShadow(bits, recipes)
    shadows.append(shadow)


######################################################################
# 4. Creating a model circuit that will learn the target process
# ----------------------------------------------------------------
#
# Now that we have the classical shadow measurements, we need to create a model circuit that
# learns to produce the same output as the target circuit.
#
# As done in [#Jerbi]_, we create a model circuit with the same gate structure as the target
# circuit. If the target quantum process were truly unknown, then we would choose a general
# variational quantum circuit like in the
# :doc:`Variational classifier demo <demos/tutorial_variational_classifier>`.
#
# .. note ::
#
#    We use *local* measurements to keep the computational complexity low and
#    because classical shadows are well-suited to estimating local observables [#Jerbi]_.
#    For this reason, the following circuit returns local density matrices for each qubit. In
#    hardware, the density matrix is obtained via state tomography using Pauli measurements or classical shadows.


@qml.qnode(dev)
def model_circuit(params, random_state):
    qml.StatePrep(random_state, wires=range(n_qubits))
    # parameterized quantum circuit with the same gate structure as the target
    for i in range(n_qubits):
        qml.RX(params[i], wires=i)

    for i in reversed(range(n_qubits - 1)):
        qml.IsingZZ(params[n_qubits + i], wires=[i, i + 1])
    return [qml.density_matrix(i) for i in range(n_qubits)]


initial_params = pnp.random.random(size=n_qubits*2-1, requires_grad=True)

qml.draw_mpl(model_circuit)(initial_params, random_states[0])
plt.show()

######################################################################
# 5. Training using classical shadows in a cost function
# ------------------------------------------------------
#
# We now have to find the optimal parameters for ``model_circuit`` to mirror the ``target_circuit``.
# We can estimate the similarity between the circuits according to this cost function (see
# Appendix B of [#Jerbi]_):
#
# .. math:: C^l_N(\theta) = 1 - \frac{1}{nN}\sum^N_{j=1}\sum^n_{i=1}Tr[U|\psi^{(j)}\rangle\langle\psi^{(j)}|U^\dagger O^{(j)}_i(\theta)],
#
# where :math:`n` is the number of qubits, :math:`N` is the number of initial states, :math:`\psi^{(j)}`
# are random states, :math:`U` is our target unitary operation, and :math:`O_i` is the local density
# matrix for qubit :math:`i` after applying the ``model_circuit`.` That is, the local states
# :math:`\rho_{i}^{(j)}` are used as the observables:
#
# .. math:: O_{i}^{(j)}(\theta) := \rho_{i}^{(j)}.
#
# We can calculate this cost for our system by using the
# `shadow measurements <#time-evolution-and-classical-shadow-measurements>`_ to estimate
# the expectation value of :math:`O_i.` Roughly, this cost function measures the fidelity between
# the model circuit and the target circuit, by proxy of the single-qubit reduced states
# :math:`\rho_{i}^{(j)}` of the model over a variety of input-output pairs.


def cost(params):
    cost = 0.0
    for idx, random_state in enumerate(random_states):
        # obtain the density matrices for each qubit
        observable_mats = model_circuit(params, random_state)
        # convert to a PauliSentence
        observable_pauli = [
            qml.pauli_decompose(observable_mat, wire_order=[qubit])
            for qubit, observable_mat in enumerate(observable_mats)
        ]
        # estimate the overlap for each qubit
        cost = cost + qml.math.sum(shadows[idx].expval(observable_pauli))
    cost = 1 - cost / n_qubits / n_random_states
    return cost


params = initial_params

optimizer = qml.GradientDescentOptimizer(stepsize=5)
steps = 50

costs = [None]*(steps+1)
params_list = [None]*(steps+1)

params_list[0]=initial_params
for i in range(steps):
    params_list[i + 1], costs[i] = optimizer.step_and_cost(cost, params_list[i])

costs[-1] = cost(params_list[-1])

print("Initial cost:", costs[0])
print("Final cost:", costs[-1])

######################################################################
#
# We can plot the cost over the iterations and compare it to the ideal cost.
#


# find the ideal parameters from the original Trotterized Hamiltonian
ideal_parameters = [
    op.decomposition()[0].parameters[0]
    for op in qml.TrotterProduct(hamiltonian, 2, 1, 1).decomposition()
]
ideal_parameters = ideal_parameters[:n_qubits][::-1] + ideal_parameters[n_qubits:]

ideal_cost = cost(ideal_parameters)

plt.plot(costs, label="Training")
plt.plot([0, steps], [ideal_cost, ideal_cost], "r--", label="Ideal parameters")
plt.ylabel("Cost")
plt.xlabel("Training iterations")
plt.legend()
plt.show()

######################################################################
# In this case, we see
# that the ideal cost is greater than 0. This is because for the ideal parameters, the model outputs
# and target outputs are equal:
#
# .. math:: \rho_{i}^{(j)} := O_i = U|\psi^{(j)}\rangle\langle\psi^{(j)}|U^\dagger.
#
# Since the single-qubit reduced states used in the
# cost function are mixed states, the trace of their square is less than one:
#
# .. math:: Tr[(\rho_{i}^{(j)})^2] < 1.
#
# The ideal cost :math:`C^l_N(\theta)` is therefore greater than 0.
#
# We can also look at the :func:`trace_distance <pennylane.math.trace_distance>` between the unitary
# matrix of the target circuit and the model circuit. As the circuits become more similar with each
# training iteration, we should see the trace distance decrease and reach a low value.
#

import scipy

target_matrix = qml.matrix(
    qml.TrotterProduct(hamiltonian, 2, 1, 1),
    wire_order=range(n_qubits),
)

zero_state = [1] + [0]*(2**n_qubits-1)

# model matrix using the all-|0> state to negate state preparation effects
model_matrices = [qml.matrix(model_circuit, wire_order=range(n_qubits))(params, zero_state) for params in params_list]
trace_distances = [qml.math.trace_distance(target_matrix, model_matrix) for model_matrix in model_matrices] 

plt.plot(trace_distances)
plt.ylabel("Trace distance")
plt.xlabel("Training iterations")
plt.show()

print("The final trace distance is: \n", trace_distances[-1])


######################################################################
# Using the Learning Dynamics Incoherently PennyLane Dataset
# ----------------------------------------------------------
#
# In Jerbi et al. [#Jerbi]_, the authors perform this procedure to learn dynamics incoherently on a
# larger, 16-qubit transverse-field Ising
# Hamiltonian, and use classical shadow samples from quantum hardware to estimate the cost function.
# The corresponding `Learning Dynamics Incoherently PennyLane Dataset <https://pennylane.ai/datasets/other/learning-dynamics-incoherently>`__
# can be downloaded via the :mod:`qml.data` module.

[ds] = qml.data.load("other", name="learning-dynamics-incoherently")

# print the available data
print(ds.list_attributes())

# print more information about the hamiltonian
print(ds.attr_info["hamiltonian"]["doc"])

######################################################################
#
# The unknown target Hamiltonian, Haar-random initial states, and resulting classical shadow
# measurements are all available in the dataset.
#
# .. note ::
#
#    We use few shadows to keep the computational time low and the dataset contains only two
#    training states.

random_states = ds.training_states

n_measurements = 10000
shadows = [qml.ClassicalShadow(shadow_meas[:n_measurements], shadow_bases[:n_measurements]) for shadow_meas, shadow_bases in zip(ds.shadow_meas,ds.shadow_bases)]

######################################################################
#
# We only need to create the model circuit, cost function, and train.
# For these we use the same model circuit as
# `above <#creating-a-model-circuit-that-will-learn-the-target-process>`_, updated to reflect
# the increased number of qubits.
#

dev = qml.device("default.qubit")

@qml.qnode(dev)
def model_circuit(params, random_state):
    # this is a parameterized quantum circuit with the same gate structure as the target unitary
    qml.StatePrep(random_state, wires=range(16))
    for i in range(16):
        qml.RX(params[i], wires=i)

    for i in reversed(range(15)):
        qml.IsingZZ(params[16 + i], wires=[i, i + 1])
    return [qml.density_matrix(i) for i in range(16)]


initial_params = pnp.random.random(size=31)

qml.draw_mpl(model_circuit)(initial_params, random_states[0])
plt.show()

######################################################################
#
# We can then minimize the cost to train the model to output the same states as the target circuit.
# For this, we can use the cost function from
# `before <#training-using-classical-shadows-in-a-cost-function>`_, 
# as long as we update the number of qubits and the number of random states.
#

n_qubits = 16
n_random_states = len(ds.training_states)

optimizer = qml.GradientDescentOptimizer(stepsize=5)
steps = 50

costs = [None]*(steps+1)
params_list = [None]*(steps+1)

params_list[0]=initial_params
for i in range(steps):
    params_list[i + 1], costs[i] = optimizer.step_and_cost(cost, params_list[i])

costs[-1] = cost(params_list[-1])

print("Initial cost:", cost(initial_params))
print("Final cost:", costs[-1])

######################################################################
# As a quick check, we can take a look at the density matrices
# to see whether the training was successful:
#

original_matrices = model_circuit(initial_params, random_states[0])
learned_matrices = model_circuit(params_list[-1], random_states[0])
target_matrices_shadow = np.mean(shadows[0].local_snapshots(), axis=0)

print("Untrained example output state\n", original_matrices[0])
print("Trained example output state\n", learned_matrices[0])
print("Target output state\n", target_matrices_shadow[0])

######################################################################
#
# After training, the model outputs are closer to the target outputs, but not quite the same.
# This is due to the limitations of this learning method. Even for a simple circuit like the
# short-time evolution of a first order single Trotter step, it requires a large number of
# shadow measurements and training states to faithfully reproduce the underlying quantum process.
# The results can be improved by increasing the number of training states and
# :doc:`classical shadow measurements <demos/tutorial_classical_shadows>`.
#


##############################################################################
#
# References
# ------------
#
# .. [#Jerbi]
#
#     Sofiene Jerbi, Joe Gibbs, Manuel S. Rudolph, Matthias C. Caro, Patrick J. Coles, Hsin-Yuan Huang, Zoë Holmes
#     "The power and limitations of learning quantum dynamics incoherently"
#     `arXiv:2303.12834 <https://arxiv.org/abs/2303.12834>`__, 2005.
#
# .. [#Huang]
#
#     Hsin-Yuan Huang, Michael Broughton, Jordan Cotler, Sitan Chen, Jerry Li, Masoud Mohseni, Hartmut Neven, Ryan Babbush, Richard Kueng, John Preskill, and Jarrod R. McClean
#     "Quantum advantage in learning from experiments"
#     `Science <http://dx.doi.org/10.1126/science.abn7293>`__, 2022
#
#
