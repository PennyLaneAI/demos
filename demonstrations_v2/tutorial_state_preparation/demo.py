r"""
.. _state_preparation:

Training a quantum circuit with PyTorch
=======================================

.. meta::
    :property="og:description": Build and optimize a circuit to prepare
        arbitrary single-qubit states, including mixed states, with PyTorch
        and PennyLane.
    :property="og:image": https://pennylane.ai/qml/_static/demonstration_assets/NOON.png

.. related::

   tutorial_qubit_rotation Basic tutorial: qubit rotation
   pytorch_noise PyTorch and noisy devices 
   tutorial_isingmodel_PyTorch 3-qubit Ising model in PyTorch

In this notebook, we build and optimize a circuit to prepare arbitrary
single-qubit states, including mixed states. Along the way, we also show
how to:

1. Construct compact expressions for circuits composed of many layers.
2. Succinctly evaluate expectation values of many observables.
3. Estimate expectation values from repeated measurements, as in real
   hardware.

"""

##############################################################################
# The most general state of a qubit is represented in terms of a positive
# semi-definite density matrix :math:`\rho` with unit trace. The density
# matrix can be uniquely described in terms of its three-dimensional
# *Bloch vector* :math:`\vec{a}=(a_x, a_y, a_z)` as:
#
# .. math:: \rho=\frac{1}{2}(\mathbb{1}+a_x\sigma_x+a_y\sigma_y+a_z\sigma_z),
#
# where :math:`\sigma_x, \sigma_y, \sigma_z` are the Pauli matrices. Any
# Bloch vector corresponds to a valid density matrix as long as
# :math:`\|\vec{a}\|\leq 1.`
#
# The *purity* of a state is defined as :math:`p=\text{Tr}(\rho^2),` which
# for a qubit is bounded as :math:`1/2\leq p\leq 1.` The state is pure if
# :math:`p=1` and maximally mixed if :math:`p=1/2.` In this example, we
# select the target state by choosing a random Bloch vector and
# renormalizing it to have a specified purity.
#
# To start, we import PennyLane, NumPy, and PyTorch for the optimization:

import pennylane as qml
qml.capture.enable()
import numpy as np
import torch
from torch.autograd import Variable
np.random.seed(42)

# we generate a three-dimensional random vector by sampling
# each entry from a standard normal distribution
v = np.random.normal(0, 1, 3)

# purity of the target state
purity = 0.66

# create a random Bloch vector with the specified purity
bloch_v = Variable(
    torch.tensor(np.sqrt(2 * purity - 1) * v / np.sqrt(np.sum(v ** 2))),
    requires_grad=False
)

# array of Pauli matrices (will be useful later)
Paulis = Variable(torch.zeros([3, 2, 2], dtype=torch.complex128), requires_grad=False)
Paulis[0] = torch.tensor([[0, 1], [1, 0]])
Paulis[1] = torch.tensor([[0, -1j], [1j, 0]])
Paulis[2] = torch.tensor([[1, 0], [0, -1]])

##############################################################################
# Unitary operations map pure states to pure states. So how can we prepare
# mixed states using unitary circuits? The trick is to introduce
# additional qubits and perform a unitary transformation on this larger
# system. By "tracing out" the ancilla qubits, we can prepare mixed states
# in the target register. In this example, we introduce two additional
# qubits, which suffices to prepare arbitrary states.
#
# The ansatz circuit is composed of repeated layers, each of which
# consists of single-qubit rotations along the :math:`x, y,` and :math:`z`
# axes, followed by three CNOT gates entangling all qubits. Initial gate
# parameters are chosen at random from a normal distribution. Importantly,
# when declaring the layer function, we introduce an input parameter
# :math:`j,` which allows us to later call each layer individually.

# number of qubits in the circuit
nr_qubits = 3
# number of layers in the circuit
nr_layers = 2

# randomly initialize parameters from a normal distribution
params = np.random.normal(0, np.pi, (nr_qubits, nr_layers, 3))
params = Variable(torch.tensor(params), requires_grad=True)

# a layer of the circuit ansatz
def layer(params, j):
    for i in range(nr_qubits):
        qml.RX(params[i, j, 0], wires=i)
        qml.RY(params[i, j, 1], wires=i)
        qml.RZ(params[i, j, 2], wires=i)

    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[0, 2])
    qml.CNOT(wires=[1, 2])


##############################################################################
# Here, we use the ``default.qubit`` device to perform the optimization, but this can be changed to
# any other supported device.

dev = qml.device("default.qubit", wires=3)

##############################################################################
# When defining the QNode, we introduce as input a Hermitian operator
# :math:`A` that specifies the expectation value being evaluated. This
# choice later allows us to easily evaluate several expectation values
# without having to define a new QNode each time.
#
# Since we will be optimizing using PyTorch, we configure the QNode
# to use the PyTorch interface:


@qml.qnode(dev, interface="torch")
def circuit(params, A):

    # repeatedly apply each layer in the circuit
    for j in range(nr_layers):
        layer(params, j)

    # returns the expectation of the input matrix A on the first qubit
    return qml.expval(qml.Hermitian(A, wires=0))


##############################################################################
# Our goal is to prepare a state with the same Bloch vector as the target
# state. Therefore, we define a simple cost function
#
# .. math::  C = \sum_{i=1}^3 \left|a_i-a'_i\right|,
#
# where :math:`\vec{a}=(a_1, a_2, a_3)` is the target vector and
# :math:`\vec{a}'=(a'_1, a'_2, a'_3)` is the vector of the state prepared
# by the circuit. Optimization is carried out using the Adam optimizer.
# Finally, we compare the Bloch vectors of the target and output state.

# cost function
def cost_fn(params):
    cost = 0
    for k in range(3):
        cost += torch.abs(circuit(params, Paulis[k]) - bloch_v[k])

    return cost


# set up the optimizer
opt = torch.optim.Adam([params], lr=0.1)

# number of steps in the optimization routine
steps = 200

# the final stage of optimization isn't always the best, so we keep track of
# the best parameters along the way
best_cost = cost_fn(params)
best_params = np.zeros((nr_qubits, nr_layers, 3))

print("Cost after 0 steps is {:.4f}".format(cost_fn(params)))

# optimization begins
for n in range(steps):
    opt.zero_grad()
    loss = cost_fn(params)
    loss.backward()
    opt.step()

    # keeps track of best parameters
    if loss < best_cost:
        best_cost = loss
        best_params = params

    # Keep track of progress every 10 steps
    if n % 10 == 9 or n == steps - 1:
        print("Cost after {} steps is {:.4f}".format(n + 1, loss))

# calculate the Bloch vector of the output state
output_bloch_v = np.zeros(3)
for l in range(3):
    output_bloch_v[l] = circuit(best_params, Paulis[l])

# print results
print("Target Bloch vector = ", bloch_v.numpy())
print("Output Bloch vector = ", output_bloch_v)
