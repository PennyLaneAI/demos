r"""How to quantum just-in-time compile VQE with Catalyst
=====================================================

The :doc:`Variational Quantum Eigensolver <demos/tutorial_vqe>` (VQE) is
a widely used quantum algorithm with applications in quantum chemistry and portfolio optimization
problems. It is an application of the `Ritz variational
principle <https://en.wikipedia.org/wiki/Ritz_method>`__, where a quantum computer is trained to
prepare the ground state of a given molecule.

Here, we will implement the VQE algorithm for the trihydrogen cation :math:`H_3^{+}` (three hydrogen
atoms sharing two electrons) using `Catalyst <https://github.com/PennyLaneAI/Catalyst>`__, a
quantum just-in-time framework for PennyLane, that allows hybrid quantum-classical workflows to be
compiled, optimized, and executed with a significant performance boost.

.. figure:: ../_static/demo_thumbnails/opengraph_demo_thumbnails/OGthumbnail_large_how-to-vqe-qjit_2024-04-23.png
    :align: center
    :width: 60%
    :target: javascript:void(0)

We will break the implementation into three steps:

1. Finding the molecular Hamiltonian for :math:`H_3^{+}.`
2. Preparing trial ground step (ansatz).
3. Optimizing the circuit to minimize the expectation value of the Hamiltonian.
"""

######################################################################
# Simple VQE workflow
# -------------------
#
# The VQE algorithm takes a molecular Hamiltonian and a parametrized circuit preparing the trial state
# of the molecule. The cost function is defined as the expectation value of the Hamiltonian computed
# in the trial state. With VQE, the lowest energy state of the Hamiltonian can be computed using an
# iterative optimization of the cost function. In PennyLane, this optimization is performed by a
# classical optimizer which (in principle) leverages a quantum computer to evaluate the cost function
# and calculate its gradient at each optimization step.
#
# Step 1: Create the Hamiltonian
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The first step is to specify the molecule we want to simulate. We can download the :math:`H_3^+`
# Hamiltonian from the `PennyLane Datasets
# service <https://pennylane.ai/datasets/qchem/h3-plus-molecule>`__:
#

import pennylane as qml
qml.capture.enable()
from pennylane import numpy as np

dataset = qml.data.load('qchem', molname="H3+")[0]
H, qubits = dataset.hamiltonian, len(dataset.hamiltonian.wires)

print(f"qubits: {qubits}")

######################################################################
# Step 2: Create the cost function
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Here, we prepare the initial state of the quantum circuit using the Hartree-Fock state, and then use
# the :class:`~pennylane.DoubleExcitation` template to create our parametrized circuit ansatz.
# Finally, we measure the expectation value of our Hamiltonian.
#

# The Hartree-Fock State
hf = dataset.hf_state

# Define the device, using lightning.qubit device
dev = qml.device("lightning.qubit", wires=qubits)

@qml.qnode(dev, diff_method="adjoint")
def cost(params):
    qml.BasisState(hf, wires=range(qubits))
    qml.DoubleExcitation(params[0], wires=[0, 1, 2, 3])
    qml.DoubleExcitation(params[1], wires=[0, 1, 4, 5])
    return qml.expval(H)

######################################################################
# Step 3: Optimize the circuit
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Since we are using AutoGrad, we can use PennyLane’s built-in
# :class:`~pennylane.GradientDescentOptimizer` to optimize the circuit.
#

init_params = np.array([0.0, 0.0])
opt = qml.GradientDescentOptimizer(stepsize=0.4)
steps = 10

params = init_params

for n in range(10):
    params, prev_energy = opt.step_and_cost(cost, params)
    print(f"--- Step: {n}, Energy: {cost(params):.8f}")

print(f"Final angle parameters: {params}")

######################################################################
# Quantum just-in-time compiling VQE
# ----------------------------------
#
# To take advantage of quantum just-in-time compilation, we need to modify the above workflow in two
# ways:
#
# 1. Instead of AutoGrad/NumPy, We need to use `JAX <https://github.com/google/jax>`__, a machine
#    learning framework that supports just-in-time compilation.
#
# 2. We need to decorate our workflow with the :func:`~pennylane.qjit` decorator, to indicate that
#    we want to quantum just-in-time compile the workflow with Catalyst.
#
# When dealing with more complex workflows, we may also need to update/adjust other aspects, including
# how we write control flow. For more details, see the `Catalyst sharp bits
# documentation <https://docs.pennylane.ai/projects/catalyst/en/stable/dev/sharp_bits.html>`__.
#
# Step 2: Create the QJIT-compiled cost function
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# When creating the cost function, we want to make sure that all parameters and arrays are created
# using JAX. We can now decorate the cost function with :func:`~pennylane.qjit`:
#
from jax import numpy as jnp

hf = np.array(dataset.hf_state)

@qml.qjit
@qml.qnode(dev)
def cost(params):
    qml.BasisState(hf, wires=range(qubits))
    qml.DoubleExcitation(params[0], wires=[0, 1, 2, 3])
    qml.DoubleExcitation(params[1], wires=[0, 1, 4, 5])
    return qml.expval(H)

init_params = jnp.array([0.0, 0.0])

cost(init_params)

######################################################################
# Step 3: Optimize the QJIT-compiled circuit
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We can now optimize the circuit. Unlike before, we cannot use PennyLane’s built-in optimizers (such
# as :func:`~.GradientDescentOptimizer`) here, as they are designed to work with AutoGrad and are
# not JAX compatible.
#
# Instead, we can use `Optax <https://github.com/google-deepmind/optax>`__, a library designed for
# optimization using JAX, as well as the :func:`~.catalyst.value_and_grad` function, which allows us to
# differentiate through quantum just-in-time compiled workflows while also returning the cost value.
# Here we use :func:`~.catalyst.value_and_grad` as we want to be able to print out and track our
# cost function during execution, but if this is not required the :func:`~.catalyst.grad` function
# can be used instead.
#

import catalyst
import optax

opt = optax.sgd(learning_rate=0.4)

@qml.qjit
def update_step(i, params, opt_state):
    """Perform a single gradient update step"""
    energy, grads = catalyst.value_and_grad(cost)(params)
    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    catalyst.debug.print("Step = {i},  Energy = {energy:.8f} Ha", i=i, energy=energy)
    return (params, opt_state)

opt_state = opt.init(init_params)
params = init_params

for i in range(10):
    params, opt_state = update_step(i, params, opt_state)

######################################################################
# Step 4: QJIT-compile the optimization
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# In the above example, we QJIT-compiled *just* the optimization update step, as well as the cost
# function. We can instead rewrite the above so that the *entire* optimization loop is QJIT-compiled,
# leading to further performance improvements:
#

@qml.qjit
def optimization(params):
    opt_state = opt.init(params)
    (params, opt_state) = qml.for_loop(0, 10, 1)(update_step)(params, opt_state)
    return params

final_params = optimization(init_params)
print(f"Final angle parameters: {final_params}")

######################################################################
# Note that here we use the QJIT-compatible :func:`~pennylane.for_loop` function, which allows
# classical control flow in hybrid quantum-classical workflows to be compiled.
#
