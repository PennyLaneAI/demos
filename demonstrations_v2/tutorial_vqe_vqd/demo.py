r"""How to implement VQD with PennyLane
===============================================================

Finding the eigenvalues of a Hamiltonian is a key task in quantum computing. Algorithms such as the variational quantum eigensolver (VQE) are used to find the smallest
eigenvalue, but sometimes we are interested in other eigenvalues. Here we will show you how to implement the variational
quantum deflation (VQD) algorithm in PennyLane
and find the first excited state energy of the `hydrogen molecule <https://pennylane.ai/datasets/qchem/h2-molecule>`__. To benefit the most from this tutorial, we recommend
a familiarization with the `variational quantum eigensolver (VQE) algorithm <demos/tutorial_vqe/>`__ first.

.. figure:: ../_static/demo_thumbnails/opengraph_demo_thumbnails/OGthumbnail_how_to_vqd_pennylane.png
    :align: center
    :width: 70%
    :target: javascript:void(0)

"""

######################################################################
#
# Variational quantum deflation
# ------------------------------
#
# The VQD algorithm [#Vqd]_ is a method used to find the excited states of a quantum system.
# It is related to the `VQE algorithm <demos/tutorial_vqe/>`__, which is often used to find the ground state energy of a quantum system.
# The main idea of the VQE algorithm is to define a quantum state ansatz that depends on adjustable parameters :math:`\theta` and minimize the energy of the system, computed as:
#
# .. math:: C_0(\theta) = \left\langle\Psi_0 (\theta)|\hat H |\Psi_0 (\theta) \right\rangle,
#
# where :math:`\Psi_0(\theta)` is the ansatz. However, this is not enough if we are interested in states other than the ground state.
# We must find a function whose minimum is no longer the ground state energy but gives the next excited state.
# This is possible by just adding a penalty term to the above function, which accounts for the orthogonality of the states:
#
# .. math:: C_1(\theta) = \left\langle\Psi(\theta)|\hat H |\Psi (\theta) \right\rangle + \beta | \left\langle \Psi (\theta)| \Psi_0 \right\rangle|^2,
#
# where :math:`\beta` is a hyperparameter that controls the penalty term and :math:`| \Psi \rangle` is the excited state.
# Note that :math:`\beta` should be larger than the energy gap between the ground and excited states.
# This function can now be minimized to give the first excited state energy.
# Similarly, we could iteratively calculate the :math:`k`-th excited states by adding the corresponding penalty term to the previous :math:`k - 1` excited state.
#
# As easy as that! Let's see how we can run this using PennyLane.
#
#
# Finding the ground state
# -------------------------------------------
#
# To implement the VQD algorithm, we first need to know the ground state of our system, and it is a breeze to use the data from `PennyLane Datasets <https://pennylane.ai/datasets/>`__  to obtain the Hamiltonian and the ground state
# of the hydrogen molecule:
#
# .. note::
#
#     To improve viewability of this tutorial, we will suppress any ``ComplexWarning``'s which may be raised during optimization.
#     The warnings do not impact the correctness of the results, but make it harder to view outputs.
#

import pennylane as qml
qml.capture.enable()
import numpy as np

import warnings
warnings.filterwarnings(action="ignore", category=np.exceptions.ComplexWarning)

# Load the dataset
h2 = qml.data.load("qchem", molname="H2", bondlength=0.742, basis="STO-3G")[0]

# Extract the Hamiltonian
H, n_qubits = h2.hamiltonian, len(h2.hamiltonian.wires)


# Obtain the ground state from the operations given by the dataset
def generate_ground_state(wires):
    qml.BasisState(np.array(h2.hf_state), wires=wires)

    for op in h2.vqe_gates:  # use the gates data from the dataset
        op = qml.map_wires(op, {op.wires[i]: wires[i] for i in range(len(wires))})
        qml.apply(op)

######################################################################
# The ``generate_ground_state`` function prepares the ground state of the molecule using the data obtained from the dataset.
# Let's use it to check the energy of that state:
#

dev = qml.device("default.qubit")

@qml.qnode(dev)
def circuit():
    generate_ground_state(range(n_qubits))
    return qml.expval(H)

print(f"Ground state energy: {circuit()}")

######################################################################
# We now use the ground state to find the first excited state.
#
# Finding the excited state
# ----------------------------
#
# To obtain the excited state we must define our ansatz that generates the state :math:`|\Psi(\theta)\rangle.`
#
# We use an ansatz constructed with :doc:`Givens rotations <demos/tutorial_givens_rotations>`, and we define the circuit for finding the excited state.
#

from functools import partial

# This line is added to better visualise the circuit
@partial(qml.transforms.decompose, max_expansion=1)

def ansatz(theta, wires):
    singles, doubles = qml.qchem.excitations(2, n_qubits)
    singles = [[wires[i] for i in single] for single in singles]
    doubles = [[wires[i] for i in double] for double in doubles]
    qml.AllSinglesDoubles(theta, wires, np.array([1,1,0,0]), singles, doubles)

theta = np.random.rand(3) # 3 parameters for the ansatz
print(qml.draw(ansatz, decimals = 2)(theta, range(4)))

######################################################################
# The ``ansatz`` function is the one that generates the state :math:`|\Psi(\theta)\rangle.`
# The next step is to calculate the overlap between our generated state and the ground state, using a technique
# known as `swap test <https://en.wikipedia.org/wiki/Swap_test>`__.


@qml.qnode(dev)
def swap_test(params):
    generate_ground_state(range(1, n_qubits + 1))
    ansatz(params, range(n_qubits + 1, 2 * n_qubits + 1))

    qml.Barrier()  # added to better visualise the circuit
    qml.Hadamard(wires=0)
    for i in range(n_qubits):
        qml.CSWAP(wires=[0, 1 + i + n_qubits, 1 + i])
    qml.Hadamard(wires=0)
    return qml.expval(qml.Z(0))

print(qml.draw(swap_test)(theta))
print(f"\nOverlap between the ground state and the ansatz: {swap_test(theta)}")

######################################################################
# The ``swap_test`` function returns the overlap between the generated state and the ground state.
#
# With this we have all the ingredients to define the loss function that we want to minimize:
#

@qml.qnode(dev)
def expected_value(theta):
    ansatz(theta, range(n_qubits))
    return qml.expval(H)

def loss_f(theta, beta):
    return expected_value(theta) + beta * swap_test(theta)

######################################################################
# The ``loss_f`` function returns the value of the cost function.
# The next step is to optimize the parameters of the ansatz to minimize the cost function.

import jax
import optax

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
print()

theta = jax.numpy.array([0.1, 0.2, 0.3])
beta = 2

# Store the values of the cost function
energy = [loss_f(theta, beta)]

conv_tol = 1e-6
max_iterations = 100

opt = optax.sgd(learning_rate=0.4)

# Store the values of the circuit parameter
angle = [theta]

opt_state = opt.init(theta)

for n in range(max_iterations):
    gradient = jax.grad(loss_f)(theta, beta)
    updates, opt_state = opt.update(gradient, opt_state)
    theta = optax.apply_updates(theta, updates)
    angle.append(theta)
    energy.append(loss_f(theta, beta))

    conv = jax.numpy.abs(energy[-1] - energy[-2])

    if n % 5 == 0:
        print(f"Step = {n},  Energy = {energy[-1]:.8f} Ha")

    if conv <= conv_tol:
        break

print(f"\nEstimated energy: {energy[-1].real:.8f}")

######################################################################
# Great! We have found a new energy value, but is this the right one?
# One way to check is to access the eigenvalues of the Hamiltonian directly:

print(np.sort(np.linalg.eigvals(H.matrix())))

######################################################################
# We have indeed found an eigenvalue of the Hamiltonian. It may seem that we have skipped the value :math:`-0.5389,`
# however the eigenvector corresponding to this eigenvalue belongs to a different particle number sector.
# The correct energy value for the first excited state of hydrogen is :math:`-0.53320939,` consistent with what we obtained with VQD!
# We have successfully found the first excited state!
#
# Conclusion
# ----------
#
# VQD is a variational method that can be used for calculating low-level excited state energies of quantum systems. Leveraging the
# orthogonality of the eigenstates, it adds a regularization penalty to the cost function to allow the search for
# the next excited state from the ground state discovered by VQE.
#
# In this tutorial we delved into the capabilities of variational quantum deflation (VQD) using PennyLane to compute
# the excited states of a hydrogen molecule.
# This illustrated how advanced quantum algorithms can extend beyond basic applications,
# offering deeper insights into quantum systems. We invite you to continue exploring these techniques and find more interesting use cases.
#
# References
# ----------
#
# .. [#Vqd]
#
#     Higgott, Oscar and Wang, Daochen and Brierley, Stephen
#     "Variational Quantum Computation of Excited States"
#     `Quantum 3, 156 (2019) <https://dx.doi.org/10.22331/q-2019-07-01-156>`__.
#
