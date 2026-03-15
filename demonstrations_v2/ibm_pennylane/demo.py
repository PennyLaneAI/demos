r"""
Using PennyLane with IBM's quantum devices and Qiskit
===================================

.. meta::
    :property="og:description": Learn how to use IBM devices with Pennylane.
    :property="og:image": https://pennylane.ai/qml/_static/demonstration_assets/thumbnail_tutorial_ibm_pennylane.png

.. related::

   quantum_volume Quantum Volume
   tutorial_vqe A brief overview of VQE


.. warning::
    This demo includes some outdated features and may not work as intended. 
    It can still be used as a guideline, but please consult the `PennyLane-Qiskit plugin documentation <https://docs.pennylane.ai/projects/qiskit/en/stable/>`__
    to get the most up-to-date information on the features and usage of this plugin.

Bigger and better quantum computers are built every year. Instead of waiting for the perfect quantum computer to be
released, we can already try out the best hardware that exists today. Experimenting on cutting-edge devices helps us
understand the technology and improve the way we develop quantum software. PennyLane is a fantastic tool for prototyping
quantum algorithms of all kinds, while IBM provides access to the newest and most powerful superconducting quantum devices
available today. Let's combine the two!

In this tutorial, we'll show you how to use PennyLane to interface with IBM's quantum computing
platform. We will learn how to:

* discover what kind of devices IBM offers;
* connect to IBM devices through PennyLane's device class;
* use Qiskit Runtime to run hybrid algorithms;
* compare different devices to improve our quantum algorithms.
"""

##############################################################################
# Using IBM devices
# -----------------
# IBM offers access to a variety of devices, both classical simulators and real quantum hardware.
# By default, these devices are not included in PennyLane, but after installing the
# `PennyLane-Qiskit plugin <https://docs.pennylane.ai/projects/qiskit/en/stable/>`__ with the command ``pip install pennylane-qiskit``,
# they can be used just like any other device offered in PennyLane!
# Currently, there are three devices available — ``Aer``, ``BasicSim`` and ``Remote`` — that can be initialized
# as follows:
import pennylane as qml
qml.capture.enable()
from qiskit_aer import AerSimulator

qubits = 4
dev_aer = qml.device("qiskit.aer", wires=qubits)
dev_basicsim = qml.device("qiskit.basicsim", wires=qubits)
try:
    dev_remote = qml.device("qiskit.remote", wires=qubits, backend=AerSimulator())
except Exception as e:
    print(e)

##############################################################################
# The last device (``qiskit.remote``) can cause an error if we don't provide a valid account
# token through Qiskit. The Remote device is used to access quantum hardware, so it also requires
# an IBM Quantum account, which can be specified using an identifying token. You can find your
# token by creating or logging into your `IBM Quantum account <https://quantum-computing.ibm.com>`__.
# Be careful not to publish code that reveals your token to other people! One way to avoid this
# is by saving your token in a `PennyLane configuration file <https://docs.pennylane.ai/en/stable/introduction/configuration.html>`__.
# To specify which machine or computational framework these devices actually connect to, we can
# use the ``backend`` argument.

dev_aer = qml.device("qiskit.aer", wires=qubits)

##############################################################################
# For the Aer device, different quantum computers can be used by changing the backend to the name
# of the specific simulator method. To see which backends exist, we can call the
# ``capabilities`` function:

from qiskit_aer import Aer

print(Aer.backends())

##############################################################################
# .. rst-class:: sphx-glr-script-out
#
#
#  .. code-block:: none
#
#      ['aer_simulator', 'aer_simulator_statevector', 'aer_simulator_density_matrix',
#      'aer_simulator_stabilizer', 'aer_simulator_matrix_product_state',
#      'aer_simulator_extended_stabilizer', 'aer_simulator_unitary', 'aer_simulator_superop',
#      'qasm_simulator', 'statevector_simulator', 'unitary_simulator', 'pulse_simulator']

##############################################################################
# You can find even more details about these devices directly from the IBM Quantum platform. You can
# find information about the size, topology, quantum volume and noise profile of all the devices
# that they have available. Currently, the smallest device has 5 qubits and the largest has 127. On
# the IBM Quantum platform you can also check which devices are free to use and whether any of them
# are temporarily unavailable.  You can even check your active jobs and estimated time in the queue
# for any programs you execute.

##############################################################################
# Qiskit Runtime
# ---------------
# Qiskit Runtime is a quantum computing service provided by IBM intended to make hybrid algorithms
# more efficient to execute. Hybrid algorithms are algorithms where a classical computer and
# quantum computer work together. This often involves the classical algorithm iteratively
# optimizing the quantum circuit, which the quantum computer repeatedly runs.
#
# One such example is the VQE algorithm, which can be used to calculate the ground state energy of
# molecules. It contains an optimization loop, which repeatedly requests the device to run a
# parameterized quantum circuit. Because the optimization algorithm changes the values of the
# parameters, the circuit requested is different each iteration. Also, the change is dependent on
# the results of the previous circuit, which means that there needs to be constant communication
# back and forth between the quantum computer and the classical computer in charge of the
# optimization.
#
# The solution that Qiskit Runtime provides is placing a classical computer in close physical
# proximity of the quantum computer. The user uploads a job to the classical computer, which runs
# the entire hybrid algorithm together with the quantum hardware, with no intermediate user input.
# This automates the iterative process, which otherwise requires time and resources for
# communication between the user and the hardware provider.

##############################################################################
# Using Qiskit Runtime
# --------------------
# The PennyLane-Qiskit plugin includes some tools to help create a Qiskit Runtime job. Since using
# Qiskit Runtime only makes sense when using real quantum hardware, we must again specify our IBM
# Quantum account details to run these jobs.
#
# First, we set up our problem as usual, and then retrieve a program ID from IBM, which gives us a
# place to upload our job:
#
# .. warning::
#
#    By default, this demo uses the online simulator (`ibmq_qasm_simulator`), which is free at the
#    time of writing. Please note that IBM Quantum's policies may change, and simulators could become paid services.
#    Always verify current pricing and access policies on the IBM Quantum platform.
#
#    This demo can also run on quantum hardware by updating the backend variable accordingly. Be aware, that access
#    to IBM Quantum hardware is not free and may result in substantial costs. Ensure you are aware of these costs
#    and comfortable with them before proceeding.

from pennylane import numpy as pnp
from qiskit_ibm_runtime import QiskitRuntimeService

import pennylane as qml
qml.capture.enable()

# Obtaining the Hamiltonian for H2 from PennyLane QChem dataset
[dataset] = qml.data.load("qchem", molname="H2", bondlength=0.742, basis="STO-3G")
H = dataset.hamiltonian
qubits = 4

# Initialize QiskitRuntimeService
service = QiskitRuntimeService()

# Use the `ibmq_qasm_simulator` available on IBM Cloud
backend = service.backend("ibmq_qasm_simulator")

try:
    # Our device supports a maximum of 31 qubits
    NUM_QUBITS_SUPPORTED = 31
    dev = qml.device("qiskit.remote", wires=NUM_QUBITS_SUPPORTED, backend=backend)
except Exception as e:
    print(e)

##############################################################################
# Next, we specify our quantum circuit. Although there are many circuits to choose from, it is
# important to know that before a circuit is executed on hardware, it undergoes a transpilation
# step, which converts your circuit into a different, but equivalent, circuit. The purpose of this
# step is to ensure that only operations that are native to the quantum computer are used. With
# parameterized gates, however, this may cause some unexpected behavior, such as the emergence of
# more parameters when the transpiler attempts to decompose a complicated gate, such as
# :class:`~pennylane.AllSinglesDoubles`. These types of issues will likely be fixed in the future, but, when in doubt,
# it is preferable to use simpler gates where possible. We will use a simple four-qubit circuit
# with one parameter that is designed specifically for the :math:`H_2` molecule:


def four_qubit_ansatz(theta):
    # initial state 1100:
    qml.PauliX(wires=0)
    qml.PauliX(wires=1)

    # change of basis
    qml.RX(pnp.pi / 2, wires=0)
    qml.Hadamard(wires=1)
    qml.Hadamard(wires=2)
    qml.Hadamard(wires=3)

    qml.CNOT(wires=[3, 2])
    qml.CNOT(wires=[2, 1])
    qml.CNOT(wires=[1, 0])

    qml.RZ(theta, wires=0)

    qml.CNOT(wires=[1, 0])
    qml.CNOT(wires=[2, 1])
    qml.CNOT(wires=[3, 2])

    # invert change of basis
    qml.RX(-pnp.pi / 2, wires=0)
    qml.Hadamard(wires=1)
    qml.Hadamard(wires=2)
    qml.Hadamard(wires=3)


##############################################################################
# Finally, we can run our example VQE algorithm. In order to query the quantum computer iteratively
# we need to initialize a `Qiskit Session <https://docs.quantum.ibm.com/guides/sessions>`__, which
# allows multiple jobs from a single algorithm to be ran sequentially without interruptions. We
# also need to provide our VQE algorithm an optimizer. In this case, we will be using the :class:`~.pennylane.GradientDescentOptimizer`.

from pennylane_qiskit import qiskit_session


@qml.qnode(dev)
def cost_fn(theta):
    four_qubit_ansatz(theta)
    return qml.expval(H)


max_iterations = 40
theta = pnp.array(0.0, requires_grad=True)
opt = qml.GradientDescentOptimizer(stepsize=0.4)
energies = []

with qiskit_session(dev) as session:
    for n in range(max_iterations):
        theta, prev_energy = opt.step_and_cost(cost_fn, theta)
        energies.append(prev_energy)

######################################################################
# .. note ::
#
#    This may take a long time depending on how busy the hardware is. Depending on the tier of
#    your IBM plan, your session may also be interrupted before the optimization can finish.
#    Luckily, the rest of the workflow demonstrates how to run VQE with a simulator instead, which
#    has no such restrictions at all.

##############################################################################
# Benchmarking
# ~~~~~~~~~~~~
# One of the reasons why we even want to have access to these various devices and backends is so
# that we can benchmark the capabilities of the algorithms that we develop. Some simulators are
# particularly good with certain types of circuits, whereas other simulators are more general and
# may provide resources for simulating noise which mimics the kind of errors that real quantum
# hardware produces. Switching between your devices helps you learn more about your algorithm and
# can potentially provide guidance on how to make it better. For example, we can compare the
# performance of the default PennyLane simulator to the Qiskit ``'aer_simulator'`` by running the same
# VQE algorithm on both. The difference between these two devices is that the ``'aer_simulator'`` uses a
# finite number of shots to estimate the energy in each iteration, rather than performing an exact
# calculation using the information hidden in the vector representation of the quantum state.

dev1 = qml.device("default.qubit", wires=4)
shots = 8000
dev2 = qml.device("qiskit.aer", wires=4)


@qml.qnode(dev1)
def cost_fn_1(theta):
    four_qubit_ansatz(theta)
    return qml.expval(H)


@qml.set_shots(shots)
@qml.qnode(dev2)
def cost_fn_2(theta):
    four_qubit_ansatz(theta)
    return qml.expval(H)


# we can also use the qnode to draw the circuit
import matplotlib.pyplot as plt

qml.draw_mpl(cost_fn_1, decimals=2)(theta=1.0)
plt.show()

##############################################################################
#
# .. figure:: ../_static/demonstration_assets/ibm_pennylane/figure_1.png
#     :align: center
#     :width: 80%
#     :alt: Circuit
#     :target: javascript:void(0);

stepsize = 0.4
max_iterations = 40
opt = qml.GradientDescentOptimizer(stepsize=stepsize)
theta_1 = pnp.array(0.0, requires_grad=True)
theta_2 = pnp.array(0.0, requires_grad=True)
energies_1 = []
energies_2 = []
for n in range(max_iterations):
    theta_1, prev_energy_1 = opt.step_and_cost(cost_fn_1, theta_1)
    theta_2, prev_energy_2 = opt.step_and_cost(cost_fn_2, theta_2)
    print(prev_energy_1, prev_energy_2)
    energies_1.append(prev_energy_1)
    energies_2.append(prev_energy_2)

##############################################################################
# .. rst-class:: sphx-glr-script-out
#
#
#  .. code-block:: none
#
#       -1.1173489211359304 -1.1190829868933736
#       -1.1279998277357466 -1.1288232086760344
#       -1.1326490062948753 -1.1301129826377698
#       -1.1346624646306156 -1.1359551977827906
#       -1.135531456349987 -1.1361153521202156
#       -1.1359059478903804 -1.1365135907218555
#       -1.1360672311675288 -1.1378046643806679
#       -1.1361366722397177 -1.135703178347981
#       -1.1361665667682972 -1.1357895389865689
#       -1.1361794357654167 -1.1369628601568447
#       -1.1361849754890518 -1.1365783784951322
#       -1.1361873601539711 -1.1367306582741445
#       -1.1361883866679017 -1.1358320382653255
#       -1.1361888285450743 -1.1357663570027223
#       -1.1361890187570935 -1.135670418637738
#       -1.1361891006364095 -1.1369084485357166
#       -1.1361891358824536 -1.139272401360956
#       -1.1361891510545838 -1.137130432389924
#       -1.136189157585629 -1.1377776180459274
#       -1.1361891603970047 -1.1358917536737867
#       -1.1361891616071986 -1.1370070425290821
#       -1.1361891621281424 -1.135792429417887
#       -1.13618916235239 -1.1350561467266231
#       -1.13618916244892 -1.1366759212135573
#       -1.1361891624904732 -1.1351253597692734
#       -1.13618916250836 -1.1362073324228987
#        -1.13618916251606 -1.1366017151897079
#       -1.136189162519374 -1.1362493563165617
#       -1.136189162520801 -1.1378309783921152
#       -1.1361891625214149 -1.1350975937163135
#       -1.1361891625216796 -1.1372437534918245
#       -1.136189162521793 -1.1361363466968788
#       -1.1361891625218425 -1.136401712436813
#       -1.1361891625218634 -1.1346185510801001
#       -1.136189162521872 -1.1351658522378076
#       -1.1361891625218763 -1.1350958264741222
#       -1.1361891625218783 -1.135516284054897
#       -1.136189162521879 -1.137538330500378
#        -1.1361891625218792 -1.1359072863719688
#       -1.1361891625218794 -1.1369053955536899


##############################################################################
# We can clearly see the difference between the two devices when we plot the energies over each
# iteration:

plt.plot(energies_1, color="r", label="default.qubit")
plt.plot(energies_2, color="b", label="qiskit.aer")

# min energy = min eigenvalue
min_energy = min(qml.eigvals(H))
z = [min_energy] * max_iterations

plt.plot(z, "--", color="k", label="Exact answer")
plt.xlabel("VQE iterations")
plt.ylabel("Energy (Ha)")
plt.legend()
plt.show()

##############################################################################
#
# .. figure:: ../_static/demonstration_assets/ibm_pennylane/figure_2.png
#     :align: center
#     :width: 80%
#     :alt: Iterations
#     :target: javascript:void(0);

##############################################################################
# The device with the finite number of shots is unable to converge to the right answer because it
# is limited by the precision of the result in each iteration. This is an effect that will
# certainly appear in real quantum devices too, and it can be instructive to study this effect
# independently of all the other limitations on real devices, such as decoherence, limited
# topology and readout errors.
#
# This tutorial has demonstrated how and why to use quantum computing hardware provided by IBM using PennyLane. To read
# more about the details and possibilities of the PennyLane-Qiskit plugin, `read the documentation <https://docs.pennylane.ai/projects/qiskit/en/latest/index.html>`__.
#
