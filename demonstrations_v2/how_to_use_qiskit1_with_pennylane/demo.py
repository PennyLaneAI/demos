r"""How to use Qiskit 1.0 with PennyLane
====================================

The PennyLane-Qiskit plugin enables you to integrate your existing Qiskit code and run circuits on IBM 
devices with PennyLane, encompassing two real-world scenarios: (1) working in PennyLane from the start 
and executing your work on an IBM device and (2) converting your existing Qiskit code to PennyLane and 
executing that on *any* device, including IBM devices, :doc:`Amazon Braket <demos/getting_started_with_hybrid_jobs>` 
— you name it!

With the `first stable release of Qiskit <https://www.ibm.com/quantum/blog/qiskit-1-0-release-summary>`__ in February 2024, we subsequently shipped some
excellent features and upgrades with the PennyLane-Qiskit plugin, allowing
anyone familiar with
Qiskit to jump into the PennyLane ecosystem and land on both feet.
In this demo, we want to show you how easy these features are to use, letting you make that
aforementioned jump like it’s nothing 😌.
"""

######################################################################
# .. note ::
#
#    To follow along, we recommend installing the PennyLane-Qiskit plugin in a separate virtual environment.
#
#    .. code-block:: none
#
#      pip install -U pennylane-qiskit 
# 
#    This will install PennyLane, the plugin, and the latest Qiskit 1.0 version that the plugin supports.
#    If you use an environment that contains a pre-1.0 version of Qiskit, upgrading to Qiskit 1.0 could
#    cause issues. In that case, we recommend following `Qiskit's upgrade instructions <https://docs.quantum.ibm.com/start/install>`__.
#

######################################################################
# Coding in PennyLane, executing on IBM Quantum devices
# -----------------------------------------------------
#
# If you want to distill how a PennyLane plugin works down to one thing, it’s all in the provided devices! In
# PennyLane, you just :doc:`create your circuit (a quantum function) </introduction/circuits>` and decorate it with
# the QNode decorator :func:`@qml.qnode(dev) <pennylane.qnode>`, where ``dev`` is (one of) the plugin’s device(s).
#
# In PennyLane and its plugins,
# `devices <https://pennylane.ai/plugins/>`__ are called upon by their short name, and can be loaded via the :func:`~pennylane.device` function:
#
# .. code-block:: 
#
#   qml.device("shortname", *device_options)
#
# If you’ve
# seen PennyLane code before, you’ve probably seen ``"default.qubit"`` or ``"lightning.qubit"`` as
# short names for our Python and C++ statevector simulators, respectively.
#
# In the PennyLane-Qiskit plugin, there are `many IBM
# devices <https://docs.pennylane.ai/projects/qiskit/en/stable/#devices>`__ you can use, but there are
# two heavy hitters for Qiskit 1.0:
#
# -  ``"qiskit.basicsim"``: uses the Qiskit ``BasicSimulator`` backend from the ``basic_provider``
#    module in Qiskit 1.0.
#
# -  ``"qiskit.remote"``: lets you run PennyLane code on any Qiskit device, where you can choose between
#    different backends — either simulators tailor-made to emulate the real hardware, or the real
#    hardware itself.
#
#
# If you want to use any of these devices in PennyLane, simply put those short names into
# ``qml.device`` and any quantum function decorated with ``@qml.qnode(dev)`` will execute on the
# corresponding device.
#

######################################################################
# To show how easy this is, let’s say we have this simple circuit in Qiskit 1.0.
#

import qiskit
from qiskit import QuantumCircuit
from qiskit.providers.basic_provider import BasicSimulator

qc = QuantumCircuit(2, 1)

qc.h(0)
qc.cx(0, 1)
qc.measure(1, 0)

backend = BasicSimulator()
counts = backend.run(qc).result().get_counts()

print(counts)

######################################################################
# .. rst-class:: sphx-glr-script-out
#
#   .. code-block:: none
#
#     {'0': 523, '1': 501}
#

######################################################################
# In PennyLane, we can execute the exact same circuit on the exact same device and backend like so:
#

import pennylane as qml
qml.capture.enable()

dev = qml.device("qiskit.basicsim", wires=2)

@qml.set_shots(1024)
@qml.qnode(dev)
def circuit():
    qml.Hadamard(0)
    qml.CNOT([0, 1])
    return qml.counts(wires=1)

print(circuit())

######################################################################
# .. rst-class:: sphx-glr-script-out
#
#   .. code-block:: none
#
#     {'0': 474, '1': 550}
#

######################################################################
# Magic! With one line of code, you can work inside PennyLane and ship the execution off to your
# favourite IBM device. It’s exactly like using Qiskit 1.0, but you interact with PennyLane instead.
#

######################################################################
# Converting Qiskit 1.0 code to PennyLane
# ---------------------------------------
#
# This is probably what a lot of the audience is wondering: “Can I combine my existing work in Qiskit
# with PennyLane?” *YES.* And don’t worry, you don’t need to import a ton of things or use a bunch of
# functions — you only need to know *two* functions:
#
# - :func:`~pennylane.from_qiskit`: converts an entire Qiskit ``QuantumCircuit`` — the whole thing — into a 
#   PennyLane quantum function. It will faithfully convert Qiskit-side measurements (even mid-circuit 
#   measurements), or you can append Pennylane-side measurements directly to it.
#
# - :func:`~pennylane.from_qiskit_op`: converts a ``SparsePauliOp`` in Qiskit 1.0 to the equivalent operator 
#   in PennyLane.
#
#
# Both of these functions give you all the functionality you need to access PennyLane’s features and
# user interface starting from the side of Qiskit 1.0. Let’s look at an example where both of these
# functions are used.
#

######################################################################
# Let’s say you’ve created the following Qiskit code that prepares a modified GHZ state for an
# arbitrary amount of qubits and measures several expectation values of ``SparsePauliOp`` operators.
#

from qiskit.quantum_info import SparsePauliOp

n = 5

def qiskit_GHZ_circuit(n):
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
        qc.rx(0.1967, i)
    return qc

qc = qiskit_GHZ_circuit(n)

operator_strings = ["I" * i + "ZZ" + "I" * (n - 2 - i) for i in range(n - 1)]
operators = [SparsePauliOp(operator_string) for operator_string in operator_strings]
print(operators)

######################################################################
# .. rst-class:: sphx-glr-script-out
#
#   .. code-block:: none
#
#     [SparsePauliOp(['ZZIII'],
#                   coeffs=[1.+0.j]), SparsePauliOp(['IZZII'],
#                   coeffs=[1.+0.j]), SparsePauliOp(['IIZZI'],
#                   coeffs=[1.+0.j]), SparsePauliOp(['IIIZZ'],
#                   coeffs=[1.+0.j])]
#

######################################################################
# With the circuit and operators defined, we can create a 
# `StatevectorEstimator <https://github.com/Qiskit/qiskit/blob/1ed5951a98b594808525c8428e06178c160cfcbb/qiskit/primitives/statevector_estimator.py#L31>`__ 
# primitive in Qiskit to execute the circuit and calculate expectation values.
#

from qiskit.primitives import StatevectorEstimator

estimator = StatevectorEstimator()
job_estimator = estimator.run([(qc, operators)])
result_estimator = job_estimator.result()[0].data.evs

print(result_estimator)

######################################################################
# .. rst-class:: sphx-glr-script-out
#
#   .. code-block:: none
#
#     [0.98071685 0.96180554 0.96180554 0.96180554]
#

######################################################################
# To convert this work into PennyLane, let’s start with the Qiskit-side ``SparsePauliOp`` operators
# and converting them to PennyLane objects with ``qml.from_qiskit_op``.
#

pl_operators = [qml.from_qiskit_op(qiskit_op) for qiskit_op in operators]
print(pl_operators)

######################################################################
# .. rst-class:: sphx-glr-script-out
#
#   .. code-block:: none
#
#     [Z(3) @ Z(4), Z(2) @ Z(3), Z(1) @ Z(2), Z(0) @ Z(1)]
#

######################################################################
# .. note ::
#
#    PennyLane wires are enumerated from left to right, while the Qiskit convention is to
#    enumerate from right to left. This means a ``SparsePauliOp`` term defined by the string “XYZ” in Qiskit
#    applies ``Z`` on wire ``0``, ``Y`` on wire ``1``, and ``X`` on wire ``2``. For more details, see
#    the `String <https://docs.quantum.ibm.com/api/qiskit/qiskit.quantum_info.Pauli>`__ representation
#    section of the Qiskit documentation for the ``Pauli`` class.
#

######################################################################
# Next, we show how to convert the Qiskit ``QuantumCircuit``, ``qc``, to PennyLane with ``qml.from_qiskit``. We
# can append the measurements — expectation values (:func:`~pennylane.expval`) of ``pl_operators`` — with the
# ``measurements`` keyword argument, which accepts a list of PennyLane measurements.
#

measurements = [qml.expval(op) for op in pl_operators]  # expectation values

qc = qiskit_GHZ_circuit(n)
pl_qfunc = qml.from_qiskit(qc, measurements=measurements)

######################################################################
# The last thing to do is make ``pl_func`` a QNode. We can’t decorate ``pl_qfunc`` with
# ``@qml.qnode``, but we can equivalently wrap it with ``qml.QNode`` and supply the device.
#

pl_circuit = qml.QNode(pl_qfunc, device=qml.device("lightning.qubit", wires=n))
print(pl_circuit())

######################################################################
# .. rst-class:: sphx-glr-script-out
#
#   .. code-block:: none
#
#     [0.9807168489852623, 0.961805537883582, 0.961805537883582, 0.961805537883582]
#

######################################################################
# What’s really useful about being able to append measurements to the end of a circuit with
# :func:`~pennylane.from_qiskit` is being able to measure something that isn’t available in Qiskit 1.0 but is
# available in PennyLane, like the :doc:`classical shadow <demos/tutorial_classical_shadows/>` measurement protocol, for example. In PennyLane,
# you can measure this with :func:`~pennylane.classical_shadow`.
#

measurements = [qml.classical_shadow(wires=range(n))]
pl_qfunc = qml.from_qiskit(qc, measurements=measurements)

pl_circuit = qml.QNode(pl_qfunc, device=qml.device("default.qubit", wires=n))
print(qml.set_shots(pl_circuit, shots=5)())

######################################################################
# .. rst-class:: sphx-glr-script-out
#
#   .. code-block:: none
#
#     [array([[[0, 0, 0, 0, 1],
#              [1, 1, 1, 0, 1],
#              [0, 1, 0, 0, 0],
#              [1, 1, 0, 0, 0],
#              [0, 0, 0, 0, 0]],
#
#             [[0, 0, 2, 2, 1],
#              [2, 0, 2, 0, 1],
#              [1, 0, 2, 0, 2],
#              [1, 0, 1, 1, 1],
#              [1, 0, 2, 0, 2]]], dtype=int8)]
#

######################################################################
# And that’s it! Now you have a copy of your work in PennyLane, where you can access
# fully-differentiable and hardware-agnostic quantum programming — the entire quantum programming
# ecosystem is at your disposal 💪.
#

######################################################################
# A real-world example
# --------------------
#
# One of the things you’ll almost certainly encounter in the wild is a cost function to optimize with
# tunable parameters belonging to a circuit — it’s common 😉! PennyLane is a great option for these
# problems because of its end-to-end differentiability and long list of `optimization methods <https://pennylane.ai/search/?contentType=DEMO&categories=optimization&sort=publication_date>`_ you can
# leverage. So, maybe you have a home-cooked :doc:`variational circuit </glossary/variational_circuit/>` written in Qiskit and you want to
# access PennyLane’s seamless differentiability — yep, the PennyLane-Qiskit plugin has you covered
# here, too 🙌.
#
# To keep things simple, let’s use the following Qiskit circuit as our variational circuit.
#

from qiskit.circuit import ParameterVector, Parameter
from matplotlib import pyplot as plt

n = 3

angles1 = ParameterVector("phis", n - 1)
angle2 = Parameter("theta")

qc = QuantumCircuit(3)
qc.rx(angles1[0], [0])
qc.ry(angles1[1], [1])
qc.ry(angle2, [2])

qc.draw("mpl")
plt.show()


######################################################################
#
# .. figure:: ../_static/demonstration_assets/how_to_use_qiskit_1_with_pennylane/qiskit_parameterized_circuit.png
#     :align: center
#     :width: 30%
#

######################################################################
# This circuit contains two sets of differentiable parameters: ``phis`` (length 2) and ``theta``
# (scalar).
#
# If we give this Qiskit circuit to ``qml.from_qiskit``, we get a quantum function that can
# subsequently be called within a circuit — it’s as if the gates and operations contained within it
# get transferred over to our new QNode.
#

import pennylane.numpy as np

pl_qfunc = qml.from_qiskit(qc)

dev = qml.device("lightning.qubit", wires=n)

@qml.qnode(dev)
def differentiable_circuit(phis, theta):
    pl_qfunc(phis, theta)
    return [qml.expval(qml.Z(i)) for i in range(n)]

phis = np.array([0.6, 0.7])
theta = np.array([0.19])

print(differentiable_circuit(phis, theta))
qml.draw_mpl(differentiable_circuit, style="pennylane")(phis, theta)
plt.show()

######################################################################
# .. rst-class:: sphx-glr-script-out
#
#   .. code-block:: none
#
#     [0.8253356149096783, 0.7648421872844883, 0.9820042351172701]
#

######################################################################
#
# .. figure:: ../_static/demonstration_assets/how_to_use_qiskit_1_with_pennylane/pl_real_world_example.png
#     :align: center
#     :width: 45%
#

######################################################################
# You’ll notice, too, that ``pl_func`` has the call signature that you would expect:
# ``pl_qfunc(phis, theta)``, just like the name we gave them when we defined them in Qiskit
# (``ParameterVector("phis", n-1)`` and ``Parameter("theta")``).
#
# With the circuit being immediately differentiable to PennyLane, let’s define a dummy cost function
# that will sum the output of ``differentiable_circuit``.
#

def cost(phis, theta):
    return np.sum(differentiable_circuit(phis, theta))

######################################################################
# Now we’re in business and can optimize! You can use any suitable optimizer in PennyLane that you
# like, and by calling its ``step_and_cost`` method you can access the updated parameters and cost
# function value after each optimization step.
#

opt = qml.AdamOptimizer(0.1)

for i in range(100):
    (phis, theta), new_loss = opt.step_and_cost(cost, phis, theta)

print(f"Optimized parameters: phis = {phis}, theta = {theta}")
print(f"Optimized cost function value: {new_loss}")

######################################################################
# .. rst-class:: sphx-glr-script-out
#
#   .. code-block:: none
#
#     Optimized parameters: phis = [3.12829384 3.12823583], theta = [3.1310224]
#     Optimized cost function value: -2.999796472821245
#

######################################################################
# As we expect, the minimum value that our cost function can take is :math:`-3` when all angles of
# rotation are :math:`\pi` (all qubits are rotated into the :math:`\vert 1 \rangle` state). Of course,
# this is just a toy example of an easy optimization problem. But, you can apply this process to a
# larger, more complicated circuit ansatze without worries.
#

######################################################################
# Further resources
# -----------------
#
# There’s so much more to learn about what’s possible in PennyLane, and if you’re coming from Qiskit
# you’re in good hands! The `PennyLany-Qiskit plugin <https://docs.pennylane.ai/projects/qiskit/en/stable/>`__ is your personal chaperone to the PennyLane
# ecosystem. You can dive deeper into what’s possible with the PennyLane-Qiskit plugin by visiting the
# plugin homepage and, in the
# mean time, if you have any questions about the plugin, PennyLane, or even Qiskit, drop a question on
# our `Discussion Forum <https://discuss.pennylane.ai>`__ and we’ll promptly respond.
#
# Now that you’ve used PennyLane, every road in the wonderful world of quantum programming SDKs is
# open with no set speed limits 🏎️. Explore our website to see
# the latest and greatest `PennyLane features </features/>`__, `Demos </qml/demonstrations/>`__ and our `blog posts </blog/>`__, and follow us on
# `LinkedIn <https://www.linkedin.com/company/pennylaneai/>`__ or `X (formerly
# Twitter) <https://twitter.com/PennyLaneAI>`__ to stay updated!
#
