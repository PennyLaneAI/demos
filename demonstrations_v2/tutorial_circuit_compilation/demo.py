r"""Compilation of quantum circuits
===============================

.. meta::
    :property="og:description": Learn about circuit transformations and quantum circuit compilation with PennyLane
    :property="og:image": https://pennylane.ai/qml/_static/demonstration_assets/thumbnail_tutorial_circuit_compilation.png

.. related::

   tutorial_quantum_circuit_cutting Introduction to circuit cutting


Quantum circuits take many different forms from the moment we design them to the point where they
are ready to run on a quantum device. These changes happen during the compilation process of the
quantum circuits, which relies on the fact that there are many equivalent representations of quantum
circuits that use different sets of gates but provide the same output.

Out of those representations, we are typically interested in finding the most suitable one for our
purposes. For example, we usually look for the one that will incur the least amount of errors in the
specific hardware on which we are compiling the circuit. This usually implies decomposing the
quantum gates in terms of the native ones of the quantum device, adapting the operations to the
hardware's topology, combining them to reduce the circuit depth, etc.

A crucial part of the compilation process consists of repeatedly performing minor circuit modifications.
In PennyLane, we can apply :mod:`~pennylane.transforms` to our quantum functions in order to obtain
equivalent ones that may be more convenient for our task. In this tutorial, we introduce the most
fundamental transforms involved in the compilation of quantum circuits.

Circuit transforms
------------------

When we implement quantum algorithms, it is typically in our best interest that the resulting
circuits are as shallow as possible, especially with the currently available noisy quantum devices.
However, they are often more complex than needed, containing multiple operations that could be
combined to reduce the circuit complexity, although it is not always obvious. Here, we
introduce three simple :mod:`~pennylane.transforms` that can be implemented together to obtain
simpler quantum circuits. The transforms are based on very basic circuit equivalences:

.. figure:: ../_static/demonstration_assets/circuit_compilation/circuit_transforms.jpg
    :align: center
    :width: 90%

To illustrate their combined effect, let us consider the following circuit.
"""

import pennylane as qml
import matplotlib.pyplot as plt

dev = qml.device("default.qubit", wires=3)

@qml.qnode(dev)
def circuit(angles):
    qml.Hadamard(wires=2)
    qml.Hadamard(wires=1)
    qml.CNOT(wires=[1, 0])
    qml.CNOT(wires=[2, 1])
    qml.RZ(angles[1], wires=2)
    qml.RY(angles[0], wires=0)
    qml.CNOT(wires=[2, 1])
    qml.RY(-angles[0], wires=0)
    qml.RX(angles[0], wires=0)
    qml.RZ(-angles[1], wires=2)
    qml.CNOT(wires=[1, 0])
    qml.RX(angles[0], 0)
    qml.Hadamard(wires=1)
    qml.CY(wires=[1, 2])
    qml.CNOT(wires=[1, 0])
    return qml.expval(qml.PauliZ(wires=0))


angles = [0.1, 0.3, 0.5]
qml.draw_mpl(circuit, decimals=1, style="sketch")(angles)
plt.show()

######################################################################
# Given an arbitrary quantum circuit, it is usually hard to clearly understand what is really
# happening. To obtain a better picture, we can shuffle the commuting operations to better distinguish
# between groups of single-qubit and two-qubit gates. We can easily do so by applying the
# :func:`~pennylane.transforms.commute_controlled` transform, which pushes all single-qubit gates
# towards a ``direction`` (defaults to the right).
#

new_circuit = qml.transforms.commute_controlled(circuit, direction="right")

qml.draw_mpl(new_circuit, decimals=1, style="sketch")(angles)
plt.show()

######################################################################
# With this rearrangement, we can clearly identify a few operations that cancel
# out. For instance, the two consecutive CNOT gates on the last two wires can
# both be removed. This can be achieved with the :func:`~pennylane.transforms.cancel_inverses`
# transform, which removes consecutive inverse operations.
#

new_circuit = qml.transforms.cancel_inverses(new_circuit)

qml.draw_mpl(new_circuit, decimals=1, style="sketch")(angles)
plt.show()

######################################################################
# Consecutive rotations along the same axis can also be merged. For example, we can combine the two
# :class:`~pennylane.RX` rotations on the first qubit into a single one with the sum of the angles.
# Additionally, the two :class:`~pennylane.RY` rotations on the first qubit and the two :class:`~pennylane.RZ`
# rotations on the third qubit will cancel each other. We can combine these rotations with the
# :func:`~pennylane.transforms.merge_rotations` transform. We can choose which rotations to merge
# with ``include_gates`` (defaults to ``None``, meaning all). Furthermore, the merged rotations
# with a resulting angle lower than ``atol`` are directly removed.
#

new_circuit = qml.transforms.merge_rotations(new_circuit, atol=1e-8)

qml.draw_mpl(new_circuit, decimals=1, style="sketch")(angles)
plt.show()

######################################################################
# Combining these simple circuit transforms, we have reduced the complexity of our original circuit.
# This will make it cheaper to execute and less prone to errors. However, there is still room for
# improvement. Let's take it a step further in the following section!
#
# As a final remark, we can directly apply the transforms to our circuit when we define it
# using their decorator forms (beware the reverse order!).
#

@qml.transforms.merge_rotations(atol=1e-8)
@qml.transforms.cancel_inverses
@qml.transforms.commute_controlled(direction="right")
@qml.qnode(dev)
def qfunc(angles):
    qml.Hadamard(wires=2)
    qml.Hadamard(wires=1)
    qml.CNOT(wires=[1, 0])
    qml.CNOT(wires=[2, 1])
    qml.RZ(angles[1], wires=2)
    qml.RY(angles[0], wires=0)
    qml.CNOT(wires=[2, 1])
    qml.RY(-angles[0], wires=0)
    qml.RX(angles[0], wires=0)
    qml.RZ(-angles[1], wires=2)
    qml.CNOT(wires=[1, 0])
    qml.RX(angles[0], 0)
    qml.Hadamard(wires=1)
    qml.CY(wires=[1, 2])
    qml.CNOT(wires=[1, 0])
    return qml.expval(qml.PauliZ(wires=0))


qml.draw_mpl(qfunc, decimals=1, style="sketch")(angles)
plt.show()

######################################################################
# Circuit compilation
# -------------------
#
# A sequence of transforms is also called a compile pipeline. We can chain
# multiple transforms into a :class:`~pennylane.CompilePipeline` that can
# be easily reused to transform a circuit:
#

pipeline = qml.CompilePipeline(
    qml.transforms.commute_controlled(direction="right"),
    qml.transforms.cancel_inverses,
    qml.transforms.merge_rotations(atol=1e-8)
)

compiled_circuit = pipeline(circuit)

qml.draw_mpl(compiled_circuit, decimals=1, style="sketch")(angles)
plt.show()

######################################################################
# In the resulting circuit, we can identify further operations that can be
# combined, such as the consecutive :class:`~pennylane.CNOT` gates on the first
# two qubits, and then recursively the two :class:`~pennylane.Hadamard` gates
# on the second qubit. Therefore, we could apply the pipeline a second time.
#
# Let us see the resulting circuit with two passes.
#

compiled_circuit = pipeline(compiled_circuit)

qml.draw_mpl(compiled_circuit, decimals=1, style="sketch")(angles)
plt.show()

######################################################################
# Compile pipelines can be easily composed and modified. Now let's create a new
# pipeline from the original by repeating the same sequence of transforms twice
# and then decomposing the optimized circuit into a gate set so that it could
# be executed on a device that can only execute single-qubit rotations and the
# :class:`~pennylane.CNOT` gate.
#

new_pipeline = pipeline * 2 + qml.transforms.decompose(gate_set={"CNOT", "RX", "RY", "RZ"})
compiled_circuit = new_pipeline(circuit)

qml.draw_mpl(compiled_circuit, decimals=1, style="sketch")(angles)
plt.show()

######################################################################
# We can see how the Hadamard and control-Y gates have been decomposed into a series of single-qubit
# rotations and CNOT gates. We're ready to run our circuit on the quantum device. Great job!
#
# Conclusion
# ----------
#
# In this tutorial, we have learned the basic principles of the compilation of quantum circuits.
# Combining simple circuit transforms that are applied repeatedly during each pass of the compiler, we can
# significantly reduce the complexity of our quantum circuits.
#
# To continue learning, you can explore the documentation for the :mod:`~pennylane.transforms` module, which
# contains other circuit transformations present in the PennyLane. You can even learn how to create your own transform!
#
