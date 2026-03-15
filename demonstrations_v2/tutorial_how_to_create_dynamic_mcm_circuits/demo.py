r"""How to create dynamic circuits with mid-circuit measurements
================================================================

Measuring qubits in the middle of a quantum circuit execution can be useful in many ways.
From understanding the inner workings of a circuit, hardware characterization,
modeling and :doc:`error mitigation <demos/tutorial_diffable-mitigation>`, to error correction,
:doc:`quantum teleportation <demos/tutorial_teleportation>`, algorithmic improvements and even up to full
computations encoded as measurements in :doc:`measurement-based quantum computation (MBQC) <demos/tutorial_mbqc>`.

Before turning to any of these advanced topics, it is worthwhile to familiarize ourselves with
the syntax and features around mid-circuit measurements (MCMs). In this how-to, we will focus on
dynamic quantum circuits that use control flow based on MCMs.
Most of the advanced concepts mentioned above incorporate MCMs in this way, making it a
key ingredient to scalable quantum computing.

If you want to learn more or are looking for how to collect statistics of mid-circuit measurements,
have a read of the :doc:`how-to on MCM statistics <demos/tutorial_how_to_collect_mcm_stats>`.

.. figure:: ../_static/demonstration_assets/how_to_create_dynamic_mcm_circuits/socialthumbnail_how_to_create_dynamic_mcm_circuits.png
    :align: center
    :width: 50%

"""

######################################################################
# Dynamic circuit with a single MCM and conditional
# -------------------------------------------------
#
# We start with a minimal dynamic circuit on two qubits.
#
# - It rotates one qubit about the ``X``-axis and applies a phase to
#   the other qubit using a :class:`~.pennylane.T` gate.
#
# - Both qubits are entangled with a :class:`~.pennylane.CNOT` gate.
#
# - The second qubit is measured with :func:`~.pennylane.measure`.
#
# - If we measured a ``1`` in the previous step, an :class:`~.pennylane.S` gate is applied
#   to the first qubit. This conditioned operation is realized with :func:`~.pennylane.cond`.
#
# - Finally, the expectation value of the Pauli ``Y`` operator on the first qubit is returned.
#

import pennylane as qml
qml.capture.enable()
import numpy as np

dev = qml.device("lightning.qubit", wires=2)


@qml.qnode(dev, interface="numpy")
def circuit(x):
    qml.RX(x, 0)
    qml.T(1)
    qml.CNOT(wires=[0, 1])
    mcm = qml.measure(1)
    qml.cond(mcm, qml.S)(wires=0)

    return qml.expval(qml.Y(0))


x = 1.361
print(circuit(x))

######################################################################
#
# After this minimal working example, we now construct a more complex circuit showcasing more
# features of MCMs and dynamic circuits in PennyLane. We start with some short preparatory
# definitions.
#
# How to dynamically prepare half-filled basis states
# ---------------------------------------------------
#
# We now turn to a more complex example of a dynamic circuit. We will build a circuit that
# non-deterministically initializes half-filled computational basis states, i.e., basis states
# with as many :math:`1`\ s as :math:`0`\ s.
#
# The procedure is as follows:
#
# - Single-qubit rotations and a layer of :class:`~.pennylane.CNOT` gates create an entangled state
#   on the first three qubits.
#
# - The first three qubits are measured and for each measured :math:`0,` another qubit is excited
#   from the :math:`|0\rangle` state to the :math:`|1\rangle` state.
#
# First, we define a quantum subprogram that creates the initial state:
#


def init_state(x):
    # Rotate the first three qubits
    for w in range(3):
        qml.RX(x[w], w)
    # Entangle the first three qubits
    qml.CNOT([0, 1])
    qml.CNOT([1, 2])
    qml.CNOT([2, 0])


######################################################################
# With this subroutine in our hands, let's define the full :class:`~.pennylane.QNode`.
# For this, we also create a shot-based device.
#

shots = 100
dev = qml.device("default.qubit", seed=0)


@qml.set_shots(shots)
@qml.qnode(dev)
def create_half_filled_state(x):
    init_state(x)
    for w in range(3):
        # Measure one qubit at a time and flip another, fresh qubit if measured 0
        mcm = qml.measure(w)
        qml.cond(~mcm, qml.X)(w + 3)

    return qml.counts(wires=range(6))


######################################################################
# Before running this ``QNode``, let's sample some random input parameters and draw the circuit:
#

np.random.seed(652)
x = np.random.random(3) * np.pi

print(qml.draw(create_half_filled_state)(x))

######################################################################
# We see an initial block of gates that prepares the starting state on the
# first three qubits, followed by pairs of measurements and conditional bit flips,
# applied to pairs of qubits.
#
# Great, now let's finally see if it works:
#

counts = create_half_filled_state(x)
print(f"Sampled bit strings:\n{list(counts.keys())}")

######################################################################
# Indeed, we created half-filled computational basis states, each with its own probability:
#

print("The probabilities for the bit strings are:")
for key, val in counts.items():
    print(f"    {key}: {val/shots*100:4.1f} %")

######################################################################
#
# Postselecting dynamically prepared states
# -----------------------------------------
# We can select only some of these half-filled states by postselecting on measurement outcomes:
#


@qml.set_shots(shots)
@qml.qnode(dev)
def postselect_half_filled_state(x, selection):
    init_state(x)
    for w in range(3):
        # Postselect the measured qubit to match the selection criterion
        mcm = qml.measure(w, postselect=selection[w])
        qml.cond(~mcm, qml.X)(w + 3)

    return qml.counts(wires=range(6))


######################################################################
# As an example, suppose we wanted half-filled states that have a :math:`0` in the first and a
# :math:`1` in the third position. We do not postselect on the second qubit, which we can indicate
# by passing ``None`` to the ``postselect`` argument of :func:`~.pennylane.measure`. Again, before
# running the circuit, let's draw it first:
#

selection = [0, None, 1]
print(qml.draw(postselect_half_filled_state)(x, selection))

######################################################################
# Note the indicated postselection values next to the drawn MCMs.
#
# Time to run the postselecting circuit:
#

counts = postselect_half_filled_state(x, selection)
postselected_shots = sum(counts.values())

print(f"Obtained {postselected_shots} out of {shots} samples after postselection.")
print("The probabilities for the postselected bit strings are:")
for key, val in counts.items():
    print(f"    {key}: {val/postselected_shots*100:4.1f} %")

######################################################################
# We successfully postselected on the desired properties of the computational basis state. Note
# that the number of returned samples is reduced, because those samples that do not meet the
# postselection criteria are discarded entirely.
#
# Dynamically correct quantum states
# ----------------------------------
#
# If we do not want to postselect the prepared states but still would like to guarantee some of
# the qubits to be in a selected state, we can instead flip the corresponding
# pairs of bits if we measured the undesired state:
#


@qml.set_shots(shots)
@qml.qnode(dev)
def create_selected_half_filled_state(x, selection):
    init_state(x)
    all_mcms = []
    for w in range(3):
        # Don't postselect on the selection criterion, but store the MCM for later
        mcm = qml.measure(w)
        qml.cond(~mcm, qml.X)(w + 3)
        all_mcms.append(mcm)

    for w, sel, mcm in zip(range(3), selection, all_mcms):
        # If the postselection criterion is not None, flip the corresponding pair
        # of qubits conditioned on the mcm not satisfying the selection criterion
        if sel is not None:
            qml.cond(mcm != sel, qml.X)(w)
            qml.cond(mcm != sel, qml.X)(w + 3)

    return qml.counts(wires=range(6))


print(qml.draw(create_selected_half_filled_state)(x, selection))

######################################################################
# We can see how the measured values are fed not only into the original conditioned operation,
# but also into the two additional bit flips for our "correction" procedure, as long as the
# selection criterion is not ``None``. Let's execute the circuit:
#

counts = create_selected_half_filled_state(x, selection)
selected_shots = sum(counts.values())

print(f"Obtained all {selected_shots} of {shots} samples because we did not postselect")
print("The probabilities for the selected bit strings are:")
for key, val in counts.items():
    print(f"    {key}: {val/selected_shots*100:4.1f} %")

######################################################################
# Note that we kept all samples because we did not postselect, and that (in this particular case)
# this evened out the probabilities of measuring either of the two possible bit strings.
# Also, note that we conditionally
# applied the bit flip operators ``qml.X`` by comparing an MCM result to
# the corresponding selection criterion (``mcm!=sel``). More generally, MCM
# results in PennyLane can be processed with standard arithmetic operations. For details,
# see the `introduction to MCMs
# <https://docs.pennylane.ai/en/stable/introduction/measurements.html#mid-circuit-measurements-and-conditional-operations>`_
# and the documentation of :func:`~.pennylane.measure`.
#
# And this is how to create dynamic circuits in PennyLane with mid-circuit measurements!
#
