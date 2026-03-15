r"""How to collect statistics of mid-circuit measurements
=========================================================

Measuring qubits in the middle of a quantum circuit execution can be useful in many ways.
From understanding the inner workings of a circuit, hardware characterization,
modeling and error mitigation, to error correction, algorithmic improvements and even up to full
computations encoded as measurements in measurement-based quantum computation (MBQC).

Before turning to any of these advanced topics, it is worthwhile to familiarize ourselves with
the syntax and features around mid-circuit measurements (MCMs). In this how-to, we will focus on
extracting statistics about measurements that are performed while a quantum circuit is up and
running --- mid-circuit measurement statistics!

.. figure:: ../_static/demonstration_assets/how_to_collect_mcm_stats/socialthumbnail_how_to_collect_mcm_stats.png
    :align: center
    :width: 50%


"""

######################################################################
# Defining the circuit ansatz
# ---------------------------
#
# We start by defining a quantum circuit ansatz that switches between a layer of simple rotation gates
# (:class:`~.pennylane.RX`), mid-circuit measurements(:func:`~.pennylane.measure`), and a layer
# of entangling two-qubit gates (:class:`~.pennylane.CNOT`) between the first and all other qubits.
# The ansatz then returns the list of four MCM values, so that we can process them further in a full quantum circuit
# As we will treat the first wire differently than all other wires, we define it as separate variable.
#
# Along the way, we perform some standard imports and set a randomness seed.
#

import pennylane as qml
qml.capture.enable()
import numpy as np

np.random.seed(511)

first_wire = 0
other_wires = [1, 2, 3]


def ansatz(x):
    mcms = []

    # Rotate all qubits
    for w, x_ in enumerate(x):
        qml.RX(x_, w)

    # Measure first qubit
    mcms.append(qml.measure(first_wire))

    # Entangle all qubits with first qubit
    for w in other_wires:
        qml.CNOT([first_wire, w])

    # Measure and reset all qubits but the first
    for w in other_wires:
        mcms.append(qml.measure(w, reset=True))

    return mcms


######################################################################
# A quantum circuit with basic MCM statistics
# -------------------------------------------
#
# Before we post-process the mid-circuit measurements in this ansatz or expand the ansatz itself,
# let's construct a simple :class:`~.pennylane.QNode` and look at the statistics of the four
# performed MCMs:
#
# 1. We compute the probability vector for the MCM on the first qubit, and
#
# 2. count the bit strings sampled from the other three MCMs.
#
# To implement the ``QNode``, we also define a shot-based qubit device.
#

dev = qml.device("default.qubit")


@qml.set_shots(100)
@qml.qnode(dev)
def simple_node(x):
    # apply the ansatz, and collect mid-circuit measurements. mcm1 is the measurement
    # of wire 0, and mcms2 is a list of measurements of the other wires.
    mcm1, *mcms2 = ansatz(x)
    return qml.probs(op=mcm1), qml.counts(mcms2)


######################################################################
# Before executing the circuit, let's draw it! For this, we sample some random  parameters, one
# for each qubit, and call the Matplotlib drawer :func:`~.pennylane.draw_mpl`.
#

x = np.random.random(4)
fig, ax = qml.draw_mpl(simple_node)(x)

######################################################################
# Neat, let's move on to executing the circuit. We apply the ``defer_measurements`` transform to
# the ``QNode`` because it allows for fast evaluation even with many shots.

probs, counts = qml.defer_measurements(simple_node)(x)
print(f"Probability vector of first qubit MCM: {np.round(probs, 5)}")
print(f"Bit string counts on other qubits: {counts}")

######################################################################
# We see that the first qubit has a probability of about :math:`20\%` to be in the state
# :math:`|1\rangle` after the rotation. We also observe that we only sampled bit strings from
# the other three qubits for which the second and third bit are identical.
# (Quiz question: Is this expected behaviour or did we just not sample often enough?
# Find the answer at the end of the how-to!)
#
# Post-processing mid-circuit measurements
# ----------------------------------------
# We now set up a more interesting ``QNode``. It executes the ``ansatz`` from above twice and
# compares the obtained MCMs (note that we did not define ``comparing_function`` yet, we will
# get to that shortly):
#


@qml.set_shots(100)
@qml.qnode(dev)
def interesting_qnode(x):
    first_mcms = ansatz(x)
    second_mcms = ansatz(-x)
    output = comparing_function(first_mcms, second_mcms)
    return qml.counts(output)


######################################################################
# Before we can run this more interesting ``QNode``, we need to actually specify the
# ``comparing_function``. We ask the following question: Is the measurement on the first qubit
# equal between the two sets of MCMs, and do the other three measured values summed together
# have the same parity, i.e. is the number of 1s odd in both sets or even in both sets?
#
# In contrast to quantum measurements at the end of a :class:`~.pennylane.QNode`,
# PennyLane supports a number of unary and binary operators for MCMs even *within*
# ``QNode``\ s. This enables us to phrase the question above as a boolean function.
# Consider the
# `introduction on measurements <https://docs.pennylane.ai/en/stable/introduction/measurements.html#mid-circuit-measurements-and-conditional-operations>`_
# and the documentation if you want to learn more about the supported operations.
#


def comparing_function(first_mcms, second_mcms):
    """A function that compares two sets of MCM outcomes."""
    equal_first = first_mcms[0] == second_mcms[0]
    # Computing the parity can be done with the bitwise "and" operator `&`
    # with the number 1. Note that Python's and is not supported between MCMs!
    first_parity = sum(first_mcms[1:]) & 1
    second_parity = sum(second_mcms[1:]) & 1
    equal_parity = first_parity == second_parity
    return equal_first & equal_parity


######################################################################
# We can again inspect this ``QNode`` by drawing it:
#

fig, ax = qml.draw_mpl(interesting_qnode)(x)

######################################################################
# Note how all mid-circuit measurements feed into the classical output variable.
#
# Finally we may run the ``QNode`` and obtain the statistics for our comparison function:
#

print(qml.defer_measurements(interesting_qnode)(x))

######################################################################
# We find that our question is answered with "yes" in about :math:`2/3` of all samples.
# Turning up the number of shots lets us compute this ratio more precisely:
#

num_shots = 10000
counts = qml.set_shots(qml.defer_measurements(interesting_qnode), shots=num_shots)(x)
p_yes = counts[True] / num_shots
p_no = counts[False] / num_shots
print(f'The probability to answer with "yes" / "no" is {p_yes:.5f} / {p_no:.5f}')

######################################################################
# This concludes our how-to on statistics and post-processing of mid-circuit measurements.
# If you would like to explore mid-circuit measurement applications, be sure to check out
# our :doc:`MBQC demo <demos/tutorial_mbqc>` and the
# :doc:`demo on quantum teleportation <demos/tutorial_teleportation>`. Or, see all available functionality in our
# `measurements quickstart page <https://docs.pennylane.ai/en/stable/introduction/measurements.html#mid-circuit-measurements-and-conditional-operations>`_.
#
# For performance considerations, take a look at
# :func:`~.pennylane.defer_measurements` and :func:`~.pennylane.dynamic_one_shot`,
# two simulation techniques that PennyLane uses under the hood to run circuits
# like the ones in this how-to.
#
# And finally, the answer to our quiz question above: It's not expected that we
# never see bit strings with differing second and third bits.
# Sampling more shots eventually reveals this, even though they remain rare:

probs, counts = qml.set_shots(qml.defer_measurements(simple_node), shots=10000)(x)
print(f"Bit string counts on last three qubits: {counts}")

######################################################################
# Supported MCM return types
# --------------------------
#
# Before finishing, we discuss the return types that are supported for (postprocessed) MCMs.
# Depending on the processing applied to the MCM results, not all return types are supported.
# ``qml.probs(mcm0 * mcm1)``, for example, is not a valid return value, because it is not clear
# which probabilities are being requested.
#
# Furthermore, available return types depend on whether or not the device is
# shot-based (``qml.sample`` can not be returned if the device is not sampling).
# Overall, **all combinations of post-processing and all of**
# :func:`~.pennylane.expval`,
# :func:`~.pennylane.var`,
# :func:`~.pennylane.probs`,
# :func:`~.pennylane.sample`, **and**
# :func:`~.pennylane.counts`,
# **are supported** for mid-circuit measurements with the following exceptions:
#
# - ``qml.sample`` and ``qml.counts`` are not supported for ``shots=None``.
# - ``qml.probs`` is not supported for MCMs collected in arithmetic expressions.
# - ``qml.expval`` and ``qml.var`` are not supported for sequences of MCMs.
#   ``qml.probs``, ``qml.sample``, and ``qml.counts`` are supported for sequences but
#   only if they do not contain arithmetic expressions of these MCMs.
#
# For more details also consider the
# `measurements quickstart page <https://docs.pennylane.ai/en/stable/introduction/measurements.html#mid-circuit-measurements-and-conditional-operations>`_
# and the documentation of :func:`~.pennylane.measure`.
#
