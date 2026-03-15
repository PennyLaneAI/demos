"""
How to quantum just-in-time (QJIT) compile Grover's algorithm with Catalyst
====================================================================

"""

######################################################################
# `Grover's algorithm </codebook/#05-grovers-algorithm>`__ is an `oracle
# </codebook/04-basic-quantum-algorithms/02-the-magic-8-ball/>`__-based quantum algorithm, first
# proposed by Lov Grover in 1996 [#Grover1996]_, to solve unstructured search problems using a
# `quantum computer <https://pennylane.ai/qml/quantum-computing/>`__. For example, we could use
# Grover's algorithm to search for a phone number in a randomly ordered database containing
# :math:`N` entries and say (with high probability) that the database contains that number by
# performing :math:`O(\sqrt{N})` queries on the database, whereas a classical search algorithm would
# require :math:`O(N)` queries to perform the same task.
#
# More formally, the *unstructured search problem* is defined as a search for a string of bits in a
# list containing :math:`N` items given an *oracle access function* :math:`f(x).` This function is
# defined such that :math:`f(x) = 1` if :math:`x` is the bitstring we are looking for (the
# *solution*), and :math:`f(x) = 0` otherwise. The generalized form of Grover's algorithm accepts
# :math:`M` solutions, with :math:`1 \leq M \leq N.`
#
# In this tutorial, we will implement the generalized Grover's algorithm using `Catalyst
# <https://docs.pennylane.ai/projects/catalyst>`__, a quantum just-in-time (QJIT) compiler framework
# for PennyLane, which makes it possible to compile, optimize, and execute hybrid quantum–classical
# workflows. We will also measure the performance improvement we get from using Catalyst with
# respect to the native Python implementation and show that the runtime performance of circuits
# compiled with Catalyst can be approximately an order of magnitude faster.
# 
# .. figure:: ../_static/demo_thumbnails/opengraph_demo_thumbnails/OGthumbnail_qjit_compile_grovers_algorithm_with_catalyst.png
#     :align: center
#     :width: 70%
#     :target: javascript:void(0)


######################################################################
# Generalized Grover's algorithm with PennyLane
# ---------------------------------------------
#
# In the :doc:`Grover's Algorithm <demos/tutorial_grovers_algorithm>` tutorial, we saw how to implement
# the generalized Grover's algorithm in PennyLane. The procedure is as follows:
#
# #. Initialize the system to an equal superposition over all states.
# #. Perform :math:`r(N, M)` *Grover iterations*:
#
#    #. Apply the unitary *oracle operator*, :math:`U_\omega,` implemented using
#       :class:`~.pennylane.FlipSign`, for each solution index :math:`\omega.`
#    #. Apply the *Grover diffusion operator*, :math:`U_D,` implemented using
#       :class:`~.pennylane.GroverOperator`.
#
# #. Measure the resulting quantum state in the computational basis.
#
# We also saw (in Ref. [#NandC2000]_) that the optimal number of Grover iterations to find the
# solution is given by
#
# .. math:: r(N, M) \approx \left \lceil \frac{\pi}{4} \sqrt{\frac{N}{M}} \right \rceil .
#
# For simplicity, throughout the rest of this tutorial we will consider the search for the :math:`M
# = 2` solution states :math:`\vert 0\rangle ^{\otimes n}` and :math:`\vert 1\rangle ^{\otimes n},`
# where :math:`n = \log_2 N` is the number of qubits, in a "database" of size :math:`N = 2^n`
# containing all possible :math:`n`-qubit states.
#
# First, we'll import the required packages and define the Grover's algorithm circuit, as we did in
# the :doc:`previous tutorial <demos/tutorial_grovers_algorithm>`.

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
qml.capture.enable()


def equal_superposition(wires):
    for wire in wires:
        qml.Hadamard(wires=wire)


def oracle(wires, omega):
    qml.FlipSign(omega, wires=wires)


def num_grover_iterations(N, M):
    return np.ceil(np.sqrt(N / M) * np.pi / 4).astype(int)


def grover_circuit(num_qubits):
    wires = list(range(num_qubits))
    omega = np.array([np.zeros(num_qubits), np.ones(num_qubits)])

    M = len(omega)
    N = 2**num_qubits

    # Initial state preparation
    equal_superposition(wires)

    # Grover iterations
    for _ in range(num_grover_iterations(N, M)):
        for omg in omega:
            oracle(wires, omg)
        qml.templates.GroverOperator(wires)

    return qml.probs(wires=wires)


######################################################################
# We'll begin with a circuit defined using the default state-simulator device, ``"default.qubit"``,
# as our baseline. See the documentation in :func:`~.pennylane.device` for a list of other supported
# devices. To run our performance benchmarks, we'll increase the number of qubits in our circuit to
# :math:`n = 12.`

NUM_QUBITS = 12

dev = qml.device("default.qubit", wires=NUM_QUBITS)


@qml.qnode(dev)
def circuit_default_qubit():
    return grover_circuit(NUM_QUBITS)


results = circuit_default_qubit()


######################################################################
# Let's quickly confirm that Grover's algorithm correctly identified the solution states
# :math:`\vert 0\rangle ^{\otimes n}` and :math:`\vert 1\rangle ^{\otimes n}` as the most likely
# states to be measured.


def most_probable_states_descending(probs, N):
    """Returns the indices of the N most probable states in descending order."""
    if N > len(probs):
        raise ValueError("N cannot be greater than the length of the probs array.")

    return np.argsort(probs)[-N:][::-1]


def print_most_probable_states_descending(probs, N):
    """Prints the most probable states and their probabilities in descending order."""
    for i in most_probable_states_descending(probs, N):
        print(f"Prob of state '{i:0{NUM_QUBITS}b}': {probs[i]:.4g}")


print_most_probable_states_descending(results, N=2)


######################################################################
# It worked! We are now ready to QJIT compile our Grover's algorithm circuit.


######################################################################
# Quantum just-in-time compiling the circuit
# ------------------------------------------
#
# Catalyst is developed natively for `PennyLane's high-performance simulators
# <https://pennylane.ai/performance/>`__ and, at the time of writing, does not support the
# ``"default.qubit"`` state-simulator device. Let's first define a new circuit using `Lightning
# <https://docs.pennylane.ai/projects/lightning>`__, which is a PennyLane plugin that provides more
# performant state simulators written in C++. See the :doc:`Catalyst documentation
# <catalyst:dev/devices>` for the full list of devices supported by Catalyst.

dev = qml.device("lightning.qubit", wires=NUM_QUBITS)


@qml.qnode(dev)
def circuit_lightning():
    return grover_circuit(NUM_QUBITS)


######################################################################
# Then, to QJIT compile our circuit with Catalyst, we simply wrap it with :func:`~pennylane.qjit`.

circuit_qjit = qml.qjit(circuit_lightning)


######################################################################
# .. note::
#
#     The Catalyst :class:`~.qjit` decorator supports capturing control flow when specified using
#     the :func:`~pennylane.for_loop`, :func:`~pennylane.while_loop`, and :func:`~pennylane.cond`
#     functions, or additionally, can automatically capture native Python control flow via
#     experimental :doc:`AutoGraph <catalyst:dev/autograph>` support.
#
#     In this tutorial, however, you'll notice that our ``grover_circuit`` function is able to use
#     native Python control flow without the need to convert the Python ``for`` loops to the
#     QJIT-compatible :func:`~.for_loop`, for instance, and without using AutoGraph. The reason we
#     are able to do so here is twofold:
#
#     * The circuit we have compiled, ``circuit_lightning``, is *unparameterized*, meaning it takes
#       in no input arguments. Thus, the control flow of the circuit does not depend on any dynamic
#       variables (whose values are known only at run time).
#
#     * The ranges of the ``for`` loops depend only on static variables (i.e., constants known at
#       compile time), in this case Python-native numerics and lists, and NumPy arrays.
#
#     Hence, the complete control flow of the circuit is known at compile time, which allows us to
#     use native Python control-flow statements.
#
#     See the :doc:`Sharp bits and debugging tips <catalyst:dev/sharp_bits>` section of the Catalyst
#     documentation for more details on this subject.


######################################################################
# We now have our QJIT object, ``circuit_qjit``. A small detail to note in this case is that because
# the function ``circuit_lightning`` takes no input arguments, Catalyst will, in fact,
# *ahead-of-time* (AOT) compile the circuit at instantiation, meaning that when we call this QJIT
# object for the first time, the compilation will have already taken place, and Catalyst will
# execute the compiled code. With JIT compilation, by contrast, the compilation is triggered at the
# first call site rather than at instantiation. See the `Compilation Modes
# <https://docs.pennylane.ai/projects/catalyst/en/stable/dev/quick_start.html#compilation-modes>`__
# documentation in the :doc:`Catalyst Quick Start <catalyst:dev/quick_start>` guide for more
# information on the difference between JIT and AOT compilation.
#
# The compilation step will incur some runtime overhead, which we will measure below. Let's first
# call the compiled circuit and confirm that we get the same results.

results_qjit = circuit_qjit()
print_most_probable_states_descending(results_qjit, N=2)


######################################################################
# Indeed, we get the same results as before: the compiled circuit has correctly identified the
# solution states :math:`\vert 0\rangle ^{\otimes n}` and :math:`\vert 1\rangle ^{\otimes n}` as the
# most likely states to be measured. We can also compare the results more rigorously by comparing
# element-wise the computed probability of every state (within the given floating-point tolerance):

results_are_equal = np.allclose(results, results_qjit, atol=1e-12)
print(f"Native-Python and compiled circuits yield same results? {results_are_equal}")


######################################################################
# Success!


######################################################################
# Benchmarking
# ------------
#
# Let's start profiling the circuits we have defined. We have four function executions in total to
# profile:
#
# #. Executing the circuit using ``"default.qubit"``.
# #. Executing the circuit using ``"lightning.qubit"``.
# #. Compiling the circuit with Catalyst, to measure the AOT compilation overhead.
# #. Calls to the QJIT-compiled circuit, to measure the circuit execution time.
#
# We'll use the `timeit <https://docs.python.org/3/library/timeit.html>`__ module part of the Python
# Standard Library to measure the runtimes. To improve the statistical precision of these
# measurements, we'll repeat the operations for items (2) and (4) five times; item (1) is slow, and
# item (3) is only run once by construction, so we will not repeat these operations.

import timeit

NUM_REPS = 5

runtimes_native_default = timeit.repeat(
    "circuit_default_qubit()",
    globals={"circuit_default_qubit": circuit_default_qubit},
    number=1,
    repeat=1,
)
runtimes_native_lightning = timeit.repeat(
    "circuit_lightning()",
    globals={"circuit_lightning": circuit_lightning},
    number=1,
    repeat=NUM_REPS,
)
runtimes_compilation = timeit.repeat(
    "qml.qjit(circuit_lightning)",
    setup="import pennylane as qml
qml.capture.enable()",
    globals={"circuit_lightning": circuit_lightning},
    number=1,
    repeat=1,
)
runtimes_qjit_call = timeit.repeat(
    "_circuit_qjit()",
    setup="import pennylane as qml
qml.capture.enable(); _circuit_qjit = qml.qjit(circuit_lightning);",
    globals={"circuit_lightning": circuit_lightning},
    number=1,
    repeat=NUM_REPS,
)

run_names = [
    "Native (default.qubit)",
    "Native (lightning.qubit)",
    "QJIT compilation",
    "QJIT call",
]
run_names_display = [name.replace(" ", "\n", 1) for name in run_names]
runtimes = [
    np.mean(runtimes_native_default),
    np.mean(runtimes_native_lightning),
    np.mean(runtimes_compilation),
    np.mean(runtimes_qjit_call),
]


def std_err(x):
    """Standard error = sample standard deviation / sqrt(sample size)"""
    if len(x) == 1:
        return np.nan
    return np.std(x, ddof=1) / np.sqrt(len(x))


runtimes_err = [
    std_err(runtimes_native_default),
    std_err(runtimes_native_lightning),
    std_err(runtimes_compilation),
    std_err(runtimes_qjit_call),
]

for i in range(len(run_names)):
    print(f"{run_names[i]} runtime: ({runtimes[i]:.4g} +/- {runtimes_err[i]:.2g}) s")


######################################################################
# Let's plot these runtimes as a bar chart to compare them visually.

fig = plt.figure(figsize=[8.0, 4.8])
plt.title("Grover's Algorithm Runtime Benchmarks")
bars = plt.bar(run_names_display, runtimes, color="#70CEFF")
plt.errorbar(
    run_names_display, runtimes, yerr=runtimes_err, fmt="None", capsize=2.0, c="k"
)
plt.bar_label(bars, fmt="{:#.2g} s", padding=5)
plt.xlabel("Function Executed")
plt.ylabel("Runtime [s]")
plt.margins(y=0.15)
plt.text(
    0.98,
    0.98,
    f"Number of qubits, $n = {NUM_QUBITS}$",
    ha="right",
    va="top",
    transform=plt.gca().transAxes,
)
plt.tight_layout()
plt.show()


######################################################################
# This plot illustrates the power of Catalyst: by simply wrapping our Grover's algorithm circuit as
# a QJIT-compiled object (or AOT-compiled in this case) with :func:`~pennylane.qjit`, we have
# achieved execution runtimes approximately an order of magnitude shorter than the PennyLane circuit
# implemented using the ``"lightning.qubit"`` device.
#
# There is one important caveat in this example, however, which is that the compilation step itself
# takes several times longer than the time it takes to run the circuit using the Lightning state
# simulator directly. This is an important factor to consider when deciding whether to QJIT compile
# your own circuits in performance-critical applications. In the case of Grover's algorithm, we only
# needed to execute the circuit once to obtain the solution, so the runtime incurred in the
# compilation step offsets the gain in the compiled circuit's execution time. However, should it be
# necessary to execute your own quantum circuit many times, the runtime savings are compounded with
# every subsequent call to the compiled circuit. Since the compilation step only needs to be
# performed once, the *total* runtime of the QJIT workflow may begin to outperform the baseline
# Lightning workflow as the number of circuit executions increases. [*]_


######################################################################
# Conclusion
# -----------
#
# This tutorial has demonstrated how to just-in-time compile a quantum circuit implementing the
# generalized Grover's algorithm using `Catalyst <https://docs.pennylane.ai/projects/catalyst>`__.
#
# For a circuit with :math:`n = 12` qubits, analogous to a search in a randomly ordered "database"
# containing :math:`N = 2^{12} = 4096` entries, Catalyst offers a circuit-execution runtime
# performance approximately an order of magnitude better than the same circuit implemented using the
# Lightning state-simulator device, with the caveat that the compilation step itself incurs some
# runtime over the workflow with a direct call to the Lightning-implemented circuit.
#
# To learn more about Catalyst and how to use it to compile and optimize your quantum programs and
# workflows, check out the :doc:`Catalyst Quick Start <catalyst:dev/quick_start>` guide.


######################################################################
# References
# ----------
#
# .. [#Grover1996]
#
#     L. K. Grover (1996) "A fast quantum mechanical algorithm for database search". `Proceedings of
#     the Twenty-Eighth Annual ACM Symposium on Theory of Computing. STOC '96. Philadelphia,
#     Pennsylvania, USA: Association for Computing Machinery: 212–219
#     <https://dl.acm.org/doi/10.1145/237814.237866>`__.
#     (arXiv: `9605043 [quant-ph] <https://arxiv.org/abs/quant-ph/9605043>`__)
#
# .. [#NandC2000]
#
#     M. A. Nielsen, and I. L. Chuang (2000) "Quantum Computation and Quantum Information",
#     Cambridge University Press.


######################################################################
# Footnotes
# ---------
#
# .. [*]
#
#     The performance improvements that can be achieved with QJIT compilation will depend on the
#     specific size and topology of your PennyLane circuit.
#
