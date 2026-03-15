r"""
QJIT compilation with Qrack and Catalyst
========================================

`Qrack <https://github.com/unitaryfund/qrack>`__ is a GPU-accelerated quantum
computer simulator with many novel optimizations, and `PyQrack
<https://github.com/unitaryfund/pyqrack>`__ is its Python wrapper, written in
pure (``ctypes``) Python language standard. Founded in 2017 by Dan Strano and
Benn Bollay, Qrack's vision was always to provide the best possible (classical)
quantum computer emulator, targeting the use case of running
industrially-relevant quantum workloads without recourse to genuine quantum
computer hardware.

In this tutorial you will learn how to use Qrack with
PennyLane and quantum just-in-time (QJIT) compilation via
`Catalyst
<https://docs.pennylane.ai/projects/catalyst/en/stable/index.html>`__ --- enabling
your hybrid quantum-classical Python program to be compiled and executed
with Qrack with significant performance boosts.

You'll learn certain suggested cases of use where Qrack might particularly excel
at delivering lightning-fast performance or minimizing required memory resources
— for example, special cases of the quantum or discrete :doc:`Fourier transform
<demos/tutorial_qft/>`, circuits with predominantly :doc:`Clifford
<demos/tutorial_clifford_circuit_simulations/>` or classical preambles,
circuits with :doc:`mid-circuit measurement
<demos/tutorial_mcm_introduction/>`, and high-width circuits with
low-complexity representations in terms of a QBDD (quantum binary decision
diagram). However, Qrack is a general-purpose simulator, so you might
employ it for all their applications and still see parity with or improvement
over available device back ends.

.. figure:: ../_static/demonstration_assets/qrack/qrack_catalyst_integration_shelf.png
    :align: center
    :width: 90%
    :target: javascript:void(0);

How Qrack works
---------------

When developing `Qrack <https://github.com/unitaryfund/qrack>`__, we wanted to
provide the emulator as open source, free of charge, agnostic to any specific
GPU or hardware accelerator provider, backwards compatible to serve those with
very limited classical computer hardware resources, but (nonetheless) capable of
scaling to supercomputer systems, as secure and free of external dependencies as
possible, and under the reasonably permissive LGPL license, with bindings and
wrappers for third-party libraries provided under even more permissive licenses
like MIT and Apache 2.0. Our hope was that the global floor of minimal access to
cost-competitive quantum workload throughput would never be lower than the
capabilities of Qrack.

When simulating quantum subroutines of varying qubit widths, Qrack will
transparently, automatically, and dynamically transition between GPU-based and
CPU-based simulation techniques for maximal execution speed, to respond to
situations when qubit registers might be too narrow to benefit from the large
parallel processing element count of a GPU (up to maybe roughly 20 qubits,
depending upon the classical hardware platform). Qrack also offers so-called
hybrid stabilizer simulation (with fallback to universal simulation) and
near-Clifford simulation with a greatly reduced memory footprint on Clifford
gate sets with the inclusion of the `RZ` variational Pauli Z-axis rotation gate.
(For more information, see the `QCE'23 report <https://arxiv.org/abs/2304.14969>`__ [#QCEReport]_ by the Qrack and `Unitary Fund <https://unitary.fund/>`__ teams.)

Particularly for systems that don't rely on GPU acceleration, Qrack offers a
quantum binary decision diagram (QBDD) simulation algorithm option that might
significantly reduce the memory footprint or execution complexity for circuits
with low entanglement, as judged by the complexity of a QBDD to represent the
state. (Qrack's implementation of QBDD is entirely original source code, but it
is based on reports like `this one <https://arxiv.org/abs/2302.04687>`__
[#Wille]_.) Qrack also offers approximation options aimed at trading off minimal
fidelity reduction for maximum reduction in simulation complexity (as opposed to
models of physical noise), including methods based on the Schmidt decomposition
rounding parameter (SDRP) [#QCEReport]_ and the near-Clifford rounding parameter
(NCRP).

The Qrack simulator doesn't fit neatly into a single canonical category of
quantum computer simulation algorithm: it optionally and by default leverages
elements of state vector simulation, tensor network simulation, stabilizer and
near-Clifford simulation, and QBDD simulation, often all at once, while it
introduces some novel algorithmic tricks for the Schmidt decomposition of
state vectors in a manner similar to matrix product state (MPS) simulation.

Demonstrating Qrack with the quantum Fourier transform
------------------------------------------------------

The :doc:`quantum Fourier transform (QFT) <demos/tutorial_qft>` is a building-block
subroutine of many other quantum algorithms. Qrack exhibits unique capability
for many cases of the QFT algorithm, and its worst-case performance is
competitive with other popular quantum computer simulators [#QCEReport]_. In this
section, you'll be presented with examples of Qrack's uniquely optimal
performance on the QFT.

In the case of a *trivial* computational basis eigenstate input, Qrack can
simulate basically any QFT width. Below, we pick a random eigenstate
initialization and perform the QFT across a width of 60 qubits, with Catalyst's `qjit <https://docs.pennylane.ai/projects/catalyst/en/latest/code/api/catalyst.qjit.html>`__.
"""

import pennylane as qml
qml.capture.enable()
from pennylane import numpy as np
from catalyst import qjit

import matplotlib.pyplot as plt

import random

qubits = 60
dev = qml.device("qrack.simulator", qubits)

@qjit
@qml.set_shots(8)
@qml.qnode(dev)
def circuit():
    for i in range(qubits):
        if random.uniform(0, 1) < 0.5:
            qml.X(wires=[i])
    qml.QFT(wires=range(qubits))
    return qml.sample(wires=range(qubits))

def counts_from_samples(samples):
    counts = {}
    for sample in samples:
        s = 0

        for bit in sample:
            s = (s << 1) | bit

        s = str(s)

        if s in counts:
            counts[s] = counts[s] + 1
        else:
            counts[s] = 1

    return counts

counts = counts_from_samples(circuit())

plt.bar(counts.keys(), counts.values())
plt.title(f"QFT on {qubits} Qubits with Random Eigenstate Init. (8 samples)")
plt.xlabel("|x⟩")
plt.ylabel("counts")
plt.show()

##############################################################################
# .. figure:: ../_static/demonstration_assets/qrack/fig1.png
#     :align: center
#     :width: 90%
#     :target: javascript:void(0);

##############################################################################
# In this image we have represented only 8 measurement samples so we can visualize the result more easily.
#
# This becomes harder if we request a non-trivial initialization. In general, Qrack will use
# Schmidt decomposition techniques to try to break up circuits into separable subsystems of
# qubits to simulate semi-independently, combining them just-in-time (JIT) with Kronecker
# products when they need to interact, according the used circuit definition.
#
# The circuit becomes much harder for Qrack if we randomly initialize the input qubits
# with Haar-random `U3 gates <https://docs.pennylane.ai/en/stable/code/api/pennylane.U3.html>`__, which you can see below,
# but the performance is still significantly better than the worst case (of
# `GHZ state <https://en.wikipedia.org/wiki/Greenberger%E2%80%93Horne%E2%80%93Zeilinger_state>`__ initialization).

qubits = 12
dev = qml.device("qrack.simulator", qubits)

@qjit
@qml.set_shots(8)
@qml.qnode(dev)
def circuit():
    for i in range(qubits):
        th = random.uniform(0, np.pi)
        ph = random.uniform(0, np.pi)
        dl = random.uniform(0, np.pi)
        qml.U3(th, ph, dl, wires=[i])
    qml.QFT(wires=range(qubits))
    return qml.sample(wires=range(qubits))

counts = counts_from_samples(circuit())

plt.bar(counts.keys(), counts.values())
plt.title(f"QFT on {qubits} Qubits with Random U3 Init. (8 samples)")
plt.xlabel("|x⟩")
plt.ylabel("counts")
plt.show()

##############################################################################
# .. figure:: ../_static/demonstration_assets/qrack/fig2.png
#     :align: center
#     :width: 90%
#     :target: javascript:void(0);

##############################################################################
# Alternate simulation algorithms (QBDD and near-Clifford)
# --------------------------------------------------------
# By default, Qrack relies on a combination of state vector simulation, "hybrid" stabilizer and
# near-Clifford simulation, and Schmidt decomposition optimization techniques. Alternatively, we could use
# pure stabilizer simulation or QBDD simulation if the circuit is at all amenable to optimization in this way.
#
# To demonstrate this, we prepare a 60-qubit GHZ state, which would commonly be intractable in the case of state vector simulation.

qubits = 60
dev = qml.device(
    "qrack.simulator",
    qubits,
    isBinaryDecisionTree=False,
    isStabilizerHybrid=True,
    isSchmidtDecompose=False,
)

@qjit
@qml.set_shots(8)
@qml.qnode(dev)
def circuit():
    qml.Hadamard(0)
    for i in range(1, qubits):
        qml.CNOT(wires=[i - 1, i])
    return qml.sample(wires=range(qubits))

counts = counts_from_samples(circuit())

plt.bar(counts.keys(), counts.values())
plt.title(f"{qubits}-Qubit GHZ preparation (8 samples)")
plt.xlabel("|x⟩")
plt.ylabel("counts")
plt.show()

##############################################################################
# .. figure:: ../_static/demonstration_assets/qrack/fig3.png
#     :align: center
#     :width: 90%
#     :target: javascript:void(0);

##############################################################################
# As you can see, Qrack was able to construct the 60-qubit GHZ state (without
# exceeding memory limitations), and the probability is peaked at bit strings of all 0 and all 1.
#
# It's trivial for Qrack to perform large GHZ state preparations with "hybrid" stabilizer
# or near-Clifford simulation if Schmidt decomposition is deactivated.
# QBDD cannot be accelerated by GPU, so its application might be limited, but it is parallel
# over CPU processing elements, hence it might be particularly well-suited for systems with no GPU at all.
# Qrack's default simulation methods will likely still outperform QBDD on BQP-complete problems
# like random circuit sampling or quantum volume certification.
#

qubits = 24
dev = qml.device(
    "qrack.simulator", qubits, isBinaryDecisionTree=True, isStabilizerHybrid=False
)

@qjit
@qml.set_shots(8)
@qml.qnode(dev)
def circuit():
    qml.Hadamard(0)
    for i in range(1, qubits):
        qml.CNOT(wires=[i - 1, i])
    return qml.sample(wires=range(qubits))

counts = counts_from_samples(circuit())

plt.bar(counts.keys(), counts.values())
plt.title(f"{qubits}-Qubit GHZ preparation (8 samples)")
plt.xlabel("|x⟩")
plt.ylabel("counts")
plt.show()

##############################################################################
# .. figure:: ../_static/demonstration_assets/qrack/fig4.png
#     :align: center
#     :width: 90%
#     :target: javascript:void(0);

##############################################################################
# If your gate set is restricted to Clifford with general :class:`~pennylane.RZ` gates
# (being mindful of the fact that compilers like Catalyst might optimize such a gate set basis into different gates),
# the time complexity for measurement samples becomes doubly exponential with near-Clifford simulation,
# but the space complexity is almost exactly that of stabilizer simulation for the logical qubits plus
# an ancillary qubit per (non-optimized) :class:`~.pennylane.RZ` gate, scaling like the square of the
# sum of the count of the logical and ancillary qubits put together.
#
# Comparing performance
# ---------------------
# .. note::
#   These results were obtained using Ubuntu 22.04 LTS, Linux kernel version 5.19.0-35-generic,
#   using one Intel(R) Core(TM) i9-10980HK CPU @ 2.40GHz, one NVIDIA GeForce RTX 3080 Laptop GPU,
#   and 32 GB of SK hynix 3200 MT/s DDR4 in 2x16 GB row configuration.
#   Results may differ if run on other system configurations.
# We've already seen that the Qrack device back end can do some tasks that most other simulators, or
# basically any other simulator, simply can't do, like 60-qubit-wide special cases of the QFT or GHZ state
# preparation with a Clifford or universal (QBDD) simulation algorithm, for example. However,
# in the worst case for circuit complexity, Qrack will tend perform similarly to state vector simulation.
#
# How does the performance of Qrack compare with other simulators' on a non-trivial problem,
# like the U3 initialization we used above for the `QFT algorithm <#demonstrating-qrack-with-the-quantum-fourier-transform>`_?

import time

def bench(n, results):
    for device in ["qrack.simulator", "lightning.qubit"]:
        dev = qml.device(device, n)

        @qjit
        @qml.set_shots(1)
        @qml.qnode(dev)
        def circuit():
            for i in range(n):
                th = random.uniform(0, np.pi)
                ph = random.uniform(0, np.pi)
                dl = random.uniform(0, np.pi)
                qml.U3(th, ph, dl, wires=[i])
            qml.QFT(wires=range(n))
            return qml.sample(wires=range(n))

        start_ns = time.perf_counter_ns()
        circuit()
        results[
            f"Qrack ({n} qb)" if device == "qrack.simulator" else f"Lightning ({n} qb)"
        ] = time.perf_counter_ns() - start_ns

    return results

results = {}
results = bench(6, results)
results = bench(12, results)
results = bench(18, results)

bar_colors = ["purple", "yellow", "purple", "yellow"]
plt.bar(results.keys(), results.values(), color=bar_colors)
plt.title("Performance comparison, QFT with U3 initialization (1 sample apiece)")
plt.xlabel("|x⟩")
plt.ylabel("Nanoseconds")
plt.show()

##############################################################################
# .. figure:: ../_static/demonstration_assets/qrack/fig5.png
#     :align: center
#     :width: 90%
#     :target: javascript:void(0);

##############################################################################
# Benchmarks will differ somewhat when running this code on your local machine, for example,
# but we tend to see that Qrack manages to demonstrate good performance compared to the
# `Lightning simulators <https://docs.pennylane.ai/projects/lightning>`__ on this task case.
# (Note that this initialization case isn't specifically the hardest case of the QFT for Qrack;
# that's probably rather a GHZ state input.)
#
# Similarly, we can use quantum just-in-time (QJIT) compilation from PennyLane's
# `Catalyst <https://docs.pennylane.ai/projects/catalyst>`__, for both Qrack and
# `Lightning <https://docs.pennylane.ai/projects/lightning>`__. How does Qrack with QJIT compare to Qrack without it?

def bench(n, results):
    dev = qml.device("qrack.simulator", n)

    @qjit
    @qml.set_shots(1)
    @qml.qnode(dev)
    def circuit():
        for i in range(n):
            th = random.uniform(0, np.pi)
            ph = random.uniform(0, np.pi)
            dl = random.uniform(0, np.pi)
            qml.U3(th, ph, dl, wires=[i])
        qml.QFT(wires=range(n))
        return qml.sample(wires=range(n))

    start_ns = time.perf_counter_ns()
    circuit()
    results[f"QJIT Qrack ({n} qb)"] = time.perf_counter_ns() - start_ns

    @qml.set_shots(1)
    @qml.qnode(dev)
    def circuit():
        for i in range(n):
            th = random.uniform(0, np.pi)
            ph = random.uniform(0, np.pi)
            dl = random.uniform(0, np.pi)
            qml.U3(th, ph, dl, wires=[i])
        qml.QFT(wires=range(n))
        return qml.sample(wires=range(n))

    start_ns = time.perf_counter_ns()
    circuit()
    results[f"PyQrack ({n} qb)"] = time.perf_counter_ns() - start_ns

    return results

# Make sure OpenCL has been initalized in PyQrack:
bench(6, results)

results = {}
results = bench(6, results)
results = bench(12, results)
results = bench(18, results)

bar_colors = ["purple", "yellow", "purple", "yellow"]
plt.bar(results.keys(), results.values(), color=bar_colors)
plt.title("Performance comparison, QFT with U3 initialization (1 sample apiece)")
plt.xlabel("|x⟩")
plt.ylabel("Nanoseconds")
plt.show()

##############################################################################
# .. figure:: ../_static/demonstration_assets/qrack/fig6.png
#     :align: center
#     :width: 90%
#     :target: javascript:void(0);

##############################################################################
# Again, your mileage may vary somewhat, depending on your local system, but Qrack tends
# to be significantly faster with Catalyst's QJIT than without!
#
# As a basic test of validity, if we compare the inner product between both simulator
# state vector outputs on some QFT case, do they agree?

def validate(n):
    results = []
    for device in ["qrack.simulator", "lightning.qubit"]:
        dev = qml.device(device, n)

        @qjit
        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(0)
            for i in range(1, n):
                qml.CNOT(wires=[i - 1, i])
            qml.QFT(wires=range(n))
            return qml.state()

        start_ns = time.perf_counter_ns()
        results.append(circuit())

    return np.abs(sum([np.conj(x) * y for x, y in zip(results[0], results[1])]))

print("Qrack cross entropy with Lightning:", validate(12), "out of 1.0")

##############################################################################
# .. rst-class:: sphx-glr-script-out
#
#  .. code-block:: none
#
#       Qrack cross entropy with Lightning: 0.9999997797266185 out of 1.0

##############################################################################
# Conclusion
# ----------
# In this tutorial, we've demonstrated the basics of using the `Qrack <https://github.com/unitaryfund/qrack>`__
# simulator back end and showed examples of special cases on which Qrack's novel optimizations can lead
# to huge increases in performance or maximum achievable qubit widths. Remember the Qrack device back
# end for PennyLane if you'd like to leverage GPU acceleration but don't want to complicate your
# choice of devices or device initialization, to handle a mixture of wide and narrow qubit registers in your subroutines.

##############################################################################
#
# References
# ----------
#
# .. [#QCEReport]
#
#     Daniel Strano, Benn Bollay, Aryan Blaauw, Nathan Shammah, William J. Zeng, Andrea Mari
#     "Exact and approximate simulation of large quantum circuits on a single GPU"
#     `arXiv:2304.14969 <https://arxiv.org/abs/2304.14969>`__, 2023.
#
# .. [#Wille]
#
#     Robert Wille, Stefan Hillmich, Lukas Burgholzer
#     "Decision Diagrams for Quantum Computing"
#     `arXiv:2302.04687 <https://arxiv.org/abs/2302.04687>`__, 2023.
#
