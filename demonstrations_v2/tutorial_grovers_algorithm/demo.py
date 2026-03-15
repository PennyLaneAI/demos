r""".. role:: html(raw)
   :format: html

Grover's Algorithm
==================

.. meta::
    :property="og:description": Learn how to find an entry in a list using Grover's algorithm 
    :property="og:image": https://pennylane.ai/qml/_static/demonstration_assets/thumbnail_tutorial_grovers_algorithm.png

.. related::

    tutorial_qft_arithmetics Basic arithmetic with the quantum Fourier transform (QFT)


`Grover's algorithm </codebook/#05-grovers-algorithm>`__ is an `oracle </codebook/04-basic-quantum-algorithms/02-the-magic-8-ball/>`__-based quantum
algorithm, proposed by Lov Grover [#Grover1996]_. In the original description, the author approaches the
following problem: suppose that we are searching for a specific phone number in a randomly-ordered
catalogue containing :math:`N` entries. To find such a number with a probability of
:math:`\frac{1}{2},` a classical algorithm will need to check the list on average
:math:`\frac{N}{2}` times.


In other words, the problem is defined by searching for an item on a list with :math:`N` items given
an Oracle access function :math:`f(x).` This function has the defining property that
:math:`f(x) = 1` if :math:`x` is the item we are looking for, and :math:`f(x) = 0`
otherwise. The solution to this black-box search problem is proposed as a quantum algorithm that
performs :math:`O(\sqrt{N})` oracular queries to the list with a high probability of finding the
answer, whereas any classical algorithm would require :math:`O(N)` queries.

In this tutorial, we are going to implement a search for an n-bit string item using a quantum
circuit based on Grover's algorithm.

The algorithm can be broken down into the following steps:

1. Prepare the initial state
2. Implement the oracle
3. Apply the Grover diffusion operator
4. Repeat steps 2 and 3  approximately :math:`\frac{\pi}{4}\sqrt{N}` times
5. Measure


Let's import the usual PennyLane and Numpy libraries to load the necessary functions:

"""

import matplotlib.pyplot as plt
import pennylane as qml
qml.capture.enable()
import numpy as np

######################################################################
# Preparing the Initial State
# ---------------------------
#
# To perform the search, we are going to create an n-dimensional system, which has :math:`N = 2^n`
# computational basis states, represented via :math:`N` binary numbers. More specifically, 
# bit strings with length :math:`n,` labelled as :math:`x_0,x_2,\cdots, x_{N-1}.`
# We initialize the system in the uniform superposition over all states, i.e.,
# the amplitudes associated with each of the :math:`N` basis states are equal:
#
# .. math:: |s\rangle ={\frac {1}{\sqrt {N}}}\sum _{x=0}^{N-1}|x\rangle .
#
#
# This can be achieved by applying a Hadamard gate to all the wires. We can inspect the circuit using
# :class:`~.pennylane.Snapshot` to see how the states change on each step. Let us check the probability of finding
# a state in the computational basis for a 2-qubit circuit, writing the following functions and
# QNodes:


NUM_QUBITS = 2
dev = qml.device("default.qubit", wires=NUM_QUBITS)
wires = list(range(NUM_QUBITS))


def equal_superposition(wires):
    for wire in wires:
        qml.Hadamard(wires=wire)


@qml.qnode(dev)
def circuit():
    qml.Snapshot("Initial state")
    equal_superposition(wires)
    qml.Snapshot("After applying the Hadamard gates")
    return qml.probs(wires=wires)  # Probability of finding a computational basis state on the wires


results = qml.snapshots(circuit)()

for k, result in results.items():
    print(f"{k}: {result}")

######################################################################
# Let's use a bar plot to better visualize the initial state amplitudes:


y = np.real(results["After applying the Hadamard gates"])
bit_strings = [f"{x:0{NUM_QUBITS}b}" for x in range(len(y))]

plt.bar(bit_strings, y, color = "#70CEFF")

plt.xticks(rotation="vertical")
plt.xlabel("State label")
plt.ylabel("Probability Amplitude")
plt.title("States probabilities amplitudes")
plt.show()

######################################################################
# As expected, they are equally distributed.
#
# The Oracle and Grover's diffusion operator
# ------------------------------------------
#
# Let's assume for now that only one index satisfies :math:`f(x) = 1`. We are going to call this index :math:`\omega.`
# To access :math:`f(x)` with an Oracle, we can formulate a unitary operator such that
#
# .. math::
#    \begin{cases}
#        U_{\omega }|x\rangle =-|x\rangle &{\text{for }}x=\omega {\text{, that is, }}f(x)=1,\\U_{\omega }|x\rangle =|x\rangle &{\text{for }}x\neq \omega {\text{, that is, }}f(x)=0,
#    \end{cases}
#
# where and :math:`U_\omega` acts by flipping the phase of the solution state while keeping the remaining states untouched. In other
# words, the unitary :math:`U_\omega` can be seen as a reflection around the set of orthogonal states
# to :math:`\vert \omega \rangle,` written as
#
# .. math:: U_\omega = \mathbb{I} - 2\vert \omega \rangle \langle \omega \vert.
#
# This can be easily implemented with :class:`~.pennylane.FlipSign`, which takes a binary array and flips the sign
# of the corresponding state.
#
# Let us take a look at an example. If we pass the array ``[0,0]``, the sign of the state
# :math:`\vert 00 \rangle = \begin{bmatrix} 1 \\0 \\0 \\0 \end{bmatrix}` will flip:

dev = qml.device("default.qubit", wires=NUM_QUBITS)

@qml.qnode(dev)
def circuit():
    qml.Snapshot("Initial state |00>")
    # Flipping the marked state
    qml.FlipSign([0, 0], wires=wires)
    qml.Snapshot("After flipping it")
    return qml.state()

results = qml.snapshots(circuit)()

for k, result in results.items():
    print(f"{k}: {result}")

y1 = np.real(results["Initial state |00>"])
y2 = np.real(results["After flipping it"])

bit_strings = [f"{x:0{NUM_QUBITS}b}" for x in range(len(y))]

plt.bar(bit_strings, y1, color = "#70CEFF")
plt.bar(bit_strings, y2, color = "#C756B2")

plt.xticks(rotation="vertical")
plt.xlabel("State label")
plt.ylabel("Probability Amplitude")
plt.title("States probabilities amplitudes")

plt.legend(["Initial state |00>", "After flipping it"])
plt.axhline(y=0.0, color="k", linestyle="-")
plt.show()

######################################################################
# We can see that the amplitude of the state :math:`\vert 01\rangle` flipped. Now, let us prepare
# the Oracle and inspect its action in the circuit.

omega = np.zeros(NUM_QUBITS)

def oracle(wires, omega):
    qml.FlipSign(omega, wires=wires)

dev = qml.device("default.qubit", wires=NUM_QUBITS)

@qml.qnode(dev)
def circuit():
    equal_superposition(wires)
    qml.Snapshot("Before querying the Oracle")

    oracle(wires, omega)
    qml.Snapshot("After querying the Oracle")

    return qml.probs(wires=wires)

results = qml.snapshots(circuit)()

for k, result in results.items():
    print(f"{k}: {result}")
##########################################

y1 = np.real(results["Before querying the Oracle"])
y2 = np.real(results["After querying the Oracle"])

bit_strings = [f"{x:0{NUM_QUBITS}b}" for x in range(len(y1))]

bar_width = 0.4

rect_1 = np.arange(0, len(y1))
rect_2 = [x + bar_width for x in rect_1]

plt.bar(
    rect_1,
    y1,
    width=bar_width,
    edgecolor="white",
    color = "#70CEFF",
    label="Before querying the Oracle",
)
plt.bar(
    rect_2,
    y2,
    width=bar_width,
    edgecolor="white",
    color = "#C756B2",
    label="After querying the Oracle",
)

plt.xticks(rect_1 + 0.2, bit_strings, rotation="vertical")
plt.xlabel("State label")
plt.ylabel("Probability Amplitude")
plt.title("States probabilities amplitudes")

plt.legend()
plt.show()

######################################################################
# We can see that the amplitude corresponding to the state :math:`\vert \omega \rangle` changed.
# However, we need an additional step to find the solution, since the probability of measuring any of
# the states remains equally distributed. This can be solved by applying the *Grover diffusion*
# operator, defined as
#
# .. math::
#    U_D = 2| s \rangle\langle s| - \mathbb{I}.
#
# The unitary :math:`U_D` also acts as a rotation, but this time through the uniform superposition :math:`\vert s \rangle.`
# Finally, the combination of :math:`U_{\omega}` with :math:`U_D` rotates the state
# :math:`\vert s \rangle` by an angle of
# :math:`\theta =2 \arcsin{\tfrac {1}{\sqrt {N}}}.` For more geometric insights
# about the oracle and the diffusion operator, please refer to this `PennyLane Codebook
# section </codebook/04-basic-quantum-algorithms/02-the-magic-8-ball/>`__.
#
#
# .. figure:: ../_static/demonstration_assets/grovers_algorithm/rotation.gif
#    :align: center
#    :width: 70%
#
#
# In a 2-qubit circuit, the diffusion operator has a specific shape:
#
# .. figure:: ../_static/demonstration_assets/grovers_algorithm/diffusion_2_qubits.svg
#    :align: center
#    :width: 90%
#
#
# Now, we have all the building blocks to implement a single-item search in a 2-qubit circuit. We can
# verify in the circuit below that applying the *Grover iterator* :math:`U_D U_\omega` once is enough
# to solve the problem.


dev = qml.device("default.qubit", wires=NUM_QUBITS)


def diffusion_operator(wires):
    for wire in wires:
        qml.Hadamard(wires=wire)
        qml.PauliZ(wires=wire)
    qml.ctrl(qml.PauliZ, 0)(wires=1)
    for wire in wires:
        qml.Hadamard(wires=wire)


@qml.qnode(dev)
def circuit():
    equal_superposition(wires)
    qml.Snapshot("Uniform superposition |s>")

    oracle(wires, omega)
    qml.Snapshot("State marked by Oracle")
    diffusion_operator(wires)

    qml.Snapshot("Amplitude after diffusion")
    return qml.probs(wires=wires)


results = qml.snapshots(circuit)()

for k, result in results.items():
    print(f"{k}: {result}")
######################################################################
# Searching for more items in a bigger list
# -----------------------------------------
#
# Now, let us consider the generalized problem with large :math:`N,` accepting :math:`M` solutions, with
# :math:`1 \leq M \leq N.` In this case, the optimal number of Grover iterations to find the solution
# is given by :math:`r \approx \left \lceil \frac{\pi}{4} \sqrt{\frac{N}{M}} \right \rceil`\ [#NandC2000]_.
#
# For more qubits, we can use the same function for the Oracle to mark the desired states, and the
# diffusion operator takes a more general form:
#
# .. figure:: ../_static/demonstration_assets/grovers_algorithm/diffusion_n_qubits.svg
#    :align: center
#    :width: 90%
#
# which is easily implemented using :class:`~.pennylane.GroverOperator`.
#
# Finally, we have all the tools to build the circuit for Grover's algorithm, as we can see in the
# code below. For simplicity, we are going to search for the states
# :math:`\vert 0\rangle ^{\otimes n}` and :math:`\vert 1\rangle ^{\otimes n},` where
# :math:`n = \log_2 N` is the number of qubits.

NUM_QUBITS = 5

omega = np.array([np.zeros(NUM_QUBITS), np.ones(NUM_QUBITS)])

M = len(omega)
N = 2**NUM_QUBITS
wires = list(range(NUM_QUBITS))

dev = qml.device("default.qubit", wires=NUM_QUBITS)

@qml.qnode(dev)
def circuit():
    iterations = int(np.round(np.sqrt(N / M) * np.pi / 4))

    # Initial state preparation
    equal_superposition(wires)

    # Grover's iterator
    for _ in range(iterations):
        for omg in omega:
            oracle(wires, omg)
        qml.templates.GroverOperator(wires)

    return qml.probs(wires=wires)


results = qml.snapshots(circuit)()

for k, result in results.items():
    print(f"{k}: {result}")
######################################################################
# Let us use a bar plot to visualize the probability to find the correct bitstring.

y = results["execution_results"]
bit_strings = [f"{x:0{NUM_QUBITS}b}" for x in range(len(y))]

plt.bar(bit_strings, results["execution_results"], color = "#70CEFF")

plt.xticks(rotation="vertical")
plt.xlabel("State label")
plt.ylabel("Probability")
plt.title("States probabilities")

plt.show()

######################################################################
# Conclusion
# -----------
#
# In conclusion, we have learned the basic steps of Grover's algorithm and how to implement it to search
# :math:`M` items in a list of size :math:`N` with high probability.
#
# Grover's algorithm in principle can be used to speed up more sophisticated computation, for
# instance, when used as a subroutine for problems that require extensive search
# and is the basis of a whole family of algorithms, such as the amplitude
# amplification technique. 
# 
# If you would like to learn more about Grover's algorithm, check out `this video <https://www.youtube.com/watch?v=EfUfwVnicP8>`__! 
#
#

######################################################################
# References
# ----------
#
# .. [#Grover1996]
#
#     L. K. Grover (1996) "A fast quantum mechanical algorithm for database search". `Proceedings of
#     the Twenty-Eighth Annual ACM Symposium on Theory of Computing. STOC '96. Philadelphia, Pennsylvania,
#     USA: Association for Computing Machinery: 212–219  
#     <https://dl.acm.org/doi/10.1145/237814.237866>`__.
#     (`arXiv <https://arxiv.org/abs/quant-ph/9605043>`__)
#
# .. [#NandC2000]
#
#     M. A. Nielsen, and I. L. Chuang (2000) "Quantum Computation and Quantum Information",
#     Cambridge University Press.
# 
