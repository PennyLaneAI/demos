r"""Intro to Quantum Phase Estimation
=============================================================

The Quantum Phase Estimation (QPE) algorithm [#qpe]_ is one of the most important tools in quantum
computing. Maybe **the** most important. It solves a deceptively simple task: given an eigenstate of a
unitary operator, find its eigenvalue. This demo explains the basics of the QPE algorithm.
After reading it, you will be able to understand
the algorithm and how to implement it in PennyLane.

.. figure:: ../_static/demonstration_assets/qpe/socialthumbnail_large_Quantum_Phase_Estimation_2023-11-27.png
    :align: center
    :width: 50%



Quantum phase estimation
------------------------

Let's define the problem more carefully. We are given a unitary
operator :math:`U` and one of its eigenstates :math:`|\psi \rangle.` The operator is unitary,
so we can write:

.. math::
    U |\psi \rangle = e^{i \phi} |\psi \rangle,

where :math:`\phi` is the *phase* of the eigenvalue (remember, unitaries have eigenvalues with an 
absolute value of 1). The goal is to estimate :math:`\phi,`
hence the name *phase estimation*. Our challenge is to design a quantum algorithm to solve this problem.
How would that work?



Part 1: Representing the phase
------------------------------
A first step is to find a quantum circuit that performs the transformation

.. math::
    |\psi \rangle |0\rangle \rightarrow  |\psi \rangle |\phi\rangle.

We could then obtain :math:`\phi` directly by measuring the second register.  We call this the **estimation register**.

But let's be more careful. Because the
complex exponential has period :math:`2\pi,` technically the phase is not unique. Instead, we
define :math:`\phi = 2\pi \theta` so that :math:`\theta` is a number between 0 and 1; this forces :math:`\phi`
to be between 0 and :math:`2\pi.` We'll refer to :math:`\theta` as the phase from now on.

How can we represent :math:`\theta` on a quantum computer? The answer is the first clever part of the algorithm: we represent
:math:`\theta` in binary. 🧠

Since you probably don't use binary fractions on a daily basis (or do you?), it's worth stopping for a moment
to make sure we're on the same page.

.. tip::
    **Binary fractions**

    When we write the number 0.15625, it is being expressed as a sum of multiples of powers of
    10:

    .. math::
        0.15625 = 1 \times 10^{-1} + 5 \times 10^{-2} + 6 \times 10^{-3} + 2 \times 10^{-4} +  5 \times 10^{-5}.

    But nothing is stopping us from using 2 instead of 10. In binary, the same number is

    .. math::
        0.00101 = 0 \times 2^{-1} + 0 \times 2^{-2} + 1 \times 2^{-3} + 0 \times 2^{-4} +  1 \times 2^{-5}.

    (You can confirm this by computing 1/8 + 1/32 on a calculator). Similarly, 0.5 is 0.1 in binary,
    and 0.3125 is 0.0101.

Ok, now back to quantum. A binary representation is useful because we can encode it using
qubits, e.g., :math:`|110010\rangle` for :math:`\theta=0.110010.` The phase is retrieved by measuring the qubits.
The **precision** of the estimate is determined by the number of qubits. We've used examples of fractions that can be
conveniently expressed exactly with just a few binary points, but this won't
always be possible. For example, the binary expansion of :math:`0.8` is :math:`0.11001100...` which does not terminate.
From now on, we'll use :math:`n` for the number of estimation qubits.

Part 2: Quantum Fourier Transform
---------------------------------

The second clever part of the algorithm is to follow advice given to many physicists:
"When in doubt, take the Fourier transform"; or in our case, "When in doubt, take the quantum Fourier transform (QFT)".

.. math::
   \text{QFT}|\theta\rangle = \frac{1}{\sqrt{2^n}}\sum^{2^n-1}_{k=0} e^{2 \pi i\theta k} |k\rangle.

Note that this results in a uniform superposition, where each basis state has an additional phase.
If we can prepare that state, then applying the *inverse* QFT would give
:math:`|\theta\rangle` in the estimation register.
This looks more promising, especially if we notice the appearance of the eigenvalues :math:`e^{2 \pi i\theta},`
although with an extra factor of :math:`k.` We can obtain this factor by applying the unitary :math:`k` times to the state :math:`|\psi\rangle:`

.. math::
   U^k|\psi\rangle =  e^{2\pi i \theta k} |\psi\rangle.

Therefore, we will use :math:`|\psi\rangle` and :math:`U` to generate the factors that are of interest to us in each of the basic states.
It would then be enough to create an operator such that:

.. math::
   |\psi\rangle |k\rangle \rightarrow  U^k |\psi\rangle |k\rangle.


In this way, if we apply this operator to the uniform superposition we obtain:

.. math::
    \frac{1}{\sqrt{2^n}}\sum_{k=0}|\psi\rangle |k\rangle \rightarrow  \frac{1}{\sqrt{2^n}}\sum_{k=0}U^k|\psi\rangle|k\rangle =  |\psi\rangle \frac{1}{\sqrt{2^n}}\sum_{k=0} e^{2 \pi i\theta k} |k\rangle

This is exactly what we want!
In PennyLane we refer to this as a :class:`~.ControlledSequence` operation. Let's see how to build it.

Part 3: Controlled sequence
---------------------------

We follow another timeless piece of physics advice: "If stuck, start with the simplest case".
Let's see what happens with two qubits. After applying the Hadamards (and omitting normalization factors),
the operator we need is

.. math::
   |\psi\rangle |00\rangle + |\psi\rangle |01\rangle + |\psi\rangle |10\rangle+  |\psi\rangle |11\rangle\rightarrow
   |\psi\rangle |00\rangle + U |\psi\rangle |01\rangle + U^2 |\psi\rangle |10\rangle+ U^3 |\psi\rangle |11\rangle.

Notice something? The power of :math:`U` is the same as the binary representation of the corresponding basis state. For example,  :math:`U^3` is applied when the estimation register is in state :math:`|11\rangle,` and 11 is just the number 3 in binary. 
Therefore, the desired operation can be implemented by applying :math:`U` controlled on the first qubit, and
:math:`U^2` controlled on the second qubit.

We can extend this idea to any number of qubits.
The following animation illustrates this effect.

.. figure:: ../_static/demonstration_assets/qpe/controlledSequence.gif
    :align: center
    :width: 80%

    Example of the controlled sequence operator applied to different 3-qubit basic states. The gates that are being actually applied are
    shown in black since the controlling qubit takes the value 1. It can be verified that the power of the final operator matches the binary input.

With six qubits, an example would be

.. math::
   |\psi\rangle |010111\rangle \rightarrow U^{16}U^4U^2U^{1}|\psi\rangle |010111\rangle = U^{23}|\psi\rangle |010111\rangle.

Note that 010111 is 23 in binary.

So we have the answer: apply :math:`U^{2^m}` controlled on the `m`-th estimation qubit.
This operator facilitates, among other things, performing :doc:`arithmetics in quantum computers <demos/tutorial_qft_arithmetics>`.

Bringing it all together, here is the quantum phase estimation algorithm in all its glory:

The QPE algorithm
-----------------

1. Start with the state :math:`|\psi \rangle |0\rangle.` Apply a Hadamard gate to all estimation qubits to implement the
   transformation

   .. math::

       |\psi \rangle |0\rangle \rightarrow |\psi\rangle \frac{1}{\sqrt{2^n}}\sum_{k=0} |k\rangle.

2. Apply a :class:`~.ControlledSequence` operation, i.e., :math:`U^{2^m}` controlled on the `m`-th estimation qubit.
   This gives

   .. math::

       |\psi\rangle \frac{1}{\sqrt{2^n}}\sum_{k=0} |k\rangle \rightarrow  |\psi\rangle \frac{1}{\sqrt{2^n}}\sum_{k=0} e^{2\pi i \theta k}|k\rangle.

3. Apply the inverse quantum Fourier transform to the estimation qubits

   .. math::

      |\psi\rangle \frac{1}{\sqrt{2^n}}\sum_{k=0} e^{2 \pi i \theta k}|k\rangle \rightarrow |\psi\rangle|\theta\rangle.

4. Measure the estimation qubits to recover :math:`\theta.`

.. figure:: ../_static/demonstration_assets/qpe/qpe.png
    :align: center
    :width: 80%

    The quantum phase estimation circuit.

QPE is doing something incredible: it can calculate eigenvalues **without ever diagonalizing
a matrix**. Wow! This is true even if we relax the assumption that the input is an eigenstate. By linearity, for an arbitrary
state expanded in the eigenbasis of :math:`U` as

.. math::
   |\Psi\rangle = \sum_i c_i |\psi_i\rangle,

QPE outputs the eigenphase :math:`\theta_i` with probability :math:`|c_i|^2.`

Most of the heavy lifting is done by the controlled sequence step. Control-U operations are the heart of the algorithm,
coupled with a clever use of quantum Fourier transforms. This feature is crucial for quantum chemistry applications,
where preparing good initial states is essential [#initial_state]_.
If you want to learn more about this check out our :doc:`demo <demos/tutorial_initial_state_preparation>`.

One more point of importance. Generally it is not possible to represent a given phase exactly using a limited number of
estimation qubits. Thus, there is typically a distribution of possible
outcomes, which induce an error in the estimation. We'll see an example in the code below.
The error *decreases* exponentially with the number of estimation qubits, but the number of controlled-U operations
*increases* exponentially. The math is such that these effects basically cancel out and the cost of estimating a phase
with error :math:`\varepsilon` is proportional to :math:`1/\varepsilon.`

All the previous ideas help to also understand the Phase KickBack algorithm in the case of one qubit.
If you want to learn more about this subroutine, take a look at this :doc:`demo <demos/tutorial_phase_kickback>`.

Time to code!
-------------
We already know the three building blocks of QPE; it is time to put them to practice.
We use a single-qubit :class:`~pennylane.PhaseShift` operator :math:`U = R_{\phi}(2 \pi / 5)`
and its eigenstate :math:`|1\rangle` with corresponding phase :math:`\theta=0.2`.

"""

import pennylane as qml
qml.capture.enable()
import numpy as np

def U(wires):
    return qml.PhaseShift(2 * np.pi / 5, wires=wires)

##############################################################################
# We construct a uniform superposition by applying Hadamard gates followed by a :class:`~.ControlledSequence`
# operation.
# Finally, we perform the adjoint of :class:`~.QFT` and
# return the probability of each computational basis state.


dev = qml.device("default.qubit")

@qml.qnode(dev)
def circuit_qpe(estimation_wires):
    # initialize to state |1>
    qml.PauliX(wires=0)

    for wire in estimation_wires:
        qml.Hadamard(wires=wire)

    qml.ControlledSequence(U(wires=0), control=estimation_wires)

    qml.adjoint(qml.QFT)(wires=estimation_wires)

    return qml.probs(wires=estimation_wires)

##############################################################################
# Let's run the circuit and plot the results. We use 4 estimation qubits.


import matplotlib.pyplot as plt

estimation_wires = range(1, 5)

results = circuit_qpe(estimation_wires)

bit_strings = [f"0.{x:0{len(estimation_wires)}b}" for x in range(len(results))]

plt.bar(bit_strings, results)
plt.xlabel("phase")
plt.ylabel("probability")
plt.xticks(rotation="vertical")
plt.subplots_adjust(bottom=0.3)

plt.show()

##############################################################################
# Since the eigenphase cannot be represented exactly using 4 bits, there is a
# distribution of possible outcomes. The peak occurs
# at :math:`\phi = 0.0011,` which is :math:`0.1875` in decimal. This is the closest value we can get with
# a 4-bit representation to the exact value :math:`0.2.`  🎊
#
# Conclusion
# ----------
# This demo presented the "textbook" version of QPE. There are multiple variations, notably iterative QPE that
# uses a single estimation qubit, as well as Bayesian versions that saturate optimal prefactors appearing in the
# total cost. There are also mathematical subtleties about cost and errors that are important but out of
# scope for this demo.
#
# Finally, there is extensive work on how to implement the unitaries themselves. In quantum chemistry,
# the main strategy is to encode a molecular Hamiltonian
# into a unitary such that the phases are invertible functions of the Hamiltonian eigenvalues. This can be done for instance
# through the mapping :math:`U=e^{-iHt},` which can be implemented using Hamiltonian simulation techniques. More advanced techniques employ a qubitization-based encoding. QPE can then
# be used to estimate eigenvalues like ground-state energies by sampling them with respect to a distribution
# induced by the input state.
#
# References
# ---------------
#
# .. [#qpe]
#
#    A.Yu.Kitaev. "Quantum measurements and the Abelian Stabilizer Problem",
#    `Arxiv <https://arxiv.org/abs/quant-ph/9511026>`__, 1995
#
# .. [#initial_state]
#
#    Stepan Fomichev et al. "Initial state preparation for quantum chemistry on quantum computers",
#    `Arxiv <https://arxiv.org/pdf/2310.18410.pdf>`__, 2023
#
