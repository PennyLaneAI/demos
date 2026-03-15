r"""Intro to the Quantum Fourier Transform
=============================================================

The quantum Fourier transform (QFT) is one of the most important building blocks in quantum algorithms, famously used in `quantum phase estimation <https://pennylane.ai/qml/demos/tutorial_qpe/>`__ and `Shor's factoring algorithm <https://en.wikipedia.org/wiki/Shor%27s_algorithm>`__.

The QFT is a quantum analog of the discrete Fourier transform --- the main tool of digital signal processing --- which is used to analyze periodic functions by mapping between time and frequency representations.

In this tutorial you will learn how to define this operation and how to build it with basic gates.

.. figure:: ../_static/demonstration_assets/qft/socialthumbnail_large_QFT_2024-04-04.png
    :align: center
    :width: 60%
    :target: javascript:void(0)

Defining the Quantum Fourier Transform
---------------------------------------

To appreciate the QFT, it will help to start with its classical counterpart.
The discrete Fourier transform (DFT) takes as input a vector :math:`(x_0, \dots, x_{N-1}) \in \mathbb{C}^N` and returns another vector :math:`(y_0, \dots, y_{N-1}) \in \mathbb{C}^N` where:

.. math::
  y_k = \sum_{j = 0}^{N-1} x_j \exp \left(-\frac{2\pi i kj}{N}\right).

For ease of comparison we assume :math:`N = 2^n`. The idea of the QFT is to perform the same operation but in a quantum state :math:`|x\rangle = \sum_{i = 0}^{N-1} x_i |i\rangle.`
In this case, the output is another quantum state :math:`|y\rangle = \sum_{i = 0}^{N-1} y_i |i\rangle` where:

.. math::
    y_k = \frac{1}{\sqrt{N}} \sum_{j = 0}^{N-1} x_j \exp \left(\frac{2\pi i kj}{N} \right).

For historical reasons, the sign of the exponent is positive in the defintion of the QFT, as opposed to a negative exponent in the DFT. Therefore the DFT technically coincides with the inverse operation :math:`\text{QFT}^{\dagger}` . Also, in the QFT we include normalization factor :math:`\frac{1}{\sqrt{N}}.`

These transformations are linear and can be represented by a matrix. Let's verify that they actually match using scipy's implementation of the DFT and PennyLane's implementation of the QFT:
"""

from scipy.linalg import dft
import pennylane as qml
qml.capture.enable()
import numpy as np

n = 2

print("DFT matrix for n = 2:\n")
print(np.round(1 / np.sqrt(2 ** n) * dft(2 ** n), 2))

qft_inverse = qml.adjoint(qml.QFT([0,1]))

print("\n inverse QFT matrix for n = 2:\n")
print(np.round(qft_inverse.matrix(), 2))

#############################################
# The QFT achieves something remarkable: it is able to transform an :math:`N`-dimensional vector encoded in a system of only :math:`n=\log_2 N` qubits. As we now explain, this is possible using only  :math:`\mathcal{O}(n^2)` operations, as opposed to  :math:`\mathcal{O}(n2^n)` steps required for the DFT.
#
# Building the Quantum Fourier Transform
# --------------------------------------
#
# To implement the QFT on a quantum computer, it is useful to express the transformation using the equivalent representation:
#
# .. math::
#
#    \text{QFT}|x\rangle = \bigotimes_{k = n-1}^{0} \left (|0\rangle + \exp \left (\frac{2\pi i 2^k}{2^n} x  \right) |1\rangle \right ),
#
# for :math:`x \in [0, \dots, N-1].` The nice thing about this formula is that it expresses the output state as a tensor product of single-qubit states. We call :math:`U_k` the unitary operator that is able to prepare the state of the k-th qubit. This operator is defined as:
#
# .. math::
#
#    U_k |x\rangle = |x_0 \dots x_{k-1}\rangle \otimes \left (|0\rangle + \exp \left (\frac{2\pi i 2^k}{2^n} x  \right) |1\rangle \right ) \otimes |x_{k+1} \dots x_{n-1}\rangle,
#
# where :math:`|x\rangle = |x_0 \dots x_{n-1}\rangle.`
# We can build :math:`U_k` with one Hadamard gate and controlled :class:`~.pennylane.PhaseShift` gates. Below we show an animation of the operator for the particular case of :math:`n = 4` and :math:`k = 1.`
# We represent the phase-shift gates with a box indicating the phase that they apply.
#
# .. figure:: ../_static/demonstration_assets/qft/qft_gif.gif
#    :align: center
#    :width: 80%
#
#
# Each gate applies a phase proportional to the position of its control qubit, and we only need to control on qubits that are "below" the target one. 
# With these operators we can now prepare the QFT by applying them sequentially:
#
# .. math::
#
#    QFT = U_{n-1} \dots U_1 U_0.
#
#
# Overall, each :math:`U_k` uses :math:`n-k-1` controlled phase-shift operations and one Hadamard,  which needs to be repeated for all :math:`n` qubits. This leads to a total of :math:`\mathcal{O}(n^2)` gates to implement the QFT. For example, the circuit implementation for the case of 3 qubits would be:
#
# .. figure::
#   ../_static/demonstration_assets/qft/qft3.jpeg
#
# Although this representation already defines the QFT, there are different conventions when writing the final result.
# In PennyLane, we rearrange the qubits in the opposite ordering; that is why we
# apply SWAP gates at the end. Let's see how the decomposition looks like using the drawer:

import pennylane as qml
qml.capture.enable()
from functools import partial
import matplotlib.pyplot as plt

plt.style.use('pennylane.drawer.plot')

# This line is to expand the circuit to see the operators
@partial(qml.transforms.decompose, max_expansion=1)

def circuit():
    qml.QFT(wires=range(4))

qml.draw_mpl(circuit, decimals = 2, style = "pennylane")()
plt.show()

#############################################
# Note that the numerical arguments are :math:`\frac{\pi}{2} \approx 1.57,` :math:`\frac{\pi}{4} \approx 0.79` and  :math:`\frac{\pi}{8} \approx 0.39` (rounded to the first two decimal places).
#
# Quantum Fourier transform in practice
# --------------------------------------
#
# We have seen how to define the QFT and how to build it with basic gates; 
# now it's time to put it into practice. Let's imagine that we have an operator that prepares the state:
#
# .. math::
#
#    |\psi\rangle = \frac{1}{\sqrt{2^5}} \sum_{x=0}^{31} \exp \left (\frac{-2 \pi i x}{10} \right)|x\rangle,
#
# whose associated period is :math:`10.` We will use the QFT in PennyLane to find that period,
# but first let's visualize the state by drawing the amplitudes:


def prep():
    """quntum function that prepares the state."""
    qml.PauliX(wires=0)
    for wire in range(1, 6):
        qml.Hadamard(wires=wire)
    qml.ControlledSequence(qml.PhaseShift(-2 * np.pi / 10, wires=0), control=range(1, 6))
    qml.PauliX(wires=0)

dev = qml.device("default.qubit")
@qml.qnode(dev)
def circuit():
    prep()
    return qml.state()

state = circuit().real[:32]

plt.bar(range(len(state)), state)
plt.xlabel("|x⟩")
plt.ylabel("Amplitude (real part)")
plt.show()

#############################################
# In this image we have represented only the real part so we can visualize it easily.
# The goal now is to compute the period of the function encoded in this five-qubit state. We will use the QFT,
# which is able to transform the state into the frequency domain. This is shown in the code below:
#

@qml.qnode(dev)
def circuit():
  prep()
  qml.QFT(wires=range(1, 6))

  return qml.probs(wires=range(1, 6))

state = circuit()[:32]

plt.bar(range(len(state)), state)
plt.xlabel("|x⟩")
plt.ylabel("probs")
plt.show()

#############################################
# The output has a clear peak at  :math:`|x\rangle = 3.`
# This value corresponds to an approximation of :math:`2^nf` where :math:`f` is the frequency and :math:`n` is the
# number of qubits.
# Once we know the frequency, we invert it to obtain the period :math:`T` of our state.
# In this case, the period is :math:`T = 2^5 / 3 \sim 10.66,` close to the real value of :math:`10.`
#
# Conclusion
# ----------
# In this tutorial, we've journeyed through the fundamentals and construction of the quantum Fourier transform, a
# cornerstone in quantum computing. We explored its mathematical
# formulation, its implementation with basic quantum gates, and its application demonstrating its usefulness in algorithms like quantum phase estimation.
# It is a technique that deserves to be mastered!
#
