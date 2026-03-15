r"""

Noisy circuits
==============

.. meta::
    :property="og:description": Learn how to simulate noisy quantum circuits
    :property="og:image": https://pennylane.ai/qml/_static/demonstration_assets/N-Nisq.png

.. related::

    tutorial_noisy_circuit_optimization Optimizing noisy circuits with Cirq
    pytorch_noise PyTorch and noisy devices

In this demonstration, you'll learn how to simulate noisy circuits using built-in functionality in
PennyLane. We'll cover the basics of noisy channels and density matrices, then use example code to
simulate noisy circuits. PennyLane, the library for differentiable quantum computations, has
unique features that enable us to compute gradients of noisy channels. We'll also explore how
to employ channel gradients to optimize noise parameters in a circuit.

We're putting the N in NISQ.

.. figure:: ../_static/demonstration_assets/noisy_circuits/N-Nisq.png
    :align: center
    :width: 20%

    ..
"""

##############################################################################
#
# Noisy operations
# ----------------
# Noise is any unwanted transformation that corrupts the intended
# output of a quantum computation. It can be separated into two categories.
#
# * **Coherent noise** is described by unitary operations that maintain the purity of the
#   output quantum state. A common source are systematic errors originating from
#   imperfectly-calibrated devices that do not exactly apply the desired gates, e.g., applying
#   a rotation by an angle :math:`\phi+\epsilon` instead of :math:`\phi.`
#
# * **Incoherent noise** is more problematic: it originates from a quantum computer
#   becoming entangled with the environment, resulting in mixed states — probability
#   distributions over different pure states. Incoherent noise thus leads to outputs that are
#   always random, regardless of what basis we measure in.
#
# Mixed states are described by `density matrices
# <https://en.wikipedia.org/wiki/Density_matrices>`__.
# They provide a more general method of describing quantum states that elegantly
# encodes a distribution over pure states in a single mathematical object.
# Mixed states are the most general description of a quantum state, of which pure
# states are a special case.
#
# The purpose of PennyLane's ``default.mixed`` device is to provide native
# support for mixed states and for simulating noisy computations. Let's use ``default.mixed`` to
# simulate a simple circuit for preparing the
# Bell state :math:`|\psi\rangle=\frac{1}{\sqrt{2}}(|00\rangle+|11\rangle).` We ask the QNode to
# return the expectation value of :math:`Z_0\otimes Z_1:`
#
import pennylane as qml
qml.capture.enable()
from jax import numpy as np
import jax
import jaxopt

jax.config.update("jax_platform_name", "cpu")
jax.config.update('jax_enable_x64', True)

dev = qml.device('default.mixed', wires=2)

@qml.qnode(dev)
def circuit():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))


print(f"QNode output = {circuit():.4f}")

######################################################################
# With a small modification of the circuit we can also ask for the density matrix. In this case, the density matrix is
# equal to :math:`|\psi\rangle\langle\psi|,`
# where :math:`|\psi\rangle=\frac{1}{\sqrt{2}}(|00\rangle + |11\rangle).`

@qml.qnode(dev)
def density_matrix_circuit():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.state()

matrix = density_matrix_circuit()
print(f"Output density matrix is = \n{np.real(matrix)}")

######################################################################
# Incoherent noise is modelled by
# quantum channels. Mathematically, a quantum channel is a linear, completely positive,
# and trace-preserving (`CPTP
# <https://www.quantiki.org/wiki/channel-cp-map>`__) map. A convenient strategy for representing
# quantum channels is to employ `Kraus operators
# <https://en.wikipedia.org/wiki/Quantum_operation#Kraus_operators>`__
# :math:`\{K_i\}` satisfying the condition
# :math:`\sum_i K_{i}^{\dagger} K_i = I.` For an initial state :math:`\rho,` the output
# state after the action of a channel :math:`\Phi` is:
#
# .. math::
#
#     \Phi(\rho) = \sum_i K_i \rho K_{i}^{\dagger}.
#
# Just like pure states are special cases of mixed states, unitary
# transformations are special cases of quantum channels. Unitary transformations are represented
# by a single Kraus operator,
# the unitary :math:`U,` and they transform a state as
# :math:`U\rho U^\dagger.`
#
# More generally, the action of a quantum channel can be interpreted as applying a
# transformation corresponding to the Kraus operator :math:`K_i` with some associated
# probability. More precisely, the channel applies the
# transformation
# :math:`\frac{1}{p_i}K_i\rho K_i^\dagger` with probability :math:`p_i = \text{Tr}[K_i \rho K_{i}^{
# \dagger}]`. Quantum
# channels therefore represent a probability distribution over different possible
# transformations on a quantum state. For
# example, consider the bit flip channel. It describes a transformation that flips the state of
# a qubit (applies an X gate) with probability :math:`p` and leaves it unchanged with probability
# :math:`1-p.` Its Kraus operators are
#
# .. math::
#
#     K_0 &= \sqrt{1-p}\begin{pmatrix}1 & 0\\ 0 & 1\end{pmatrix}, \\
#     K_1 &= \sqrt{p}\begin{pmatrix}0 & 1\\ 1 & 0\end{pmatrix}.
#
# This channel can be implemented in PennyLane using the :class:`qml.BitFlip <pennylane.BitFlip>`
# operation.
#
# Let's see what happens when we simulate this type of noise acting on
# both qubits in the circuit. We'll evaluate the QNode for different bit flip probabilities.
#


@qml.qnode(dev)
def bitflip_circuit(p):
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.BitFlip(p, wires=0)
    qml.BitFlip(p, wires=1)
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)), qml.state()


ps = [0.001, 0.01, 0.1, 0.2]
for p in ps:
    result = bitflip_circuit(p)
    print(f"QNode output for bit flip probability {p} is {result[0]:.4f}")


######################################################################
# The circuit behaves quite differently in the presence of noise! This will be familiar to anyone
# that has run an algorithm on quantum hardware. It is also highlights why error
# mitigation and error correction are so important. We can use PennyLane to look under the hood and
# see the output state of the circuit for the largest noise parameter

print(f"Output state for bit flip probability {p} is \n{result[1]}")

######################################################################
# Besides the bit flip channel, PennyLane supports several other noisy channels that are commonly
# used to describe experimental imperfections: :class:`~.pennylane.PhaseFlip`,
# :class:`~.pennylane.AmplitudeDamping`, :class:`~.pennylane.GeneralizedAmplitudeDamping`,
# :class:`~.pennylane.PhaseDamping`, and the :class:`~.pennylane.DepolarizingChannel`. You can also
# build your own custom channel using the operation :class:`~.pennylane.QubitChannel` by
# specifying its Kraus operators, or even submit a `pull request
# <https://pennylane.readthedocs.io/en/stable/development/guide.html>`__ introducing a new channel.
#
# Let's take a look at another example. The depolarizing channel is a
# generalization of
# the bit flip and phase flip channels, where each of the three possible Pauli errors can be
# applied to a single qubit. Its Kraus operators are given by
#
# .. math::
#
#     K_0 &= \sqrt{1-p}\begin{pmatrix}1 & 0\\ 0 & 1\end{pmatrix}, \\
#     K_1 &= \sqrt{p/3}\begin{pmatrix}0 & 1\\ 1 & 0\end{pmatrix}, \\
#     K_2 &= \sqrt{p/3}\begin{pmatrix}0 & -i\\ i & 0\end{pmatrix}, \\
#     K_3 &= \sqrt{p/3}\begin{pmatrix}1 & 0\\ 0 & -1\end{pmatrix}.
#
#
# A circuit modelling the effect of depolarizing noise in preparing a Bell state is implemented
# below.


@qml.qnode(dev)
def depolarizing_circuit(p):
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.DepolarizingChannel(p, wires=0)
    qml.DepolarizingChannel(p, wires=1)
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))


ps = [0.001, 0.01, 0.1, 0.2]
for p in ps:
    print(f"QNode output for depolarizing probability {p} is {depolarizing_circuit(p):.4f}")

######################################################################
# As before, the output deviates from the desired value as the amount of
# noise increases.
# Modelling the noise that occurs in real experiments requires careful consideration.
# PennyLane
# offers the flexibility to experiment with different combinations of noisy channels to either mimic
# the performance of quantum algorithms when deployed on real devices, or to explore the effect
# of more general quantum transformations.
#
# Channel gradients
# -----------------
#
# The ability to compute gradients of any operation is an essential ingredient of 
# :doc:`quantum differentiable programming </glossary/quantum_differentiable_programming>`.
# In PennyLane, it is possible to
# compute gradients of noisy channels and optimize them inside variational circuits.
# PennyLane supports analytical
# gradients for channels whose Kraus operators are proportional to unitary
# matrices [#johannes]_. In other cases, gradients are evaluated using finite differences.
#
# To illustrate this property, we'll consider an elementary example. We aim to learn the noise
# parameters of a circuit in order to reproduce an observed expectation value. So suppose that we
# run the circuit to prepare a Bell state
# on a hardware device and observe that the expectation value of :math:`Z_0\otimes Z_1` is
# not equal to 1 (as would occur with an ideal device), but instead has the value 0.7781. In the
# experiment, it is known that the
# major source of noise is amplitude damping, for example as a result of photon loss.
# Amplitude damping projects a state to :math:`|0\rangle` with probability :math:`p` and
# otherwise leaves it unchanged. It is
# described by the Kraus operators
#
# .. math::
#
#     K_0 = \begin{pmatrix}1 & 0\\ 0 & \sqrt{1-p}\end{pmatrix}, \quad
#     K_1 = \begin{pmatrix}0 & \sqrt{p}\\ 0 & 0\end{pmatrix}.
#
# What damping parameter (:math:`p`) explains the experimental outcome? We can answer this question
# by optimizing the channel parameters to reproduce the experimental
# observation! 💪 Since the parameter :math:`p` is a probability, we use a sigmoid function to
# ensure that the trainable parameters give rise to a valid channel parameter, i.e., a number
# between 0 and 1.
#
ev = 0.7781  # observed expectation value

def sigmoid(x):
    return 1/(1+np.exp(-x))

@qml.qnode(dev)
def damping_circuit(x):
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.AmplitudeDamping(sigmoid(x), wires=0)  # p = sigmoid(x)
    qml.AmplitudeDamping(sigmoid(x), wires=1)
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

######################################################################
# We optimize the circuit with respect to a simple cost function that attains its minimum when
# the output of the QNode is equal to the experimental value:


def cost(x, target):
    return (damping_circuit(x) - target)**2

######################################################################
# All that remains is to optimize the parameter. We use a straightforward gradient descent
# method.

steps = 35

gd = jaxopt.GradientDescent(cost, maxiter=steps, tol=1e-5)

x = np.array(0.01)
res = gd.run(x, ev)

print(f"QNode output after optimization = {damping_circuit(res.params):.4f}")
print(f"Experimental expectation value = {ev}")
print(f"Optimized noise parameter p = {sigmoid(x.take(0)):.4f}")

######################################################################
# Voilà! We've trained the noisy channel to reproduce the experimental observation. 😎
#
# Developing quantum algorithms that leverage the power of NISQ devices requires serious
# consideration of the effects of noise. With PennyLane, you have access to tools that can
# help you design, simulate, and optimize noisy quantum circuits. We look forward to seeing what
# the quantum community can achieve with them! 🚀 🎉 😸
#
#
# References
# ----------
#
# .. [#johannes]
#
#     Johannes Jakob Meyer, Johannes Borregaard, and Jens Eisert, "A variational toolbox for quantum
#     multi-parameter estimation." `arXiv:2006.06303 (2020) <https://arxiv.org/abs/2006.06303>`__.
#
