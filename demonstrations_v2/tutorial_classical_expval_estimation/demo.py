r"""
Pauli Propagation: Classically estimating expectation values from parametrized quantum circuits
===============================================================================================

In the race between classical and `quantum computing <https://pennylane.ai/qml/quantum-computing/>`__, an important question
is whether there exist efficient classical algorithms to simulate quantum
circuits.
Probably the most widely-known result of this type is the Gottesman–Knill
theorem [#gottesman]_. It states that quantum circuits consisting of Clifford
gates alone can be simulated classically, provided that the initial state
of the circuit is "nice enough" (also see the :doc:`PennyLane Demo on Clifford simulation
<demos/tutorial_clifford_circuit_simulations>`).

In this demo we will showcase a new result on classical simulation of quantum
circuits at a glance! For this, we will learn about *Pauli propagation*, how to
truncate it, and how it results in an efficient classical algorithm for
estimating expectation values of parametrized quantum circuits (on average
across parameter settings).
We will implement a basic variant in PennyLane, and discuss limitations and the
important details that lie at the heart of the new preprint called
*Classically estimating observables of noiseless quantum circuits*
by Angrisani et al. [#angrisani]_.

This result is important, as it casts doubt on the usefulness of generic parametrized quantum
circuits in quantum computations. Read on if you are wondering whether this
preprint just dequantized your work!

.. figure:: ../_static/demo_thumbnails/opengraph_demo_thumbnails/OGthumbnail_classical_expval_estimation.png
    :align: center
    :width: 60%
    :target: javascript:void(0)

Our target: Estimating expectation values from quantum circuits
---------------------------------------------------------------

Let's start by looking at the type of quantum computations that we want to
simulate classically.
Given an initial state :math:`|\psi_0\rangle,` a parametrized quantum
circuit :math:`U(\theta)`, and an observable :math:`H,` a common task in
many variational quantum algorithms is to compute the expectation value

.. math::

    E(\theta) = \langle \psi_0 | U^\dagger(\theta) H U(\theta) |\psi_0\rangle

for various parameter settings :math:`\theta.`
Being able to estimate such an expectation value efficiently is required
to train the parametrized quantum circuit in applications such as
:doc:`QAOA <demos/tutorial_qaoa_intro>`,
the :doc:`variational quantum eigensolver <demos/tutorial_vqe>` and a wide
range of `quantum machine learning <https://pennylane.ai/qml/whatisqml/>`__ tasks.

For simplicity, we will assume the initial state to be
:math:`|\psi_0\rangle=|0\rangle` throughout.
Similarly, we will work with a particular example for the circuit
:math:`U(\theta)` and discuss the class of :doc:`parametrized circuits </glossary/variational_circuit/>` the algorithm
can tackle further below.

For our demo, we choose the widely used hardware-efficient ansatz, which alternates
arbitrary single-qubit rotations with layers of entangling gates.
Lastly, the observable :math:`H` can be decomposed into a sum of Pauli words as

.. math::

    H = \sum_{\ell=1}^L h_\ell P_\ell, \quad P_\ell \in \{I, X, Y, Z\}^{\otimes N}.

Here, :math:`N` is the total number of qubits.
The number of qubits on which a Pauli word is supported (i.e., has a non-identity
component) is called its *weight*.
For the algorithm to work, the weights of the Hamiltonian's Pauli words may not
exceed a set threshold, see the truncation step below.
For our example we pick the Heisenberg model Hamiltonian

.. math::

    H_{XYZ} = \sum_{j=1}^{N-1} h_j^{(X)} X_j X_{j+1} + h_j^{(Y)} Y_j Y_{j+1} + h_j^{(Z)} Z_j Z_{j+1}

with random coefficients :math:`h_j^{(X|Y|Z)}.` In :math:`H_{XYZ},` each Pauli word
has weight two.

We will discuss the type of Hamiltonians that the algorithm can tackle
further below, but for now, let's define the Hamiltonian and the circuit ansatz in code.
We pick a :math:`\operatorname{CNOT}` layer on a ring as entangler for the ansatz.
We transform the ``ansatz`` function with :func:`~.pennylane.transforms.make_tape`, making
the function into one that returns a tape containing the gates (and expectation value).
This also allows us to draw the ansatz easily with :func:`~.pennylane.drawer.tape_text`.
To maintain an overview, we set the number of qubits and layers to just 4 and 3, respectively.
"""

import pennylane as qml
import numpy as np
from itertools import combinations, product


def _ansatz(params, num_qubits, H):
    """Parametrized quantum circuit ansatz that alternates arbitrary single-qubit
    rotations with strongly entangling CNOT layers. The depth of the ansatz and the
    number of qubits are given by the first dimension of the input parameters."""

    for i, params_layer in enumerate(params):
        # Execute arbitrary parametrized single-qubit rotations
        for j, params_qubit in enumerate(params_layer):
            qml.RZ(params_qubit[0], j)
            qml.RY(params_qubit[1], j)
            qml.RZ(params_qubit[2], j)
        # If we are not in the last layer, execute an entangling CNOT layer
        if i < len(params) - 1:
            for j in range(num_qubits):
                qml.CNOT([j, (j + 1) % num_qubits])

    return qml.expval(H)


ansatz = qml.transforms.make_tape(_ansatz)

num_qubits = 4
num_layers = 3
np.random.seed(852)
H_coeffs = np.random.random((num_qubits - 1) * 3)
H_ops = [op(j) @ op(j + 1) for j in range(num_qubits - 1) for op in [qml.X, qml.Y, qml.Z]]
H = qml.dot(H_coeffs, H_ops)

# Smaller parameter set to get smaller circuit to draw
params = np.random.random((num_layers, num_qubits, 3))
tape = ansatz(params, num_qubits, H)
print(qml.drawer.tape_text(tape))

##############################################################################
# Now that we have our example set up, let's look at the core technique behind
# the simulation algorithm.
#
# Pauli propagation
# -----------------
#
# A standard classical simulation technique for quantum circuits is based on
# state vector simulation, which updates the state vector of the quantum system
# with each gate applied to it. From a physics perspective, this is the evolution of
# a quantum state in the Schrödinger picture. To conclude the simulation, this approach
# then contracts the evolved state with the observable :math:`H.`
#
# Here we will use a technique based on the Heisenberg picture, which
# describes the evolution of the measurement observable :math:`H` instead.
# This technique is called *Pauli propagation* and has also been used by
# related simulation algorithms [#aharonov]_, [#lowesa]_, [#begusic]_.
#
# In the Heisenberg picture, each gate :math:`V,` be it parametrized or not,
# acts on the observable via
#
# .. math::
#
#     H' = V^\dagger H V.
#
# The evolved observable can then be contracted with the initial state at the end
# of the simulation.
#
# Pauli propagation tracks the Pauli words :math:`P_\ell` in the Hamiltonian throughout
# this Heisenberg picture evolution, requiring us to only determine how a gate
# :math:`V` acts on any Pauli word. For Clifford gates, including the Hadamard
# gate, :math:`\operatorname{CNOT},` :math:`\operatorname{CZ}` and Pauli gates themselves, any Pauli word is mapped
# to another Pauli word. As a matter of fact, this is a standard way to *define*
# Clifford gates. As an example, consider a :math:`\operatorname{CNOT}` gate acting on the Pauli
# word :math:`Z\otimes Z` in the Heisenberg picture. We can evaluate (note that
# :math:`\operatorname{CNOT}^\dagger=\operatorname{CNOT}`)
#
# .. math::
#
#     \operatorname{CNOT} (Z\otimes Z) \operatorname{CNOT}
#     &=(P_0 \otimes \mathbb{I}+P_1\otimes X) (Z\otimes Z) (P_0 \otimes \mathbb{I}+P_1\otimes X)\\
#     &=(Z \otimes \mathbb{I}) (P_0 \otimes Z - P_1 \otimes Z)\\
#     &=\mathbb{I}\otimes Z.
#
# Here we abbreviated the projectors :math:`P_i=|i\rangle\langle i|` and used simple
# operator arithmetic. We can similarly look at the action of :math:`\operatorname{CNOT}`
# on any other two-qubit Pauli word. However, this might get tedious, so let us
# do it in code.

cnot = qml.CNOT([0, 1])

for op0, op1 in product([qml.Identity, qml.X, qml.Y, qml.Z], repeat=2):
    original_op = op0(0) @ op1(1)
    new_op = cnot @ original_op @ cnot
    new_op = qml.pauli_decompose(new_op.matrix())
    print(f"CNOT transformed {original_op} to {new_op}")

##############################################################################
# This fully specifies the action of :math:`\operatorname{CNOT}` on any Pauli word,
# because any tensor factors that the gate does not act on are simply left unchanged.
# We will use these results as a lookup table for the simulation below,
# so let's store them in a string-based dictionary:

cnot_table = {
    ("I", "I"): (("I", "I"), 1),
    ("I", "X"): (("I", "X"), 1),
    ("I", "Y"): (("Z", "Y"), 1),
    ("I", "Z"): (("Z", "Z"), 1),
    ("X", "I"): (("X", "X"), 1),
    ("X", "X"): (("X", "I"), 1),
    ("X", "Y"): (("Y", "Z"), 1),
    ("X", "Z"): (("Y", "Y"), -1),
    ("Y", "I"): (("Y", "X"), 1),
    ("Y", "X"): (("Y", "I"), 1),
    ("Y", "Y"): (("X", "Z"), -1),
    ("Y", "Z"): (("X", "Y"), 1),
    ("Z", "I"): (("Z", "I"), 1),
    ("Z", "X"): (("Z", "X"), 1),
    ("Z", "Y"): (("I", "Y"), 1),
    ("Z", "Z"): (("I", "Z"), 1),
}


##############################################################################
# Now, on to some non-Clifford gates, namely Pauli rotation gates. They
# have the important property of mapping a Pauli word to two Pauli words whenever
# the rotation generator and the transformed Pauli word do not commute.
#
# As an example, we can compute the action of :class:`~.pennylane.RZ` on the Pauli
# :class:`~.pennylane.X` operator:
#
# .. math::
#
#     R_Z^\dagger(\theta) X R_Z(\theta)
#     &= (\cos(\theta/2) \mathbb{I} + i\sin(\theta/2) Z) X (\cos(\theta/2) \mathbb{I} - i\sin(\theta/2) Z)\\
#     &= (\cos(\theta/2)^2 - \sin(\theta/2)^2) X + i\sin(\theta/2)\cos(\theta/2)[Z, X]\\
#     &= \cos(\theta) X - \sin(\theta) Y.
#
# Here we used trigonometric identities and :math:`[Z, X]=2iY` in the last step.
# Again, the tensor factors of the Pauli word on which the rotation does not act is
# left unchanged.
# And if the rotation generator commutes with the transformed Pauli word, the
# gate will of course leave the word unchanged.
#
# Great, now that we know how a :math:`\operatorname{CNOT}` gate and single-qubit
# Pauli rotations act on a Pauli word, we will be able to tackle our example
# circuit.
#
# But before we move on to implementing it, there is still a *crucial* point missing!
#
# Truncating the Pauli propagation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Pauli propagation is a neat way to approach the task of estimating expectation
# values that we are after, like :math:`E(\theta)`\ . However, if we were to run
# a classical simulation using this approach as is, we would quickly run into
# a scaling problem. There are :math:`4^N` Pauli words on :math:`N` qubits, an
# unfeasibly large number if we hit all of them during our simulation.
#
# Both existing algorithms and the new work [#angrisani]_ therefore use
# truncation methods to keep the number of Pauli words that need to be tracked
# below a reasonable threshold.
# The algorithm we discuss here does this based on the weight of the tracked Pauli
# words. For a chosen threshold :math:`k,` it simply discards all Pauli words with
# non-trivial tensor factors on more than :math:`k` qubits.
#
# This is clearly an approximation, and in principle we could introduce a large
# error in this truncation. However, Angrisani et al. show that this is not the case
# for a wide class of parametrized circuits at most parameter settings.
#
# As anticipated above, the truncation step requires the Pauli words of the initial :math:`H`
# to be at most :math:`k`-local, as they get truncated away otherwise. Alternatively,
# non-local terms need to make a negligible contribution to the expectation value
# in order for the truncation to be a good approximation.
#
# So let's move on to implementing truncated Pauli propagation technique!
# We will make use of :class:`~.pennylane.pauli.PauliWord` and
# :class:`~.pennylane.pauli.PauliSentence` objects that allow us to handle the
# observable easily. Let's start with two functions that implement a
# single-qubit rotation and a :math:`\operatorname{CNOT}` gate in the Heisenberg
# picture, respectively. Note that single-qubit rotation gates do not
# require us to implement truncation, because they never increase the weight of
# a Pauli word.
#

from pennylane.pauli import PauliWord, PauliSentence


def apply_cnot(wires, H, k=None):
    """Apply a CNOT gate on given wires to operator H in the Heisenberg picture.
    Truncate all Pauli words in the transformed operator that have weight larger than k."""
    new_H = PauliSentence()
    for pauli_word, coeff in H.items():
        # Extract the Pauli tensor factors on the wires of the CNOT
        op_pw_0 = pauli_word.get(wires[0], "I")
        op_pw_1 = pauli_word.get(wires[1], "I")
        # Look up the prefactor and new Pauli tensor factors in our lookup table
        (new_op_pw_0, new_op_pw_1), factor = cnot_table[(op_pw_0, op_pw_1)]
        # Create new Pauli word from old one and update it with new tensor factors
        new_pw = pauli_word.copy()
        new_pw.update({wires[0]: new_op_pw_0, wires[1]: new_op_pw_1})
        new_pw = PauliWord(new_pw)

        # Truncation: Only add to the new H if the new Pauli word is small enough
        if (k is None) or len(new_pw) <= k:
            new_H[new_pw] += factor * coeff

    return new_H


def apply_single_qubit_rot(pauli, wire, param, H):
    """Apply a single-qubit rotation about the given ``pauli`` on the given ``wire``
    by a rotation angle ``param`` to an operator ``H``."""
    new_H = PauliSentence()
    rot_pauli_word = PauliWord({wire: pauli})
    for pauli_word, coeff in H.items():
        if pauli_word.commutes_with(rot_pauli_word):
            # Rotation generator commutes with Pauli word from H, the word is unchanged
            new_H[pauli_word] += coeff
        else:
            # Rotation generator does not commute with Pauli word from H;
            # Multiply old coefficient by cosine, and add new term with modified Pauli word
            new_H[pauli_word] += qml.math.cos(param) * coeff
            new_pauli_word, factor = list((rot_pauli_word @ pauli_word).items())[0]
            new_H[new_pauli_word] += (qml.math.sin(param) * coeff * factor * 1j).real

    return new_H


##############################################################################
# Completing the simulation with the initial state
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# With those two essential functions implemented, we're almost ready to put the algorithm together.
# Before doing so, we need a function that computes the expectation value of the evolved observable
# with respect to the initial state :math:`|0\rangle.` This is simple, though, because we know for
# each Pauli word :math:`P_\ell` in the Hamiltonian that it will contribute its coefficient :math:`h_\ell`
# to the expectation value if (and only if) all tensor factors are :math:`I` or :math:`Z.`


def initial_state_expval(H):
    """Compute the expectation value of an operator ``H`` in the state |0>."""
    expval = 0.0
    for pauli_word, coeff in H.items():
        if all(pauli in {"I", "Z"} for pauli in pauli_word.values()):
            expval += coeff
    return expval


##############################################################################
# Putting the pieces together
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Now let's combine everything into a function that can handle the tape of our quantum
# circuit from the beginning. It simply extracts the measurement observable and then
# goes through the circuit backwards, propagating the Pauli words of the observable
# according to the Heisenberg picture. Finally, it evaluates the expectation value
# with respect to :math:`|0\rangle.` The truncation threshold :math:`k` for ``apply_cnot``
# is a hyperparameter of the execution function.


def execute_tape(tape, k=None):
    """Classically simulate a tape and estimate the expectation value
    of its output observable using truncated Pauli propagation."""
    H = tape.measurements[0].obs.pauli_rep
    for op in reversed(tape.operations):
        if isinstance(op, qml.CNOT):
            # Apply CNOT
            H = apply_cnot(op.wires, H, k=k)
        elif isinstance(op, (qml.RZ, qml.RX, qml.RY)):
            # Extract the Pauli rotation generator, wire, and parameter from the gate
            pauli = op.name[-1]
            wire = op.wires[0]
            param = op.data[0]
            H = apply_single_qubit_rot(pauli, wire, param, H)
        else:
            raise NotImplementedError

    return initial_state_expval(H)


##############################################################################
# Great! So let's run it on our circuit, but now on 25 qubits and with 5 layers,
# and compare the result to the exact value from PennyLane's `fast statevector simulator,
# Lightning Qubit <https://docs.pennylane.ai/projects/lightning/en/stable/lightning_qubit/device.html>`__.
# We also set the truncation threshold to :math:`k=8`\ .

num_qubits = 25
num_layers = 5
k = 7
H_coeffs = np.random.random((num_qubits - 1) * 3)
H_ops = [op(j) @ op(j + 1) for j in range(num_qubits - 1) for op in [qml.X, qml.Y, qml.Z]]
H = qml.dot(H_coeffs, H_ops)
params = np.random.random((num_layers, num_qubits, 3))


def run_estimate(params, H):
    tape = ansatz(params, num_qubits, H)
    expval = execute_tape(tape, k=k)
    return expval


expval = run_estimate(params, H)


@qml.qnode(qml.device("lightning.qubit", wires=num_qubits))
def run_lightning(params, H):
    return _ansatz(params, num_qubits, H)


exact_expval = run_lightning(params, H)

print(f"Expectation value estimated by truncated Pauli propagation: {expval:.6f}")
print(f"Numerically exact expectation value:                        {exact_expval:.6f}")


##############################################################################
# Wonderful, we have a working approximate simulation of the circuit that is scalable (for
# fixed :math:`k`) that estimates the expectation value of :math:`H!`
# Note that a single estimate neither is a proof that the algorithm works in general,
# nor is it the subject of the main results by Angrisani et al.
# However, a full-fledged benchmark goes beyond this demo.
#
# Fine print: Which circuits can be simulated?
# --------------------------------------------
#
# As anticipated multiple times throughout the demo, we now want to consider some details on
# the circuits for which the truncated Pauli propagation is guaranteed to work.
#
# It's a statistical thing
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# First, it is very important to note that the guarantee of good approximations does not
# apply to any single instance of a quantum circuit, but to the full parametrized *family*
# defined by a circuit ansatz, together with a probability distribution to pick the parameters.
# The guarantee then is that the approximation error of truncated Pauli propagation
# *on average across the sampled parameter settings* can be suppressed exponentially by
# increasing the truncation threshold :math:`k.`
#
# This can be rephrased as follows: the probability of obtaining an error larger than some
# tolerance can be suppressed exponentially by increasing :math:`k`\ .
# Note that this does not prevent the simulation
# algorithm to be *very* wrong at some rare parameter settings!
#
# The circuit needs to be locally scrambling
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Second, there is a requirement for the structure and gates of the parametrized circuit:
# We need to be able to divide the circuit into layers such that each layer, together with
# the distribution of parameters we consider for the parametrized family, does not change under
# random single-qubit rotations. The circuit and its parameter distribution are said to be
# *locally scrambling* in this case.
#
# This may sound complicated, but is easy to understand for a
# small example: Consider an arbitrary rotation :class:`~.pennylane.Rot` on a single qubit,
# together with a distribution for its three angles that leads to Haar random rotations
# (see the :doc:`PennyLane Demo on the Haar measure <demos/tutorial_haar_measure>` for details).
# This parametrized rotation then is unchanged if we apply another Haar random rotation!
# That is, even though an individual rotation does get modified, the *distribution* of rotations
# remains the same (an important property of the Haar measure!).
#
# For our purposes it is sufficient
# to note that the hardware-efficient layers from above do indeed satisfy this requirement.
# Please take a look at the original paper for further details [#angrisani]_, in particular
# Sections II and VIII and Appendix A.
#
# Scrambling layers need to be shallow
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Third, the parametrized circuit may not "branch too much." That is, if a Pauli word has
# weight :math:`r,` none of the locally scrambling layers of the circuit
# may produce more than :math:`n^r` different Pauli words under the Heisenberg evolution.
# Intuitively, this requirement limits the layers of the circuit to be sufficiently shallow.
#
# For our hardware-efficient circuit, we can bound this amount of branching directly:
# Each :math:`\operatorname{CNOT}` in the entangling layer can at most double the weight of the Pauli word, e.g.,
# if each :math:`\operatorname{CNOT}` gate hits a :math:`Z` on its target qubit.
# Afterwards, each single-qubit rotation can create all three Pauli operators on each
# qubit in the support of the enlarged Pauli word, leading to a factor of three.
# Taken together, a Pauli word with weight :math:`r` is transformed into at most
# :math:`3^{2r}=9^r` Pauli words with weights at most :math:`2r.` The requirement
# of not "branching too much" therefore is satisfied, because :math:`9^r<n^r`
# from the complexity theoretic perspective.
#
# Counterexample
# --------------
#
# As emphasized in the fine print above, we should not expect the classical algorithm
# to produce precise expectation value estimates for all parameter settings.
# So let's evaluate our circuit for a specific parameter setting with all parameters set to
# :math:`\frac{\pi}{4}.` In addition, we will even *reduce* the number of qubits and layers,
# which makes the task easier for other classical simulation tools (such as
# `Lightning Qubit <https://docs.pennylane.ai/projects/lightning/en/stable/lightning_qubit/device.html>`__),
# but does not help the truncated Pauli propagation.

num_qubits = 15
num_layers = 3
H_coeffs = np.random.random((num_qubits - 1) * 3)
H_ops = [op(j) @ op(j + 1) for j in range(num_qubits - 1) for op in [qml.X, qml.Y, qml.Z]]
H = qml.dot(H_coeffs, H_ops)
specific_params = np.ones((num_layers, num_qubits, 3)) * np.pi / 4

expval = run_estimate(specific_params, H)
exact_expval = run_lightning(specific_params, H)

print(f"Expectation value estimated by truncated Pauli propagation: {expval:.6f}")
print(f"Numerically exact expectation value:                        {exact_expval:.6f}")

##############################################################################
# As we can see, the estimation error became quite large, although we reduced the
# qubit and layer count while keeping :math:`k` fixed.
# This is a manifestation of the statement that the algorithm only will estimate expectation
# values successfully *on average* over the parameter domain.
#
# Conclusion
# ----------
# We learned about Pauli propagation, how to truncate it, and how it results in an efficient
# classical algorithm that estimates expectation values of parametrized quantum circuits.
# This result is important, as it casts doubt on the usefulness of generic parametrized quantum
# circuits in quantum computations. Instead of such generic circuits, we will need to make
# use of specialized circuit architectures and smart parametrization and initialization
# techniques if we want to employ parametrized circuits in a useful manner.
#
# Besides the caveat that the approximation is only guaranteed across the full distribution
# of parameters, it is important to note that expectation value estimation is not the only
# interesting task on a quantum computer.
# Similar to the result discussed here, there already exist other classes of quantum circuits
# for which this estimation task is easy, but sampling from the quantum state prepared by
# the circuit is hard.
# One prominent example are so-called instantaneous quantum polynomial-time (IQP) circuits
# [#bremner]_, [#bremner2]_, which arise in the context of :doc:`Boson sampling <demos/gbs>`.
#
# Finally, it is important to note that while truncated Pauli propagation scales
# polynomially with the qubit count, the exponent of this scaling contains :math:`k,`
# which still can lead to impractical computational cost, e.g., for deep circuits.
#
# References
# ----------
#
# .. [#gottesman]
#
#     Daniel Gottesman
#     "The Heisenberg Representation of Quantum Computers"
#     `arXiv:quant-ph/9807006 <https://arxiv.org/abs/quant-ph/9807006>`__, 1998.
#
# .. [#angrisani]
#
#     Armando Angrisani, Alexander Schmidhuber, Manuel S. Rudolph, M. Cerezo, Zoë Holmes, Hsin-Yuan Huang
#     "Classically estimating observables of noiseless quantum circuits"
#     `arXiv:2409.01706 <https://arxiv.org/abs/2409.01706>`__, 2024.
#
# .. [#aharonov]
#
#     Dorit Aharonov, Xun Gao, Zeph Landau, Yunchao Liu, Umesh Vazirani
#     "A polynomial-time classical algorithm for noisy random circuit sampling"
#     `arXiv:2211.03999 <https://arxiv.org/abs/2211.03999>`__, 2022.
#
# .. [#lowesa]
#
#     Manuel S. Rudolph, Enrico Fontana, Zoë Holmes, Lukasz Cincio
#     "Classical surrogate simulation of quantum systems with LOWESA"
#     `arXiv:2308.09109 <https://arxiv.org/abs/2308.09109>`__, 2023.
#
# .. [#begusic]
#
#     Tomislav Begušić, Johnnie Gray, Garnet Kin-Lic Chan
#     "Fast and converged classical simulations of evidence for the utility of quantum computing before fault tolerance"
#     `arXiv:2308.05077 <https://arxiv.org/abs/2308.05077>`__, 2023.
#
# .. [#bremner]
#
#     M. Bremner, R. Jozsa, D. Shepherd.
#     "Classical simulation of commuting quantum computations implies collapse of the polynomial hierarchy."
#     `arXiv:1005.1407 <https://arxiv.org/abs/1005.1407>`__, 2010.
#
# .. [#bremner2]
#
#     Michael J. Bremner, Ashley Montanaro, Dan J. Shepherd
#     "Achieving quantum supremacy with sparse and noisy commuting quantum computations."
#     `arXiv:1610.01808 <https://arxiv.org/abs/1610.01808>`__, 2016.
#