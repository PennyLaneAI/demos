r"""
Stabilizer codes for quantum error correction
=================================================

State-of-the-art quantum devices, such as IBM's Condor and Atom Computing's optical lattices, contain more than
a thousand qubits. Does this qubit count suffice for valuable quantum algorithms with clear speedups?
The reality is that there is more to the story than the sheer number of qubits. As we currently stand, quantum
devices are still prone to errors that increase with device size. For this reason, **quantum error correction**-one of the most important domains in the universe of quantum computing-has
been gaining traction.

Quantum error correction is implemented through **error correction codes,** which come in many varieties that address different error types.
Is there a unified way to understand all these codes? The **stabilizer formalism** provides such a framework for
a large class of quantum error correction codes [#Gottesman1997]_. The so-called
**stabilizer codes**, such as the repetition, Shor, Steane, and surface codes,
all fall under this formalism.

In this demo, we will introduce the stabilizer formalism using a bottom-up approach. We build
some well-known codes using **stabilizer generators,** from which
the other elements of the code (codewords, syndrome measurements, etc.) can be reconstructed. Then, we
represent these codes as quantum circuits and implement them in PennyLane.

.. figure:: ../_static/demo_thumbnails/opengraph_demo_thumbnails/pennylane-demo-stabilizer-codes-open-graph.png
    :align: center
    :width: 50%


A toy example: the repetition code
-----------------------------------

To start with, let us explore the general structure of error correction codes using a simple example: the **three-qubit repetition code.**
We will introduce this code as a quantum circuit with definite steps to gain some intuition on how it corrects errors on qubits.
We represent the states of the qubits in the circuit using **state vectors**. In this formalism, error correction codes follow a simple structure:

- Qubit encoding
- Error detection
- Error correction

Let us describe each element in detail.

Qubit encoding
~~~~~~~~~~~~~~~

The first step in an error correction code is **encoding** one abstract or **logical qubit** into a set of many on-device **physical qubits.**
The rationale is that, if some external factor changes the state of one of the qubits, the remaining qubits still provide information about the original logical qubit.
For example, in the three-qubit repetition code, the logical basis-state
qubits, or **logical codewords**, :math:`\vert \bar{0}\rangle` ("logical 0") and :math:`\vert \bar{1}\rangle` ("logical 1") are encoded into three physical qubits via

.. math::

    \vert \bar{0} \rangle \mapsto \vert 000 \rangle, \quad \vert \bar{1} \rangle \mapsto \vert 111 \rangle.

A general qubit :math:`\vert \psi\rangle = \alpha \vert 0\rangle + \beta \vert 1\rangle` is then encoded as

.. math::

    \alpha \vert 000 \rangle + \beta \vert 111\rangle \mapsto  \alpha \vert \bar{0}\rangle + \beta \vert \bar{1}\rangle = \vert \bar{\psi} \rangle.

This encoding can be done via the following quantum circuit.

.. figure:: ../_static/demonstration_assets/stabilizer_codes/three_qubit_encode.png
    :align: center
    :width: 70%

    ..

Let's code this below and verify the output

"""

import pennylane as qml
qml.capture.enable()
from pennylane import numpy as np


def encode(alpha, beta):
    qml.StatePrep([alpha, beta], wires=0)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[0, 2])


def encoded_state(alpha, beta):
    encode(alpha, beta)
    return qml.state()


encode_qnode = qml.QNode(encoded_state, qml.device("default.qubit"))

alpha = 1 / np.sqrt(2)
beta = 1 / np.sqrt(2)

encode_qnode = qml.QNode(encoded_state, qml.device("default.qubit"))

print("|000> component: ", encode_qnode(alpha, beta)[0])
print("|111> component: ", encode_qnode(alpha, beta)[7])

##############################################################################
#
# .. note::
#     Why do we encode qubits in this way, instead of preparing many copies of the state? If the quantum state is known, we could do this,
#     but this only increases the amount of quantum resources we need! If the quantum state is not known, the no-cloning
#     theorem states it is impossible to make a copy of the state.
#
# Error detection
# ~~~~~~~~~~~~~~~~
#
# Now, suppose that a **bit-flip** error occurs on the second qubit, meaning that the qubit is randomly flipped. This can be modelled as
# an unwanted Pauli-$X$ operator being applied on :math:`\vert \bar{\psi}\rangle:`
#
# .. math::
#
#     X_2 \vert \bar{\psi}\rangle = \alpha \vert 010 \rangle + \beta \vert 101 \rangle
#
# How do we detect this error? As we already know, measuring the state collapses it, so we cannot measure the state to detect the error.
#
# To detect a bit-flip error on one of the physical qubits without disturbing the encoded logical state, we perform a **parity measurement.**
# This checks whether all physical qubits are in the same state by comparing them two at a time, without directly measuring them.
# Instead, auxiliary qubits are used and measured. For the three-qubit repetition code, this involves measuring two auxiliary qubits
# in the computational basis after applying a series of :math:`\textrm{CNOT}` gates, as illustrated in the circuit below.
# The auxiliary qubits record whether each pair of data qubits are the same or different: if they are the same, the auxiliary stays at 0,
# and if they differ, the auxiliary flips to 1. Thus, if all physical qubits are identical, the auxiliary qubits remain at 0. This is the parity measurement.
#
# .. figure:: ../_static/demonstration_assets/stabilizer_codes/parity_measurements.png
#    :align: center
#    :width: 100%
#
#    ..
# The result of these parity measurements is known as the **syndrome**. Since we use two auxiliary qubits, there are four possible measurement outcomes (00, 01, 10, 11).
# Each outcome tells us whether a bit-flip error occurred in the encoded state :math:`\vert \bar{\psi} \rangle` and, if so, which qubit was flipped.
# The following table shows how to interpret the error and the syndromes.
#
# .. figure:: ../_static/demonstration_assets/stabilizer_codes/syndrome_table3.png
#    :align: center
#    :width: 100%
#
#    ..
#
# When there is no error, the syndrome measurement will yield 0 on both auxiliary qubits.
# Let us verify the full table by implementing the syndrome measurement in PennyLane.


def error_detection():
    qml.CNOT(wires=[0, 3])
    qml.CNOT(wires=[1, 3])
    qml.CNOT(wires=[1, 4])
    qml.CNOT(wires=[2, 4])


@qml.set_shots(1)
@qml.qnode(qml.device("default.qubit", wires=5))  # A single sample flags error
def syndrome_measurement(error_wire):
    encode(alpha, beta)

    qml.PauliX(wires=error_wire)  # Unwanted Pauli Operator

    error_detection()

    return qml.sample(wires=[3, 4])


print("Syndrome if error on wire 0: ", syndrome_measurement(0))
print("Syndrome if error on wire 1: ", syndrome_measurement(1))
print("Syndrome if error on wire 2: ", syndrome_measurement(2))

##############################################################################
#
# The measurement outputs confirm the syndrome table.
#
# .. note::
#     How many auxiliary qubits are needed for a repetition code of n qubits? Try it out for yourself!
#
# Error Correction
# ~~~~~~~~~~~~~~~~~
#
# Once a single bit-flip error is detected, correction is straightforward. Since the Pauli-X operator is its own inverse
# (i.e., :math:`X^2 = \mathbb{I}`), re-applying the :math:`X` operator to the erroneous qubit restores the original state. For example,
# if the syndrome measurement shows the error occurred on the second qubit (qubits are labelled 0, 1, and 2), we apply
#
# .. math::
#
#     X_1 (X_1 \vert \bar{\psi}\rangle) = \vert \bar{\psi} \rangle.
#
# By applying the appropriate corrective operation, the repetition code effectively protects and repairs the encoded quantum information.
# The full workflow is shown in the circuit below.
#
# .. figure:: ../_static/demonstration_assets/stabilizer_codes/3_qubit_code.png
#    :align: center
#    :width: 100%
#
#    ..
#
# We can use PennyLane's mid-circuit measurement features to implement the full three-qubit repetition code.


@qml.qnode(qml.device("default.qubit", wires=5))
def error_correction(error_wire):
    encode(alpha, beta)

    qml.PauliX(wires=error_wire)

    error_detection()

    # Mid circuit measurements

    m3 = qml.measure(3)
    m4 = qml.measure(4)

    # Operations conditional on measurements

    qml.cond(m3 & ~m4, qml.PauliX)(wires=0)
    qml.cond(m3 & m4, qml.PauliX)(wires=1)
    qml.cond(~m3 & m4, qml.PauliX)(wires=2)

    return qml.density_matrix(
        wires=[0, 1, 2]
    )  # qml.state not supported, but density matrices are.


##############################################################################
#
# At the time of writing, PennyLane circuits with mid-circuit measurements cannot return a quantum state vector, so we return the density matrix instead.
# With this result, we can verify that the fidelity of the encoded state is the same as the final state after correction
# as follows.

dev = qml.device("default.qubit", wires=5)
error_correction_qnode = qml.QNode(error_correction, dev)
encoded_state = qml.math.dm_from_state_vector(encode_qnode(alpha, beta))

# Compute fidelity of final corrected state with initial encoded state

print(
    "Fidelity when error on wire 0: ",
    qml.math.fidelity(encoded_state, error_correction_qnode(0)).round(2),
)
print(
    "Fidelity when error on wire 1: ",
    qml.math.fidelity(encoded_state, error_correction_qnode(1)).round(2),
)
print(
    "Fidelity when error on wire 2: ",
    qml.math.fidelity(encoded_state, error_correction_qnode(2)).round(2),
)

##############################################################################
#
# The error is corrected no matter which qubit was flipped!
#
# Revisiting the three-qubit repetition code with Pauli operators
# ---------------------------------------------------------------
#
# We have worked with a simple example (repetition code), but it is quite limited. Indeed, the three-qubit code only works for
# a single bit flip error. There are more powerful codes, but they also need more resources. For example, Shor's code
# can correct any error on a single logical qubit, but it needs 9 qubits. To make matters worse, to avoid errors at an acceptable level,
# one might need as much as 1000 physical qubits per logical qubit. Even with a few qubits, the encoded states and protocols can become increasingly
# complex. Writing a 1000 qubit state vector seems quite daunting! To deal with these situations, we resort to a different
# representation of error correction codes, using **Pauli operators** instead of state vectors.
#
# .. note::
#     It is a great time to look at `Pauli operators <https://pennylane.ai/codebook/single-qubit-gates>`_ and their properties now,
#     if you are not familiar with them.
#
# To gain some intuition about the operator picture, let us express the three-qubit repetition code in a different way. Using the
# identity below,
#
# .. figure:: ../_static/demonstration_assets/stabilizer_codes/cnot_identity.png
#    :align: center
#    :width: 100%
#
#    ..
#
# the three-qubit repetition code can be expressed in the following way.
#
# .. figure:: ../_static/demonstration_assets/stabilizer_codes/3_qubit_stabilizer_circ.png
#    :align: center
#    :width: 100%
#
#    ..
#
# This is the same circuit, but the controls are all now in the auxiliary qubits, while the logical qubits act as target qubits.
# This does not seem desirable--we do not want to change the state of the logical qubits! However, let us observe that the operators
# that act on the logical qubits are :math:`Z_0 Z_1 I_2` and :math:`I_0 Z_1 Z_2,` which leave the logical codewords invariant.
#
# .. math::
#
#     Z_0 Z_1 I_2 \vert 000 \rangle = \vert 000 \rangle, \quad Z_0 Z_1 I_2 \vert 111 \rangle = \vert 111 \rangle.
#
# .. math::
#
#     I_0 Z_1 Z_2 \vert 000 \rangle = \vert 000 \rangle, \quad I_0 Z_1 Z_2 \vert 111 \rangle = \vert 111 \rangle.
#
# This is great news as the operators :math:`Z_0 Z_1 I_2` and :math:`I_0 Z_1 Z_2` leave the logical states unchanged.
# Any state of the form :math:`\alpha \lvert 000 \rangle + \beta \lvert 111 \rangle` is a +1 eigenstate of both, which means
# the entire logical code space is invariant under their action.
# A single bit-flip error breaks this invariance: at least one operator will then return eigenvalue -1.
# Measuring the two operators therefore reveals whether an error occurred and, as shown in the syndrome table below,
# identifies the qubit on which it happened. In this sense, detecting and correcting errors requires only two Pauli operators
# and the outcomes of their eigenvalue measurements.
#
# .. figure:: ../_static/demonstration_assets/stabilizer_codes/table_eigenvalues.png
#    :align: center
#    :width: 100%
#
#    ..
#
# This table is related to the previous syndrome table. If we know that the initial state was :math:`\lvert 000\rangle` and assume that only one flip occurred (states with two zeros and a one),
# then the pairs of eigenvalues uniquely determine the erroneous state. Therefore, we can determine which qubit was flipped.
#
# This gives us a new option for characterizing error correction codes. What if instead of building codewords and trying to find
# the syndrome measurement operators from them, we went the other way round? Namely, we could start by specifying a set of operators, find the states
# that remain invariant under their action, and make these our codewords. The operators in this initial set are known as **stabilizer generators.**
#
# The stabilizer formalism
# -------------------------
#
# Stabilizer generators
# ~~~~~~~~~~~~~~~~~~~~~~
#
# .. admonition:: Groups
#     :class: note
#
#     It is important to have some familiarity with group theory to understand the stabilizer formalism. As a refresher, here is
#     the definition of a group.
#
#     A group is a set of elements that has:
#       1. an operation that maps two elements a and b of the set into a third element of the set, for example c = a + b,
#       2. an "identity element" e such that e + a = a for any element a, and
#       3. an inverse -a for every element a, such that a + (-a) = e.
#
# .. admonition:: Notation
#     :class: note
#
#     In the stabilizer formalism, we often omit explicit tensor product symbols (:math:`\otimes`) for brevity.
#     For example, :math:`X_0 Z_1` denotes :math:`X \otimes Z` acting on qubits 0 and 1, respectively.
#     When identity operators are omitted, we use subscripts to indicate which qubits the non-identity Pauli operators act on.
#     If all positions are filled (e.g., :math:`XZI`), the position implicitly indicates the qubit index (qubit 0, 1, 2).
#
# The stabilizer formalism is a powerful framework for constructing quantum error-correcting codes using the algebraic structure of *Pauli operators*.
# It focuses on subgroups of the *Pauli group* on :math:`n` qubits—denoted :math:`\mathcal{P}_n`—which consists of all tensor products of single-qubit Pauli operators :math:`\{I, X, Y, Z\}` (with overall phases :math:`\pm1, \pm i`).
# A **stabilizer group** :math:`S` is defined as a subgroup of :math:`\mathcal{P}_n` that satisfies the following:
#
# 1. It contains the identity operator :math:`I_0 \otimes I_1 \otimes \cdots \otimes I_{n-1}`.
# 2. All elements of :math:`S` commute with each other.
# 3. The product of any two elements in :math:`S` is also in :math:`S` (i.e., it forms a group under matrix multiplication).
# 4. It does not contain the negative identity operator :math:`-I^{\otimes n}`.
#
# Rather than listing all the elements of a stabilizer group explicitly, the group can be more succinctly specified via a
# set of **generators**: a minimal set of operators in the stabilizer group that can produce all
# the other elements through finite products of generators. As a simple example, consider the stabilizer set
#
# .. math::
#
#     S = \left\lbrace I_0  I_1  I_2, \ Z_0 Z_1  I_2, \ Z_0  I_1  Z_2, \ I_0  Z_1  Z_2 \right\rbrace.
#
# We can check that it satisfies the defining properties 1 to 4. The most cumbersome to check is property 3, where we have to take all
# possible products of the elements and check whether the result is in :math:`S.` For example,
#
# .. math::
#
#    (Z_0  Z_1  I_2)\cdot (I_0  Z_1  Z_2) = Z_0  I_1  Z_2, \\
#    (Z_0 Z_1 I_2 )^2 = I_0  I_1  I_2,
#
# and so on. Note that we can obtain all the elements in :math:`S` just from :math:`Z_0  Z_1  I_2` and :math:`I_0  Z_1  Z_2.` Because
# of this property, these elements are **stabilizer generators** for :math:`S`. We write this fact as
#
# .. math::
#
#     S = \left\langle Z_0  Z_1  I_2, \ I_0  Z_1  Z_2 \right\rangle,
#
# which reads ":math:`S` *is the stabilizer group generated by the elements* :math:`Z_0  Z_1  I_2` *and* :math:`I_0  Z_1  Z_2.`"
#
# It turns out that specifying these generators is sufficient to completely define the stabilizer group, and
# thereby the corresponding quantum error-correcting code.
#
# Now that we know how stabilizer generators work, let us create a tool for later use that creates the full stabilizer group from its generators.

import itertools
from pennylane import X, Y, Z
from pennylane import Identity as I


def generate_stabilizer_group(gens, num_wires):
    group = []
    init_op = I(0)
    for i in range(1, num_wires):
        init_op = init_op @ I(i)
    for bits in itertools.product([0, 1], repeat=len(gens)):
        op = init_op
        for i, bit in enumerate(bits):
            if bit:
                op = qml.prod(op, gens[i]).simplify()
        group.append(op)
    return set(group)


generators = [Z(0) @ Z(1) @ I(2), I(0) @ Z(1) @ Z(2)]
generate_stabilizer_group(generators, 3)


##############################################################################
#
# Indeed, we obtain all the elements of the group by inputting the generators only. Feel free to try out this code with different generators!
#
# Defining the codespace
# ~~~~~~~~~~~~~~~~~~~~~~~
#
# At this point, the pressing question is how to define the error correction code given a set of stabilizer generators. As far as
# we know, stabilizer generators are only a bunch of Pauli strings satisfying some properties. Let us recall that the property that inspired using stabilizer generators
# is that they must leave the codewords invariant. For any stabilizer element :math:`S` and codeword :math:`\vert \psi \rangle`, we must
# have
#
# .. math::
#
#     S\vert \psi \rangle = \vert \psi \rangle.
#
# The **codespace** is defined **uniquely** as the set made up of all states such that :math:`S_i \vert\psi\rangle = \vert \psi\rangle` for all stabilizer
# group elements :math:`S_i.` The **codewords** can then be recovered by choosing an orthogonal basis of the codespace.
# For example, for the three-qubit repetition code, the codewords  (:math:`\vert 000 \rangle` and :math:`\vert 111 \rangle`)
# can be recovered from the stabilizer generators :math:`Z_0  Z_1  I_2` and :math:`I_0  Z_1  Z_2` from table above.
#
# There is a **one-to-one correspondence** between stabilizer groups and the quantum error-correcting codes they define.
# This means we can describe a code entirely by its stabilizer group, using operators rather than listing the codewords
# directly as state vectors.
#
# Logical operators
# ~~~~~~~~~~~~~~~~~~
#
# So far, we have introduced the stabilizer generators, which define the syndrome measurements, and the codewords,
# which are the states left unchanged by all stabilizers. What remains is to understand how to perform computation on the
# encoded qubits, specifically, how to apply gates that act on logical qubits without leaving the codespace.
# These operators must act non-trivially on the codewords, so they cannot be part of the stabilizer group.
# However, to preserve the codespace, they **must commute** with all stabilizer generators.  If a logical operator does not commute with a stabilizer,
# it can map valid code states outside the codespace or change the syndrome, thus corrupting the encoded information.
# In particular, we are interested in the logical Pauli operators :math:`\bar{X}` and :math:`\bar{Z}`, defined by:
#
# .. math::
#
#     \bar{X}\vert \bar{0} \rangle = \vert \bar{1} \rangle, \quad \bar{X}\vert \bar{1} \rangle = \vert \bar{0} \rangle, \\
#     \ \bar{Z}\vert \bar{0} \rangle = \vert \bar{0} \rangle, \quad \bar{Z}\vert \bar{1} \rangle = - \vert \bar{1} \rangle.
#
# For example, in the three-qubit bit-flip error correcting code, the logical operators are :math:`\bar{X} = X_0 X_1 X_2` and
# :math:`\bar{Z} = Z_0 Z_1 Z_2,` but they will not always be this simple.  In general, given a stabilizer set $S$, the logical
# operators for the code satisfy the following properties:
#
# 1. They commute with all elements in :math:`S,` so they leave the codespace invariant,
# 2. They are not in the stabilizer group,
# 3. Logical operators corresponding to the same logical qubit (e.g., :math:`\bar{X}_1` and :math:`\bar{Z}_1`) *anticommute, meaning they act non-trivially on the codewords.
#
# LSD Theorem
# ~~~~~~~~~~~
# Remember that the stabilizer group is a subgroup of the Pauli group with some properties. In the stabilizer formalism,
# every Pauli operator acting on the qubits can be categorized based on how it interacts with the stabilizer group.
# The **LSD theorem** states that Pauli operators on encoded qubits can be divided into three types:
#
#
# * **L**: Logical operators – commute with all stabilizers but act non-trivially on the codewords.
# * **S**: Stabilizers – leave all codewords unchanged.
# * **D**: Destabilizers (errors) – do not commute with at least one stabilizer and take the state out of the codespace.
#
# This classification helps distinguish between correctable errors, harmless stabilizer actions, and useful logical operations.
#
# So let us now write some code to classify Pauli operators based on the LSD theorem for a given a set of stabilizer generators.


def classify_pauli(operator, logical_ops, generators, n_wires):
    allowed_wires = set(range(n_wires))
    operator_wires = set(operator.wires)

    assert operator_wires.issubset(allowed_wires), (
        "Operator has wires not allowed by the code"
    )

    operator_names = set([op.name for op in operator.decomposition()])
    allowed_operators = set(["Identity", "PauliX", "PauliY", "PauliZ", "SProd"])

    assert operator_names.issubset(allowed_operators), (
        "Operator contains an illegal operation"
    )

    stabilizer_group = generate_stabilizer_group(generators, n_wires)

    if operator.simplify() in stabilizer_group:
        return f"{operator} is a Stabilizer."

    if all(qml.is_commuting(operator, g) for g in generators):
        if operator in logical_ops:
            return f"{operator} is a Logical Operator."
        else:
            return f"{operator} commutes with all stabilizers — it's a Logical Operator (or a multiple of one)."

    return f"{operator} is an Error Operator (Destabilizer)."


generators = [Z(0) @ Z(1) @ I(2), I(0) @ Z(1) @ Z(2)]
logical_ops = [X(0) @ X(1) @ X(2), Z(0) @ Z(1) @ Z(2)]
print(classify_pauli(Z(0) @ I(1) @ Z(2), logical_ops, generators, 3))
print(classify_pauli(Y(0) @ Y(1) @ Y(2), logical_ops, generators, 3))
print(classify_pauli(X(0) @ Y(1) @ Z(2), logical_ops, generators, 3))

##############################################################################
#
# .. note::
#     In the literature, you may have come across an error correction code being called an ":math:`[n,k]`-stabilizer code." In this notation, the number :math:`n` represents
#     the number of physical qubits used to encode the logical qubit. The integer :math:`k` is the number of logical qubits and it is
#     equal to :math:`1` for all the codes in this demo. It is possible to show that the number minimal of stabilizer generators :math:`m`
#     is related to :math:`n` and :math:`k` via :math:`m = n - k.`
#
# Example: Five-qubit stabilizer code
# -----------------------------------
#
# The 5-qubit code, also called as Laflamme's code [#Laflamme1996]_, holds a special place as the smallest
# error correcting code capable of correcting arbitrary Pauli Errors---unwanted applications of :math:`X,` :math:`Y,` or :math:`Z`
# gates on a single qubit. In this section, we will build and implement the complete error correction procedure
# starting from its stabilizer generators:
#
# .. math::
#
#     S = \langle S_0, \ S_1, \ S_2, \ S_3 \rangle,
#
# with
#
# .. math::
#
#    S_0 = X_0 Z_1 Z_2 X_3 I_4,\\
#    S_1 = I_0 X_1 Z_2 Z_3 X_4,\\
#    S_2 = X_0 I_1 X_2 Z_3 Z_4,\\
#    S_3 = Z_0 X_1 I_2 X_3 Z_4.
#
# Encoding the logical qubit
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# First, we need to prepare a data qubit that we want to protect. In this tutorial, we will use the qubit
# :math:`\vert \psi \rangle = \alpha \vert 0 \rangle + \beta \vert 1 \rangle,` as our data qubit.
#
# The next step is to encode this data qubit into a logical qubit. This is done by **encoding circuit** given below. Notice that we
# do not need to know the logical operators to implement the encoding circuit. The circuit is completely determined by the stabilizer generators.
# It is beyond the scope of this tutorial to explain how the circuit is constructed from the stabilizer generators. The state after encoding
# is given by [#Chandak2018]_:
#
# .. math::
#
#     \vert \bar{\psi}\rangle = \alpha \vert \bar{0} \rangle + \beta \vert \bar{1} \rangle
#
# .. figure:: ../_static/demonstration_assets/stabilizer_codes/five_qubit_encode.png
#    :align: center
#    :width: 100%
#
#    ..
#
# The calculations are a bit cumbersome, but with some patience we can find the common :math:`+1`-eigenspace of the stabilizer generators,
# which are the codewords.
#
# .. math::
#
#     \begin{align*}
#     \vert \bar{0}\rangle = &\frac{1}{4}\left(\vert 00000 \rangle \vert 10010 \rangle + \vert 01001 \rangle + \vert 10100 \rangle + \vert 01010 \rangle - \vert 11011 \rangle - \vert 00110 \rangle \right.\\
#                            &\left. - \vert 11101 \rangle - \vert 00011\rangle - \vert 11110 \rangle - \vert 01111 \rangle - \vert 10001 \rangle - \vert 01100 \rangle - \vert 10111 \rangle + \vert 00101 \rangle \right)
#     \end{align*}
#
# .. math::
#
#     \vert \bar{1}\rangle = X_1 X_2 X_3 X_4 X_5 \vert \bar{0} \rangle.
#
# The logical operators bit-flip and phase-flip are for this code are :math:`\bar{X}= X^{\otimes 5}` and :math:`\bar{Z}=Z^{\otimes 5}.`
#
# Let us implement this encoding circuit in PennyLane.


def five_qubit_encode(alpha, beta):
    qml.StatePrep([alpha, beta], wires=4)
    qml.Hadamard(wires=0)
    qml.S(wires=0)
    qml.CZ(wires=[0, 1])
    qml.CZ(wires=[0, 3])
    qml.CY(wires=[0, 4])
    qml.Hadamard(wires=1)
    qml.CZ(wires=[1, 2])
    qml.CZ(wires=[1, 3])
    qml.CNOT(wires=[1, 4])
    qml.Hadamard(wires=2)
    qml.CZ(wires=[2, 0])
    qml.CZ(wires=[2, 1])
    qml.CNOT(wires=[2, 4])
    qml.Hadamard(wires=3)
    qml.S(wires=3)
    qml.CZ(wires=[3, 0])
    qml.CZ(wires=[3, 2])
    qml.CY(wires=[3, 4])


##############################################################################
#
# Pauli Errors and syndrome measurements
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# After encoding, the qubit is exposed to noise and decoherence in a real system.
# To simulate this, we introduce an artificial error by randomly acting on one of the physical
# qubits on wires 0, 1, 2, 3, or 4 with Pauli X, Y, or Z operations. Then, we proceed with the syndrome measurements, which
# in this case, amounts to acting with the controlled stabilizer generators on the work wires, as follows
#
# .. figure:: ../_static/demonstration_assets/stabilizer_codes/general_stabilizer_circuit.png
#    :align: center
#    :width: 100%
#
# with the corresponding stabilizer operators.
#

stabilizers = [
    X(0) @ Z(1) @ Z(2) @ X(3) @ I(4),
    I(0) @ X(1) @ Z(2) @ Z(3) @ X(4),
    X(0) @ I(1) @ X(2) @ Z(3) @ Z(4),
    Z(0) @ X(1) @ I(2) @ X(3) @ Z(4),
]


def five_qubit_error_detection():
    for wire in range(5, 9):
        qml.Hadamard(wires=wire)

    for i in range(len(stabilizers)):
        qml.ctrl(stabilizers[i], control=[i + 5])

    for wire in range(5, 9):
        qml.Hadamard(wires=wire)


#############################################################################
#
# We can now combine this with the encoding circuit and the application of the Pauli errors to obtain the circuit that
# measures the syndrome, as we did in the three-qubit code.
#

dev = qml.device("default.qubit", wires=9)


@qml.set_shots(1)
@qml.qnode(dev)
def five_qubit_syndromes(alpha, beta, error_op, error_wire):
    five_qubit_encode(alpha, beta)

    error_op(wires=error_wire)

    five_qubit_error_detection()

    return qml.sample(wires=range(5, 9))


#############################################################################
#
# Now we need to build the syndrome table, which maps measurement outcomes to
# specific errors, allowing us to detect and correct errors on the encoded qubits.

ops_and_syndromes = []

for wire in (0, 1, 2, 3, 4):
    for error_op in (qml.PauliX, qml.PauliY, qml.PauliZ):
        ops_and_syndromes.append(
            (
                error_op,
                wire,
                five_qubit_syndromes(1 / 2, np.sqrt(3) / 2, error_op, wire),
            )
        )

        print(
            f"{error_op(wire).name[-1]}{wire}",
            five_qubit_syndromes(1 / 2, np.sqrt(3) / 2, error_op, wire),
        )


##############################################################################
#
# The syndrome table is printed, and with it we can apply the necessary operators to fix the corresponding Pauli errors. The script above is straightforward
# to generalize to any valid set of stabilizers.
#
# Error correction
# ~~~~~~~~~~~~~~~~
#
# The last step is to correct the error by applying the appropriate Pauli operators to the encoded qubits. This time, we have many
# possible syndrome measurement outcomes. Let us write a helper function to encode the possible syndromes in a way amiable to
# PennyLane's mid-circuit measurement capabilities, which only allows for Boolean operators.
#
def syndrome_booleans(syndrome, measurements):
    syndrome = qml.math.squeeze(syndrome)
    if syndrome[0] == 0:
        m = ~measurements[0]
    else:
        m = measurements[0]

    for i, elem in enumerate(syndrome[1:]):
        if elem == 0:
            m = m & ~measurements[i + 1]
        else:
            m = m & measurements[i + 1]

    return m


#############################################################################
#
# Combining all these pieces, we can write the full error correcting code.
#
dev = qml.device("default.qubit", wires=9)


@qml.qnode(dev)
def five_qubit_code(alpha, beta, error_op, error_wire):
    five_qubit_encode(alpha, beta)

    error_op(wires=error_wire)

    five_qubit_error_detection()

    m5 = qml.measure(5)
    m6 = qml.measure(6)
    m7 = qml.measure(7)
    m8 = qml.measure(8)

    measurements = [m5, m6, m7, m8]

    for op, wire, synd in ops_and_syndromes:
        qml.cond(syndrome_booleans(synd, measurements), op)(wires=wire)

    return qml.density_matrix(wires=[0, 1, 2, 3, 4])


#############################################################################
#
# Let us check that the fidelity between the output state and the initial encoded state is equal to 1 for arbitrary Pauli errors on one
# qubit. Indeed:
#


@qml.qnode(qml.device("default.qubit", wires=5))
def five_qubit_encoded_state(alpha, beta):
    five_qubit_encode(alpha, beta)
    return qml.state()


encoded_state = qml.math.dm_from_state_vector(five_qubit_encoded_state(alpha, beta))
for wire in range(5):
    for error_op in (qml.PauliX, qml.PauliY, qml.PauliZ):
        print(
            f"Fidelity when error {error_op(wire).name[-1]}{wire}:",
            qml.math.fidelity(
                encoded_state, five_qubit_code(alpha, beta, error_op, wire)
            ).round(2),
        )

#############################################################################
#
# The fidelity is 1.0 after error correction, which means the output state is the same!
#
# Note that to build the encoding, syndrome measurement, and error correction circuits, we did only use the stabilizer generators.
# This is a powerful feature of the stabilizer formalism. It allows us to construct the code from its stabilizer generators
# and then use the code to correct errors. However, we can also find the codewords  and logical operators directly from the stabilizer generators by
# finding the common +1-eigenspace of the stabilizer generators.
#
# Conclusion
# -----------
#
# In this tutorial, we introduced the stabilizer formalism and showed how it can be used to construct quantum error correction codes.
# We applied it to a concrete example—the five-qubit code—using a PennyLane implementation.
# However, finding the codewords, logical operators, and the encoding circuit directly from the stabilizer generators
# is not straightforward. Take a look at the types of gates used in the encoding circuit: you’ll notice they are all Clifford gates.
# In fact, any standard stabilizer code can be encoded using only Clifford gates, which is a major advantage. Why?
# Because Clifford gates are easy to implement and measure! This means that, while the qubit cost of error correction may be high,
# the gates needed for stabilizer codes can be robustly implemented on quantum hardware.  Quantum error correction
# is not out of reach as soon as the scalability challenges are addressed!
#
# The authors of this demo would like to thank Austin Daniel for the insightful discussions and Priya Nadkarni for her helpful
# technical feedback.
#

#######################################################################
# References
# ------------
#
# .. [#Gottesman1997]
#    D. Gottesman.
#    "Stabilizer Codes and Quantum Error Correction",
#    `<https://arxiv.org/abs/quant-ph/9705052>`__, 1997.
#
# .. [#Laflamme1996]
#    R. Selinger, C. Miquel, J.P. Paz, W.H Surek.
#    "Perfect Quantum Error Correcting Code",
#    `<https://doi.org/10.1103/PhysRevLett.77.198>`__, Phys. Rev. Lett, vol. 77, no. 1, Jul 1996.
#
# .. [#Chandak2018]
#    S.Chandak, J. Mardia, M. Tolunay.
#    "Implementation and Analysis of stabilizer codes in PyQuil."
#    `<https://shubhamchandak94.github.io/reports/stabilizer_code_report.pdf>`__,
#
