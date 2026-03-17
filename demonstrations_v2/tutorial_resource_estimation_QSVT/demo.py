r"""
How to estimate the resource cost of QSVT
=========================================

The Quantum Singular Value Transformation (QSVT) is a versatile algorithm that is applicable to a wide range of
problems, including unstructured search, Hamiltonian simulation, matrix inversion, and many more [#chuang2021]_.
PennyLane makes it easy to build circuits and experiment with QSVT using the :func:`~.pennylane.qsvt` function.
For more information on how to use PennyLane's QSVT functionality checkout our other demos:

- `Intro to QSVT <tutorial_intro_qsvt>`_
- `QSVT in Practice <tutorial_apply_qsvt>`_
- `How to implement QSVT on hardware <tutorial_qsvt_hardware>`_

It's important to understand the quantum resource cost of the QSVT algorithm for a variety of system sizes.
Fortunately, PennyLane's resource :mod:`~.pennylane.estimator` module makes that easy, even if the QSVT problem
you're interested in is too big to simulate right now. If you are new to resource estimation in PennyLane or need
a quick refresher, checkout this demo on `how to use PennyLane for Resource Estimation <re_how_to_use_pennylane_for_resource_estimation>`_.

.. figure:: ../_static/demo_thumbnails/opengraph_demo_thumbnails/pennylane-demo-resource-estimation-qsvt-open-graph.png
    :align: center
    :width: 70%
    :target: javascript:void(0)

In this demo, you will learn how to use PennyLane's :mod:`~.pennylane.estimator` module to easily estimate the
cost of QSVT. There are two ways of doing so: the Executable workflow and the Estimator workflow. The
Estimator workflow involves expressing our QSVT circuit using :mod:`~.pennylane.estimator` operators. This
workflow scales for *any* system size, has a simpler UI and produces tighter resource estimates. For users
who have already built a standard PennyLane circuit, the Executable workflow allows for resource estimation
with only one extra line of code.

Estimating the cost of QSVT
---------------------------
Let's estimate the cost of performing a quintic (5th degree) polynomial transformation to the matrix
:math:`A`:

.. math::

    A = \begin{bmatrix}
        0.1 &  0.0 &  0.3 &  0.2 \\
        0.0 & -0.1 &  0.2 & -0.3 \\
        0.3 &  0.2 & -0.1 &  0.0 \\
        0.2 & -0.3 &  0.0 &  0.1 \\
        \end{bmatrix},


This particular matrix can be expressed as a linear combination of unitaries (LCU) :math:`A = 0.1 \cdot Z_{0}Z_{1} + 0.2 \cdot X_{0}X_{1} + 0.3 \cdot X_{0}Z_{1}`.
The LCU representation is crucial for building the **block encoding** operator using the standard method of LCUs.
For a recap on this technique, see our demo on `linear combination of unitaries and block encodings <tutorial_lcu_blockencoding>`_.
"""

import pennylane as qml

A = 0.1 * (qml.Z(0) @ qml.Z(1)) + 0.2 * (qml.X(0) @ qml.X(1)) + 0.3 * (qml.X(0) @ qml.Z(1))

print(qml.matrix(A, wire_order=[0, 1]))

##############################################################################
# Resources from an Executable Workflow
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Suppose we already had a PennyLane circuit which used QSVT to apply the quintic polynomial transformation to
# :math:`A`. We can obtain the resource estimate with only a couple of lines of code with
# `estimate() <https://docs.pennylane.ai/en/stable/code/api/pennylane.estimator.estimate.estimate.html>`_.
#

import pennylane.numpy as qnp
import pennylane.estimator as qre

## --- QSVT Workflow: ---
num_terms = len(A)
num_encoding_wires = int(qnp.ceil(qnp.log2(num_terms)))
encoding_wires = [f"e_{i}" for i in range(num_encoding_wires)]

poly = (0, 0, 0, 0, 0, 1)  # f(x) = x^5
def circ():
    qml.qsvt(A, poly, encoding_wires=encoding_wires)
    return

## --- Resource Estimation: ---
gs = {"X", "Y", "Z", "S", "T", "Hadamard", "CNOT", "Toffoli"}
resources = qre.estimate(circ, gate_set=gs)()
print(resources)

##############################################################################
# This works well for small systems. For larger system sizes, we can use some of the other functionalities
# from the :mod:`~.pennylane.estimator` module that are designed for scale to estimate the cost of QSVT.
#
# Resources from an Estimator Workflow
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The LCU representation of :math:`A` is efficiently stored using the
# :class:`~.pennylane.estimator.compact_hamiltonian.PauliHamiltonian` class. This produces a compact object
# specifically for resource estimation. The block encoding operator is built with the
# `ChangeOpBasis <https://docs.pennylane.ai/en/stable/code/api/pennylane.estimator.ops.ChangeOpBasis.html>`_
# class which uses the compute-uncompute pattern to implement the
# :math:`\text{Prep}^{\dagger} \circ \text{Select} \circ \text{Prep}` operator.
#
# The resources for `QSVT <https://docs.pennylane.ai/en/stable/code/api/pennylane.estimator.templates.QSVT.html>`_
# can then be obtained using this block encoding. Note that these operators are specifically designed for resource
# estimation and are *not supported* for execution or simulation in a circuit.

## --- LCU representation of A: ---
lcu_A = qre.PauliHamiltonian(
    num_qubits=2,
    pauli_terms={"ZZ": 1, "XX": 1, "XZ": 1},
)  # represents A = 0.1*ZZ + 0.2*XX + 0.3*XZ
num_terms = lcu_A.num_terms
num_qubits = lcu_A.num_qubits

## --- Block Encoding operator: ---
num_encoding_wires = int(qnp.ceil(qnp.log2(num_terms)))
encoding_wires = [f"e_{i}" for i in range(num_encoding_wires)]
lcu_wires = [f"t_{i}" for i in range(num_qubits)]

Prep = qre.QubitUnitary(  # Prep the coeffs of the LCU
    num_encoding_wires,
    wires=encoding_wires,
)

Select = qre.SelectPauli(  # Select over ops in the LCU
    lcu_A,
    wires=lcu_wires + encoding_wires,
)

BlockEncoding = qre.ChangeOpBasis(Prep, Select)  # Prep ○ Sel Prep^t

## --- QSVT operator: ---
qsvt_op = qre.QSVT(
    block_encoding=BlockEncoding,
    encoding_dims=(4, 4),  # The shape of matrix A
    poly_deg=5,  # quintic
)

## --- Resource Estimation: ---
gs = {"X", "Y", "Z", "S", "T", "Hadamard", "CNOT", "Toffoli"}
resources = qre.estimate(qsvt_op, gate_set=gs)
print(resources)

##############################################################################
# Representing the QSVT workflow like this allows us to easily perform resource estimation larger system sizes
# without any computational overheads. Let's extend this example to a **50 qubit** system with an LCU of **2000
# terms** and a **100th degree** polynomial transformation. Notice how simple it is to update the code
# and obtain the cost of this larger system:

## --- LCU representation of A: ---
lcu_A = qre.PauliHamiltonian(
    num_qubits=50,
    pauli_terms={"ZZ": 250, "XX": 750, "XZ": 1000},
)  # 2000 terms !
num_terms = lcu_A.num_terms
num_qubits = lcu_A.num_qubits

## --- Block Encoding operator: ---
num_encoding_wires = int(qnp.ceil(qnp.log2(num_terms)))
encoding_wires = [f"e_{i}" for i in range(num_encoding_wires)]
lcu_wires = [f"t_{i}" for i in range(num_qubits)]

Prep = qre.QROMStatePreparation(  # Efficient Prep for large systems
    num_encoding_wires,
    wires=encoding_wires,
)

Select = qre.SelectPauli(  # Select over ops in the LCU
    lcu_A,
    wires=lcu_wires + encoding_wires,
)

BlockEncoding = qre.ChangeOpBasis(Prep, Select)  #  Prep ○ Sel Prep^t

## --- QSVT operator: ---
qsvt_op = qre.QSVT(
    block_encoding=BlockEncoding,
    encoding_dims=(2**50, 2**50),  # The shape of matrix A
    poly_deg=100,
)

## --- Resource Estimation: ---
gs = {"X", "Y", "Z", "S", "T", "Hadamard", "CNOT", "Toffoli"}
resources = qre.estimate(qsvt_op, gate_set=gs)
print(resources)

##############################################################################
# With PennyLane's resource estimation functionality we can analyze the cost of QSVT workflows for large
# system sizes consisting of hundreds of qubits and millions of gates!
#
# Conclusion
# ----------
# In this demo, you learned how to use PennyLane's :mod:`~.pennylane.estimator` module to determine the
# resource requirements for **QSVT**. Now that you are armed with these tools for resource estimation,
# I challenge you to find another problem where polynomial transformations may be helpful, and figure out:
# what are the logical resource requirements of solving this on a quantum computer?
#
# References
# ----------
#
# .. [#chuang2021]
#
#     John M. Martyn, Zane M. Rossi, Andrew K. Tan, and Isaac L. Chuang,
#     "A Grand Unification of Quantum Algorithms"
#     `arxiv.2105.02859 <https://arxiv.org/abs/2105.02859>`__, 2021.
#
