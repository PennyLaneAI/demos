r"""
Optimizing QSVT data loading by exploiting structure
====================================================

Solving systems of linear equations is important for a wide range of industries, such as healthcare,
transportation, finance, chemistry, and even quantum computing. The Quantum Singular Value Transformation (QSVT) algorithm can implement matrix inversion to solve such
equations on a quantum computer [#chuang2021]_. For more information on how to use PennyLane's
:func:`~.pennylane.qsvt` functionality to run matrix inversion on a quantum computer see our demo on `QSVT in
Practice <tutorial_apply_qsvt>`_.

It turns out that the bottleneck of QSVT is
typically the cost of encoding the matrix onto a quantum computer in the first place! While this **data loading** cost is
significant for any general matrix, in real life our problems often have patterns and structure!

By exploiting the structure of a problem, we can significantly reduce the quantum resource cost of the algorithm, thereby making QSVT based matrix inversion more accessible to implement on nearer term fault-tolerant quantum
hardware.

.. figure:: ../_static/demo_thumbnails/opengraph_demo_thumbnails/pennylane-demo-resource-estimation-qsvt-2-open-graph.png
    :align: center
    :width: 70%
    :target: javascript:void(0)

This demo, based on our recent paper `Quantum compilation framework for data loading
<https://arxiv.org/abs/2512.05183>`_ [#linaje2025]_, will showcase an optimized block encoding strategy that
uses the sparsity of the matrix to significantly reduce the cost of QSVT. We will focus on a particular matrix
inversion problem that arises in computational fluid dynamics (CFD) [#lapworth2022]_. We will quantify the resource cost using PennyLane's :mod:`~.pennylane.estimator` module. 

Problem Setup
-------------
The two-dimensional lid-driven cavity flow (2D-LDC) is a canonical benchmark in CFD for verifying numerical
schemes that solve the incompressible Navier–Stokes equations. Determining the
fluid flow within the cavity requires solving the pressure correction equations by inverting the associated
pressure correction matrix (:math:`A`). A key observation is that :math:`A` is highly structured and sparse.
The figure below highlights the non-zero entries of :math:`A` with a dimensionality of :math:`64 \times 64`
[#lapworth2022]_.

|

.. figure:: ../_static/demonstration_assets/resource_estimation_QSVT_CFD/sparse_diagonal_matrix.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

|

To invert this matrix using QSVT, we will need to **load** it onto the quantum computer using a block encoding.
The standard technique for block encoding any (square) matrix is the method of `linear combination of
unitaries (LCUs) <tutorial_lcu_blockencoding>`_. However, it suffers from a few fatal flaws (try saying that five
times fast).

The number of terms in the LCU scales as :math:`O(4^{n})` in the number of qubits, and thus the cost of the block
encoding also scales exponentially. Even computing the LCU decomposition becomes a computational bottleneck.
Furthermore, there is no way of knowing a priori how many terms there will be in the LCU. For these reasons, a
general "one-size-fits-all" block encoding scheme is usually too expensive for our systems of interest.

.. _naive_block_encoding:
Resource cost of naive block encoding
------------------------------------------
Let's explore the average cost of block encoding a matrix in the standard way. We suppose that this matrix has size :math:`2^{20} \times 2^{20}` and can be written as the superposition of $4^{20}$ Pauli words. Note that not all matrices of that size can be written in this manner, so the resulting resource estimate is not necessarily general. However, it does illustrate the cost of naively block encoding a particular matrix of this size. 

"""

import pennylane.estimator as qre

num_qubits = 20
matrix_size = 2**num_qubits

lcu_A = qre.PauliHamiltonian(
   num_qubits = num_qubits,
   pauli_terms = {"Z"*(num_qubits//2): 4**num_qubits},
) # 4^20 Pauli words comprise this matrix

def Standard_BE(prep, sel):
   return qre.ChangeOpBasis(prep, sel, qre.Adjoint(prep))

Prep = qre.QROMStatePreparation(num_qubits)  # Preparing a single qubit in the target state
Select = qre.SelectPauli(lcu_A)  # Select the operators in the LCU

gate_set = {"X", "Y", "Z", "Hadamard", "T", "S", "CNOT"}
resources = qre.estimate(Standard_BE, gate_set)(Prep, Select) # Estimate the resource requirement
print(resources)



##############################################################################
# With one call to :mod:`qre.estimate`, we can see that that the estimated $T$ gate cost of naive block encoding this matrix is :math:`4 \times 10^{12}`. This block encoding is called many times within an instance of the QSVT algorithm, and can be the dominant cost. Now that we have established a baseline of the 'standard' cost, we ask: Can we do better? 
# 
# Yes! We can leverage the **structure** of our matrix to implement a much more efficient block encoding operator.
# 
# 
# Exploiting structure in the block encoding
# ------------------------------------------
# The matrix :math:`A` can be block encoded using a *d-diagonal encoding* technique [#linaje2025]_ developed
# by my colleagues here at Xanadu. The method loads each diagonal in parallel, then shifts them to their respective
# ranks in the matrix. The values along each diagonal can be sparsely represented in a different basis, which
# further reduces resources than existing state-of-the-art methods. The quantum circuit that implements the
# d-diagonal block encoding is presented below. To learn more, read our paper: `"Quantum compilation framework for
# data loading" <https://arxiv.org/abs/2512.05183>`_.
# 
# |
# 
# .. figure:: ../_static/demonstration_assets/resource_estimation_QSVT_CFD/SparseBE.png
#     :align: center
#     :width: 80%
#     :target: javascript:void(0)
# 
# |
# 
# To estimate the resource cost for this approach, we can use
# PennyLane's quantum resource :mod:`~.pennylane.estimator` module!
# 
# Diagonal Matrices & the Walsh Transform
# ------------------------------------------------
# Each :math:`D_{k}` is a block-diagonal operator that contains the normalised entries from the :math:`k^{\text{th}}`
# diagonal of our d-diagonal matrix :math:`A`. By multiplexing over the :math:`D_{k}` operators, we can load all of
# the diagonals in *parallel*.
# 
# Instead of implementing each :math:`D_{k}` as a product of :class:`~.pennylane.ControlledPhaseShift` gates,
# we leverage the `Walsh transform <https://mathworld.wolfram.com/WalshTransform.html>`_ [#zylberman2025]_. The Walsh transform allows us to naturally
# optimize the cost of our block encoding by tuning the number of Walsh coefficients within :math:`[1, N]`,
# where :math:`N` is the size of the matrix. If the entries in our diagonal are sparse in the Walsh basis,
# as is the case for the CFD example, then we can get away with far fewer Walsh coefficients. This results in
# a much more efficient encoding. The circuit below, with :math:`H_{y} = SH`, prepares a single such Walsh diagonal block encoding,
# ultimately we need to prepare as many operators as non-zero diagonals in :math:`A`.
# 
# |
# 
# .. figure:: ../_static/demonstration_assets/resource_estimation_QSVT_CFD/WH_a.png
#     :align: center
#     :width: 90%
#     :target: javascript:void(0)
# 
# |
# 
# For more details, refer to Appendix B's Walsh transform encoding section in
# [#linaje2025]_.
# 

import pennylane.numpy as qnp
import pennylane.estimator as qre


def Walsh_Dk(num_diags, size_diagonal, num_walsh_coeffs):
    num_diag_wires = int(qnp.ceil(qnp.log2(size_diagonal)))
    list_of_diagonal_ops = []

    for _ in range(num_diags):
        compute_op = qre.Prod([qre.Hadamard(), qre.S()])

        zero_ctrl_wh = qre.Controlled(
            qre.Prod(((qre.MultiRZ(num_diag_wires // 2), num_walsh_coeffs),)),
            num_ctrl_wires=1,
            num_zero_ctrl=1,
        )
        one_ctrl_wh = qre.Controlled(
            qre.Prod(((qre.MultiRZ(num_diag_wires // 2), num_walsh_coeffs),)),
            num_ctrl_wires=1,
            num_zero_ctrl=0,
        )
        target_op = qre.Prod([zero_ctrl_wh, one_ctrl_wh])

        list_of_diagonal_ops.append(qre.ChangeOpBasis(compute_op, target_op))

    return list_of_diagonal_ops


##############################################################################
# Here the `Prod <https://docs.pennylane.ai/en/stable/code/api/pennylane.estimator.ops.Prod.html>`_
# class is used to represent a product of operators, and the
# `Controlled <https://docs.pennylane.ai/en/stable/code/api/pennylane.estimator.ops.Controlled.html>`_
# class represents the controlled operator.
#
# Shift Operator
# --------------
# Next, we will implement the Shift operator. This is a product of three subroutines which
# shift the diagonal entries into the correct rank of the d-diagonal block encoding. Each rank
# has an associated shift value, :math:`k`, the main diagonal has shift value $0$. Each diagonal going down
# is labelled :math:`(+1, +2, \ldots)` respectively and each diagonal going up is labelled
# with its negative counterpart. The Shift operator works in three steps:
#
# 1. **Load** the binary representation of the shift value :math:`k` for each non-zero diagonal in :math:`A`.
#
# 2. **Shift** each of the diagonals in parallel using an :math:`Adder` subroutine.
#
# 3. **Unload** the shift values to restore the quantum register.
#


def ShiftOp(num_shifts, num_load_wires, wires):
    prep_wires, load_shift_wires, load_diag_wires = wires

    Load = qre.QROM(
        num_bitstrings=num_shifts,
        size_bitstring=num_load_wires,
        restored=False,
        select_swap_depth=1,
        wires=prep_wires + load_shift_wires,
    )

    Adder = qre.SemiAdder(
        max_register_size=num_load_wires,
        wires=load_shift_wires + load_diag_wires,
    )

    Unload = qre.Adjoint(
        qre.QROM(
            num_bitstrings=num_shifts,
            size_bitstring=num_load_wires,
            restored=False,
            select_swap_depth=1,
            wires=prep_wires + load_shift_wires,
        )
    )
    return qre.Prod([Load, Adder, Unload])


##############################################################################
# d-Diagonal Block Encoding
# -------------------------
# Now that we have developed all of the pieces, we'll bring them together to implement
# the full d-diagonal block encoding operator. Note that the circuit appears more involved
# than a simple PrepSelPrep circuit does, but we will soon show that resource cost can be decreased
# significantly as a result.
#
# |
#
# .. figure:: ../_static/demonstration_assets/resource_estimation_QSVT_CFD/SparseBE.png
#     :align: center
#     :width: 80%
#     :target: javascript:void(0)
#
# |
#


def WH_Diagonal_BE(num_diagonals, matrix_size, num_walsh_coeffs):
    # Initialize qubit registers:
    num_prep_wires = int(qnp.ceil(qnp.log2(num_diagonals)))
    num_diag_wires = int(qnp.ceil(qnp.log2(matrix_size)))

    prep_wires = [f"prep{i}" for i in range(num_prep_wires)]
    load_diag_wires = [f"load_d{i}" for i in range(num_diag_wires)]
    load_shift_wires = [f"load_s{i}" for i in range(num_diag_wires)]

    # Prep:
    Prep = qre.QROMStatePreparation(num_prep_wires, wires=prep_wires)

    # Shift:
    Shift = ShiftOp(
        num_shifts=num_diagonals,
        num_load_wires=num_diag_wires,
        wires=(prep_wires, load_diag_wires, load_shift_wires),
    )

    # Multiplexed - Dk:
    diagonal_ops = Walsh_Dk(num_diagonals, matrix_size, num_walsh_coeffs)
    Dk = qre.Select(diagonal_ops, wires=load_diag_wires + prep_wires + ["target"])

    # Prep^t:
    Prep_dagger = qre.Adjoint(qre.QROMStatePreparation(num_prep_wires, wires=prep_wires))

    return qre.Prod([Prep, Shift, Dk, Prep_dagger])

##############################################################################
# With this special block encoding method ready, we can estimate the resource cost of block encoding with the same hyperparameters as used in the CFD example below. 

matrix_size = 2**20  # 2^20 x 2^20
num_diagonals = 3
num_walsh_coeffs = 512  # empirically determined
resources = qre.estimate(WH_Diagonal_BE)(num_diagonals, matrix_size, num_walsh_coeffs)
print(resources)

##############################################################################
# Although this matrix :math:`A` is not the same as the matrix considered in the :ref:`naive_block_encoding` section, it is the same size. We can see that exploiting the structure of the matrix has decreased the order of magnitude of the $T$ gate cost from :math:`4 \times 10^{12}` to :math:`3 \times 10^{5}` for the task of block encoding. 



##############################################################################
# Putting it all together: QSVT for matrix inversion
# --------------------------------------------------
# In this section we use all of the tools we have developed to **estimate the resource requirements
# for solving a practical matrix inversion problem**. Following the CFD example [#linaje2025]_, we
# will estimate the resources required to invert a tri-diagonal matrix of size :math:`2^{20} \times 2^{20}`.
# This system requires a :math:`10^{8}`-degree polynomial approximation of the inverse function. We will
# also restrict the gateset to Clifford + $T$ gates.

degree = 10**8
matrix_size = 2**20  # 2^20 x 2^20
num_diagonals = 3
num_walsh_coeffs = 512  # empirically determined


def matrix_inversion(degree, matrix_size, num_diagonals):
    num_state_wires = int(qnp.ceil(qnp.log2(matrix_size)))  # 20 qubits
    b_wires = [f"load_d{i}" for i in range(num_state_wires)]

    # Prepare |b> vector:
    qre.MPSPrep(
        num_state_wires,
        max_bond_dim=2**5,  # bond dimension = 2**5
        wires=b_wires,
    )

    # Apply A^-1:
    qre.QSVT(
        block_encoding=WH_Diagonal_BE(num_diagonals, matrix_size, num_walsh_coeffs),
        encoding_dims=(matrix_size, matrix_size),
        poly_deg=degree,
    )
    return


gate_set = {"X", "Y", "Z", "Hadamard", "T", "S", "CNOT"}
res = qre.estimate(matrix_inversion, gate_set)(degree, matrix_size, num_diagonals)
print(res)

##############################################################################
# The estimated $T$ gate count of this matrix inversion workflow matches
# the reported :math:`3 \times 10^{13}` from the reference.
#
# Conclusion
# ----------
# In this demo, we showed how we can exploit knowledge about the structure of our matrices to build a more
# efficient block encoding. This work brings us one step closer to implementing quantum linear solvers on near
# term quantum hardware. If you are interested in other subroutines for data loading, including state preparation,
# and how to optimize them, check out our paper: `Quantum compilation framework for data loading <https://arxiv.org/abs/2512.05183>`_.
#
# Along the way we showcased PennyLane's :mod:`~.pennylane.estimator` module to determine the resource
# requirements for **matrix inversion by QSVT**. You learned how to build complex circuits from our library of
# primitives, making resource estimation as simple as putting together Lego blocks. Of course, QSVT has many other
# applications, from unstructured search to phase estimation, and even to Hamiltonian simulation. We challenge you to
# think of smarter data loading techniques to reduce the quantum cost of QSVT for these applications as well!
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
# .. [#linaje2025]
#
#     Guillermo Alonso-Linaje, Utkarsh Azad, Jay Soni, Jarrett Smalley,
#     Leigh Lapworth, and Juan Miguel Arrazola,
#     "Quantum compilation framework for data loading"
#     `arxiv.2512.05183 <https://arxiv.org/abs/2512.05183>`__, 2025.
#
# .. [#lapworth2022]
#
#     Leigh Lapworth
#     "A Hybrid Quantum-Classical CFD Methodology with Benchmark HHL Solutions"
#     `arxiv.2206.00419 <https://arxiv.org/abs/2206.00419>`__, 2022.
#
# .. [#zylberman2025]
#
#     Julien Zylberman, Ugo Nzongani, Andrea Simonetto, and Fabrice Debbasch,
#     "Efficient Quantum Circuits for Non-Unitary and Unitary Diagonal Operators with Space-Time-Accuracy trade-offs"
#     `arxiv.2404.02819 <https://arxiv.org/abs/2404.02819>`__, 2025.
#
