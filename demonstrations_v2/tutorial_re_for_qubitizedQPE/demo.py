r"""Qubit and gate trade-offs in Qubitized Quantum Phase Estimation
======================================================================


`Quantum Phase Estimation (QPE) <https://pennylane.ai/qml/demos/tutorial_qpe>`_ is a powerful quantum algorithm
that allows us to estimate the eigenvalues of a Hamiltonian with high precision.
The most advanced variants of QPE rely on
`qubitization <https://pennylane.ai/qml/demos/tutorial_qubitization>`_  to encode chemical Hamiltonians  as unitary operators. This leverages a `linear combination of unitaries (LCU) <https://pennylane.ai/qml/demos/tutorial_lcu_blockencoding>`_ decomposition to create a
`block encoding <https://pennylane.ai/qml/demos/tutorial_lcu_blockencoding>`_ of the Hamiltonian , which is then used to construct a "quantum walk" operator that is used as input to QPE.

We focus on the Tensor Hypercontraction (THC) representation, a state-of-the-art LCU decomposition for quantum chemistry
that approximates the interaction tensor via a low-rank factorization.

**But is implementing this quantum algorithm feasible on early fault-tolerant hardware?**
To answer this, we must move beyond asymptotic scaling and determine the concrete resource requirements.
In this demo, we use PennyLane's logical resource :mod:`estimator <pennylane.estimator>`
to calculate the precise costs and demonstrate how to optimize the algorithm to fit on constrained devices with
a few hundred logical qubits. In particular, we show how to implement **QPE for the 76-orbital active space of FeMoco, using fewer than 500 logical qubits**.

.. figure:: ../_static/demo_thumbnails/opengraph_demo_thumbnails/OGthumbnail_re_for_qubitization.png
    :align: center
    :width: 70%
    :target: javascript:void(0)
"""

######################################################################
# The key to this optimization lies in the specific method to build the quantum walk operator,
# which is constructed from two primary subroutines:
# the ``Prepare`` oracle, which prepares a state encoding the Hamiltonian coefficients, and the
# ``Select`` oracle, which applies the Hamiltonian terms controlled by that state. The implementation of these subroutines
# offers the flexibility to trade off qubits for gates, and vice versa. Specifically, we can tune two algorithmic knobs to perform this trade-off: batched Givens rotations and
# `Quantum Read-Only Memory (QROM) <https://pennylane.ai/qml/demos/tutorial_intro_qrom>`_ Select-Swap. Let's see these two in detail.
#
# Knob #1: Batched Givens rotations
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# In the ``Select`` operator, we need to implement a series of Givens rotations to change the single-particle basis.
# Naively, to store all angles simultaneously, we require a register size equal to the number of rotations multiplied by the number of bits used for precision for each angle.
# However, we can choose to load these angles in batches instead of loading all of them at once [#Caesura]_.
# The tunable knob here is the **number of batches** in which the rotation angles are loaded. By increasing the number of batches,
# register size is decreased, leading to a reduction of qubit requirements, but in exchange we need more repetitions of the QROM
# subroutine for each batch, which increases the Toffoli count.
#
# .. figure:: ../_static/demonstration_assets/qubitization_re/pennylane-demo-image-circuit-batching-fig.png
#    :align: center
#    :width: 1100px
#    :target: javascript:void(0)
#
# In the left panel (a), we load all angles at once using a single call to QROM (pink), but this requires four ancilla registers.
# In the right panel (c), a single ancilla register is used, but we need four calls to QROM. The middle panel (b) shows an
# intermediate strategy with two ancilla registers and two QROM calls.
#
# Knob #2: QROM SelectSwap
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The second major optimization strategy is through `QROM <https://pennylane.ai/qml/demos/tutorial_intro_qrom>`_ itself. Crucially, both ``Prepare`` and ``Select``
# rely on QROM to access Hamiltonian coefficients and rotation angles, respectively. We can use
# the select-swap variant of QROM, which allows us to trade the depth of the circuit for width, as shown in the diagrams below:
#
# .. figure:: ../_static/demonstration_assets/qubitization_re/selswap_combine.jpeg
#    :align: center
#    :width: 100%
#    :target: javascript:void(0)
#
# The configuration on the right achieves lower gate complexity by employing auxiliary work wires to enable block-wise data loading.
# This approach replaces expensive multi-controlled operations with simpler controlled-swap gates, significantly reducing the Toffoli
# count while requiring additional qubits.
#
# Standard resource estimates often treat these oracles as fixed "black boxes", yielding a single cost value.
# However, our quantum resource :mod:`estimator <pennylane.estimator>` provides much more than a static cost report.
# We demonstrate how PennyLane exposes these
# tunable knobs of the circuit implementation, allowing us to actively navigate the circuit design and trade-off between gates and
# logical qubits to suit different constraints. As a concrete example, let's estimate the resources needed to simulate the FeMoco molecule.
#
# Resource estimation for FeMoco
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Determining the resources necessary for large-scale chemical system simulation is often bottlenecked by the challenge of constructing and storing the full Hamiltonian tensor.
# PennyLane's resource estimator allows us to sidestep this bottleneck by
# using a `compact representation <https://docs.pennylane.ai/en/stable/code/api/pennylane.estimator.compact_hamiltonian.THCHamiltonian.html>`__ of the THC
# Hamiltonian, where we capture only the essential structural parameters: the number of spatial orbitals :math:`(N)`,
# the THC factorization rank :math:`(M)`, and the Hamiltonian one-norm :math:`(\lambda)`.
#
# While calculating the exact one-norm typically requires full Hamiltonian construction, this compact form is particularly
# useful for well-known benchmarks where these values are already reported in the literature. Furthermore, it allows
# us to rapidly generate estimates for different ranges of one-norms, enabling sensitivity analysis without
# the need to build the full operator for every case, which would be computationally expensive.
#
# .. note::
#     It is important to acknowledge that while the reference used here represents a seminal work in algorithmic development, the
#     current state-of-the-art for such simulations is achieved by methods utilizing Block-Invariant Symmetry Shift (BLISS)-THC
#     Hamiltonians [#Caesura]_ or sum-of-squares spectral amplification (SOSSA) [#SOSSA]_.
#     Here we focus on the THC implementation as it provides a straightforward and intuitive framework
#     for understanding the fundamental trade-offs between qubit and gate resource requirements.
#
# Using parameters obtained from the literature, let's compactly describe the THC representation
# of the FeMoco Hamiltonian with a 76-orbital active space [#lee2021]_:

from pennylane import estimator as qre

femoco = qre.THCHamiltonian(num_orbitals=76, tensor_rank=450, one_norm=1201.5)

#######################################################################
# Next we need to determine the precision with which we want to simulate these systems, and how it translates to the circuit parameters.
#
# Defining the error budget
# -------------------------
# We begin by fixing the target accuracy for the Quantum Phase Estimation (QPE) routine to :math:`\epsilon_{QPE} = 0.0016 \textrm{Ha}` ,
# which dictates the total number of QPE iterations required:
#
# .. math::
#     n_{iter} = \lceil \log_2\left(\frac{2\pi \lambda}{\epsilon_{QPE}}\right) \rceil.
#
# This choice also dictates the required bit precision for the circuit's subroutines. Specifically, to maintain this
# overall accuracy, we fix the numerical precision for expressing the Hamiltonian coefficients in ``Prepare``
# and the rotation angles in ``Select``.
#
# Using the error bounds derived in `Lee et al. (2021) <https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.2.030305>`__ (Appendix C),
# we can calculate the required number of bits for loading coefficients (:math:`n_{coeff}`) and rotation angles (:math:`n_{angle}`) as:
#
# .. math::
#    n_{coeff} = \left\lceil 2.5 + \log_2\left(\frac{10 \lambda}{\epsilon_{QPE}}\right) \right\rceil, \quad
#    n_{angle} = \left\lceil 5.652 + \log_2\left(\frac{20 \lambda N}{\epsilon_{QPE}}\right) \right\rceil.
#
# Since we are following the analysis in Lee et al. (2021) [#lee2021]_, we use the same constants as the reference:

import numpy as np

epsilon_qpe = 0.0016  # Ha
n_iter = int(np.ceil(2 * np.pi * femoco.one_norm / epsilon_qpe))  # QPE iterations
n_coeff = 10
n_angle = 20

########################################################################
# Estimating the cost of qubitized QPE
# -----------------------------
# With these parameters in hand, we can estimate the total resources required by our algorithm. The full algorithm consists of the Walk Operator,
# constructed via `QubitizeTHC <https://docs.pennylane.ai/en/stable/code/api/pennylane.estimator.templates.QubitizeTHC.html>`_, running within a QPE routine.
#
# We  note that `SelectTHC <https://docs.pennylane.ai/en/stable/code/api/pennylane.estimator.templates.SelectTHC.html>`_
# oracle implementation is based on the description in
# von Burg et al. [#vonburg]_. This work uses the phase gradient technique to implement Givens rotations, and thus requires an
# auxiliary resource state for phase addition.
#
# Let's estimate the total resources for Qubitized QPE for FeMoco:

wo_femoco = qre.QubitizeTHC(femoco, coeff_precision=n_coeff, rotation_precision=n_angle)

phase_grad_cost = qre.estimate(qre.PhaseGradient(n_angle))

qpe_cost = qre.estimate(qre.UnaryIterationQPE(wo_femoco, num_iterations=n_iter))

total_cost = qpe_cost.add_parallel(phase_grad_cost)  # add cost of phase gradient
print(f"Resources for Qubitized QPE for FeMoco(76): \n {total_cost}\n")

######################################################################
# Analyzing the results
# ---------------------
# This version of QPE thus requires 2188 qubits and :math:`8.8 \times 10^{10}` Toffoli gates (not to mention around :math:`1 \times 10^{13}`
# CNOT gates, which are often overlooked).
# But logical qubits are a precious resource. Could we implement a variant of the algorithm that uses only
# 500 logical qubits? Yes, we can actively trade qubits for gates by modifying the circuit architecture using the "tunable knobs" we discussed
# earlier.
#
#
# Exploring trade-offs
# ^^^^^^^^^^^^^^^^^^^^
#
# Step 1: Reducing qubits with batching
# -------------------------------------
# Let's first explore the impact of **batched Givens rotations** by varying the number of batches in which rotation angles
# are loaded.
#
# .. note::
#    To strictly isolate the effect of batching, we fix the ``select_swap_depth`` to 4 here.
#    While this does not represent the optimal gate count, it allows us to
#    observe the pure trade-off between batch size and qubit count without confounding factors.
#
# This particular argument is accessible through the :class:`~.pennylane.estimator.templates.SelectTHC` operator as
# ``num_batches``. Let's see how the resources change for FeMoco as we vary this parameter:

batch_sizes = [1, 2, 3, 5, 10, 75]
qubit_counts = []
toffoli_counts = []

for i in batch_sizes:
    prep_thc = qre.PrepTHC(femoco, coeff_precision=n_coeff, select_swap_depth=4)
    select_thc = qre.SelectTHC(femoco, rotation_precision=n_angle, num_batches=i)
    wo_batched = qre.QubitizeTHC(
        femoco,
        prep_op=prep_thc,
        select_op=select_thc,
        coeff_precision=n_coeff,
        rotation_precision=n_angle,
    )

    qpe_cost = qre.estimate(qre.UnaryIterationQPE(wo_batched, n_iter))
    total_cost = qpe_cost.add_parallel(phase_grad_cost)
    qubit_counts.append(total_cost.total_wires)
    toffoli_counts.append(total_cost.gate_counts["Toffoli"])

######################################################################
# Let's visualize the results by plotting the qubit and Toffoli counts against the batch size:
#
# .. figure:: ../_static/demonstration_assets/qubitization_re/batching_tradeoff.jpeg
#    :align: center
#    :width: 85%
#    :target: javascript:void(0)
#
# The plot illustrates a clear crossover in resource requirements. At the left extreme (a single batch),
# we minimize Toffoli counts but require over 1800 logical qubits, which far exceeds
# our hypothetical 500 qubit limit.
# As we increase the number of batches, the qubit count plummets, eventually dipping below
# the 500 qubit limit. However, there is no free lunch: the Toffoli count rises steadily as qubits decrease,
# because we must repeat the QROM readout for every additional batch.
# To verify feasibility, let's examine the resource requirements of the two extremes:
#
print("Resource counts with batch size: 1")
print(f"  Qubits: {qubit_counts[0]}")
print(f"  Toffolis: {toffoli_counts[0]:.3e}\n")
print("Resource counts with batch size: 75")
print(f"  Qubits: {qubit_counts[-1]}")
print(f"  Toffolis: {toffoli_counts[-1]:.3e}\n")

######################################################################
#
# **Crucially, while the qubit requirements are reduced by nearly a factor of 5, the Toffoli count remains in the same order of magnitude.**
# This favorable trade-off allows us to fit the algorithm on constrained hardware without
# making the runtime prohibitively long.
#
#
# Step 2: Circuit optimization with Select-Swap
# -----------------------------------------------
# We have successfully brought the qubit count down using batching. Now, can we optimize the gate count
# without incurring extra qubit costs?
# To do this, we use the **Select-Swap QROM** strategy. Normally, this involves trading qubits for Toffoli gates -
# but we have a useful trick: the register used to store rotation angles in the :class:`~.pennylane.resource.SelectTHC`
# operator is idle during the ``Prepare`` step. We can reuse these idle qubits to implement the
# ``QROM`` for the :class:`~.pennylane.resource.PrepareTHC` operator.
# This should allow us to decrease the Toffoli gates without increasing the logical
# qubit count, at least until we run out of reusable space.
#
# Let's verify this by sweeping through different ``select_swap_depth`` values:

swap_depths = [1, 2, 4, 8, 16]
qubit_counts = []
toffoli_counts = []

for depth in swap_depths:
    select_thc_qrom = qre.SelectTHC(
        femoco, rotation_precision=n_angle, num_batches=10, select_swap_depth=1
    )
    prepare_thc_qrom = qre.PrepTHC(femoco, coeff_precision=n_coeff, select_swap_depth=depth)

    wo_qrom = qre.QubitizeTHC(
        femoco,
        select_op=select_thc_qrom,
        prep_op=prepare_thc_qrom,
    )

    qpe_cost = qre.estimate(qre.UnaryIterationQPE(wo_qrom, n_iter))
    total_cost = qpe_cost.add_parallel(phase_grad_cost)

    qubit_counts.append(total_cost.total_wires)
    toffoli_counts.append(total_cost.gate_counts["Toffoli"])


######################################################################
#
# .. figure:: ../_static/demonstration_assets/qubitization_re/qrom_selswap.jpeg
#    :align: center
#    :width: 90%
#    :target: javascript:void(0)
#
# The data confirms our intuition. For depths 1, 2, and 4, the logical qubit count stays exactly the same, while the
# Toffoli count decreases. However, moving to depth 8, the qubit count jumps as the swap network becomes too large to fit
# entirely within the reused register, forcing the allocation of additional qubits. This marks
# the point where the "free" optimization ends and the standard trade-off resumes.
# To summarize the impact of our optimizations, let's compare the resources required for the naive implementation and our
# final configuration, optimized using a choice of 10 batches and a Select-Swap depth of 4:
#
# .. list-table::
#    :widths: 30 35 35
#    :width: 80%
#    :header-rows: 1
#    :align: center
#
#    * - Configuration
#      - Baseline
#      - Optimized
#    * - Logical Qubits
#      - 2188
#      - 466
#    * - Toffoli Gates
#      - 8.88e10
#      - 3.45e11
#
# Conclusion
# ^^^^^^^^^^
#
# In this demo, we estimated the logical resources needed to simulate FeMoco, a complex molecule central to understanding
# biological nitrogen fixation. Our baseline estimate revealed a requirement of nearly 2000 logical qubits,
# which underscores the magnitude of the challenge facing early fault-tolerant hardware.
#
# However, these calculations tell only half the story. As we demonstrated, these resource requirements are not
# immutable constants. By actively navigating the architectural trade-offs between logical qubits
# and Toffoli gates, we can reshape the cost profile of the algorithm.
#
# This is where the flexibility of PennyLane's resource :mod:`~.pennylane.estimator`
# shines. Rather than treating subroutines like :class:`~.pennylane.estimator.templates.PrepTHC`
# and :class:`~.pennylane.estimator.SelectTHC` as black boxes,
# PennyLane allows us to tune the internal circuit configurations.
# This transforms resource estimation from a passive reporting tool into an active design process,
# enabling researchers to optimize their algorithmic implementations even before the hardware is available.
#
# Now it is your turn! Try plugging in the parameters for a molecule of your interest,
# and experiment with different architectural choices to see if you can simulate your system
# on the near-term hardware. With just a few lines of code, you can start building the blueprint for the
# fault-tolerant algorithms of tomorrow.
#
#
# References
# ^^^^^^^^^^
#
# .. [#Caesura]
#
#    A Caesura et al.
#    "Faster quantum chemistry simulations on a quantum computer with improved tensor factorization and active volume compilation."
#    `arXiv:2501.06165 (2025). <https://arxiv.org/abs/2501.06165>`__
#
# .. [#SOSSA]
#
#    Robbie King et al.
#    "Quantum simulation with sum-of-squares spectral amplification."
#    `arXiv:2505.01528 (2025). <https://arxiv.org/abs/2505.01528>`__
#
# .. [#lee2021]
#
#     Joonho Lee et al.
#     "Even More Efficient Quantum Computations of Chemistry Through Tensor Hypercontraction."
#     `PRX Quantum 2, 030305 (2021). <https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.030305>`__
#
# .. [#vonburg]
#
#     Vera von Burg et al.
#     "Quantum computing enhanced computational catalysis."
#     `arXiv:2007.14460 (2020). <https://arxiv.org/abs/2007.14460>`__
#
