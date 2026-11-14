r"""Simulating Singlet Fission Dynamics with GPU-Accelerated Tensor Networks
=========================================================================================================================

.. tip::
    *This is a joint demo by* `Qoro Quantum <https://qoroquantum.net>`__ *and Xanadu*
"""

######################################################################
# Introduction
# ------------
# 
# What if a single photon could generate *two* electron-hole pairs instead of one? That's the promise
# of `singlet fission <https://pennylane.ai/blog/2025/02/material-discovery-quantum-dynamics/>`__ — a quantum mechanical process in organic semiconductors where a high-energy
# singlet exciton splits into two lower-energy triplet excitons. If harnessed in solar cells, singlet
# fission could push efficiencies past the `Shockley–Queisser limit <https://en.wikipedia.org/wiki/Shockley%E2%80%93Queisser_limit>`__, the theoretical ceiling for
# conventional single-junction devices [#singletfission]_.
# 
# The challenge is that understanding singlet fission requires simulating the coupled dynamics of
# electronic states and nuclear vibrations — a *vibronic* problem where quantum effects are essential.
# In a recent work [#quantumalgorithm]_, a quantum algorithm was developed for simulating vibronic dynamics and applied
# to singlet fission in anthracene dimers, a prototypical organic photovoltaic material. The algorithm
# maps the vibronic Hamiltonian onto qubits and evolves it using `Trotterized time evolution <https://pennylane.ai/codebook/hamiltonian-simulation/trotterization>`__.
# 
# In this demo, we take that quantum algorithm and simulate it classically using `Matrix Product
# State (MPS) <demos/tutorial_mps>`__ tensor networks on `Qoro Quantum's <https://qoroquantum.net>`__
# `Maestro <https://github.com/qoroquantum/maestro>`__ simulator, accessed through PennyLane. We
# show that:
# 
# 1. The vibronic dynamics algorithm from [#quantumalgorithm]_ can be efficiently simulated at scale using MPS methods
# 2. Bond dimension convergence analysis reveals the entanglement structure of the singlet fission
#    process
# 3. **GPU-accelerated MPS** delivers up to **16.3× speedup** over CPU at the bond dimensions required
#    for converged physics
# 
# Along the way, we’ll see how PennyLane’s resource estimation tools can quantify the computational
# cost of the algorithm, and how Maestro’s GPU backend makes large-scale tensor network simulation
# practical.
# 

######################################################################
# The vibronic Hamiltonian
# ------------------------
# 
# The system we’re simulating is an ab initio active-space model of the :math:`\text{(NO)}_4\text{-anthracene}`
# chromophore — an organic photovoltaic material whose 5 electronic states are coupled to 19 dominant
# vibrational (phonon) modes. The vibronic Hamiltonian takes the form [#quantumalgorithm]_:
# 
# .. math::
# 
#    H = H_{\text{el}} \otimes I + \sum_m \frac{\omega_m}{2}(p_m^2 + q_m^2) + \sum_m \kappa_m \otimes q_m
# 
# where:
# 
# * :math:`H_{\text{el}}` is the **electronic Hamiltonian** describing 5 states: the ground
#   state :math:`S_0`, singlet excited state :math:`S_1`, correlated triplet pair :math:`{}^1(TT)`,
#   separated triplets :math:`T_1 T_1`, and charge-separated state :math:`CS`.
# * :math:`\omega_m` are the harmonic **vibrational frequencies** of each mode.
# * :math:`q_m` and :math:`p_m` are the position and momentum operators for mode :math:`m`.
# * :math:`\kappa_m` are the **vibronic coupling tensors**, which encode how electronic transitions
#   are driven by nuclear motion.
# 
# The key physics: starting in the singlet excited state :math:`S_1`, vibronic coupling drives
# population transfer into the triplet-pair state :math:`{}^1(TT)` — this *is* singlet fission. The rate
# and efficiency of this process depend on all 19 vibrational modes, making it a genuinely
# high-dimensional quantum dynamics problem.
# 
# Qubit encoding
# ~~~~~~~~~~~~~~
# 
# The electronic register requires :math:`\lceil \log_2 5 \rceil = 3` qubits. Each vibrational mode is
# discretized on a grid of :math:`2^{n_q}` points, requiring :math:`n_q` qubits per mode. The total
# qubit count is:
# 
# .. math::
# 
# 
#    N = 3 + 19 \times n_q
# 
# For :math:`n_q = 3` (8 grid points per mode): **60 qubits**. For :math:`n_q = 4` (16 grid points per
# mode): **79 qubits**.
# 
# Both are well beyond the reach of exact statevector simulation, but within comfortable range of MPS
# methods — provided we choose the right bond dimension.
# 

######################################################################
# Resource estimation
# -------------------
# 
# Before running any simulation, it’s valuable to understand the computational cost of the algorithm.
# How many gates does each Trotter step require? How does this scale with system size?
# 
# Gate counting for classical simulation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# From the classical simulation perspective, we care about the number of **Pauli rotation gates** per
# Trotter step, since each one translates to tensor network contractions in the MPS backend.
# 
# The vibronic Hamiltonian is decomposed into Pauli terms using PennyLane's :func:`~pennylane.pauli_decompose`. The
# second-order Trotter step (Eq. 5 of [#quantumalgorithm]_) applies:
# 
# 1. A **forward half-step** of potential + coupling terms (Pauli rotations)
# 2. A **full kinetic step** in momentum space (QFT → Pauli rotations → adjoint QFT)
# 3. A **backward half-step** of potential + coupling (reversed)
# 

import pennylane as qp
from vibronic_utils import build_pauli_hamiltonian, electronic_pop_observable, _FREQS, _H_EL, _KAPPA

# Build Pauli decomposition from embedded Hamiltonian data
pot_coup_terms, kinetic_modes = build_pauli_hamiltonian(_FREQS, _H_EL, _KAPPA, n_q=3)

n_pot_coup = len(pot_coup_terms)
n_kinetic = sum(len(kt) for _, kt in kinetic_modes)

print(f"Potential + coupling Pauli terms: {n_pot_coup}")
print(f"Kinetic Pauli terms: {n_kinetic}")
print(f"PauliRots per Trotter step: {2 * n_pot_coup + n_kinetic}")

######################################################################
# Choosing :math:`n_q = 3` instead of :math:`n_q = 4` is a tractability decision: it reduces the gate
# count by **~28%** per step (1,732 vs 2,400 PauliRots) and the qubit count from 79 to 60. Combined
# with fewer Trotter steps (10 vs 20), the total circuit is **~64% shallower**. This keeps the problem
# within reach of MPS simulation while preserving the essential physics of singlet fission.
# 
# The Pauli weight distribution tells us about the entanglement cost:
# 
# ============ ===== ==================
# Pauli Weight Count CNOT Cost per Gate
# ============ ===== ==================
# 1            63    0
# 2            216   2
# 3            314   4
# 4            216   6
# ============ ===== ==================
# 
# The majority of terms are weight-3 and weight-4, arising from the vibronic coupling
# :math:`\kappa_m \otimes q_m` which entangles the 3-qubit electronic register with each 3-qubit mode
# register. These multi-qubit rotations are precisely what generates entanglement across the MPS chain
# and drives up the required bond dimension.
# 

######################################################################
# Simulating with Maestro MPS
# ---------------------------
# 
# The Maestro device
# ~~~~~~~~~~~~~~~~~~
# 
# `Maestro <https://github.com/qoroquantum/maestro>`__ is `Qoro Quantum's <https://qoroquantum.net>`__ high-performance quantum
# circuit simulator, available as a PennyLane plugin. It supports multiple simulation backends
# including statevector, :doc:`MPS <demos/tutorial_mps>`, :doc:`stabilizer <demos/tutorial_clifford_circuit_simulations>`, and :doc:`Pauli propagation <demos/tutorial_classical_expval_estimation>`. For this demo, we use the **Matrix
# Product State (MPS)** backend, which represents the quantum state as a chain of tensors with tunable
# bond dimension :math:`\chi`.
# 
# The key advantage of MPS: while a full statevector for 60 qubits would require
# :math:`2^{60} \approx 10^{18}` complex amplitudes (exabytes of memory), an MPS with bond dimension
# :math:`\chi = 256` uses only :math:`60 \times 256^2 \times 2 \approx 8 \times 10^6` parameters
# (~64 MB at double precision) — a compression factor of :math:`10^{11}`.
# 
# This demo requires an additional package beyond PennyLane:
# 
# -  `pennylane-maestro <https://github.com/qoroquantum/pennylane-maestro>`__ — the PennyLane
#    plugin for Maestro
# 
# .. code:: bash
# 
#    pip install pennylane-maestro
# 
# Once installed, setting up the Maestro device in PennyLane is straightforward:
# 

# (pennylane already imported above as qp)

# CPU MPS backend
dev = qp.device(
    "maestro.qubit",
    wires=60,
    simulator_type="QCSim",
    simulation_type="MatrixProductState",
    max_bond_dimension=256,
)

# GPU MPS backend — same interface, just change simulator_type
dev_gpu = qp.device(
    "maestro.qubit",
    wires=60,
    simulator_type="Gpu",
    simulation_type="MatrixProductState",
    max_bond_dimension=256,
)

######################################################################
# The circuit construction uses PennyLane's standard API. Each Trotter step is built from
# ``qp.PauliRot`` gates for the Hamiltonian terms and ``qp.QFT`` / ``qp.adjoint(QFT)`` for the kinetic
# energy in momentum space.
#
# .. note::
#
#     The simulation results shown below are precomputed from dedicated benchmark runs — while
#     Maestro's GPU backend completes the full 60-qubit simulation at :math:`\chi = 256` in just
#     23 minutes (and under 10 minutes at :math:`\chi \le 64`), running on a CPU takes over 6 hours,
#     exceeding standard documentation build limits. The circuit definition and gate-counting
#     code above execute live to verify the algorithm structure.
#

n_steps = 10
dt = 1.0
sample_steps = set(range(1, n_steps + 1))
state_projectors = [electronic_pop_observable(s) for s in range(5)]


@qp.qnode(dev)
def circuit():
    # Initialize in S₁ (singlet excited state)
    qp.PauliX(wires=2)

    # Time evolution via second-order Trotter with intermediate snapshots
    for step in range(1, n_steps + 1):
        # Forward half-step: potential + coupling
        for _, coeff, pauli_word, wires in pot_coup_terms:
            if abs(coeff * dt) > 1e-8:
                qp.PauliRot(coeff * dt, pauli_word, wires=wires)

        # Full kinetic step in momentum basis
        for mode_wires, ke_terms in kinetic_modes:
            qp.QFT(wires=mode_wires)
            qp.PauliX(wires=mode_wires[0])
            for coeff, pauli_word, wires in ke_terms:
                if abs(2 * coeff * dt) > 1e-8:
                    qp.PauliRot(2 * coeff * dt, pauli_word, wires=wires)
            qp.PauliX(wires=mode_wires[0])
            qp.adjoint(qp.QFT)(wires=mode_wires)

        # Backward half-step: potential + coupling (reversed)
        for _, coeff, pauli_word, wires in reversed(pot_coup_terms):
            if abs(coeff * dt) > 1e-8:
                qp.PauliRot(coeff * dt, pauli_word, wires=wires)

        # Capture intermediate populations without rebuilding circuits
        if step in sample_steps:
            for s, projector in enumerate(state_projectors):
                qp.Snapshot(f"step_{step}_s{s}", measurement=qp.expval(projector))

    # Measure electronic state populations
    return [qp.expval(projector) for projector in state_projectors]

######################################################################
# .. note::
#
#     **PennyLane Snapshot Acceleration**: In conventional simulation workflows, measuring dynamics
#     at :math:`K` intermediate time points requires running :math:`K` separate circuits from :math:`t=0`,
#     causing total gates to scale quadratically as :math:`\mathcal{O}(N^2)`. By embedding ``qp.Snapshot``
#     inside the Trotter loop, ``pennylane-maestro`` automatically executes the evolution via Maestro's
#     persistent-state simulation engine, keeping the MPS in memory and achieving true linear
#     :math:`\mathcal{O}(N)` scaling.
#
# To execute all time points in a single pass on the Maestro MPS backend:
#
# .. code-block:: python
#
#     # Execute all time points via Maestro persistent evolution
#     results = qp.snapshots(circuit)()
#
#     # Extract populations at final step (t = 10)
#     final_pops = [results[f"step_{n_steps}_s{s}"] for s in range(5)]
#     for label, pop in zip(["S₀", "S₁", "¹TT", "T₁T₁", "CS"], final_pops):
#         print(f"{label}: {pop:.4f}")

######################################################################
# Bond dimension convergence
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# The critical question for any MPS simulation: *how large does the bond dimension need to be?* If
# :math:`\chi` is too small, the MPS truncates entanglement and gives incorrect dynamics. If it’s
# unnecessarily large, we waste computation.
# 
# We ran the full 60-qubit, 19-mode system at :math:`\chi = 32, 64, 128, 256` and tracked the
# electronic state populations over 10 a.u. of evolution time:
# 
# .. figure:: ../_static/demonstration_assets/singlet_fission_tensor_networks/vibronic_gpu_convergence.png
#    :alt: Bond dimension convergence
# 
# *Bond dimension convergence showing S₁ decay and ¹TT growth across χ = 32, 64, 128, 256. The curves
# separate at low χ and converge at χ ≥ 128.*
# 
# The results tell a clear story:
# 
# ============ =================== ======================== ===================
# :math:`\chi` :math:`S_1` (final) :math:`{}^1(TT)` (final) Converged?
# ============ =================== ======================== ===================
# 32           0.502               0.289                    ✗
# 64           0.388               0.288                    ✗
# 128          0.325               0.278                    ~
# 256          0.299               0.266                    :math:`t \le 4`
# ============ =================== ======================== ===================
# 
# At :math:`\chi = 32`, the :math:`S_1` population is **~68% higher** than the :math:`\chi=256` value — a
# qualitatively wrong picture of the singlet fission dynamics. With :math:`\chi = 256`, the dynamics
# are strictly converged for the initial fission event (:math:`t \le 4` a.u., where differences from
# :math:`\chi=128` remain :math:`< 0.002`). At later times (:math:`t > 6` a.u.), ongoing entanglement
# growth leaves a modest residual drift (:math:`\Delta S_1 \approx 0.026` between :math:`\chi=128` and
# :math:`\chi=256`), providing qualitative stability while highlighting where tensor networks begin to
# encounter the entanglement barrier.
# 
# This makes physical sense: the vibronic coupling tensors :math:`\kappa_m` have off-diagonal elements
# that entangle the electronic register with all 19 mode registers simultaneously. Combined with the
# QFT operations (which create long-range correlations within each mode), the entanglement entropy
# grows enough to require :math:`\chi \sim 200` for faithful representation.
# 

######################################################################
# GPU acceleration: where it matters
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# The GPU advantage depends strongly on bond dimension. Each gate in the MPS simulation involves contracting and
# re-decomposing tensors of dimension :math:`\chi \times \chi`. At small :math:`\chi`, these are tiny
# matrix operations where GPU kernel launch overhead dominates. But at large :math:`\chi`, they become
# substantial linear algebra problems — exactly what GPUs are built for.
# 
# We ran the identical convergence study on both CPU and GPU backends:
# 
# ============ ======== ======== =================
# :math:`\chi` CPU Time GPU Time GPU Speedup
# ============ ======== ======== =================
# 32           3.0 min  6.6 min  0.5× (CPU faster)
# 64           13.6 min 9.3 min  **1.5×**
# 128          1.1 h    13.8 min **4.8×**
# 256          6.3 h    23.0 min **16.3×**
# ============ ======== ======== =================
# 
# The hardware used for this experiment was based on standard VMs on Google Cloud:
# 
# -  CPU: c2-standard-30 (30 vCPUs, 120 GB memory)
# -  GPU: a2-highgpu-1g (12 vCPUs, 85 GB memory, 1 NVIDIA A100 40 GB)
# 
# .. figure:: ../_static/demonstration_assets/singlet_fission_tensor_networks/vibronic_cpu_vs_gpu.png
#    :alt: CPU vs GPU comparison
# 
# *CPU vs GPU wall-clock time comparison across bond dimensions. GPU becomes faster already at χ = 64,
# reaching 16.3× speedup at χ = 256.*
# 
# The crossover happens between :math:`\chi = 32` and :math:`\chi = 64`. Below that, CPU wins because
# the tensor operations are too small to justify GPU overhead. Above it, GPU advantage grows rapidly:
# 
# -  **CPU scales ~5× per doubling** of :math:`\chi` (approaching the :math:`O(\chi^3)` theoretical
#    cost)
# -  **GPU scales ~1.6× per doubling** of :math:`\chi` (parallelism absorbs the cubic growth)
# 
# At :math:`\chi = 256` — the bond dimension needed for converged physics — GPU turns a **6.3-hour CPU
# job into a 23-minute GPU run** (**16.3× speedup**). For research workflows where you need to iterate on parameters, run
# convergence studies, or explore different molecular systems, this speedup becomes essential.
# 

######################################################################
# Singlet fission dynamics
# ~~~~~~~~~~~~~~~~~~~~~~~~
# 
# Putting it all together, the high-precision simulation (:math:`\chi = 256`) reveals the following
# dynamics of singlet fission in the anthracene dimer:
# 
# .. figure:: ../_static/demonstration_assets/singlet_fission_tensor_networks/vibronic_gpu_populations.png
#    :alt: Population dynamics
# 
# *Population dynamics of the five electronic states over 10 a.u. of evolution, showing S₁ → ¹TT
# singlet fission and subsequent vibrational redistribution.*
# 
# Starting from the photoexcited :math:`S_1` state:
# 
# 1. **Rapid initial decay** (0–2 a.u.): :math:`S_1` drops from 1.0 to ~0.45, with population flowing
#    primarily into the triplet-pair state :math:`{}^1(TT)`. This is the singlet fission event.
# 
# 2. **Vibrational redistribution** (2–6 a.u.): Population oscillates between :math:`S_1` and
#    :math:`{}^1(TT)` as vibrational modes exchange energy with the electronic subsystem. The
#    charge-separated state :math:`CS` gradually accumulates population.
# 
# 3. **Quasi-equilibrium** (6–10 a.u.): The system approaches a quasi-steady state with
#    :math:`S_1 \approx 0.30`, :math:`{}^1(TT) \approx 0.27`, and significant population in
#    :math:`CS \approx 0.22`.
# 
# The trace (sum of all populations) is preserved to better than :math:`\Sigma = 0.9997` throughout
# (reaching :math:`\Sigma = 0.9998` at :math:`t = 10`), confirming the accuracy of the MPS simulation.
# 

######################################################################
# Where classical simulation meets its limits — and where QPUs become essential
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# While GPU-accelerated MPS efficiently simulates this 60-qubit active-space model, classical tensor
# networks face steep scaling limits. Quantum hardware becomes indispensable across three key frontiers:
# 
# -  **Simulation time**: Entanglement entropy grows linearly with time (:math:`S \propto t`), causing
#    the required bond dimension to explode exponentially (:math:`\chi \propto e^{vt}`). We observe this
#    directly in our convergence data: while :math:`\chi = 256` strictly converges the initial fission event
#    (:math:`t \le 4` a.u.), ongoing entanglement growth introduces residual drift at later times.
#    Pushing to longer timescales (relaxation, transport) demands quantum computers, where circuit depth
#    scales linearly (:math:`\mathcal{O}(t)`).
# 
# -  **System size & environment**: 19 modes capture the active space of an isolated chromophore.
#    Scaling to 50+ modes (~150–200+ qubits) to capture secondary vibrational couplings and environmental
#    baths rapidly overwhelms 1D tensor networks with dense long-range entanglement, with full interfaces
#    reaching hundreds of modes (such as the 246-mode, 1,053-qubit system in Motlagh et al. [#quantumalgorithm]_).
# 
# -  **Grid discretization**: Increasing resolution (:math:`n_q = 4`, 79 qubits) doubles the local
#    Hilbert space, rapidly compounding entanglement per Trotter step.
# 
# .. list-table::
#    :header-rows: 1
# 
#    * - Regime
#      - Method
#      - Qubits / Modes
#      - Timescale
#      - Bottleneck
#    * - **Exact Statevector**
#      - Classical CPU/GPU
#      - < 30 qubits (<= 6 modes)
#      - Any
#      - Memory (2^N)
#    * - **Active-Space MPS**
#      - **Maestro + PennyLane**
#      - **30–80 qubits (~19 modes)**
#      - **Ultrafast (<= 100 fs)**
#      - **Bond dimension growth**
#    * - **Complex Systems / Bulk**
#      - **Fault-Tolerant QPU**
#      - **150 to 1,000+ qubits (50+ modes)**
#      - **Long-time (ps+)**
#      - **Physical gate depth / errors**
# 
# GPU-accelerated MPS pushes the classical baseline for molecular active spaces, clarifying exactly where
# quantum advantage begins for full-scale material simulation.
# 

######################################################################
# Conclusion
# ----------
# 
# In this demo, we simulated the active-space vibronic dynamics of singlet fission in an organic
# chromophore using classical tensor network methods on Qoro Quantum’s Maestro simulator via PennyLane.
# The key takeaways:
# 
# -  **Tensor networks can simulate quantum algorithms at scale.** The 60-qubit, 19-mode singlet
#    fission circuit runs efficiently with MPS, enabling detailed convergence analysis and parameter
#    exploration that would be impractical with exact statevector methods.
# 
# -  **Bond dimension matters.** The vibronic coupling structure of singlet fission generates enough
#    entanglement to require :math:`\chi \geq 128` for qualitatively sound dynamics, with :math:`\chi = 256`
#    achieving strict convergence for the initial fission transition (:math:`t \le 4` a.u.) — low bond
#    dimension gives qualitatively incorrect dynamics.
# 
# -  **GPU acceleration is essential at high bond dimension.** At :math:`\chi = 256`, Maestro’s GPU
#    backend delivers 16.3× speedup over CPU, reducing simulation time from 6.3 hours to 23 minutes. The
#    advantage grows with :math:`\chi`, making GPU-accelerated MPS the practical choice for
#    production-quality tensor network simulations.
# 
# -  **Classical baselines sharpen quantum advantage.** By pushing MPS simulation to its limits on
#    active spaces, we establish a rigorous classical baseline that clarifies the boundary for quantum
#    computing: while GPU-accelerated tensor networks can efficiently tackle isolated molecules at
#    ultrafast timescales, full condensed-phase environments and long-time transport remain the essential
#    domain of fault-tolerant quantum hardware.
# 
# -  **PennyLane makes it seamless.** Switching between CPU and GPU backends requires changing a
#    single parameter (``simulator_type``). The same PennyLane circuit code runs on statevector, CPU
#    MPS, and GPU MPS — enabling rapid prototyping and cross-validation.
#

######################################################################
# References
# ----------
# 
# .. [#singletfission]
# 
#     M. B. Smith and J. Michl,
#     "Singlet Fission",
#     *Chem. Rev.* **110**, 6891 (2010).
#     `DOI: 10.1021/cr1002613 <https://doi.org/10.1021/cr1002613>`__.
# 
# .. [#quantumalgorithm]
# 
#     D. Motlagh, R. A. Lang, W. Maxwell, T. Zeng, P. Jain, J. A. Campos-Gonzalez-Angulo, A. Aspuru-Guzik, and J. M. Arrazola,
#     "Quantum Algorithm for Vibronic Dynamics: Case Study on Singlet Fission Solar Cell Design",
#     `arXiv:2411.13669 <https://arxiv.org/abs/2411.13669>`__, 2024.
#