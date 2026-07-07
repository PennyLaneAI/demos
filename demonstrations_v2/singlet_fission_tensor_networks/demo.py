r"""Simulating Singlet Fission Dynamics with Tensor Networks: From Quantum Algorithms to GPU-Accelerated Classical Simulation
=========================================================================================================================

*A joint demo by Qoro Quantum and Xanadu*
"""

######################################################################
# Introduction
# ------------
# 
# What if a single photon could generate *two* electron-hole pairs instead of one? That’s the promise
# of **singlet fission** — a quantum mechanical process in organic semiconductors where a high-energy
# singlet exciton splits into two lower-energy triplet excitons. If harnessed in solar cells, singlet
# fission could push efficiencies past the Shockley-Queisser limit, the theoretical ceiling for
# conventional single-junction devices [1].
# 
# But here’s the catch: understanding singlet fission requires simulating the coupled dynamics of
# electronic states and nuclear vibrations — a *vibronic* problem where quantum effects are essential.
# In a recent work [2], a quantum algorithm was developed for simulating vibronic dynamics and applied
# to singlet fission in anthracene dimers, a prototypical organic photovoltaic material. The algorithm
# maps the vibronic Hamiltonian onto qubits and evolves it using Trotterized time evolution.
# 
# In this demo, we take that quantum algorithm and simulate it classically using **Matrix Product
# State (MPS)** tensor networks on Qoro Quantum’s
# `Maestro <https://www.github.com/qoroquantum/maestro>`__ simulator, accessed through PennyLane. We
# show that:
# 
# 1. The vibronic dynamics algorithm from [2] can be efficiently simulated at scale using MPS methods
# 2. Bond dimension convergence analysis reveals the entanglement structure of the singlet fission
#    process
# 3. **GPU-accelerated MPS** delivers up to **6.2× speedup** over CPU at the bond dimensions required
#    for converged physics
# 
# Along the way, we’ll see how PennyLane’s resource estimation tools can quantify the computational
# cost of the algorithm, and how Maestro’s GPU backend makes large-scale tensor network simulation
# practical.
# 

######################################################################
# The Vibronic Hamiltonian
# ------------------------
# 
# The system we’re simulating is an anthracene dimer — two anthracene molecules whose electronic
# states are coupled to 19 vibrational (phonon) modes. The vibronic Hamiltonian takes the form [2]:
# 
# .. math::
# 
# 
#    H = H_{\text{el}} \otimes I + \sum_m \frac{\omega_m}{2}(p_m^2 + q_m^2) + \sum_m \kappa_m \otimes q_m
# 
# where: - :math:`H_{\text{el}}` is the **electronic Hamiltonian** describing 5 states: the ground
# state :math:`S_0`, singlet excited state :math:`S_1`, correlated triplet pair :math:`^1(TT)`,
# separated triplets :math:`T_1 T_1`, and charge-separated state :math:`CS` - :math:`\omega_m` are the
# harmonic **vibrational frequencies** of each mode - :math:`q_m` and :math:`p_m` are the position and
# momentum operators for mode :math:`m` - :math:`\kappa_m` are the **vibronic coupling tensors** —
# these encode how electronic transitions are driven by nuclear motion
# 
# The key physics: starting in the singlet excited state :math:`S_1`, vibronic coupling drives
# population transfer into the triplet-pair state :math:`^1(TT)` — this *is* singlet fission. The rate
# and efficiency of this process depend sensitively on all 19 vibrational modes, making it a genuinely
# high-dimensional quantum dynamics problem.
# 
# Qubit Encoding
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
# Resource Estimation
# -------------------
# 
# Before running any simulation, it’s valuable to understand the computational cost of the algorithm.
# How many gates does each Trotter step require? How does this scale with system size?
# 
# Gate Counting for Classical Simulation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# From the classical simulation perspective, we care about the number of **Pauli rotation gates** per
# Trotter step, since each one translates to tensor network contractions in the MPS backend.
# 
# The vibronic Hamiltonian is decomposed into Pauli terms using PennyLane’s ``pauli_decompose``. The
# second-order Trotter step (Eq. 5 of [2]) applies:
# 
# 1. A **forward half-step** of potential + coupling terms (Pauli rotations)
# 2. A **full kinetic step** in momentum space (QFT → Pauli rotations → iQFT)
# 3. A **backward half-step** of potential + coupling (reversed)
# 

import numpy as np
import pennylane as qml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =====================================================================
# Constants
# =====================================================================

STATE_LABELS = ["S₀", "S₁", "¹TT", "T₁T₁", "CS"]
STATE_COLORS = ["#2d3436", "#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
N_STATES = 5
N_EL = 3  # ceil(log2(5)) = 3 qubits for electronic register

# =====================================================================
# Pre-computed Hamiltonian data for 19 strongest vibrational modes
# of the anthracene dimer (from arXiv:2411.13669)
# =====================================================================

_FREQS = np.array([
    0.03578249487813, 0.05873191729548, 0.06594583484775, 0.13219188100833,
    0.15137422752738, 0.15920649163848, 0.19255756165074, 0.20687542312866,
    0.17194162530831, 0.18889706105985, 0.20533325818434, 0.10062141371997,
    0.15998005992969, 0.19512658063041, 0.06017811474675, 0.12776997342252,
    0.15252188368989, 0.17320466066904, 0.17783003347395,
])

_H_EL = np.array([
    [ 0.000000000, -0.201832325,  0.000000000,  0.000000000,  0.000000000],
    [-0.201832325,  2.233503436,  0.000000000,  0.000000000,  0.000000000],
    [ 0.000000000,  0.000000000,  2.767689809,  0.000000000,  0.000000000],
    [ 0.000000000,  0.000000000,  0.000000000,  2.801189770,  0.000000000],
    [ 0.000000000,  0.000000000,  0.000000000,  0.000000000,  3.161450102],
])

_KAPPA = np.array([
    [[ 1.35063174e-02,  1.92369935e-03, -1.91538358e-02,  9.78518673e-03,
       3.17519552e-02, -1.09525310e-02, -2.35132039e-02,  5.01132956e-03,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
     [-1.63408000e-02, -1.22641000e-03, -1.57635000e-02, -2.29069000e-02,
      -7.70572000e-03,  9.21896000e-02,  3.23569000e-02,  2.36942000e-02,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
       7.90575000e-02, -1.32722000e-02, -9.64010000e-02,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  3.06805000e-02,
      -2.32997000e-02, -2.41684000e-02,  0.00000000e+00,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  6.22161000e-02, -9.54081000e-02,
      -3.22895000e-01,  5.82345000e-02,  4.38246000e-02]],
    [[-1.63408000e-02, -1.22641000e-03, -1.57635000e-02, -2.29069000e-02,
      -7.70572000e-03,  9.21896000e-02,  3.23569000e-02,  2.36942000e-02,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
     [ 6.45562725e-02,  8.27697164e-02, -1.36982688e-02,  8.62196427e-02,
       1.38434002e-01, -2.24468417e-01, -7.57498341e-02, -1.23543520e-01,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
      -9.07046000e-02,  5.28769000e-02,  1.56963000e-01,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  2.76204000e-02,
      -1.07366000e-02, -4.98571000e-02,  0.00000000e+00,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  1.23914000e-02, -1.00906000e-02,
      -8.83992000e-02, -2.10482000e-04,  5.75267000e-03]],
    [[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
       7.90575000e-02, -1.32722000e-02, -9.64010000e-02,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
      -9.07046000e-02,  5.28769000e-02,  1.56963000e-01,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
     [ 2.49920331e-02,  6.16755610e-02,  3.23331109e-02,  3.64320824e-02,
       5.66157776e-03, -1.20296584e-01,  2.74897101e-02, -2.88676034e-02,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  1.04710000e-01, -4.52008000e-02,
      -2.85642000e-02,  8.87648000e-02, -3.70079000e-02],
     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.27972000e-02,
      -4.67057000e-02, -1.20023000e-01,  0.00000000e+00,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00]],
    [[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  3.06805000e-02,
      -2.32997000e-02, -2.41684000e-02,  0.00000000e+00,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  2.76204000e-02,
      -1.07366000e-02, -4.98571000e-02,  0.00000000e+00,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  1.04710000e-01, -4.52008000e-02,
      -2.85642000e-02,  8.87648000e-02, -3.70079000e-02],
     [-8.80993650e-03,  8.00110783e-02,  7.66765411e-02,  2.15016704e-02,
      -9.19148851e-03,  8.42125922e-03,  8.41615186e-02,  2.48395321e-02,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
       6.50941000e-02,  9.04485000e-02, -4.44717000e-02,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00]],
    [[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  6.22161000e-02, -9.54081000e-02,
      -3.22895000e-01,  5.82345000e-02,  4.38246000e-02],
     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  1.23914000e-02, -1.00906000e-02,
      -8.83992000e-02, -2.10482000e-04,  5.75267000e-03],
     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.27972000e-02,
      -4.67057000e-02, -1.20023000e-01,  0.00000000e+00,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
       6.50941000e-02,  9.04485000e-02, -4.44717000e-02,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
     [ 5.60436170e-02,  2.04079317e-02,  7.25826237e-03,  4.89400782e-02,
       2.26525477e-02, -2.18039659e-01, -1.13237518e-01,  1.85791714e-02,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
       0.00000000e+00,  0.00000000e+00,  0.00000000e+00]],
])


def _op_to_pauliword(op):
    """Convert a PennyLane Pauli operator to (word, wires) tuple."""
    if isinstance(op, qml.Identity):
        return None, None
    if isinstance(op, (qml.PauliX, qml.PauliY, qml.PauliZ)):
        return op.name[-1], list(op.wires)
    if hasattr(op, "operands"):
        word, wires = "", []
        for o in op.operands:
            if isinstance(o, qml.Identity):
                continue
            w, ws = _op_to_pauliword(o)
            if w is not None:
                word += w
                wires.extend(ws)
        return (word, wires) if word else (None, None)
    return None, None


def _decompose_matrix(mat, wire_order):
    """Decompose Hermitian matrix into (coeff, pauli_word, wires) tuples."""
    decomp = qml.pauli_decompose(mat, wire_order=wire_order)
    terms = []
    for c, op in zip(decomp.coeffs, decomp.ops):
        if abs(c) < 1e-12:
            continue
        pw, ws = _op_to_pauliword(op)
        if pw is None:
            continue
        terms.append((float(c.real), pw, ws))
    return terms


def _coalesce_terms(terms):
    """Merge Pauli terms with identical (pauli_word, wires) by summing coeffs.

    Reduces gate count by ~17% for the singlet fission Hamiltonian.
    """
    merged = {}
    for label, coeff, pw, ws in terms:
        key = (pw, tuple(ws))
        if key in merged:
            merged[key] = (merged[key][0], merged[key][1] + coeff, pw, ws)
        else:
            merged[key] = (label, coeff, pw, ws)
    return [v for v in merged.values() if abs(v[1]) > 1e-12]


def build_pauli_hamiltonian(freqs, H_el, kappa, n_q):
    """Decompose the full vibronic Hamiltonian into Pauli rotation terms.

    The vibronic Hamiltonian (arXiv:2411.13669) is:
        H = H_el ⊗ I + Σ_m (ω_m/2)(p_m² + q_m²) + Σ_m κ_m ⊗ q_m

    This is split into potential+coupling fragments (position-basis diagonal
    or Pauli-decomposable) and a kinetic fragment (diagonal in momentum
    basis, accessed via QFT).

    Parameters
    ----------
    freqs : (n_modes,) — harmonic frequencies (a.u.)
    H_el  : (5, 5)    — electronic Hamiltonian
    kappa : (5, 5, n_modes) — vibronic coupling tensors
    n_q   : int        — qubits per vibrational mode

    Returns
    -------
    pot_coup_terms : list of (label, coeff, pauli_word, wires)
        Electronic + potential + coupling Pauli terms.
    kinetic_modes  : list of (mode_wires, ke_pauli_terms)
        Per-mode kinetic energy Pauli terms (applied in Fourier basis).
    """
    n_modes = len(freqs)
    grid = 2**n_q
    x_mat = np.diag([float(k - grid // 2) for k in range(grid)])
    x2_mat = x_mat @ x_mat

    pot_coup_terms = []

    # 1) Electronic Hamiltonian on qubits [0, 1, 2]
    H_el_full = np.zeros((2**N_EL, 2**N_EL), dtype=complex)
    H_el_full[:N_STATES, :N_STATES] = H_el
    for c, pw, ws in _decompose_matrix(H_el_full, list(range(N_EL))):
        pot_coup_terms.append(("el", c, pw, ws))

    # 2) Per-mode: harmonic potential + vibronic coupling
    kinetic_modes = []
    for m in range(n_modes):
        mode_wires = list(range(N_EL + m * n_q, N_EL + (m + 1) * n_q))
        el_wires = list(range(N_EL))

        # Harmonic potential: ω q²/2  (position-basis diagonal)
        pot_mat = freqs[m] * x2_mat / 2.0
        for c, pw, ws in _decompose_matrix(pot_mat.astype(complex), mode_wires):
            pot_coup_terms.append(("pot", c, pw, ws))

        # Vibronic coupling: κ ⊗ q
        K = np.zeros((2**N_EL, 2**N_EL), dtype=complex)
        K[:N_STATES, :N_STATES] = kappa[:, :, m]
        coup_mat = np.kron(K, x_mat)
        for c, pw, ws in _decompose_matrix(
            coup_mat.astype(complex), el_wires + mode_wires
        ):
            pot_coup_terms.append(("coup", c, pw, ws))

        # Kinetic energy: ω p²/2  (same x² form, but applied in momentum basis)
        ke_mat = freqs[m] * x2_mat / 2.0
        ke_terms = _decompose_matrix(ke_mat.astype(complex), mode_wires)
        kinetic_modes.append((mode_wires, ke_terms))

    # Coalesce terms with same (pauli_word, wires) to reduce gate count
    pot_coup_terms = _coalesce_terms(pot_coup_terms)

    return pot_coup_terms, kinetic_modes


# =====================================================================
# Circuit building blocks
# =====================================================================

def apply_qft(wires):
    """QFT using only native gates (no Adjoint wrappers)."""
    n = len(wires)
    for j in range(n):
        qml.Hadamard(wires=wires[j])
        for k in range(j + 1, n):
            qml.ControlledPhaseShift(
                np.pi / 2 ** (k - j), wires=[wires[k], wires[j]]
            )
    for i in range(n // 2):
        qml.SWAP(wires=[wires[i], wires[n - 1 - i]])


def apply_iqft(wires):
    """Inverse QFT using only native gates (no Adjoint wrappers).

    QFT = S · U  =>  QFT† = U† · S  (since S† = S for SWAPs).
    U† reverses the gate order and negates ControlledPhaseShift angles.
    """
    n = len(wires)
    # SWAP layer first (same as forward)
    for i in range(n // 2):
        qml.SWAP(wires=[wires[i], wires[n - 1 - i]])
    # Reversed H + CPhase with negated angles
    for j in range(n - 1, -1, -1):
        for k in range(n - 1, j, -1):
            qml.ControlledPhaseShift(
                -np.pi / 2 ** (k - j), wires=[wires[k], wires[j]]
            )
        qml.Hadamard(wires=wires[j])


def apply_trotter_step(pot_coup_terms, kinetic_modes, dt):
    """One second-order Trotter step (Eq. 5 of arXiv:2411.13669).

    Implements:
        U₂(dt) = [∏_m exp(-i H_m dt/2)] · exp(-i H_T dt) · [∏_m exp(-i H_m dt/2)]†

    where H_m are potential+coupling fragments and H_T is kinetic energy.
    The kinetic energy is applied in momentum space via QFT.
    """
    # ── Forward half-step: potential + coupling ──
    for _, coeff, pw, ws in pot_coup_terms:
        theta = coeff * dt
        if abs(theta) > 1e-8:
            qml.PauliRot(theta, pw, wires=ws)

    # ── Full kinetic energy step (momentum basis via QFT) ──
    for mode_wires, ke_terms in kinetic_modes:
        # Transform to momentum basis
        apply_qft(mode_wires)
        # X on MSB to center the momentum grid around zero
        qml.PauliX(wires=mode_wires[0])
        # Apply kinetic diagonal: ω/2 · p² (full step, factor of 2 in theta)
        for coeff, pw, ws in ke_terms:
            theta = 2.0 * coeff * dt
            if abs(theta) > 1e-8:
                qml.PauliRot(theta, pw, wires=ws)
        # Undo X on MSB
        qml.PauliX(wires=mode_wires[0])
        # Transform back to position basis
        apply_iqft(mode_wires)

    # ── Backward half-step: potential + coupling (reversed) ──
    for _, coeff, pw, ws in reversed(pot_coup_terms):
        theta = coeff * dt
        if abs(theta) > 1e-8:
            qml.PauliRot(theta, pw, wires=ws)


def electronic_pop_observable(state_idx):
    """Build projector |state⟩⟨state| as a Pauli Hamiltonian on el wires."""
    el_wires = list(range(N_EL))
    coeffs, ops = [], []
    for mask in range(2**N_EL):
        c = 1.0 / (2**N_EL)
        paulis = []
        for w in range(N_EL):
            if (mask >> w) & 1:
                bit_pos = N_EL - 1 - w
                sign = -1 if ((state_idx >> bit_pos) & 1) else 1
                c *= sign
                paulis.append(qml.PauliZ(el_wires[w]))
        if not paulis:
            ops.append(qml.Identity(el_wires[0]))
        elif len(paulis) == 1:
            ops.append(paulis[0])
        else:
            ops.append(qml.prod(*paulis))
        coeffs.append(c)
    return qml.Hamiltonian(coeffs, ops)


# =====================================================================
# Plotting utilities
# =====================================================================

BG_COLOR = "#0d1117"
LEGEND_KWARGS = dict(
    fontsize=11, framealpha=0.3, facecolor="#1a1a2e",
    edgecolor="#444", labelcolor="white",
)


def _style_ax(ax):
    """Apply dark-theme styling to an axis."""
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors="white", labelsize=10)
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.grid(True, alpha=0.15, color="white")


def _dark_fig(*args, **kwargs):
    """Create a figure with dark background."""
    fig, axes = plt.subplots(*args, **kwargs)
    fig.patch.set_facecolor(BG_COLOR)
    return fig, axes


def plot_populations(times, populations, title, filename, meta=""):
    """Publication-quality electronic state population dynamics plot.

    Parameters
    ----------
    times : (n_steps+1,) array
    populations : (n_steps+1, N_STATES) array
    title : str — plot title
    filename : str or Path — output file
    meta : str — annotation text (bottom-left corner)
    """
    fig, ax = _dark_fig(figsize=(12, 6))
    _style_ax(ax)

    for s in range(N_STATES):
        ax.plot(times, populations[:, s], color=STATE_COLORS[s],
                linewidth=2.5, label=STATE_LABELS[s],
                marker="o", markersize=4, alpha=0.9)

    # Sum conservation band
    total = populations.sum(axis=1)
    ax.fill_between(times, total - 0.001, total + 0.001,
                    alpha=0.1, color="white",
                    label=f"Σ = {total[-1]:.4f}")

    ax.set_xlabel("Time (a.u.)", fontsize=14, color="white")
    ax.set_ylabel("Electronic State Population", fontsize=14, color="white")
    ax.set_title(title, fontsize=16, color="white", fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(times[0], times[-1])
    ax.legend(fontsize=12, loc="upper right", **{
        k: v for k, v in LEGEND_KWARGS.items() if k != "fontsize"
    })

    if meta:
        ax.text(0.02, 0.02, meta, transform=ax.transAxes,
                fontsize=10, color="#888", va="bottom")

    plt.tight_layout()
    plt.savefig(filename, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Plot saved: {filename}")


def plot_convergence(times, results_by_chi, suptitle, filename):
    """Two-panel convergence plot: S₁ decay and ¹TT growth vs χ.

    Parameters
    ----------
    times : (n_steps+1,) array
    results_by_chi : dict of {chi_or_"exact": (n_steps+1, N_STATES) array}
    suptitle : str — figure super-title
    filename : str or Path — output file
    """
    fig, axes = _dark_fig(1, 2, figsize=(16, 6))
    cmap = plt.cm.viridis

    chi_values = sorted(
        (k for k in results_by_chi if k != "exact"),
    ) + (["exact"] if "exact" in results_by_chi else [])

    panels = [
        (axes[0], 1, "S₁ Decay — Bond Dimension Convergence"),
        (axes[1], 2, "¹TT Growth — Singlet Fission Signature"),
    ]
    for ax, state_idx, panel_title in panels:
        _style_ax(ax)
        for i, chi in enumerate(chi_values):
            color = cmap(i / max(len(chi_values) - 1, 1))
            label = f"χ = {chi}" if chi != "exact" else "Exact (SV)"
            ls = "-" if chi != "exact" else "--"
            lw = 2.0 if chi != "exact" else 2.5
            ax.plot(times, results_by_chi[chi][:, state_idx],
                    color=color, linewidth=lw, linestyle=ls,
                    label=label, alpha=0.9)
        ax.set_xlabel("Time (a.u.)", fontsize=13, color="white")
        ax.set_ylabel(f"{STATE_LABELS[state_idx]} Population",
                       fontsize=13, color="white")
        ax.set_title(panel_title, fontsize=14, color="white")
        ax.legend(**LEGEND_KWARGS)

    fig.suptitle(suptitle, fontsize=16, color="white",
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {filename}")


def plot_validation(times, pops_mps, pops_sv, chi, filename):
    """Overlay MPS (solid) vs Statevector (dashed) with error annotation.

    Parameters
    ----------
    times : (n_steps+1,) array
    pops_mps : (n_steps+1, N_STATES) — MPS populations
    pops_sv  : (n_steps+1, N_STATES) — statevector populations
    chi : int — bond dimension used for MPS
    filename : str or Path — output file
    """
    max_err = np.max(np.abs(pops_mps - pops_sv))

    fig, ax = _dark_fig(figsize=(12, 6))
    _style_ax(ax)

    for s in range(N_STATES):
        ax.plot(times, pops_sv[:, s], "--", color=STATE_COLORS[s],
                linewidth=1.5, alpha=0.6)
        ax.plot(times, pops_mps[:, s], "-", color=STATE_COLORS[s],
                linewidth=2.5, label=STATE_LABELS[s],
                marker="o", markersize=3)

    ax.set_xlabel("Time (a.u.)", fontsize=14, color="white")
    ax.set_ylabel("Population", fontsize=14, color="white")
    ax.set_title(
        f"MPS (χ={chi}, solid) vs Statevector (dashed) — "
        f"max error = {max_err:.2e}",
        fontsize=14, color="white",
    )
    ax.legend(fontsize=12, **{
        k: v for k, v in LEGEND_KWARGS.items() if k != "fontsize"
    })

    plt.tight_layout()
    plt.savefig(filename, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"    Plot saved: {filename}")
    return max_err


def plot_scaling(scaling_data, filename, title=None):
    """Bar chart: CPU vs GPU timing at different χ values.

    Parameters
    ----------
    scaling_data : list of dicts with keys "chi", "gpu_time", optional "cpu_time"
    filename : str or Path — output file
    """
    fig, ax = _dark_fig(figsize=(12, 7))
    _style_ax(ax)

    chi_vals = [d["chi"] for d in scaling_data]
    cpu_times = [d.get("cpu_time", 0) for d in scaling_data]
    gpu_times = [d["gpu_time"] for d in scaling_data]

    x = np.arange(len(chi_vals))
    width = 0.35

    if any(t > 0 for t in cpu_times):
        ax.bar(x - width / 2, cpu_times, width, label="CPU",
               color="#e74c3c", alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.bar(x + width / 2, gpu_times, width, label="GPU (CUDA)",
           color="#2ecc71", alpha=0.85, edgecolor="white", linewidth=0.5)

    # Speedup annotations
    for i, d in enumerate(scaling_data):
        if d.get("cpu_time", 0) > 0:
            speedup = d["cpu_time"] / d["gpu_time"]
            ax.text(x[i], max(d["cpu_time"], d["gpu_time"]) * 1.05,
                    f"{speedup:.0f}×", ha="center", fontsize=13,
                    color="#f39c12", fontweight="bold")

    ax.set_xlabel("Bond Dimension (χ)", fontsize=14, color="white")
    ax.set_ylabel("Wall-clock Time (seconds)", fontsize=14, color="white")
    ax.set_title(title or "CPU vs GPU MPS Performance", fontsize=16,
                  color="white", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in chi_vals])
    ax.legend(fontsize=13, **{
        k: v for k, v in LEGEND_KWARGS.items() if k != "fontsize"
    })
    ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(filename, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Plot saved: {filename}")


# =====================================================================
# Build Pauli decomposition from embedded Hamiltonian data
# =====================================================================

pot_coup_terms, kinetic_modes = build_pauli_hamiltonian(_FREQS, _H_EL, _KAPPA, n_q=3)


n_pot_coup = len(pot_coup_terms)
n_kinetic = sum(len(kt) for _, kt in kinetic_modes)

print(f"Potential + coupling Pauli terms: {n_pot_coup}")
print(f"Kinetic Pauli terms: {n_kinetic}")
print(f"PauliRots per Trotter step: {2 * n_pot_coup + n_kinetic}")

######################################################################
# .. rst-class:: sphx-glr-script-out
# 
# .. code-block:: none
# 
#    Potential + coupling Pauli terms: 809
#    Kinetic Pauli terms: 114
#    PauliRots per Trotter step: 1,732

######################################################################
# Using :math:`n_q = 3` instead of :math:`n_q = 4` reduces the gate count by **~28%** per step (1,732
# vs 2,400 PauliRots). Combined with fewer Trotter steps (10 vs 20), the total circuit is **~64%
# shallower** — a significant reduction that makes the simulation more practical while preserving the
# essential physics.
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
# The Maestro Device
# ~~~~~~~~~~~~~~~~~~
# 
# `Maestro <https://www.github.com/qoroquantum/maestro>`__ is Qoro Quantum’s high-performance quantum
# circuit simulator, available as a PennyLane plugin. It supports multiple simulation backends
# including statevector, MPS, stabilizer, and Pauli propagation. For this demo, we use the **Matrix
# Product State (MPS)** backend, which represents the quantum state as a chain of tensors with tunable
# bond dimension :math:`\chi`.
# 
# The key advantage of MPS: while a full statevector for 60 qubits would require
# :math:`2^{60} \approx 10^{18}` complex amplitudes (exabytes of memory), an MPS with bond dimension
# :math:`\chi = 256` uses only :math:`60 \times 256^2 \times 2 \approx 8 \times 10^6` parameters — a
# compression factor of :math:`10^{11}`.
# 
# This demo requires an additional package beyond PennyLane:
# 
# -  ```pennylane-maestro`` <https://www.github.com/qoroquantum/pennylane-maestro>`__ — the PennyLane
#    plugin for Maestro
# 
# .. code:: bash
# 
#    pip install pennylane-maestro
# 
# Once installed, setting up the Maestro device in PennyLane is straightforward:
# 

import pennylane as qml

# CPU MPS backend
dev = qml.device(
    "maestro.qubit",
    wires=60,
    simulator_type="QCSim",
    simulation_type="MatrixProductState",
    max_bond_dimension=256,
)

# GPU MPS backend — same interface, just change simulator_type
dev_gpu = qml.device(
    "maestro.qubit",
    wires=60,
    simulator_type="Gpu",
    simulation_type="MatrixProductState",
    max_bond_dimension=256,
)

######################################################################
# The circuit construction uses PennyLane’s standard API. Each Trotter step is built from
# ``qml.PauliRot`` gates for the Hamiltonian terms and ``qml.QFT`` / ``qml.IQFT`` for the kinetic
# energy in momentum space:
# 

@qml.qnode(dev)

def circuit():
    # Initialize in S₁ (singlet excited state)
    qml.PauliX(wires=2)

    # Time evolution via second-order Trotter
    for _ in range(n_steps):
        # Forward half-step: potential + coupling
        for coeff, pauli_word, wires in pot_coup_terms:
            qml.PauliRot(coeff * dt, pauli_word, wires=wires)

        # Full kinetic step in momentum basis
        for mode_wires, ke_terms in kinetic_modes:
            qml.QFT(wires=mode_wires)
            for coeff, pauli_word, wires in ke_terms:
                qml.PauliRot(2 * coeff * dt, pauli_word, wires=wires)
            qml.adjoint(qml.QFT)(wires=mode_wires)

        # Backward half-step: potential + coupling (reversed)
        for coeff, pauli_word, wires in reversed(pot_coup_terms):
            qml.PauliRot(coeff * dt, pauli_word, wires=wires)

    # Measure electronic state populations
    return [qml.expval(projector) for projector in state_projectors]

######################################################################
# Bond Dimension Convergence
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# The critical question for any MPS simulation: *how large does the bond dimension need to be?* If
# :math:`\chi` is too small, the MPS truncates entanglement and gives incorrect dynamics. If it’s
# unnecessarily large, we waste computation.
# 
# We ran the full 60-qubit, 19-mode system at :math:`\chi = 32, 64, 128, 256` and tracked the
# electronic state populations over 10 a.u. of evolution time:
# 
# .. figure:: figures/vibronic_gpu_convergence.png
#    :alt: Bond dimension convergence
# 
# *Bond dimension convergence showing S₁ decay and ¹TT growth across χ = 32, 64, 128, 256. The curves
# separate at low χ and converge at χ ≥ 128.*
# 
# The results tell a clear story:
# 
# ============ =================== ====================== ==========
# :math:`\chi` :math:`S_1` (final) :math:`^1(TT)` (final) Converged?
# ============ =================== ====================== ==========
# 32           0.429               0.276                  ✗
# 64           0.324               0.280                  ✗
# 128          0.299               0.268                  ~
# 256          0.289               0.263                  ✓
# ============ =================== ====================== ==========
# 
# At :math:`\chi = 32`, the :math:`S_1` population is **50% higher** than the converged value — a
# qualitatively wrong picture of the singlet fission dynamics. The populations only stabilize at
# :math:`\chi \geq 128`, with the :math:`\chi = 128 \to 256` change dropping below 0.01.
# 
# This makes physical sense: the vibronic coupling tensors :math:`\kappa_m` have off-diagonal elements
# that entangle the electronic register with all 19 mode registers simultaneously. Combined with the
# QFT operations (which create long-range correlations within each mode), the entanglement entropy
# grows enough to require :math:`\chi \sim 200` for faithful representation.
# 

######################################################################
# GPU Acceleration: Where It Matters
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# Here’s where things get interesting. Each gate in the MPS simulation involves contracting and
# re-decomposing tensors of dimension :math:`\chi \times \chi`. At small :math:`\chi`, these are tiny
# matrix operations where GPU kernel launch overhead dominates. But at large :math:`\chi`, they become
# substantial linear algebra problems — exactly what GPUs are built for.
# 
# We ran the identical convergence study on both CPU and GPU backends:
# 
# ============ ======== ======== =================
# :math:`\chi` CPU Time GPU Time GPU Speedup
# ============ ======== ======== =================
# 32           19 min   1.0 h    0.3× (CPU faster)
# 64           63 min   1.4 h    0.7×
# 128          5.3 h    2.3 h    **2.3×**
# 256          ~27 h\*  4.3 h    **~6×**
# ============ ======== ======== =================
# 
# *\*CPU :math:`\chi = 256` estimated from scaling trend (5.1× per doubling at
# :math:`\chi \geq 128`).*
# 
# The hardware used for this experiment were based on standard VMs on Google Cloud:
# 
# -  CPU: c2-standard-30 (30 vCPUs, 120 GB Memory)
# -  GPU: a2-highgpu-1g (12 vCPUs, 85 GB Memory, 1 NVIDIA A100 40GB)
# 
# .. figure:: figures/vibronic_cpu_vs_gpu.png
#    :alt: CPU vs GPU comparison
# 
# *CPU vs GPU wall-clock time comparison across bond dimensions. GPU becomes faster at χ ≥ 128,
# reaching 6.2× speedup at χ = 256.*
# 
# The crossover happens between :math:`\chi = 64` and :math:`\chi = 128`. Below that, CPU wins because
# the tensor operations are too small to justify GPU overhead. Above it, GPU advantage grows rapidly:
# 
# -  **CPU scales ~5× per doubling** of :math:`\chi` (approaching the :math:`O(\chi^3)` theoretical
#    cost)
# -  **GPU scales ~2× per doubling** of :math:`\chi` (parallelism absorbs the cubic growth)
# 
# At :math:`\chi = 256` — the bond dimension needed for converged physics — GPU turns a **~27-hour CPU
# job into a 4-hour GPU run**. For research workflows where you need to iterate on parameters, run
# convergence studies, or explore different molecular systems, this is the difference between waiting
# a day and getting results before lunch.
# 

######################################################################
# The Singlet Fission Story
# -------------------------
# 
# Putting it all together, here’s what the converged simulation (:math:`\chi = 256`) tells us about
# singlet fission in the anthracene dimer:
# 
# .. figure:: figures/vibronic_gpu_populations.png
#    :alt: Population dynamics
# 
# *Population dynamics of the five electronic states over 10 a.u. of evolution, showing S₁ → ¹TT
# singlet fission and subsequent vibrational redistribution.*
# 
# Starting from the photoexcited :math:`S_1` state:
# 
# 1. **Rapid initial decay** (0–2 a.u.): :math:`S_1` drops from 1.0 to ~0.45, with population flowing
#    primarily into the triplet-pair state :math:`^1(TT)`. This is the singlet fission event.
# 
# 2. **Vibrational redistribution** (2–6 a.u.): Population oscillates between :math:`S_1` and
#    :math:`^1(TT)` as vibrational modes exchange energy with the electronic subsystem. The
#    charge-separated state :math:`CS` gradually accumulates population.
# 
# 3. **Quasi-equilibrium** (6–10 a.u.): The system approaches a quasi-steady state with
#    :math:`S_1 \approx 0.29`, :math:`^1(TT) \approx 0.26`, and significant population in
#    :math:`CS \approx 0.22`.
# 
# The trace (sum of all populations) is preserved to better than :math:`\Sigma = 0.9995` throughout,
# confirming the accuracy of the MPS simulation.
# 

######################################################################
# Where Classical Simulation Meets Its Limits
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# An important question emerges from these results: if MPS can simulate the 60-qubit vibronic dynamics
# circuit efficiently, where does classical simulation *stop* working?
# 
# In this demo we used :math:`n_q = 3` qubits per vibrational mode (8 grid points). The converged bond
# dimension was :math:`\chi = 256` — large enough to require GPU acceleration, but ultimately
# tractable. What happens when we push further?
# 
# -  **Higher grid resolution** (:math:`n_q = 4`, 79 qubits): Doubling the Hilbert space per mode
#    increases the number of Pauli rotation gates and deepens the entanglement generated per Trotter
#    step. The required :math:`\chi` could grow significantly.
# 
# -  **Longer evolution times**: Entanglement entropy typically grows with simulation time. Our 10
#    a.u. evolution already shows :math:`\chi`-sensitivity — longer simulations may push bond
#    dimension requirements beyond what even GPU-accelerated MPS can handle.
# 
# -  **Larger molecular systems**: Multi-dimer or multi-chromophore systems introduce inter-molecular
#    electronic couplings that create long-range entanglement across the MPS chain — precisely the
#    regime where MPS compression breaks down.
# 
# Establishing where this boundary lies — where the best classical methods fail — is valuable in its
# own right. It defines the regime where quantum hardware offers a genuine advantage, and provides a
# rigorous classical baseline against which quantum results can be validated. GPU-accelerated MPS
# pushes that baseline as far as classical hardware allows, making the case for quantum advantage
# stronger and more precise.
# 

######################################################################
# Conclusion
# ----------
# 
# In this demo, we simulated the vibronic dynamics of singlet fission — a quantum algorithm originally
# designed for future quantum hardware — using classical tensor network methods on Qoro Quantum’s
# Maestro simulator via PennyLane. The key takeaways:
# 
# -  **Tensor networks can simulate quantum algorithms at scale.** The 60-qubit, 19-mode singlet
#    fission circuit runs efficiently with MPS, enabling detailed convergence analysis and parameter
#    exploration that would be impractical with exact statevector methods.
# 
# -  **Bond dimension matters.** The vibronic coupling structure of singlet fission generates enough
#    entanglement to require :math:`\chi \geq 128` for converged results — low bond dimension gives
#    qualitatively incorrect dynamics.
# 
# -  **GPU acceleration is essential at high bond dimension.** At :math:`\chi = 256`, Maestro’s GPU
#    backend delivers ~6× speedup over CPU, reducing simulation time from ~27 hours to ~4 hours. The
#    advantage grows with :math:`\chi`, making GPU-accelerated MPS the practical choice for
#    production-quality tensor network simulations.
# 
# -  **Classical baselines sharpen quantum advantage.** By pushing MPS simulation to its limits, we
#    can identify exactly where classical methods fail — and where quantum hardware becomes essential.
#    This is a critical step in building the case for practical quantum advantage in chemistry.
# 
# -  **PennyLane makes it seamless.** Switching between CPU and GPU backends requires changing a
#    single parameter (``simulator_type``). The same PennyLane circuit code runs on statevector, CPU
#    MPS, and GPU MPS — enabling rapid prototyping and cross-validation.
# 
# Want to try it yourself? Install ``pennylane-maestro`` and run:
# 
# .. code:: bash
# 
#    pip install pennylane pennylane-maestro
# 
#    # Quick test (12 qubits, ~5 seconds)
#    python vibronic_demo_gpu.py --n-modes 3 --n-steps 5 --chi 32
# 
#    # Full GPU convergence study (60 qubits, ~9 hours)
#    python vibronic_demo_gpu.py --convergence --gpu
# 

######################################################################
# References
# ----------
# 
# [1] M. B. Smith and J. Michl, “Singlet Fission,” *Chem. Rev.* **110**, 6891 (2010). `DOI:
# 10.1021/cr1002613 <https://doi.org/10.1021/cr1002613>`__
# 
# [2] J. Huh *et al.*, “Quantum Algorithm for Vibronic Dynamics: Case Study on Singlet Fission Solar
# Cell Design,” arXiv:2411.13669 (2024). `arXiv: 2411.13669 <https://arxiv.org/abs/2411.13669>`__
# 
# [3] Qoro Quantum, “Maestro Quantum Simulator.” https://www.github.com/qoroquantum/maestro
# 
# [4] V. Bergholm *et al.*, “PennyLane: Automatic differentiation of hybrid quantum-classical
# computations,” arXiv:1811.04968 (2018). `arXiv: 1811.04968 <https://arxiv.org/abs/1811.04968>`__
# 