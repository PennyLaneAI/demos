"""Supporting subroutines for the singlet fission tensor-network demo."""
import numpy as np
import pennylane as qp
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
    if isinstance(op, qp.Identity):
        return None, None
    if isinstance(op, (qp.PauliX, qp.PauliY, qp.PauliZ)):
        return op.name[-1], list(op.wires)
    if hasattr(op, "operands"):
        word, wires = "", []
        for o in op.operands:
            if isinstance(o, qp.Identity):
                continue
            w, ws = _op_to_pauliword(o)
            if w is not None:
                word += w
                wires.extend(ws)
        return (word, wires) if word else (None, None)
    return None, None


def _decompose_matrix(mat, wire_order):
    """Decompose Hermitian matrix into (coeff, pauli_word, wires) tuples."""
    decomp = qp.pauli_decompose(mat, wire_order=wire_order)
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
        qp.Hadamard(wires=wires[j])
        for k in range(j + 1, n):
            qp.ControlledPhaseShift(
                np.pi / 2 ** (k - j), wires=[wires[k], wires[j]]
            )
    for i in range(n // 2):
        qp.SWAP(wires=[wires[i], wires[n - 1 - i]])


def apply_iqft(wires):
    """Inverse QFT using only native gates (no Adjoint wrappers).

    QFT = S · U  =>  QFT† = U† · S  (since S† = S for SWAPs).
    U† reverses the gate order and negates ControlledPhaseShift angles.
    """
    n = len(wires)
    # SWAP layer first (same as forward)
    for i in range(n // 2):
        qp.SWAP(wires=[wires[i], wires[n - 1 - i]])
    # Reversed H + CPhase with negated angles
    for j in range(n - 1, -1, -1):
        for k in range(n - 1, j, -1):
            qp.ControlledPhaseShift(
                -np.pi / 2 ** (k - j), wires=[wires[k], wires[j]]
            )
        qp.Hadamard(wires=wires[j])


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
            qp.PauliRot(theta, pw, wires=ws)

    # ── Full kinetic energy step (momentum basis via QFT) ──
    for mode_wires, ke_terms in kinetic_modes:
        # Transform to momentum basis
        apply_qft(mode_wires)
        # X on MSB to center the momentum grid around zero
        qp.PauliX(wires=mode_wires[0])
        # Apply kinetic diagonal: ω/2 · p² (full step, factor of 2 in theta)
        for coeff, pw, ws in ke_terms:
            theta = 2.0 * coeff * dt
            if abs(theta) > 1e-8:
                qp.PauliRot(theta, pw, wires=ws)
        # Undo X on MSB
        qp.PauliX(wires=mode_wires[0])
        # Transform back to position basis
        apply_iqft(mode_wires)

    # ── Backward half-step: potential + coupling (reversed) ──
    for _, coeff, pw, ws in reversed(pot_coup_terms):
        theta = coeff * dt
        if abs(theta) > 1e-8:
            qp.PauliRot(theta, pw, wires=ws)


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
                paulis.append(qp.PauliZ(el_wires[w]))
        if not paulis:
            ops.append(qp.Identity(el_wires[0]))
        elif len(paulis) == 1:
            ops.append(paulis[0])
        else:
            ops.append(qp.prod(*paulis))
        coeffs.append(c)
    return qp.Hamiltonian(coeffs, ops)


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
