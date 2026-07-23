r"""
How to do Hamiltonian Simulation with GQSP in PennyLane
=======================================================

.. tip::

    This demo assumes familiarity with block encoding, qubitization, and quantum signal
    processing (QSP). If any of these are new, the
    :doc:`introduction to qubitization <demos/tutorial_qubitization>` [#qubitization]_ and
    :doc:`QSVT in practice <demos/tutorial_apply_qsvt>` [#qsvt]_ demos are good starting points.

Generalized Quantum Signal Processing (GQSP), introduced by Motlagh and Wiebe [#motlagh]_,
applies an arbitrary complex polynomial :math:`P` of a unitary :math:`U` using a single extra
control qubit. Its flagship application is **Hamiltonian simulation**, which is mainly concerned
with implementing the time-evolution operator :math:`e^{-iHt}`.

Hamiltonian simulation is the original motivation for quantum signal processing, and GQSP is a
modern variant of it. For GQSP, a single ancilla qubit and one complex polynomial of the
qubitization walk :math:`W` (the block encoding of :math:`H`) are enough; the complex target is
applied directly, with no need to build its real and imaginary parts separately as ordinary QSP
requires. This keeps the angle-finding step simple while still reaching :math:`e^{-iHt}` with a
query cost that grows
favourably in the evolution time and the target accuracy.

PennyLane provides :class:`~pennylane.GQSP` as a runnable circuit primitive and a resource-estimation demo
(:doc:`Resource estimation for Hamiltonian simulation with GQSP <demos/tutorial_estimator_hamiltonian_simulation_gqsp>`
[#estimator]_), which counts gates for cost analysis. This demo is the **executable
counterpart**. We will build and run the GQSP circuit and verify :math:`e^{-iHt}` against
``scipy.linalg.expm``, spelling out the recipe end to end.

By the end of this demo, you will be able to carry out a
:doc:`qubitization <demos/tutorial_qubitization>` [#qubitization]_ walk for a Pauli Hamiltonian,
derive the GQSP polynomial for :math:`e^{-iHt}` from the Jacobi-Anger expansion, run
:class:`~pennylane.GQSP` and recover the evolution, and confirm that the error converges with the
truncation order, all with PennyLane. The three pieces are:

1. **Block-encode** :math:`H` as a qubitization walk operator :math:`W` whose eigenvalues are

   .. math::

       e^{\pm i\arccos(E/\lambda)}

   for each eigenvalue :math:`E` of :math:`H`, with :math:`\lambda` the :math:`\ell_1`-norm of
   the coefficients.

2. **Choose your target polynomial** :math:`P` so that :math:`P(W) = e^{-iHt}`. Writing a walk
   eigenvalue as :math:`z=e^{i\theta}` with :math:`\cos\theta = E/\lambda`, we need

   .. math::

       P(e^{i\theta}) = e^{-i\lambda t\cos\theta} = e^{-iEt},

   which the Jacobi-Anger expansion gives as a Laurent series in :math:`z`.

3. **Run** :class:`~pennylane.GQSP` with the angles from ``qp.poly_to_angles(..., "GQSP")``, undo the
   Laurent shift, and read :math:`e^{-iHt}` from the top-left block.
"""

###############################################################################
# The Hamiltonian and the Exact Target
# -------------------------------------
#
# We will use a small two-qubit Heisenberg-type Hamiltonian. This is a convenient choice for this
# demo since it is already a sum of Pauli terms (the form
# :class:`~pennylane.Qubitization` block-encodes directly), its terms do not all commute (so :math:`e^{-iHt}`
# is non-trivial), and it is small enough that we can form the full matrix
# :math:`e^{-iHt}` and compare against it directly. In general, though, the methods in this demo
# work for any Pauli Hamiltonian. :math:`\lambda=\sum_k|c_k|` is the normalization used by the
# qubitization walk.

import warnings

warnings.filterwarnings("ignore", message=".*JAX.*")  # JAX is unused here
import numpy as np
import pennylane as qp
from scipy.linalg import expm
from scipy.special import jv  # Bessel functions of the first kind
import matplotlib.pyplot as plt

coeffs = [0.5, 0.3, 0.4, 0.2]
obs = [qp.X(0) @ qp.X(1), qp.Y(0) @ qp.Y(1), qp.Z(0) @ qp.Z(1), qp.Z(0)]
H = qp.Hamiltonian(coeffs, obs)
lam = sum(abs(c) for c in coeffs)  # l1-norm of the coefficients
t = 0.7  # evolution time

H_matrix = qp.matrix(H, wire_order=[0, 1])
U_exact = expm(-1j * H_matrix * t)

# show the Hamiltonian explicitly
print("H =", H)
print(f"{len(coeffs)} terms on 2 qubits   lambda = sum|c_k| = {lam}   t = {t}")
print("H as a matrix:")
print(np.round(H_matrix, 3))

###############################################################################
# Step 1: The Qubitization Walk
# ------------------------------
#
# ``qp.Qubitization(H, control)`` builds the Low and Chuang [#low]_
# :doc:`qubitization <demos/tutorial_qubitization>` walk operator
#
# .. math::
#
#     W = \text{Prep}^\dagger\,\text{Sel}\,\text{Prep}\,(2|0\rangle\langle 0| - I),
#
# which applies a reflection about the control :math:`|0\rangle` and then the block encoding of
# :math:`H/\lambda`. Its eigenphases are
#
# .. math::
#
#     \pm\arccos(E/\lambda)
#
# for each eigenvalue :math:`E` of :math:`H`, so a walk eigenvalue :math:`z = e^{i\theta}` satisfies
# :math:`\cos\theta = E/\lambda`. The control register needs :math:`\lceil \log_2 L \rceil` qubits
# for :math:`L` Hamiltonian terms.
#
# The printout below confirms this, since every :math:`\arccos(E/\lambda)` appears among the walk's
# eigenphases. The walk also carries a few extra phases (here :math:`0` and :math:`\pi`) coming
# from the complementary subspace where the control ancillas are not :math:`|0\rangle`. These
# encode no information about :math:`H` and are projected out when we read off the top-left block
# in Step 3.

n_ctrl = int(np.ceil(np.log2(len(coeffs))))
anc = [f"a{i}" for i in range(n_ctrl)]


@qp.qnode(qp.device("default.qubit"))
def walk():
    qp.Qubitization(H, control=anc)
    return qp.state()


W = qp.matrix(walk, wire_order=anc + [0, 1])()
walk_phases = np.sort(np.unique(np.round(np.abs(np.angle(np.linalg.eigvals(W))), 4)))
arccos_E = np.sort(
    np.unique(np.round(np.arccos(np.clip(np.linalg.eigvalsh(H_matrix) / lam, -1, 1)), 4))
)
print("walk eigenphases   :", walk_phases)
print("arccos(E / lambda) :", arccos_E)

###############################################################################
# Step 2: The Jacobi-Anger Polynomial
# ------------------------------------
#
# We need a polynomial :math:`P` with :math:`P(e^{i\theta}) = e^{-i\lambda t\cos\theta}`. The
# Jacobi-Anger expansion provides exactly this, as the Laurent series in :math:`z=e^{i\theta}`:
#
# .. math::
#
#     e^{-i a\cos\theta} = \sum_{k=-\infty}^{\infty} (-i)^k J_k(a)\, e^{ik\theta},
#     \qquad a = \lambda t,
#
# with :math:`J_k` being the Bessel functions of the first kind. The series converges
# super-exponentially once :math:`K \gtrsim a`, so we truncate at :math:`|k|\le K`, where
# :math:`K` is the **truncation order** (the highest power of :math:`z` we keep).
#
# Two practical points that the GQSP machinery requires:
#
# - :func:`~pennylane.poly_to_angles` needs a polynomial in **non-negative** powers, so we shift the
#   Laurent series by :math:`z^{K}` (we undo this shift in the circuit later).
# - :func:`~pennylane.poly_to_angles` also requires :math:`|P(e^{i\theta})|\le 1`, so we rescale the
#   coefficients by a constant :math:`s<1` (and divide it back out at the end).


def jacobi_anger_poly(a, K):
    "shifted, rescaled Jacobi-Anger coefficients for exp(-i a cos theta), plus the scale s."
    laurent = {k: (-1j) ** k * jv(k, a) for k in range(-K, K + 1)}
    p = [laurent[j - K] for j in range(2 * K + 1)]  # shift to powers 0..2K
    grid = np.exp(1j * np.linspace(0, 2 * np.pi, 400))
    s = 0.99 / max(abs(np.polyval(p[::-1], z)) for z in grid)  # enforce |P| <= 1
    return [c * s for c in p], s


K = 8
poly, s = jacobi_anger_poly(lam * t, K)
print(f"K = {K}: polynomial degree {len(poly) - 1}, scale s = {s:.4f}")

###############################################################################
# Step 3: Run :class:`~pennylane.GQSP` and Recover :math:`e^{-iHt}`
# -----------------------------------------------------------------
#
# ``qp.GQSP(U, angles, control)`` applies the single polynomial :math:`P(U)` with ``angles``
# from ``qp.poly_to_angles(poly, "GQSP")``. Recall how we built that polynomial in Step 2. The
# truncated Jacobi-Anger series already approximates :math:`e^{-iHt}` on the walk eigenvalues, and
# we made two changes to it: multiplying by :math:`z^{K}` to remove the negative powers, and
# scaling by :math:`s` to enforce :math:`|P|\le 1`. So, the coefficients in ``poly`` are those of
# :math:`P(z) = s\, z^{K}\, [\text{Jacobi-Anger series}]`, and GQSP applies this single polynomial
# of :math:`W` in one step, giving
#
# .. math::
#
#     P(W) \approx s\, W^{K}\, e^{-iHt}.
#
# The :math:`s` and :math:`W^{K}` terms are not separate circuits we multiply on afterwards -- they
# are the two corrections already baked into the polynomial, which we now simply undo. We cancel the
# :math:`W^{K}` factor by applying the **adjoint walk** :math:`W^\dagger` a total of :math:`K`
# times after the GQSP block, and we divide out :math:`s` at read-out. The evolution then sits in
# the all-ancilla-zero block (the top-left :math:`\dim\times\dim` corner), up to a global phase.


def gqsp_evolution(K):
    # recovered exp(-iH t) on the system, read from the GQSP top-left block.
    poly, s = jacobi_anger_poly(lam * t, K)
    angles = qp.poly_to_angles(poly, "GQSP")

    @qp.qnode(qp.device("default.qubit"))
    def circuit():
        qp.GQSP(qp.Qubitization(H, control=anc), angles, control="g")
        for _ in range(K):  # undo the z^K Laurent shift
            qp.adjoint(qp.Qubitization(H, control=anc))
        return qp.state()

    M = qp.matrix(circuit, wire_order=["g"] + anc + [0, 1])()
    block = M[:4, :4] / s  # all-ancilla-zero (top-left) block
    return block


block = gqsp_evolution(K)
ph = np.exp(-1j * np.angle(block[0, 0] / U_exact[0, 0]))  # match global phase
err = np.linalg.norm(block * ph - U_exact, 2)
print(f"||GQSP block - exp(-iHt)||_2 = {err:.2e}   (K = {K})")
print(
    "recovered block is unitary?  ||B^dag B - I|| =",
    f"{np.linalg.norm(block.conj().T @ block - np.eye(4), 2):.2e}",
)

###############################################################################
# Convergence
# -----------
#
# The truncation error falls super-exponentially with the order :math:`K` once
# :math:`K\gtrsim \lambda t`, which is the hallmark of the Jacobi-Anger approach. We plot the
# 2-norm distance to the exact evolution against :math:`K`.


def gqsp_error(K, t_val):
    "spectral-norm error of the GQSP evolution at order K and time t_val."
    poly, s = jacobi_anger_poly(lam * t_val, K)
    angles = qp.poly_to_angles(poly, "GQSP")

    @qp.qnode(qp.device("default.qubit"))
    def circuit():
        qp.GQSP(qp.Qubitization(H, control=anc), angles, control="g")
        for _ in range(K):  # undo the z^K Laurent shift
            qp.adjoint(qp.Qubitization(H, control=anc))
        return qp.state()

    M = qp.matrix(circuit, wire_order=["g"] + anc + [0, 1])()
    block = M[:4, :4] / s
    U = expm(-1j * H_matrix * t_val)
    ph = np.exp(-1j * np.angle(block[0, 0] / U[0, 0]))
    return np.linalg.norm(block * ph - U, 2)


Ks = list(range(2, 21))
lambda_t_targets = [1, 3, 5]  # show lambda*t = 1, 3, 5 exactly
t_values = [m / lam for m in lambda_t_targets]  # so lam * t = m

plt.style.use("pennylane.drawer.plot")
fig, ax = plt.subplots(figsize=(5.4, 3.4))
for t_val in t_values:
    errs = [gqsp_error(K, t_val) for K in Ks]
    a = lam * t_val
    ax.semilogy(Ks, errs, "o-", label=rf"$\lambda t = {round(a)}$")
    print(f"lambda t = {round(a)}:", [f"{e:.1e}" for e in errs])

ax.set_xticks(range(2, 21, 2))
ax.set_xlabel("Jacobi-Anger order $K$")
ax.set_ylabel(r"$\|U_{\mathrm{GQSP}} - e^{-iHt}\|_2$")
ax.set_title("GQSP error convergence")
ax.legend()
fig.tight_layout()
plt.show()

###############################################################################
# **Figure: GQSP error convergence.** Spectral-norm distance between the GQSP evolution and the
# exact :math:`e^{-iHt}`, versus the Jacobi-Anger order :math:`K`, for three evolution
# times :math:`\lambda t = 1, 3, 5`. Each curve falls super-exponentially once :math:`K` exceeds
# its :math:`\lambda t`, then floors at machine precision; larger :math:`\lambda t` needs a larger
# :math:`K` to reach the same accuracy.
#
# Understanding the Results
# -------------------------
#
# Running :class:`~pennylane.GQSP` on the qubitization walk reproduced :math:`e^{-iHt}` to within
# :math:`\sim 10^{-8}` at truncation order :math:`K=8` (a polynomial of degree :math:`2K=16`),
# matching ``scipy.linalg.expm`` to machine precision. Because of the fast Bessel decay seen in the
# convergence plot, useful accuracy needs only :math:`K = \mathcal{O}(\lambda t + \log(1/\varepsilon))`,
# i.e. :math:`\mathcal{O}(\lambda t + \log(1/\varepsilon))` applications of the walk.
#
# There are two practical things to keep in mind:
#
# - **One control qubit, one polynomial.** GQSP needs a single ancilla and a single complex
#   polynomial, applied directly rather than as separate real and imaginary parts, which is what
#   makes its angle synthesis simpler and more stable than ordinary QSP for this task.
# - **Where the cost lives.** The accuracy knob is the truncation order :math:`K`; the polynomial
#   GQSP applies has degree :math:`2K`, so the circuit uses :math:`2K` applications of the walk
#   :math:`W` (plus :math:`K` adjoint-walk applications to undo the :math:`z^{K}` shift). The depth
#   therefore grows quickly with :math:`K`. The
#   :doc:`resource-estimation demo <demos/tutorial_estimator_hamiltonian_simulation_gqsp>`
#   [#estimator]_ is a good reference to explore this cost.
#
# In practice, this makes GQSP a natural choice when you need an accurate :math:`e^{-iHt}` over a
# longer evolution time :math:`t`. The super-exponential convergence means the order :math:`K`
# (and hence the number of walk applications) grows only mildly as you tighten the error, so the
# cost stays close to the :math:`\lambda t` set by the evolution itself. This points to an
# accuracy-versus-resource trade-off rather than one method dominating: GQSP (and block-encoding
# methods generally) reach high accuracy with a query count that grows only mildly as the error
# shrinks, while product-formula (Trotter) approaches are often cheaper in qubit and gate counts
# for a coarse target but need rapidly more steps to reach the same accuracy. Within the
# block-encoding / quantum-signal-processing family, GQSP's particular appeal is economy of
# ancillas, since a single control qubit carries one complex polynomial applied directly, instead of
# building the polynomial's real and imaginary parts separately as ordinary QSP must.
#
# References
# ----------
#
# .. [#motlagh] Danial Motlagh, Nathan Wiebe,
#     "Generalized Quantum Signal Processing",
#     `PRX Quantum 5, 020368 <https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.5.020368>`__, 2024.
#
#
# .. [#low] Guang Hao Low, Isaac L. Chuang,
#     "Hamiltonian simulation by qubitization",
#     `Quantum 3, 163 <https://quantum-journal.org/papers/q-2019-07-12-163/>`__, 2019.
#
#
# .. [#estimator] PennyLane demo,
#     "Resource estimation for Hamiltonian simulation with GQSP",
#     `pennylane.ai <https://pennylane.ai/demos/tutorial_estimator_hamiltonian_simulation_gqsp>`__.
#
#
# .. [#qubitization] PennyLane demo,
#     "Introduction to qubitization",
#     `pennylane.ai <https://pennylane.ai/demos/tutorial_qubitization>`__.
#
#
# .. [#qsvt] PennyLane demo,
#     "QSVT in practice",
#     `pennylane.ai <https://pennylane.ai/demos/tutorial_apply_qsvt>`__.
