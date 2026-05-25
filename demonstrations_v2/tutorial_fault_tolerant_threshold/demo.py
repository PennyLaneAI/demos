r"""Understanding the Fault-tolerant Threshold Theorem in Practice
===============================================================

The current generation of quantum hardware faces a critical roadblock, namely physical
instability. Even though modern processors feature hundreds of qubits, they are highly
susceptible to stray environmental interactions and imperfect gate operations. This constant
barrage of noise causes delicate quantum states to rapidly decohere, corrupting the system
with computational errors.

Building a fault-tolerant quantum computer requires quantum error correction (QEC),
which involves redundantly encoding a single logical qubit across many physical qubits
so that errors can be detected and corrected without ever directly reading the encoded
information. But because the physical operations performing this encoding are themselves
noisy, QEC introduces new opportunities for errors, raising a fundamental question:
*Can we ever get ahead of the noise?*

Think of QEC like bailing water from a leaky boat. If the hole is too large, water rushes
in faster than you can throw it out, and the effort of bailing only wears you down. But if
the hole is small enough, your bailing outpaces the leak and the boat stays dry indefinitely.
The fault-tolerant threshold theorem is the mathematical proof that such a regime exists,
i.e., below a critical physical error rate, adding more qubits exponentially suppresses the
logical error rate.

In this demo, we will learn the theoretical framework of the fault-tolerant threshold
theorem and observe this phenomenon in action. We will construct a QEC scheme called the
*rotated surface code*, and subject it to simulated quantum noise across a range of physical
error rates, computing two key metrics: (i) the pseudo-threshold (:math:`p_{pseudo}`), which
is the break-even point where a specific error-correcting code becomes better than doing
nothing at all, and (ii) the fault-tolerant threshold (:math:`p_{th}`), the fundamental
crossing point below which increasing our code size guarantees an exponential suppression
of errors. For a hardware engineer eager to implement near-term algorithms, achieving the 
pseudo-threshold is the immediate milestone. In contrast, reaching the fault-tolerant 
threshold is the ultimate goal for building a utility-scale quantum computer.

.. _fig-cartoon-thresholds:

.. figure::
    ../_static/demonstration_assets/fault-tolerant-threshold/cartoon_thresholds.png
    :align: center
    :width: 85%
    :target: javascript:void(0)

    Figure 1: Schematic of the fault-tolerant thresholds for rotated surface codes, where
    :math:`p` is the error probability per physical operation and :math:`p_{L}` is the
    probability that the decoded logical state of the encoded qubit is wrong.

The Fault-tolerant Threshold Theorem
---------------------------------

The threshold theorem is the mathematical bedrock of scalable quantum computing.
Intuitively, it states that a fault-tolerant quantum computation of any size can be 
accurately executed on imperfect hardware, provided that the base error rate of the 
physical operations, :math:`p`, remains strictly below a specific, non-zero constant 
known as the *threshold*, :math:`p_{th}`, the leftmost crossing point in the
:ref:`Figure 1 <fig-cartoon-thresholds>`.

More rigorously, assuming a local stochastic error model, we can take any
mathematically ideal (noiseless) circuit :math:`\mathcal{C}` of size :math:`N` and
construct a corresponding fault-tolerant circuit :math:`\mathcal{C}^{\prime}` to execute
on real hardware. Provided that the physical operations comprising this fault-tolerant
circuit have an error rate :math:`p < p_{th}`, the computation is guaranteed to yield an
output that deviates from the ideal outcome by no more than an arbitrarily small
tolerance, :math:`\epsilon > 0`.

Crucially, the theorem ensures that this correction is practically achievable. The
required hardware overhead, i.e., the total number of physical qubits and time steps
needed for :math:`\mathcal{C}^{\prime}`, grows at most by a polylogarithmic factor,
:math:`\mathcal{O}(\log^{c}(N/\epsilon))` for some positive constant :math:`c` [#threshold]_.
Although the original theoretical framework relied on specific assumptions like independent
stochastic noise, the threshold theorem is robust enough to apply to highly realistic,
correlated noise environments as well.

In simpler terms, operating in the green region of :ref:`Figure 1 <fig-cartoon-thresholds>`
(where :math:`p < p_{th}`) guarantees that the required number of physical qubits grows
non-exponentially with the size of the computation. This assures us that there is no
fundamental physical barrier standing in the way of large-scale quantum computers.
At least, theoretically!

To test the threshold theorem in practice, we use one of the leading candidates for
quantum error correction, the *surface code*, which is a topological code where qubits
are arranged on a 2D grid, with stabilizer measurements locally checking for parity
errors among nearest neighbors. If you would like to deepen your understanding
of surface codes before proceeding, our `Introducing Lattice Surgery <tutorial_lattice_surgery>`_
and `A Game of Surface Codes <tutorial_game_of_surface_codes>`_ demos are a great starting point.
Here, we specifically look at its *rotated* variant, which requires only :math:`d^2`
data qubits to achieve the same distance :math:`d`, giving a 50% reduction in qubit
overhead compared to the standard surface code.

.. figure::
    ../_static/demonstration_assets/fault-tolerant-threshold/rotated_surface_code.jpg
    :align: center
    :width: 85%
    :target: javascript:void(0)

    Figure 2: Rotated surface code for :math:`d=3` and :math:`d=5` codes.

As shown above for :math:`d=3` and :math:`d=5` rotated surface codes, the bulk plaquettes
alternate between :math:`Z`- and :math:`X`-type stabilizers in a checkerboard pattern,
while the boundaries host weight-2 stabilizers: :math:`X`-type on the top and bottom edges,
:math:`Z`-type on the left and right edges [#fowler]_. The logical :math:`X` operator runs
top-to-bottom along the left column, and the logical :math:`Z` operator runs left-to-right
along the top row, each forming a string of length :math:`d` that cannot be detected by any
stabilizer.

The function below encodes this geometry directly, returning the qubit indices
involved in each stabilizer and logical operator for any distance :math:`d` rotated
surface code:
"""

import numpy as np

def rotated_surface_code(d: int):
    """Generate stabilizers and logical operators for a rotated surface code."""
    # Create a grid of row and column indices for the plaquettes
    grid_idxs = np.arange(d - 1)
    row, col = np.meshgrid(grid_idxs, grid_idxs, indexing='ij')
    plaquette_qubits = np.stack([
        row * d + col, row * d + col + 1, # Top-left/right
        (row + 1) * d + col, (row + 1) * d + col + 1# Bottom-left/right
     ], axis=-1)

    # Assign X and Z stabilizers in a checkerboard pattern
    is_x_plaquette = (row + col) % 2 == 1
    x_stabilizers = plaquette_qubits[is_x_plaquette].tolist()
    z_stabilizers = plaquette_qubits[~is_x_plaquette].tolist()

    # Top boundary X-stabilizers sit on the even indices
    top_edges = np.stack([grid_idxs, grid_idxs + 1], axis=-1)
    x_stabilizers += top_edges[(grid_idxs % 2) == 0].tolist()

    # Bottom boundary X-stabilizers sit on the odd indices
    bottom_edges = np.stack([(d-1)*d + grid_idxs, (d-1)*d + grid_idxs + 1], axis=-1)
    x_stabilizers += bottom_edges[(grid_idxs % 2) != 0].tolist()

    # Left boundary Z-stabilizers sit on the odd indices
    left_edges = np.stack([grid_idxs * d, (grid_idxs + 1) * d], axis=-1)
    z_stabilizers += left_edges[(grid_idxs % 2 != 0)].tolist()

    # Right boundary Z-stabilizers sit on the even indices
    right_edges = np.stack([grid_idxs*d + (d-1), grid_idxs*d + 2*d - 1], axis=-1)
    z_stabilizers += right_edges[(grid_idxs % 2 == 0)].tolist()

    # Logical X runs top-to-bottom along the left-most column
    # Logical Z runs left-to-right along the top-most row
    logical_x, logical_z = list(range(0, d * d, d)), list(range(d))

    return x_stabilizers, z_stabilizers, logical_x, logical_z

dist = 3
x_stabs, z_stabs, log_x, log_z = rotated_surface_code(dist)
print(f"X stabilizers: {x_stabs}")
print(f"Z stabilizers: {z_stabs}")

######################################################################
# Let's verify our construction for a distance :math:`d=3` code, for which
# we expect 4 :math:`X`-type stabilizers, 4 :math:`Z`-type stabilizers, and
# 1 logical qubit encoded in 9 data qubits (:math:`9 - 8 = 1`).
#

nx, nz = len(x_stabs), len(z_stabs)
print(f"Number of X / Z stabilizers: {nx} | {nz}")
print(f"Logical X / Z operator: {log_x} | {log_z}")

n_qubits = max(map(max, (x_stabs + z_stabs))) + 1
print(f"\nNumber of data qubits: {n_qubits}")
print(f"Number of logical qubits: {n_qubits - nx - nz}")

######################################################################
# The Pseudo-Threshold
# --------------------
#
# Before sweeping over many code distances to locate the true asymptotic threshold, it is
# instructive to first ask a simpler question: Is this specific code worth using at all for
# my hardware? The answer is given by the *pseudo-threshold*. As seen in :ref:`Figure 1
# <fig-cartoon-thresholds>`, each code distance has its own distinct pseudo-threshold,
# represented by the points on the right where the individual QEC curves cross the dashed
# unencoded baseline. For a single code of distance :math:`d`, it is referred to as
# :math:`p_{\text{pseudo}}^{(d)}`, the physical error rate at which the encoded logical
# error rate (LER) precisely equals the unencoded physical error rate, where the LER is
# defined as the probability that an error persists on the encoded, logical qubit *after*
# the entire QEC decoding process is applied.
#
# Below :math:`p_{\text{pseudo}}^{(d)}`, encoding actively suppresses errors, i.e., the LER
# sits beneath the unencoded line, and QEC is doing *something* helpful. Above it, the extra
# circuit operations introduce more noise than they correct, making the code a net liability.
# Therefore, encoding with a distance-:math:`d` code is only worthwhile if the hardware's
# physical error rate stays strictly below :math:`p_{\text{pseudo}}^{(d)}`. Moreover, if
# :math:`p_{\text{pseudo}}^{(d)}` decreases with increasing :math:`d`, we can assess that
# the code family is scalable, even before computing the more expensive asymptotic threshold.
#
# For the rotated surface code, we compute the pseudo-threshold by evaluating the logical
# error rate of the minimum-distance code (:math:`d=3`) and comparing it against the raw
# physical noise level. To do so, we need syndrome extraction circuits that measure all
# stabilizers and return a syndrome for the decoder. For simplicity, these circuits assume
# instantaneous, noiseless syndrome extraction. In real hardware, each stabilizer measurement
# requires typically 4 CNOTs followed by a noisy readout, each introducing additional error
# sources. This places our simulation in the *code-capacity* regime, which yields a higher
# threshold than what is achievable in practice, a key difference we will quantify at the end
# of the `Simulating The Threshold <#simulating-the-threshold>`__ section.
#
# The ``syndrome_extraction`` function below uses the stabilizers and logical operators
# from the previous section to build these circuits, which are then executed on the
# ``default.clifford`` `device
# <https://docs.pennylane.ai/en/latest/code/api/pennylane.devices.default_clifford.html>`_
# with depolarizing noise at a specified number of shots.
#

import pennylane as qp

def syndrome_extraction(stabilizers, logical_ops, num_wires, noise_param, n_shots):
    """Extract the syndrome from the stabilizers and logical operators."""
    x_stabs, z_stabs = stabilizers
    x_lg_op, z_lg_op = logical_ops

    # Build the measurement operators for the X and Z stabilizers
    z_stab_ops = [qp.prod(*[qp.Z(q) for q in s]) for s in z_stabs]
    x_stab_ops = [qp.prod(*[qp.X(q) for q in s]) for s in x_stabs]
    z_logic_op = qp.prod(*[qp.Z(q) for q in z_lg_op])
    x_logic_op = qp.prod(*[qp.X(q) for q in x_lg_op])
    z_meas_ops = z_stab_ops + [z_logic_op]
    x_meas_ops = x_stab_ops + [x_logic_op]

    # Build the syndrome extraction circuit
    @qp.set_shots(n_shots)
    @qp.qnode(qp.device("default.clifford", wires=num_wires))
    def syndrome_circuit(meas_ops, x_basis=False):
        for q in range(num_wires):
            if x_basis:
                qp.H(wires=q)
            qp.DepolarizingChannel(noise_param, wires=q)
        return [qp.sample(op) for op in meas_ops]

    # Circuit 1: |+...+⟩ -> noise -> measure X-stabs + X-logical  (Z-error syndrome)
    # Circuit 2: |0...0⟩ -> noise -> measure Z-stabs + Z-logical  (X-error syndrome)
    z_stab_res = np.column_stack(syndrome_circuit(x_meas_ops, x_basis=True))
    x_stab_res = np.column_stack(syndrome_circuit(z_meas_ops, x_basis=False))
    return z_stab_res, x_stab_res

######################################################################
# The results from the above syndrome extraction circuits are then decoded using
# the minimum weight perfect matching (MWPM) decoding algorithm from the
# `PyMatching <https://github.com/oscarhiggott/PyMatching>`__ [#pymatching]_
# library, as shown below, to give the corrected syndromes :math:`\vec{c}`.
#

from pymatching import Matching

def syndrome_decoding(stabilizers, syndrome_results, num_wires, noise_param):
    """Decode the syndrome using PyMatching and compute corrections."""
    x_stabs, z_stabs = stabilizers
    nx, nz = len(x_stabs), len(z_stabs)

    # Z-error and X-error syndromes from the results
    z_stab_res, x_stab_res = syndrome_results
    z_syndrome = ((1 - z_stab_res[:, :nx]) // 2).astype(np.uint8)
    x_syndrome = ((1 - x_stab_res[:, :nz]) // 2).astype(np.uint8)

    # Build the parity check matrices for the X and Z stabilizers
    H_x, H_z = np.zeros((nx, num_wires)), np.zeros((nz, num_wires))
    for ix, sx in enumerate(x_stabs):
        H_x[ix, sx] = 1
    for iz, sz in enumerate(z_stabs):
        H_z[iz, sz] = 1

    # Decode the syndrome using PyMatching and compute corrections
    q_eff = 2 * noise_param / 3
    wt = np.log((1 - q_eff) / q_eff) if 0 < q_eff < 1 else 1.0
    z_corr = Matching(H_x, weights=wt).decode_batch(z_syndrome)
    x_corr = Matching(H_z, weights=wt).decode_batch(x_syndrome)
    return z_corr, x_corr

######################################################################
# So overall, we model the surface code such that all the qubits independently suffer a
# depolarizing error with probability (``noise_param``). We then compute the :math:`Z`/
# :math:`X`-stabilizer syndromes from the :math:`X`/:math:`Z`-stabilizer measurements,
# which are then decoded using the MWPM decoder defined above [#gottesman]_. We additionally
# measure the logical operators to obtain the values of logical observables that are used
# to compute the logical errors as a final step. More specifically, the ``ler_eval`` function
# below does this by computing the residual :math:`\vec{p} = \vec{e} \oplus \vec{c} \cdot \vec{l}`.
# It reconstructs the :math:`\vec{e} \cdot \vec{l}` from the circuit output and uses the
# corrected syndromes from the function above to compute the LER as
# :math:`p_{L} = 1 - (1 - p_{x}) * (1 - p_{z})`.
#

def ler_eval(stabilizers, logical_ops, noise_param, num_shots=10_000):
    """Evaluate LER for a given set of stabilizers and logical operators."""
    num_wires = 2 * max(len(stabilizers[0]), len(stabilizers[1])) + 1

    # Extract the syndrome measurements
    z_stab_res, x_stab_res = syndrome_extraction(
        stabilizers, logical_ops, num_wires, noise_param, num_shots
    )

    # Decode the syndrome and compute the logical corrections
    syndrome_results = (z_stab_res[:, :-1], x_stab_res[:, :-1])
    z_corr, x_corr = syndrome_decoding(
        stabilizers, syndrome_results, num_wires, noise_param
    )

    # Build the logical operators for the X and Z stabilizers
    log_x_vec, log_z_vec = np.zeros((2, num_wires), dtype=np.uint8)
    log_x_vec[logical_ops[0]], log_z_vec[logical_ops[1]] = 1, 1

    # Pauli frame tracking: compute error from the circuit output
    x_log_meas, z_log_meas = (x_stab_res[:, -1], z_stab_res[:, -1])
    lx_raw = ((1 - x_log_meas) // 2).astype(np.uint8)
    lz_raw = ((1 - z_log_meas) // 2).astype(np.uint8)
    p_lx = (lx_raw ^ (x_corr @ log_z_vec % 2)).mean()
    p_lz = (lz_raw ^ (z_corr @ log_x_vec % 2)).mean()
    return (1 - (1 - p_lx) * (1 - p_lz))

######################################################################
# We can now evaluate the pseudo-threshold for a given set of stabilizers and
# logical operators by sweeping over a range of noise parameters and evaluating
# the logical error rate.
#

from matplotlib import pyplot as plt

lerror_rates = []
p_noise = np.geomspace(0.025, 0.25, 21)
for p in p_noise:
    ler = ler_eval((x_stabs, z_stabs), (log_x, log_z), p)
    lerror_rates.append(ler)

# Approximating the pseudo-threshold by linear interpolation
diff = np.array(lerror_rates) - p_noise
p_idx = np.where(diff <= 0)[0][-1]
p0, p1 = p_noise[p_idx], p_noise[p_idx + 1]
d0, d1 = diff[p_idx], diff[p_idx + 1]
pseudo_threshold = p0 - d0 * (p1 - p0) / (d1 - d0)

plt.figure(figsize=(6, 3))
plt.axvline(x=pseudo_threshold, color="black", linestyle="--", linewidth=1)
plt.text(pseudo_threshold, 0.04, r" p$_{pseudo}$="+f"{pseudo_threshold:.3f}")
plt.plot(p_noise, lerror_rates, marker="o", label=f"Surface Code (d={dist})", color="b")
plt.plot(p_noise, p_noise, linestyle="--", label="Unencoded", color="r")
plt.xlabel("Physical Error Rate (p)", fontsize=10)
plt.ylabel(r"Logical Error Rate (p$_{L}$)", fontsize=10)
plt.yscale("log")
plt.xscale("log")
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.legend(fontsize=10)
caption_text = "Figure 3: Simulated pseudo-threshold for the rotated surface code."
plt.figtext(
    0.5, 0.008, caption_text, wrap=True, horizontalalignment='center', fontsize=10
)
plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.show()

######################################################################
# The red dashed line is the baseline, i.e., the error rate you would see with
# no error correction at all. On the right side of the graph (high noise), the
# blue curve sits *above* the red line, meaning QEC is making things worse because
# the extra circuit operations introduce more noise than they correct. Moving
# leftward to lower physical error rates, the blue curve eventually dips
# *below* the red line. That crossing point is the *pseudo-threshold* for
# our distance :math:`d=3` code.
#
# Simulating The Threshold
# -------------------------
#
# While the pseudo-threshold we computed previously tells us when a
# specific code distance starts helping, the true *asymptotic* fault-tolerant
# threshold tells us something deeper for the entire code family, i.e., the
# physical error rate below which we can keep improving by increasing the
# code distance. Here, we sweep over distances :math:`d = 3, 5, 7` and a
# range of depolarizing noise strengths.
#

def eval_threshold(distances, p_noise, num_shots):
    """Evaluates the threshold for a given set of distances and noise levels."""
    results = {d: [] for d in distances}
    for dist in distances:
        x_stabs, z_stabs, log_x, log_z = rotated_surface_code(dist)
        for p in p_noise:
            ler = ler_eval((x_stabs, z_stabs), (log_x, log_z), p, num_shots)
            results[dist].append(ler)
    return results

distances = [3, 5, 7]
p_noise = np.geomspace(0.036, 0.36, 15)
results = eval_threshold(distances, p_noise, num_shots=20_000)

######################################################################
# We visualize the results on a log-log plot. Below the threshold, errors
# are suppressed *exponentially* with increasing distance, so the curves
# fan out dramatically on a logarithmic scale.
#

plt.figure(figsize=(6, 4))

ler_vals = []
for d in distances:
    result = np.array(results[d])
    plt.plot(p_noise, result, label=f"Distance {d}", marker="o")
    ler_vals.append(result)

# Approximation of the threshold by linear interpolation
diff = np.diff(np.array(ler_vals), axis=0)
idxs = np.argmax(np.diff(np.sign(diff), axis=1) != 0, axis=1)[0]
p0, p1 = p_noise[idxs], p_noise[idxs + 1]
d0, d1 = diff[np.arange(len(diff)), idxs], diff[np.arange(len(diff)), idxs + 1]
p_th  = np.mean(p0 - d0 * (p1 - p0) / (d1 - d0))
plt.axvline(x=p_th, color="black", linestyle="--", linewidth=1)
plt.text(p_th, 0.04, r" p$_{th}$="+f"{p_th:.3f}")

plt.xlabel("Physical Error Rate (p)", fontsize=10)
plt.ylabel("Logical Error Rate (p$_{L}$)", fontsize=10)
plt.yscale("log")
plt.xscale("log")
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.legend(fontsize=10)
caption_text = "Figure 4: Simulated fault-tolerant threshold for the rotated surface codes."
plt.figtext(
    0.5, 0.008, caption_text, wrap=True, horizontalalignment='center', fontsize=10
)
plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.show()

######################################################################
# The curves for different distances cross at a single point, the *threshold*.
# Because standard error models behave mathematically like a phase transition,
# all logical error rates for a code family converge at this single critical
# point. This creates three distinct regions, mirroring our schematic in
# :ref:`Figure 1 <fig-cartoon-thresholds>`, where below the fault-tolerant threshold
# (the green region), larger codes perform exponentially better; above the rightmost
# pseudo-threshold (the red region), all QEC is actively harmful and raw qubits are
# better, and in the transition region between them (the amber region), the noise
# is too high to support the massive gate overhead  of a :math:`d=7` code, making
# it perform worse than doing nothing, while a smaller :math:`d=3` code carries less
# overhead and can still manage to break even.
#
# Note that this code-capacity threshold (``~15%``) is considerably higher than the
# circuit-level threshold (``~0.8%``) reported in the pseudo-threshold section. This
# gap arises directly from our code-capacity assumption of instantaneous, perfect
# syndrome extraction; the real hardware noise would, in fact, drive the threshold
# lower. Nevertheless, the qualitative picture remains the same: logical error
# rate curves for increasing code distances cross at a single threshold point.
#
# Conclusion
# ----------
#
# In this tutorial, we demonstrated the threshold theorem in practice by simulating both
# the pseudo-threshold and the asymptotic threshold for the rotated surface code. We saw
# that below the threshold point :math:`p_{th}`, increasing the code distance :math:`d`
# leads to an exponential suppression of logical errors. Visually, this qualitative behavior
# gets captured by the curves for different code distances crossing at a single distinct
# point :math:`p_{th}`, which is the hallmark of the threshold theorem. For topological
# codes like the surface code, this suppression follows the power-law relationship
# :math:`p_L \propto C \cdot (p / p_{th})^{(d + 1)/2}`, where :math:`C > 0` is a constant.
# Since the base is :math:`p / p_{th} < 1`, when operating below the threshold point
# :math:`p < p_{th}`, the logical error rate :math:`p_L` drops exponentially as :math:`d`
# increases. Therefore, by simply engineering a larger code, we can suppress the logical
# error rate to arbitrarily low levels.
#
# As we saw earlier, our code-capacity simulations represent a theoretical upper bound;
# real circuit-level thresholds are considerably lower once noisy syndrome extraction is
# accounted for. Nevertheless, the qualitative picture is preserved, and the threshold
# theorem guarantees that we are fighting a winnable battle. While significant engineering
# challenges remain, such as scaling up the number of physical qubits and executing fast,
# efficient logical operations, we can unlock the path to arbitrarily complex, reliable
# quantum computations by keeping physical error rates below the threshold.
#
# References
# ----------
#
# .. [#threshold]
#
#     D. Aharonov, M. Ben-Or,
#     "Fault-Tolerant Quantum Computation With Constant Error Rate",
#     `SIAM J. Comput., 38(4), 1207–1282 <https://epubs.siam.org/doi/10.1137/S0097539799359385>`__, 2008.
#
# .. [#gottesman]
#
#     D. Gottesman,
#     "An Introduction to Quantum Error Correction and Fault-Tolerant Quantum Computation",
#     `arXiv:0904.2557 <https://arxiv.org/abs/0904.2557>`__, 2009.
#
# .. [#fowler]
#
#     A. G. Fowler, M. Mariantoni, J. M. Martinis, A. N. Cleland,
#     "Surface codes: Towards practical large-scale quantum computation",
#     `Phys. Rev. A 86, 032324 <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.86.032324>`__, 2012.
#
# .. [#pymatching]
#
#     O. Higgott,
#     "PyMatching: A Python package for decoding quantum codes with minimum-weight perfect matching",
#     `ACM Trans. Quantum Comput. 3(3), 1–16 <https://dl.acm.org/doi/10.1145/3505637>`__, 2022.
#
