r"""Understanding Fault-tolerant Threshold Theorem in Practice
===============================================================

Quantum mechanics offers a revolutionary framework for computation, unlocking the ability
to solve highly complex problems well beyond the reach of classical supercomputers. Yet,
the current generation of quantum hardware faces a critical roadblock, namely physical
instability. Even though modern processors feature hundreds of qubits, they are highly
susceptible to stray environmental interactions and imperfect gate operations. This constant
barrage of noise causes delicate quantum states to rapidly decohere, corrupting the system
with computational errors.

To build a quantum computer that can run indefinitely with negligible errors, we must utilize
Quantum Error Correction (QEC). QEC works by redundantly encoding a single "logical" qubit into
many "physical" qubits. However, because the operations used to perform this encoding are
themselves noisy, QEC introduces new opportunities for errors to occur. This leads to a
fundamental question: Can we ever get ahead of the noise? This is where the *fault-tolerant
threshold theorem* comes in.

.. figure::    
    ../_static/demo_thumbnails/opengraph_demo_thumbnails/pennylane-demo-stabilizer-codes-open-graph.png
    :align: center
    :width: 50%
    :target: javascript:void(0)


Fault-tolerant Threshold Theorem
---------------------------------

The Threshold Theorem is the mathematical bedrock of scalable quantum computing.
Intuitively, it states that a fault-tolerant quantum computation of size :math:`N`
can be accurately executed on imperfect hardware, provided that the base error rate
of the physical operations, :math:`p`, remains strictly below a specific, non-zero
constant known as the threshold, :math:`p_{th}`.

To state this more rigorously: assuming a local stochastic error model where :math:`p<p_{th}`,
we can take any ideal circuit :math:`\mathcal{C}` and construct a corresponding fault-tolerant
circuit :math:`\mathcal{C}^{\prime}`. Even when subjected to continuous noise, the latter is
guaranteed to yield an output that is statistically indistinguishable from the ideal
outcome—deviating by no more than an arbitrarily small tolerance, :math:`\epsilon > 0`.
Furthermore, the theorem ensures that this error correction is practically achievable,
i.e., the required hardware overhead is efficient. The total number of physical qubits and
time steps needed for the fault-tolerant circuit :math:`\mathcal{C}^{\prime}`
grows at most by a polylogarithmic factor, :math:`\mathcal{O}(\log^{c}(N/\epsilon))`
for some positive constant :math:`c`.

In simpler terms, this means that as long as your physical hardware is "good enough",
i.e., the error rate per physical gate or time step is below the threshold :math:`p_{th}`,
you can build reliable quantum circuits of any size. The required number of physical
qubits would grow non-exponentially with the size of the computation.

Although the original theoretical framework relied on specific assumptions like
independent stochastic noise, the threshold theorem is robust enough to apply to
highly realistic, correlated noise environments as well. It assures us that there
is no fundamental physical barrier standing in the way of large-scale quantum computers.

The Pseudo-Threshold
--------------------

While the asymptotic threshold :math:`p_{th}` guarantees scalability in the long run,
experimentalists working with near-term hardware often focus on a different metric:
the *pseudo-threshold*. It is defined as the physical error rate below which the
logical error rate of a specific code distance becomes lower than the physical error
rate of a single, unencoded physical operation (:math:`p_{L} < p_{phys}`).

To test the threshold theorem in practice, we look to the leading candidate for
near-term QEC: the Surface Code, which is a topological code where qubits
are arranged on a 2D grid, with stabilizer measurements locally checking for
parity errors among nearest neighbors. For practical implementation, we specifically
look at the Rotated Surface Code, which requires only :math:`d^2` data qubits to
achieve the exact same distance :math:`d`. This gives a 50% reduction in qubit overhead
when compared to the standard surface code. This reduction is crucial and makes it the
ideal candidate for near-term QEC.

To find the pseudo-threshold, we don't need to test a bunch of different distances.
We just need to focus on a single, near-term implementation—like a distance-3
(:math:`d=3`) surface code—and compare its performance to the raw physical noise.

"""

######################################################################
# Setting Up the Simulation
# -------------------------
#
# We simulate surface code circuits under a circuit-level depolarizing noise
# model. We use `Stim <https://github.com/quantumlib/Stim>`__ [#stim]_ for
# fast stabilizer circuit simulation and
# `PyMatching <https://github.com/oscarhiggott/PyMatching>`__ [#pymatching]_
# for decoding via Minimum Weight Perfect Matching (MWPM).
#
# Our core utility function takes a noisy circuit, samples detection events
# from it, runs the MWPM decoder, and counts how many shots the decoder
# failed to correct.
#

import stim
import pymatching
import numpy as np
import matplotlib.pyplot as plt


def count_logical_errors(circuit: stim.Circuit, num_shots: int) -> int:
    """Samples a noisy circuit and counts decoding failures using MWPM."""
    sampler = circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(
        num_shots, separate_observables=True
    )

    detector_error_model = circuit.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(detector_error_model)
    predictions = matcher.decode_batch(detection_events)

    num_errors = int(np.any(predictions != observable_flips, axis=1).sum())
    return num_errors


######################################################################
# Evaluating the Pseudo-Threshold
# --------------------------------
#
# We start by evaluating a single distance-3 rotated surface code across
# a range of physical noise levels. The noise model applies depolarizing
# errors after every Clifford gate, data-qubit depolarization before each
# syndrome extraction round, and bit-flip errors on measurements and
# resets—all at the same rate :math:`p`.
#

def evaluate_pseudo_threshold():
    """Evaluates a d=3 surface code to find its pseudo-threshold."""
    d = 3
    noise_levels = [0.001, 0.003, 0.005, 0.008, 0.012, 0.015]
    num_shots = 20_000

    logical_error_rates = []

    for p in noise_levels:
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=d,
            rounds=d,
            after_clifford_depolarization=p,
            before_round_data_depolarization=p,
            before_measure_flip_probability=p,
            after_reset_flip_probability=p,
        )

        errors = count_logical_errors(circuit, num_shots)
        ler = errors / num_shots
        logical_error_rates.append(ler)

        print(f"  p = {p:.3f}  ->  d=3 logical error rate = {ler:.4f}")

    return noise_levels, logical_error_rates


######################################################################
# Next, we plot the encoded logical error rate against the *unencoded*
# physical error rate (the :math:`y = x` diagonal). The crossing point
# of these two curves is the pseudo-threshold: below it, our distance-3
# code actually outperforms a bare physical qubit.
#

def plot_pseudo_threshold(noise_levels, logical_error_rates):
    """Plots d=3 logical error rate against unencoded physical error rate."""
    plt.figure(figsize=(8, 6))

    plt.plot(
        noise_levels, logical_error_rates,
        marker="o", label="Encoded d=3 (Logical Error)", color="blue", linewidth=2,
    )
    plt.plot(
        noise_levels, noise_levels,
        linestyle="--", color="red", label="Unencoded (Physical Error)", linewidth=2,
    )

    plt.title("Surface Code Pseudo-Threshold (d=3)", fontsize=14)
    plt.xlabel("Physical Error Rate (p)", fontsize=12)
    plt.ylabel("Error Rate", fontsize=12)
    plt.yscale("log")
    plt.xscale("log")
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()


######################################################################
# Let's run the evaluation and visualize the result.

p_levels, d3_results = evaluate_pseudo_threshold()
plot_pseudo_threshold(p_levels, d3_results)

######################################################################
# The red dashed line is the baseline: the error rate you would see with no
# error correction at all. On the right side of the graph (high noise), the
# blue curve sits *above* the red line—QEC is making things worse because the
# extra circuit operations introduce more noise than they correct. Moving
# leftward to lower physical error rates, the blue curve eventually dips
# *below* the red line. That crossing point is the **pseudo-threshold** for
# our distance-3 code.
#
# Observing the Threshold
# -----------------------
#
# The pseudo-threshold tells us when a *specific* code distance starts
# helping. The true threshold tells us something deeper: it is the physical
# error rate below which we can *keep improving* by increasing the code
# distance.
#
# To observe this, we sweep over multiple distances (:math:`d = 3, 5, 7`)
# and look for a crossing point in the logical-vs-physical error rate
# curves. The circuit-level threshold for the rotated surface code under
# depolarizing noise is approximately 0.6–0.8 %, so we sample noise levels
# around that region to capture the crossing.
#

def evaluate_surface_code_threshold():
    """Evaluates the rotated surface code across varying distances and noise levels."""
    distances = [3, 5, 7]
    noise_levels = [0.004, 0.006, 0.008, 0.010, 0.012]
    num_shots = 20_000

    results = {}

    for d in distances:
        results[d] = []
        for p in noise_levels:
            circuit = stim.Circuit.generated(
                "surface_code:rotated_memory_z",
                distance=d,
                rounds=d,
                after_clifford_depolarization=p,
                before_round_data_depolarization=p,
                before_measure_flip_probability=p,
                after_reset_flip_probability=p,
            )

            errors = count_logical_errors(circuit, num_shots)
            logical_error_rate = errors / num_shots
            results[d].append(logical_error_rate)

            print(f"  d = {d}, p = {p:.3f}  ->  logical error rate = {logical_error_rate:.4f}")

    return distances, noise_levels, results


######################################################################
# We visualize the results on a log-log plot. Below the threshold, errors
# are suppressed *exponentially* with increasing distance, so the curves
# fan out dramatically on a logarithmic scale.

def plot_threshold(distances, noise_levels, results):
    """Plots logical error rate vs. physical error rate across code distances."""
    plt.figure(figsize=(8, 6))

    markers = ["o", "s", "^"]

    for i, d in enumerate(distances):
        plt.plot(
            noise_levels, results[d],
            marker=markers[i], label=f"Distance {d}", linestyle="-", markersize=8,
        )

    plt.title("Rotated Surface Code Threshold", fontsize=14)
    plt.xlabel("Physical Error Rate (p)", fontsize=12)
    plt.ylabel("Logical Error Rate", fontsize=12)
    plt.yscale("log")
    plt.xscale("log")
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()


######################################################################
# Let's run the multi-distance evaluation and visualize the threshold.

distances, noise_levels, results = evaluate_surface_code_threshold()
plot_threshold(distances, noise_levels, results)

######################################################################
# The curves for different distances cross at a single point—this is the
# **threshold**. To the right of the crossing (high noise), larger codes
# perform *worse*: the added circuit complexity introduces more errors than
# it corrects. To the left (low noise), larger codes perform *better*, and
# the improvement is exponential with distance. This is exactly the behavior
# predicted by the threshold theorem.
#
# Conclusion
# ----------
#
# The Threshold Theorem transforms quantum computing from an abstract
# mathematical curiosity into a viable engineering discipline. By proving
# that noise can be systematically managed, it provides the foundation on
# which modern quantum architectures are built. As we demonstrated, 2D
# topological codes like the Rotated Surface Code make fault tolerance an
# achievable reality, bringing the necessary thresholds within reach of
# current hardware capabilities.
#
# While engineering challenges remain—particularly in scaling up the number
# of physical qubits and implementing efficient logical operations—the
# threshold theorem guarantees that we are fighting a winnable battle.
# By keeping physical gate errors below the threshold, we unlock the path
# to arbitrarily complex, reliable quantum computations.
#
# References
# ----------
#
# .. [#threshold]
#
#     D. Aharonov, M. Ben-Or,
#     "Fault-tolerant quantum computation with constant error rate",
#     `SIAM J. Comput., 38(4), 1207–1282 <https://arxiv.org/abs/quant-ph/9906129>`__, 2008.
#
# .. [#gottesman]
#
#     D. Gottesman,
#     "An Introduction to Quantum Error Correction and Fault-Tolerant Quantum Computation",
#     `arXiv:0904.2557 <https://arxiv.org/abs/0904.2557>`__, 2009.
#
# .. [#kitaev]
#
#     A. Kitaev,
#     "Fault-tolerant quantum computation by anyons",
#     `Annals of Physics, 303(1), 2–30 <https://arxiv.org/abs/quant-ph/9707021>`__, 2003.
#
# .. [#dennis]
#
#     E. Dennis, A. Kitaev, A. Landahl, J. Preskill,
#     "Topological quantum memory",
#     `J. Math. Phys. 43, 4452–4505 <https://arxiv.org/abs/quant-ph/0110143>`__, 2002.
#
# .. [#fowler]
#
#     A. G. Fowler, M. Mariantoni, J. M. Martinis, A. N. Cleland,
#     "Surface codes: Towards practical large-scale quantum computation",
#     `Phys. Rev. A 86, 032324 <https://arxiv.org/abs/1208.0928>`__, 2012.
#
# .. [#stim]
#
#     C. Gidney,
#     "Stim: a fast stabilizer circuit simulator",
#     `Quantum 5, 497 <https://quantum-journal.org/papers/q-2021-07-06-497/>`__, 2021.
#
# .. [#pymatching]
#
#     O. Higgott,
#     "PyMatching: A Python package for decoding quantum codes with minimum-weight perfect matching",
#     `ACM Trans. Quantum Comput. 3(3), 1–16 <https://arxiv.org/abs/2105.13082>`__, 2022.
#
