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
when compared to the standard surface code. This reduction is crucial makes it the
ideal candidate for near-term QEC.

To find the pseudo-threshold, we don't need to test a bunch of different distances.
We just need to focus on a single, near-term implementation—like a distance-3
():math:`d=3`) surface code—and compare its performance to the raw physical noise.

"""
def evaluate_pseudo_threshold():
    """Evaluates a d=3 surface code to find its pseudo-threshold."""
    d = 3
    # We use a slightly wider noise range to ensure we catch the crossover point
    noise_levels = [0.001, 0.003, 0.005, 0.008, 0.012, 0.015]
    num_shots = 20000 
    
    logical_error_rates = []
    
    for p in noise_levels:
        # Generate the d=3 rotated surface code circuit
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=d,
            rounds=d,
            after_clifford_depolarization=p,
            before_round_data_depolarization=p,
            before_measure_flip_probability=p,
            after_reset_flip_probability=p
        )
        
        # Count errors (reusing our previously defined count_logical_errors function)
        errors = count_logical_errors(circuit, num_shots)
        ler = errors / num_shots
        logical_error_rates.append(ler)
        
        print(f"Physical Noise p={p:.3f} -> d=3 Logical Error Rate: {ler:.4f}")
            
    return noise_levels, logical_error_rates

######################################################################
# Next, we need a function to plot this data. The key addition here is the "Unencoded Physical Error" line. When our d=3 line crosses below this unencoded line, we know our error correction is actually helping.
#

def plot_pseudo_threshold(noise_levels, logical_error_rates):
    """Plots the d=3 logical error rate against the unencoded physical error rate."""
    plt.figure(figsize=(8, 6))
    
    # Plot the logical error rate for our d=3 code
    plt.plot(noise_levels, logical_error_rates, marker='o', label='Encoded d=3 (Logical Error)', color='blue', linewidth=2)
    
    # Plot the break-even line (y=x) representing the unencoded physical error rate
    plt.plot(noise_levels, noise_levels, linestyle='--', color='red', label='Unencoded (Physical Error)', linewidth=2)
    
    plt.title('Surface Code Pseudo-Threshold (d=3)', fontsize=14)
    plt.xlabel('Physical Error Rate (p)', fontsize=12)
    plt.ylabel('Error Rate', fontsize=12)
    
    # Log scale for clear visualization
    plt.yscale('log')
    plt.xscale('log')
    
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

######################################################################
# Finally, we execute our functions to see the results.

# Run the pseudo-threshold evaluation and plot
p_levels, d3_results = evaluate_pseudo_threshold()
plot_pseudo_threshold(p_levels, d3_results)

######################################################################
# When you look at this graph, the red dashed line is your baseline. If your hardware's noise
# is high (on the right side of the graph), the blue line (d=3 QEC) sits above the red line,
# meaning QEC is actually making things worse because the extra gates are introducing too
# much noise. But as hardware improves and moves to the left, the blue line dips below the
# red line. That exact crossing point is your pseudo-threshold!
#
# Simulating the Threshold in Practice
# -------------------------------------
#
# To see the threshold theorem in action, we need to simulate a quantum circuit,
# inject noise, and attempt to correct those errors. We will use stim for
# lightning-fast simulation of our surface code circuits and pymatching to
# decode the errors.
#
# Let's start by importing our libraries and defining our core decoding function.
# This function takes a noisy circuit, samples it, and uses a Minimum Weight
# Perfect Matching (MWPM) decoder to see if our error correction succeeded or failed.
#
import stim
import pymatching
import numpy as np
import matplotlib.pyplot as plt

def count_logical_errors(circuit: stim.Circuit, num_shots: int) -> int:
    """Samples the circuit and decodes errors using PyMatching."""
    # Sample the circuit to get detection events and actual logical observable flips.
    sampler = circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(
        num_shots, separate_observables=True
    )

    # Configure a decoder using the circuit's specific error model.
    detector_error_model = circuit.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(detector_error_model)

    # Run the decoder to predict if a logical flip occurred.
    predictions = matcher.decode_batch(detection_events)

    # Count the mistakes by comparing predictions to the actual observable flips.
    num_errors = 0
    for shot in range(num_shots):
        actual_for_shot = observable_flips[shot]
        predicted_for_shot = predictions[shot]
        if not np.array_equal(actual_for_shot, predicted_for_shot):
            num_errors += 1
            
    return num_errors

######################################################################
# Next, we need a function to evaluate our rotated surface code across different
# sizes—known as the code distance :math:`d`—and various physical noise levels :math:`p`.
# We will use distances of 3, 5, and 7. For the noise levels, we know the circuit-level
# threshold for a surface code is generally around 0.8% (or 0.008). We will sweep our
# noise parameter right across that bridge to capture the crossing point.
#

def evaluate_surface_code_threshold():
    """Evaluates the rotated surface code across varying distances and noise levels."""
    distances = [3, 5, 7]
    noise_levels = [0.004, 0.006, 0.008, 0.010, 0.012]
    num_shots = 20000 
    
    results = {}
    
    for d in distances:
        results[d] = []
        for p in noise_levels:
            # Use Stim to instantly generate a rotated surface code circuit
            # incorporating reset, measurement, and depolarizing errors.
            circuit = stim.Circuit.generated(
                "surface_code:rotated_memory_z",
                distance=d,
                rounds=d,
                after_clifford_depolarization=p,
                before_round_data_depolarization=p,
                before_measure_flip_probability=p,
                after_reset_flip_probability=p
            )
            
            # Count the errors and calculate the logical error rate (LER)
            errors = count_logical_errors(circuit, num_shots)
            logical_error_rate = errors / num_shots
            results[d].append(logical_error_rate)
            
            print(f"Distance {d}, Physical Noise p={p:.3f} -> Logical Error Rate: {logical_error_rate:.4f}")
            
    return distances, noise_levels, results

######################################################################
# Now that we have the data, we need to visualize it. The visual hallmark of the
# threshold theorem is a specific crossing point on a graph. We will plot the
# Logical Error Rate (LER) against the Physical Error Rate (p) using matplotlib.
# We use a logarithmic scale for this plot because below the threshold, errors
# are suppressed exponentially as we increase the code distance!

def plot_threshold(distances, noise_levels, results):
    """Plots the logical error rate vs. physical error rate to visualize the threshold."""
    plt.figure(figsize=(8, 6))
    
    markers = ['o', 's', '^']
    
    for i, d in enumerate(distances):
        plt.plot(noise_levels, results[d], marker=markers[i % len(markers)], label=f'Distance {d}', linestyle='-', markersize=8)
        
    plt.title('Rotated Surface Code Threshold Evaluation', fontsize=14)
    plt.xlabel('Physical Error Rate (p)', fontsize=12)
    plt.ylabel('Logical Error Rate (LER)', fontsize=12)
    
    # Using log scale helps visualize the exponential suppression of errors below threshold
    plt.yscale('log')
    plt.xscale('log')
    
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

######################################################################
# Finally, we just need to call our functions to run the simulation and render the plot.
#

# Run the evaluation and plot the results
distances, noise_levels, results = evaluate_surface_code_threshold()
plot_threshold(distances, noise_levels, results)

######################################################################
#
# When you run this code, you will clearly see the lines intersecting.
# Above that crossing point, a larger code distance makes your logical qubit worse.
# But below that threshold, the magic of the Threshold Theorem kicks in, and
# increasing the code distance successfully suppresses the logical errors.
#
# Conclusion
# ----------
#
# The Threshold Theorem transitions quantum computing from an abstract mathematical
# curiosity into a viable engineering discipline. By proving that noise can be
# systematically managed, it provides the foundation upon which modern quantum
# architecture is built. As we demonstrated, 2D topological models like the
# Rotated Surface Code make fault tolerance an achievable reality, bringing the
# necessary thresholds into the realm of current hardware capabilities.
#
# While engineering challenges remain—particularly in executing efficient logical
# measurements and building the physical hardware to support the required number
# of qubits—the threshold theorem guarantees that we are fighting a winnable battle.
# By keeping our physical gate errors below the threshold, we unlock the path to
# arbitrarily complex, reliable quantum computations.
#
# References
# ----------
#
# .. [#qldpc1]
#
#     N. P. Breuckmann, J. N. Eberhardt,
#     "Quantum Low-Density Parity-Check Codes",
#     `PRX Quantum 2, 040101 <https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.040101>`__, 2021.
#
# .. [#CSS]
#
#     T. Rakovszky, V. Khemani,
#     "The Physics of (good) LDPC Codes I. Gauging and dualities",
#     `arXiv:2310.16032 <https://arxiv.org/abs/2310.16032>`__, 2023.
#
# .. [#HGP]
#
#     J.-P. Tillich, G. Zémor,
#     "Quantum LDPC Codes With Positive Rate and Minimum Distance Proportional to the Square Root of the Blocklength",
#     `IEEE Transactions on Information Theory 60(1), 119–136 <https://ieeexplore.ieee.org/document/6671468>`__, 2014.
#
# .. [#LPCodes]
#
#     F. G. Jeronimo, T. Mittal, R. O'Donnell, P. Paredes, M. Tulsiani,
#     "Explicit Abelian Lifts and Quantum LDPC Codes",
#     `arXiv:2112.01647 <https://arxiv.org/abs/2112.01647>`__, 2021.
#
# .. [#QTCodes]
#
#     A. Leverrier, G. Zémor,
#     "Quantum Tanner codes",
#     `arXiv:2202.13641 <https://arxiv.org/abs/2202.13641>`__, 2022.
#
# .. [#BBCodes]
#
#     S. Bravyi, A. W. Cross, J. M. Gambetta, D. Maslov, P. Rall, T. J. Yoder,
#     "High-threshold and low-overhead fault-tolerant quantum memory",
#     `Nature <https://www.nature.com/articles/s41586-024-07107-7>`__, 2024.
#
# .. [#BProp]
#
#     J. Old, M. Rispler,
#     "Generalized Belief Propagation Algorithms for Decoding of Surface Codes",
#     `Quantum 7, 1037 <https://quantum-journal.org/papers/q-2023-06-07-1037/>`__, 2023.
#
# .. [#OSD0]
#
#     J. Valls, F. Garcia-Herrero, N. Raveendran, B. Vasic,
#     "Syndrome-Based Min-Sum vs OSD-0 Decoders: FPGA Implementation and Analysis for Quantum LDPC Codes",
#     `IEEE Access <https://ieeexplore.ieee.org/document/9562513>`__, 2021.
#
# .. [#Transversal]
#
#     H. Leitch, A. Kay,
#     "Transversal Gates for Highly Asymmetric qLDPC Codes",
#     `arXiv:2506.15905 <https://arxiv.org/abs/2506.15905>`__, 2025.
#
# .. [#LMHM]
#
#     B. Ide, M. G. Gowda, P. J. Nadkarni, G. Dauphinais,
#     "Fault-tolerant logical measurements via homological measurement",
#     `Phys. Rev. X 15, 021088 <https://arxiv.org/abs/2410.02753>`__, 2024.
#