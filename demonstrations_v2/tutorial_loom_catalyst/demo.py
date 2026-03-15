r"""Loom x Catalyst: designing, orchestrating, and automating quantum error correction experiments
==================================================

Running computations on physical systems inevitably means confronting imperfections---what we commonly
refer to as noise. Noise, by definition, is any deviation from the ideal behaviour of a system as it
evolves, and it’s a constant feature of our physical world. This reality becomes even more
pronounced in quantum computing. Quantum states are intrinsically fragile, which impedes the
implementation of practical algorithms on quantum computers. While quantum hardware companies are
building increasingly large and complex multi-qubit systems, a significant challenge lies in
managing the errors caused by noise in quantum information processing and devising ways to correct
them.

.. figure:: ../_static/demonstration_assets/loom_catalyst/entropica-QEC-hero-image.png
    :align: center
    :width: 70% 

**Quantum Error Correction (QEC)** is a method to protect quantum information using redundant
encoding. In QEC, a “code” is used to store a single qubit of information in the state of multiple
physical qubits, so that it is possible to determine and recover errors on the individual qubits
protecting the logical information. The sapient use of entanglement allows us to be as close as
possible to perfect, noiseless qubits. The difficulty in QEC is that the implementation of error
correction is itself error-prone. When the error detection and correction procedures can correct
errors faster than they cascade, we speak of Fault-Tolerant Quantum Error Correction (FTQEC). This
design methodology ensures that a quantum computation can tolerate errors, so that QEC can remain
effective in time throughout the execution of a quantum algorithm.

Quantum error correction stands as one of the great frontiers in quantum computing. It gives the
ability to take noisy and imperfect physical qubits, and create perfect and ideal logical
qubits, paving the way for truly useful quantum computations. This promise hinges on three key
technical components:

- Low-noise physical qubits
- A control system of exceptional precision
- Software capable of executing the QEC protocol

In this demo, we’ll delve into foundational concepts in QEC theory, with a particular focus on the
software layer. We’ll start here by implementing a simple, naive QEC protocol directly in Catalyst,
leveraging one of its greatest strengths---seamlessly integrating and compiling classical and quantum 
routines within a single program through its connection with JAX. From there, we’ll explore the limitations
of this naive approach and then move to Loom, Entropica Labs’ platform for designing, orchestrating,
and automating quantum error correction experiments, to refine the protocol and scale it up.

Classical error correction
--------------------------

At the heart of quantum error correction (QEC) lies a powerful concept: using many imperfect
physical systems to simulate a single qubit that is far more resilient to noise than any one system
on its own. In doing so, QEC distinguishes between two layers of abstraction: **noisy physical qubits**
and **ideal logical qubits** protected from noise through error correction. 

To ground this idea, we can borrow intuition from classical error correction. One of the simplest schemes 
is the **repetition code**. The concept is straightforward: replicate the information multiple times. For instance, 
you could use three bits to represent a single logical bit by encoding:

.. math::  0 \rightarrow 000, \quad 1 \rightarrow 111, 

where on the right side, you have the encoded bit. If you want to perform a logical NOT operation,
you simply flip all the bits:

.. math::  000 \rightarrow 111, \quad 111 \rightarrow 000. 

Now consider a noisy environment, where each physical bit has a probability :math:`p` of flipping.
When reading a logical bit, you check all three physical bits and take a **majority vote**. If
:math:`p` is small, then seeing something like :math:`010` likely means the middle bit was corrupted
by noise. You can confidently correct it by flipping it back, restoring the original :math:`000`
state.

This works well in classical systems—but quantum computing introduces a twist: **you cannot create
an exact copy of an arbitrary quantum state**. The no-cloning theorem forbids creating identical
copies of a quantum system while preserving the original. So, we can’t just copy a quantum state and
vote on the outcome later. However, all is not lost. Quantum mechanics offers a uniquely powerful
alternative: **entanglement** coupled with a clever protocol to detect and correct errors. Instead
of duplicating quantum states, we use entanglement to **distribute** quantum information across
multiple physical qubits.

The simplest QEC code: the quantum repetition scheme
----------------------------------------------------

A QEC scheme is a method for constructing logical qubits that are more robust to noise than the
underlying *physical qubits*. To understand how this works, let’s begin with the simplest example:
the **quantum repetition code**.

Unlike classical bits, qubits can experience several types of errors:

- Bit-flip :math:`(X)`: swaps :math:`\vert 0 \rangle  \leftrightarrow  \vert 1 \rangle`
- Phase-flip :math:`(Z)`: swaps :math:`\vert + \rangle  \leftrightarrow  \vert - \rangle`
- Combined :math:`(Y)`: a simultaneous bit- and phase-flip, represented as :math:`iXZ`
- Other: such as amplitude damping, depolarizing, etc

Repetition codes are the most basic form of quantum error correction. While they’re conceptually
easy to understand, they are not very powerful and can only correct *bit-flip* errors. We’ll see why
this limitation exists by the end of the demo.

In classical error correction, redundancy is added by directly copying bits. But in quantum
computing, the no-cloning theorem prevents copying arbitrary quantum states. Instead, redundancy is
introduced *indirectly*, using **entanglement** and **syndrome extraction**. Setting up a QEC scheme
involves three main steps:

1. **Define how logical qubits are encoded** within the physical system.
2. **Extract information about possible errors**, without disturbing the logical state.
3. **Infer the error from that information** and apply the appropriate correction.

Drawing from the classical repetition code, we can encode one logical qubit across three physical
qubits:

.. math::  \alpha \vert 0 \rangle + \beta \vert 1 \rangle \rightarrow  \alpha \vert 000 \rangle + \beta \vert 111\rangle. 

This entangled state now represents a single logical qubit across three physical ones.

The circuit below shows how to generate the logical qubit. As discussed, this is done by creating an
entangled state across three physical qubits---laying the groundwork for robust error correction
through QEC.

.. figure:: ../_static/demonstration_assets/loom_catalyst/encode_qubits.png
    :align: center
    :width: 70% 

To detect and later correct any errors in this system, we introduce a **syndrome extraction
circuit**, which operates in three phases:

1. **Introduce auxiliary qubits**: These auxiliary qubits don’t carry any logical information.
   Instead, they’re used solely to probe for errors.
2. **Entangle data and auxiliary qubits**: This step allows the auxiliary register to “pick up”
   error information---called the syndrome---without directly measuring the data qubits.
3. **Measure the auxiliary qubits**: The measurement reveals a pattern (the syndrome) that tells us
   where and what kind of error has likely occurred.

Based on the syndrome measurements, we infer the specific error---say, a bit-flip on qubit 2---and apply
the corresponding recovery operation to restore the logical state.

After encoding the logical qubit, we add another two qubits, the auxiliary qubits. 

.. figure:: ../_static/demonstration_assets/loom_catalyst/parity_meas.png
    :align: center
    :width: 70%

At this point, we’re working with **five qubits** in total: three data qubits that define the
logical qubit, and two auxiliary qubits used for extracting the error syndrome (we have gone a long
way to get a single, ideal logical qubit!). The circuit in the previous image concludes with
measurements on the two auxiliary qubits, yielding two classical bits of information---the **syndrome
signature**.

For this simple enough case, we can go through all the possible errors and see what happens to the
syndrome signature. Recall that a Pauli-:math:`X` gate flips :math:`Z`-basis eigenstates
:math:`(\vert 0 \rangle  \leftrightarrow  \vert 1 \rangle).` To simulate an error, we insert an
:math:`X` gate after preparing the logical state but *before* running the syndrome extraction
circuit. We obtain four cases illustrated below.

.. figure:: ../_static/demonstration_assets/loom_catalyst/syndrome00.png
    :align: center
    :width: 75%

.. figure:: ../_static/demonstration_assets/loom_catalyst/syndrome01.png
    :align: center
    :width: 75%

.. figure:: ../_static/demonstration_assets/loom_catalyst/syndrome10.png
    :align: center
    :width: 75%

.. figure:: ../_static/demonstration_assets/loom_catalyst/syndrome11.png
    :align: center
    :width: 75%

As it is easy to verify, the three locations where the error may arise give rise to an independent
signature over the measurement outcomes. Explicitly writing the match between syndromes and
corrections is called generating the “lookup table”.

Catalyst implementation
-----------------------

Let’s walk through how to implement a *quantum memory* experiment using **Catalyst** and **Loom**. A
memory experiment involves running several cycles of a quantum error correction code, recording the
resulting *syndrome measurements*, and using them to determine the appropriate error corrections. To
build such an experiment, we need three key components:

1. **Syndrome extraction circuit** responsible for capturing information about any errors affecting
   the data qubits.
2. **Syndrome decoder** which analyzes that information and identifies the most likely error that
   occurred.
3. **A mechanism to link the two** enabling the decoded syndrome to be translated into real-time
   corrective actions.

Let’s implement this natively using the PennyLane and Catalyst frameworks, relying on the hard-coded
solution or lookup table.
"""

from catalyst import qjit, cond, measure, debug
from jax import random, numpy as jnp
import pennylane as qml
qml.capture.enable()

distance = 3

data_qubits = [i for i in range(distance)]
aux_qubits = [i + distance for i in range(distance - 1)]
n_qubits = len(data_qubits) + len(aux_qubits)


######################################################################
# This instantiates the 5 qubits that we need, 3 data and 2 auxiliary. We then proceed to instantiate
# the Pennylane backend.
# 

dev = qml.device("qrack.simulator", wires=n_qubits)

######################################################################
# Then, we proceed to generate the circuit. Note that we start by introducing an error by hand, and
# later try to run a noisy simulation.
# 

@qjit()
@qml.qnode(dev)
def circuit(seed : int):

    # Based on the seed, apply an X gate to a random data qubit
    # If the result is -1, then don't apply anything
    random_qubit = random.randint(random.PRNGKey(seed),(1,),-1,distance)[0]

    # Define conditional noise application and apply it only if random_qubit is not -1
    @cond(random_qubit != -1)
    def apply_noise():
        debug.print("Applying noise to qubit: {}", random_qubit)
        qml.X(random_qubit)
    
    apply_noise()

    # Syndrome extraction routine: entangle data and auxiliary, and measure auxiliary
    for i in range(distance-1):
        qml.CNOT(wires=[data_qubits[i], aux_qubits[i]])
        qml.CNOT(wires=[data_qubits[i+1], aux_qubits[i]])
    syndrome = [measure(aux_qubit) for aux_qubit in aux_qubits]


    # Fix the data qubits based on the auxiliary qubit measurements
    @cond(jnp.logical_and(syndrome[0] == 0, syndrome[1] == 1))
    def fix_data_qubits():
        debug.print("Applying correction on data qubit 2")
        qml.X(data_qubits[2])
    @fix_data_qubits.else_if( jnp.logical_and(syndrome[0] == 1, syndrome[1] == 0))
    def fix_data_qubits():
        debug.print("Applying correction on data qubit 0")
        qml.X(data_qubits[0])
    @fix_data_qubits.else_if(jnp.logical_and(syndrome[0] == 1, syndrome[1] == 1))
    def fix_data_qubits():
        debug.print("Applying correction on data qubit 1")
        qml.X(data_qubits[1])
    
    # Apply the the fix
    fix_data_qubits()


    # Measure the data qubits
    data_qubit_measurements = [measure(data_qubit) for data_qubit in data_qubits]

    return data_qubit_measurements, syndrome

######################################################################
# Let’s break it down. We start by applying a random :math:`X` gate to one of the data qubits---this
# serves as our (deliberately introduced) noise for the experiment. Next, we run the syndrome
# extraction routine, implemented by applying CNOT gates between the data and auxiliary qubits,
# followed by measurements on the auxiliary qubits. Finally, we use a conditional statement:
# ``@cond(jnp.logical_and(syndrome[0] == 0, syndrome[1] == 1))``. This checks the specific signature
# of the measured syndromes and applies the corresponding correction when that condition is met.
# 
# If we then run the script (with a fixed seed, for reproducibility),
# 

logical_measurement, syndrome = circuit(0)

######################################################################
# we obtain:
# ``Logical_measurement = [Array(False, dtype=bool), Array(False, dtype=bool), Array(False, dtype=bool)]``
# and ``Syndrome = [Array(False, dtype=bool), Array(True, dtype=bool)]``
# 
# This result is exactly what we expected: the logical state has been restored to the logical “0”
# state—i.e., :math:`000`. The syndrome signature is :math:`01`, which corresponds to an :math:`X`
# error on the third data qubit (or the second qubit, if you follow Python’s convention of zero-based
# indexing).
# 
# Extending it all with Loom
# --------------------------
# 
# Lookup tables can be useful, but their scalability quickly becomes a problem as the system grows.
# For instance, if you want to run multiple QEC cycles and stack them together, the number of possible
# syndrome combinations you’d need to precompute increases **exponentially** with the number of
# measurements. To address this challenge, we can turn to Loom, Entropica Labs’ solution for
# designing, automating, and orchestrating quantum error correction experiments 
# (if interested in knowing more and how to access, email `info@entropicalabs.com <info@entropicalabs.com>`__).
# 

distance = 3

lattice = Lattice.linear((distance,))

initial_patch = RepetitionCode.create(
    d=distance, check_type="Z", lattice=lattice, unique_label="alpha", position=(0,)
)

# Define Caterpillar workflow
initial_state_rc = "0"
n_cycles = distance


operations_rc = [
    ResetAllDataQubits("alpha", initial_state_rc),
    MeasurePatchSyndromes("alpha", n_cycles=n_cycles),
    MeasureLogicalZ("alpha"),
]

# Define the eka
eka_rc = LSCRD(lattice=lattice, patches=[initial_patch], operations=operations_rc)
# Interpret the eka
interpreted_eka_rc = interpret_lscrd(eka_rc)


######################################################################
# Breaking it down, we begin by initialising the logical state of the repetition code in the
# logical “0” state—i.e., :math:`000`. From there, we run the repetition code circuit for three QEC
# cycles, during which we repeatedly measure the syndrome to monitor for potential errors.
# At the end of the experiment, we measure the logical state of the encoded qubit. 
#
# If the name of variables left you wondering, EKA is loom’s qec-centric internal representation. From Sanskrit एक
# (eka, “one, first”), EKA provides a data structure that serves as a single source of truth
# throughout the entire execution of a quantum error correction code. If no errors have occurred
# throughout the process, we expect the final state to remain :math:`0_L`, that is, the encoded
# version of :math:`000`.
# 
# So far, we have to define everything in the abstract layer represented by Loom. If we want to obtain
# the actual circuit, then we simply invoke
# 

# Convert the circuit to a PennyLane circuit runnable on catalyst
circuit_rc, reg_rc = convert_circuit_to_pennylane(
    interpreted_eka_rc.final_circuit, is_catalyst=True
)

######################################################################
# This function returns another function that can be invoked to execute the PennyLane circuit. Since
# the ``is_catalyst=True`` flag is set, it must be used within the ``@qjit`` decorator. Additionally,
# it returns a dictionary called ``reg_rc``, which contains the qubit register associated with the
# PennyLane circuit. If we try to plot the circuit, we can clearly see the 3 QEC cycles:
# 
# .. figure:: ../_static/demonstration_assets/loom_catalyst/circuit_3QEC.png
#    :align: center
#    :width: 90%
# 
# We generate a PennyLane device, as we always do:
# 

# Define device
dev = qml.device(
    "qrack.simulator",
    wires=len(reg_rc),
    shots=1,
    noise=0.05,
    isTensorNetwork=False,
    isStabilizerHybrid=True,
)


######################################################################
# Next, we need to define the decoding algorithm. Note that we are using qrack with a noise flag now!
# Catalyst allows us to create classical routines using JAX, so that’s exactly what we’ll use here.
# 

# Get the decoding function in JAX
union_find_decoder_rc = get_uf_decoding_function(
    edges=edges_rc, boundary_nodes=boundary_nodes_rc
)
# and accelerate it with Catalyst
union_find_decoder_rc_accelerated = accelerate(union_find_decoder_rc)

######################################################################
# Finally, we are ready to run it all together and let the Catalyst compiler do its magic and weave
# all into a single compiled circuit.
# 

@qjit()
@qml.qnode(dev)
def repetition_code_circuit():

    ## Run the circuit
    # Run the circuit and get the measurement results
    m_res = circuit_rc()

    ## Logical Observables
    # Obtain all the bits associated with each observable
    observables = jnp.array(
        [
            get_value_from_register(m_res, obs)
            for obs in interpreted_eka_rc.logical_observables
        ]
    )

    ## Detectors and decoding
    # Calculate detector values
    detector_values = jax.numpy.array(
        [get_value_from_register(m_res, det) for det in interpreted_eka_rc.detectors]
    )

    # Decode the detector values using the union-find decoder
    edge_corrections = union_find_decoder_rc_accelerated(detector_values)

    ## Final Observables
    observables_after_decoding = observables.copy()
    observables_after_decoding = jax.lax.fori_loop(
        0,
        edge_corrections.shape[0],
        # For every edge, apply the corrections to the observables if the edge is faulty
        # i.e., if the edge_corrections[edge_idx] == True
        lambda edge_idx, decoded_obs: decoded_obs
        ^ (fault_ids_rc[edge_idx] * edge_corrections[edge_idx]),
        observables_after_decoding,
    )

    return observables, observables_after_decoding


######################################################################
# This main function executes the circuit on the desired PennyLane backend, here qrack. Then, it
# extracts the syndromes for the 3 QEC cycles that we have performed, decodes them, and applies a
# correction if the error has occurred.
# 

repetition_code_circuit()

######################################################################
# We obtain: ``(Array([0], dtype=int64), Array([0], dtype=int64))``
# 
# In this case, we have been lucky! No logical error has occurred. Sure enough, we can try to run the
# circuit many times, and check how often the logical observable was changed (and correctly
# recovered).
# 

# Repeat the experiment multiple times to get a better estimate of the logical observables
n_repeats = 10000

results_rc = [repetition_code_circuit() for _ in range(n_repeats)]

final_state_rc = initial_state_rc

n_log_errors_wo_decoding_rc = sum(
    1 for obs, _ in results_rc if obs[0] != int(final_state_rc)
)
n_log_errors_w_decoding_rc = sum(
    1
    for _, obs_after_decoding in results_rc
    if obs_after_decoding[0] != int(final_state_rc)
)

print(
    f"Probability of logical error without decoding: {n_log_errors_wo_decoding_rc / n_repeats:.4f}"
)
print(
    f"Probability of logical error with decoding: {n_log_errors_w_decoding_rc / n_repeats:.4f}"
)
######################################################################
# ``Probability of logical error without decoding: 0.0903``
# 
# ``Probability of logical error with decoding: 0.0360``
# 
# Indeed, the probability of getting a logical error without decoding is much higher. Interestingly
# enough, though, even with the QEC scheme, the probability of error did not go to zero! Wonder why?
# 
# When we defined the ``qml.device``, we set the noise level at at 0.05:
# 

# Define device
dev = qml.device(
    "qrack.simulator",
    wires=len(reg_rc),
    shots=1,
    noise=0.05,
    isTensorNetwork=False,
    isStabilizerHybrid=True,
)


######################################################################
# So, we can correct some of the errors, but not all. The main limitation is that our current QEC
# scheme can only handle bit-flip errors---and only those that occur after the syndrome extraction
# circuit and before measurement. The Qrack noise model, however, can introduce errors anywhere in the
# circuit, and some of these will inevitably propagate to the logical qubit. To achieve greater
# resilience, we’ll need to explore more advanced codes and introduce the concept of fault tolerance.
# 
# Conclusions
# -----------
# 
# We like to think of quantum error correction as a classic steam engine, steadily chugging through
# three essential steps: entangling the data and auxiliary qubits, extracting the syndrome by
# measuring the auxiliaries, and then decoding and correcting any detected errors. Once this engine is
# up and running, it keeps going until the computation is complete, tirelessly counteracting the
# physical noise that plagues any real-world quantum system.
# 
# In this demo, we explored some of the core concepts behind QEC theory. More importantly, we
# demonstrated how to integrate these components into a single hybrid workflow. We used Loom to design
# the experiment at a high level, and Catalyst to compile both quantum and classical routines into a
# unified program---exactly what’s needed to keep this error-correcting engine in motion.
# 
# What next? Well, one can start exploring how to perform actual computations on these logical
# qubits---and, step by step, transition from simple repetition codes to more sophisticated quantum
# error correction engines. I mean, codes!
# 
# References
# -----------
#
# .. [#Munro2013]
#    Simon J. Devitt, Kae Nemoto, William J. Munro.
#    "Quantum Error Correction for Beginners.",
#    `<https://arxiv.org/abs/0905.2794>`__, 2013.
