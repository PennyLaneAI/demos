r"""
Decoding bivariate bicycle codes with Backline, Pelagos, and Relay-BP
=====================================================================

Quantum low-density parity-check (qLDPC) codes promise to protect logical qubits with far fewer
physical qubits than the surface code. But there is no such thing as a free lunch: in exchange for
the lower qubit overhead, the classical side of the computer has to work much harder. Their
decoders operate on large Tanner graphs, receive syndrome data that no longer fits in a single
machine word, and are often developed and released as complete software packages, with their own
GPU kernels and their own driver loops.

In our previous demos, we saw how to :doc:`program Backline from local CPUs all the way to FPGAs and
GPUs <backline>`, and how to :doc:`bring your own decoder <backline-ising-decoders>` in the form of
an ONNX model. But what if your decoder isn't a single kernel or a single model, but an entire
package --- a Python driver loop orchestrating a set of GPU kernels? In this demo, we'll bring
exactly that to Backline:

- **Pelagos** [#diamantopoulos2026pelagos]_, an open-source GPU implementation of the Relay-BP
  decoder, written in `Triton <https://triton-lang.org/main/index.html>`__ and PyTorch; and
- the :math:`[[144, 12, 12]]` **bivariate bicycle code** [#bravyi2024high]_, decoded at the circuit
  level, with 1,728 detectors per shot.

Along the way, we will compile Pelagos --- exactly as it is published --- into a single native
library from Python, and let Catalyst ship it to a remote GPU node. The best part? The coprocessor
never runs a Python interpreter: just Pelagos's own GPU kernels, driven by a small compiled engine.

.. note::

    This demo uses the ``pelagos_native`` package, a prototype that compiles Pelagos into a
    Backline coprocessing function, alongside the latest development versions of PennyLane and
    Catalyst. Executing the full demo requires a server with an AMD Instinct™ or NVIDIA GPU and an
    RDMA-capable NIC; the first half of the demo (defining the decoding problem and compiling the
    decoder) only requires a CPU.

From qLDPC codes to a decoding problem
--------------------------------------

Before we start writing code, let's recap what the decoder actually needs to do. A circuit-level
memory experiment repeatedly measures the code's stabilizers, and compares consecutive
measurements of the same stabilizer to produce **detectors**. A detector is 0 when the two
measurements agree, and 1 (a *detection event*) when they don't, signalling that an error occurred
nearby.

Every possible fault in the circuit --- a faulty gate, a faulty measurement, an idling error ---
flips a particular set of detectors, and might flip one or more of the code's logical observables.
Collecting this information for all faults gives us the circuit's **detector error model** (DEM),
which we can write as two binary matrices and a vector of probabilities:

- the check matrix :math:`H \in \mathbb{F}_2^{C \times V}`, where :math:`H_{ij} = 1` if fault
  :math:`j` flips detector :math:`i`;
- the observables matrix :math:`O \in \mathbb{F}_2^{K \times V}`, where :math:`O_{kj} = 1` if
  fault :math:`j` flips logical observable :math:`k`; and
- the priors :math:`p_j`, the probability of fault :math:`j`.

Given the detectors :math:`s \in \mathbb{F}_2^C` from one shot, the decoder looks for a likely set
of faults :math:`e \in \mathbb{F}_2^V` with :math:`He = s`. What we actually care about, however,
is the effect of those faults on the logical qubits, :math:`Oe`: these are the **logical
corrections** we need to apply.

Let's build this decoding problem for the :math:`[[144, 12, 12]]` bivariate bicycle code (also
known as the *gross code*), with 12 rounds of syndrome measurement and uniform circuit noise at a
physical error rate of :math:`p = 0.003`. Pelagos ships the `Stim <https://github.com/quantumlib/stim>`__
circuits for this experiment in its test data, along with a helper to convert a Stim detector error
model into the matrices above:
"""

import sys

# A local clone of github.com/quantum-supercomputing/pelagos, for its bundled Stim circuits.
PELAGOS_REPO = "/path/to/pelagos"
sys.path.insert(0, f"{PELAGOS_REPO}/tests")

from testdata import get_test_circuit
from pelagos.stim.sinter.check_matrices import CheckMatrices

circuit = get_test_circuit("bicycle_bivariate_144_12_12_memory_X", 0.003, distance=12, rounds=12)
dem = circuit.detector_error_model(decompose_errors=False)
cm = CheckMatrices.from_dem(dem)

H, O, priors = cm.check_matrix, cm.observables_matrix, cm.error_priors

print(f"detectors C = {H.shape[0]}, faults V = {H.shape[1]}, nonzeros = {H.nnz}")
print(f"logical observables K = {O.shape[0]}")

######################################################################
# .. rst-class:: sphx-glr-script-out
#
#   .. code-block:: none
#
#       detectors C = 1728, faults V = 67824, nonzeros = 391464
#       logical observables K = 12
#
# That's a Tanner graph with almost 400,000 edges --- and we need to decode one of these every
# syndrome cycle. To put this into perspective, here is how the problem grows across the bivariate
# bicycle codes that Pelagos ships with:
#
# .. rst-class:: docstable
#
#   +------------------------------------+-----------+---------+-----------------------+-------------------+
#   | Code (memory-X, :math:`p = 0.003`) | Detectors | Faults  | Nonzeros in :math:`H` | Syndrome per shot |
#   +====================================+===========+=========+=======================+===================+
#   | :math:`[[72, 12, 6]]`              | 432       | 16,200  | 91,188                | 54 bytes          |
#   +------------------------------------+-----------+---------+-----------------------+-------------------+
#   | :math:`[[144, 12, 12]]`            | 1,728     | 67,824  | 391,464               | 216 bytes         |
#   +------------------------------------+-----------+---------+-----------------------+-------------------+
#   | :math:`[[288, 12, 18]]`            | 5,184     | 206,496 | 1,201,104             | 648 bytes         |
#   +------------------------------------+-----------+---------+-----------------------+-------------------+
#
# Notice the last column. Packed into bits, the syndrome of the gross code is 216 bytes per shot.
# The reply, on the other hand, can be tiny: if the decoder returns the 12 logical corrections
# :math:`Oe` rather than the full set of 67,824 faults, it fits in just 2 bytes. Keep these two
# numbers in mind --- we will see them again shortly.
#
# Relay-BP in a nutshell
# ----------------------
#
# Belief propagation (BP) decodes by passing messages back and forth along the edges of the Tanner
# graph (see :doc:`tutorial_bp_catalyst` for a gentle introduction). Relay-BP
# [#muller2025improved]_ makes two changes to the standard min-sum variant of BP, which together
# make it accurate enough for circuit-level qLDPC decoding while staying just as parallel.
#
# **Memory.** Each fault :math:`j` mixes its prior log-likelihood ratio
# :math:`\lambda_j = \log\frac{1-p_j}{p_j}` with its marginal from the previous iteration,
# :math:`M_j(t-1)`, using a *memory strength* :math:`\gamma_j`:
#
# .. math::
#
#     \Lambda_j(t) = (1 - \gamma_j)\,\lambda_j + \gamma_j\, M_j(t-1).
#
# The rest of the iteration is ordinary min-sum. Each detector :math:`i` sends a message to each of
# its faults,
#
# .. math::
#
#     \nu_{i \to j} = (-1)^{s_i} \prod_{j' \in N(i) \setminus j} \operatorname{sgn}\left(\mu_{j' \to i}\right)
#     \min_{j' \in N(i) \setminus j} \left|\mu_{j' \to i}\right|,
#
# and each fault replies with :math:`\mu_{j \to i} = \Lambda_j(t) + \sum_{i' \in N(j)\setminus i}
# \nu_{i' \to j}`. The marginals :math:`M_j(t) = \Lambda_j(t) + \sum_{i \in N(j)} \nu_{i \to j}`
# give the hard decision :math:`e_j = [M_j(t) \le 0]`, and we stop as soon as :math:`He = s`.
#
# **Relaying.** The first *leg* runs up to :math:`T_0` iterations with a uniform memory strength
# :math:`\gamma_0`. Each of the following :math:`R` legs keeps the marginals of the previous leg,
# draws new, *disordered* memory strengths :math:`\gamma_j` (which may be negative!), and runs for
# up to :math:`T_r` more iterations. Each leg that converges contributes a candidate correction,
# and after :math:`S` of them, the decoder returns the one with the lowest weight
# :math:`\sum_j e_j \lambda_j`.
#
# Pelagos implements exactly this, batched across shots, as a set of fused Triton kernels. A Python
# loop launches them --- five kernels per iteration, followed by a check of whether every shot has
# converged --- and moves on to the next leg when needed. And this is where things get interesting
# for real-time decoding.
#
# From Python package to coprocessing function
# --------------------------------------------
#
# On Backline, a decoder runs on a **coprocessor** as a **coprocessing function**: a compiled
# function that Catalyst's runtime calls with each message the controller sends. Previously, we
# defined these in three ways:
#
# - by pointing ``qp.CoprocessorFunction`` at a pre-compiled library;
# - by writing a Triton kernel and compiling it with
#   `qp.backline.triton_decoder <https://docs.pennylane.ai/en/latest/code/api/pennylane.backline.triton_decoder.html>`__; and
# - by exporting a model to ONNX and loading it with
#   `qp.backline.onnx_decoder <https://docs.pennylane.ai/en/latest/code/api/pennylane.backline.onnx_decoder.html>`__.
#
# None of these fit Pelagos as-is. Its kernels are designed to be launched one after another from
# Python, with device buffers allocated by PyTorch, rather than inlined into a single persistent
# kernel. And while we *could* run the whole package on the coprocessor inside a Python
# interpreter, this would put Python --- and the GIL --- in the middle of every decoding round, on
# a machine whose job is to answer as fast as possible.
#
# Instead, we will compile Pelagos. The ``pelagos_native.export`` function takes the same arguments
# as Pelagos's ``RelayBPDecoder``, and returns a ``qp.CoprocessorFunction`` backed by a single
# shared library:

import pelagos_native

relay_bp = pelagos_native.export(
    H,
    priors,
    observables=O,              # reply with the 12 logical corrections, not all 67,824 faults
    platform="hip:gfx942:64",   # an AMD Instinct MI300X; use "cuda:90:32" for an NVIDIA H100
    pre_iter=80,                # T0: iterations of the first leg
    num_sets=100,               # R: relay legs
    set_max_iter=60,            # Tr: iterations per leg
    gamma0=0.1,                 # memory strength of the first leg
    gamma_dist_interval=(-0.24, 0.66),
    stop_nconv=5,               # S: candidate corrections to collect
    cflags=("-DPN_HIP", "-I/opt/rocm/include", "-L/opt/rocm/lib", "-lamdhip64"),
)

print(relay_bp.name, relay_bp.message_bytes)

######################################################################
# .. rst-class:: sphx-glr-script-out
#
#   .. code-block:: none
#
#       pelagos_native_decode (216, 2)
#
# This takes about 25 seconds on a 4-core CPU --- no GPU is required to compile the decoder. The
# returned coprocessing function already knows the size of the messages it handles: 216 bytes of
# packed detectors in, and 2 bytes of packed logical corrections out. When we attach it to a
# backline, Backline sizes its transport for these messages automatically.
#
# .. note::
#
#     Messages of up to 8 bytes travel in Backline's 16-byte ``BL16`` frames. Anything larger, such
#     as our 216-byte syndrome, automatically switches to variable-length ``BLV`` frames, which carry
#     messages of any size --- no extra configuration needed.
#
# What happens inside ``export``?
# -------------------------------
#
# Under the hood, ``export`` performs three steps:
#
# 1. **Record.** It builds the decoder with Pelagos, and runs Pelagos's own ``decode()`` method with
#    a recorder attached. The recorder logs every Triton kernel launch (with its buffers, grid, and
#    compile-time parameters), every operation on a device buffer, and every point where the
#    Python loop reads a flag back from the device to decide what to do next. No kernel actually
#    runs during recording: the two kernels that set the loop's convergence flags receive scripted
#    values instead, so that the recording walks every path through the loop in a few seconds, no
#    matter how large the code.
#
# 2. **Compile.** It compiles each kernel launch ahead of time with Triton, directly from Pelagos's
#    unmodified kernel source, for the requested GPU.
#
# 3. **Link.** It links the compiled kernels with a small C engine, along with the Tanner graph,
#    priors, and memory strengths (embedded as data), into a shared library that exposes Catalyst's
#    coprocessing function interface.
#
# The recording is organized into the segments of Pelagos's decode loop: a prologue, the first leg's
# iterations, then the relay legs, each ending in a flag read. We can take a look at what was
# recorded for our decoder:

import json
from collections import Counter
from pathlib import Path

program = json.loads((Path(relay_bp.lib_path).parent / "program.json").read_text())

for segment, ops in program["segments"].items():
    if ops:
        kernels = [op["kernel"].removesuffix("_kernel") for op in ops if op["op"] == "launch"]
        print(f"{segment:9s} {dict(Counter(op['op'] for op in ops))} {kernels or ''}")

######################################################################
# .. rst-class:: sphx-glr-script-out
#
#   .. code-block:: none
#
#       PROLOGUE  {'copy': 5, 'fill': 28, 'launch': 1, 'write': 2} ['init_messages_with_priors_EB']
#       PRE_TOP   {'fill': 1}
#       PRE_BODY  {'launch': 5, 'sync': 1} ['c2v_min_sum_btile', 'v2c_and_gamma_btile', 'parity_from_hard_compact_with_zero', 'check_and_select', 'freeze_finished_lanes']
#       DONE_TAIL {'fill': 2}
#       POST_PRE  {'copy': 1, 'launch': 2, 'sync': 1} ['freeze_finished_lanes', 'reduce_all_ge']
#       UNFREEZE  {'launch': 1} ['unfreeze_finished_lanes']
#       LEG_HEAD  {'fill': 5, 'add_scalar': 1, 'copy': 1, 'write': 1, 'launch': 1} ['init_messages_with_priors_EB']
#       LEG_BODY  {'launch': 5, 'sync': 1} ['c2v_min_sum_btile', 'v2c_and_gamma_btile', 'parity_from_hard_compact_with_zero', 'check_and_select', 'freeze_finished_lanes']
#       LEG_TAIL  {'launch': 1, 'sync': 1} ['reduce_all_ge']
#
# Here we can see Relay-BP's structure directly. Each iteration (``PRE_BODY`` and ``LEG_BODY``)
# launches the same five kernels --- check-to-variable messages, variable-to-check messages with
# memory, the parity check, solution selection, and lane freezing --- and then reads back a single
# convergence flag (``sync``). Each relay leg starts by resetting the messages and copying in a new
# row of memory strengths (``LEG_HEAD``).
#
# Only the *shape* of this loop is written by hand in the engine; everything inside the segments is
# taken from Pelagos. The exporter also checks that every repetition of a segment was recorded
# identically. If a future version of Pelagos changes its loop in a way the engine doesn't
# understand, the export fails loudly, rather than producing a decoder that silently behaves
# differently.
#
# The kernels themselves are small. Here are the resources of each compiled kernel, as reported by
# the Triton compiler for an MI300X (``gfx942``) and an H100 (``sm_90``):
#
# .. rst-class:: docstable
#
#   +----------------------------------------+-------+--------------+-----------------+
#   | Pelagos kernel                         | Warps | gfx942 VGPRs | sm_90 registers |
#   +========================================+=======+==============+=================+
#   | ``init_messages_with_priors_EB``       | 2     | 7            | 16              |
#   +----------------------------------------+-------+--------------+-----------------+
#   | ``c2v_min_sum_btile``                  | 1     | 16           | 24              |
#   +----------------------------------------+-------+--------------+-----------------+
#   | ``v2c_and_gamma_btile``                | 1     | 22           | 26              |
#   +----------------------------------------+-------+--------------+-----------------+
#   | ``parity_from_hard_compact_with_zero`` | 1     | 36           | 30              |
#   +----------------------------------------+-------+--------------+-----------------+
#   | ``check_and_select``                   | 1     | 8            | 36              |
#   +----------------------------------------+-------+--------------+-----------------+
#   | ``freeze_finished_lanes``              | 4     | 13--14       | 23              |
#   +----------------------------------------+-------+--------------+-----------------+
#   | ``reduce_all_ge``                      | 4     | 7            | 15              |
#   +----------------------------------------+-------+--------------+-----------------+
#   | ``unfreeze_finished_lanes``            | 4     | 2            | 6               |
#   +----------------------------------------+-------+--------------+-----------------+
#
# None of them spill registers, and none need scratch memory. The graph, priors, and 32 rows of
# memory strengths are embedded in the library, which comes to 18 MiB for the gross code.
#
# .. note::
#
#     Pelagos selects each kernel's launch configuration by autotuning on the GPU. Since we recorded
#     on a CPU, the autotuner picked its first valid candidate for each kernel. To compile tuned
#     configurations into the library, record on the target GPU instead.
#
#     Similarly, Pelagos draws each relay leg's memory strengths from PyTorch's GPU random number
#     generator by default. The engine uses Pelagos's ``explicit_gammas`` mode instead, where leg
#     :math:`r` takes row :math:`(r + 1) \bmod 32` of a fixed table.
#
# Building the backline
# ---------------------
#
# With our decoder compiled, we can put it on a backline. As in our :doc:`previous demo
# <backline>`, we start with the details of the remote server --- here, a server with an AMD
# Instinct™ GPU and an RDMA-capable NIC --- and a remote CPU controller:

import pennylane as qp

SERVER = {
    "host": "192.168.3.14",              # the address of the remote server
    "user": "username",                  # the SSH account on the remote machine
    "triple": "x86_64-unknown-linux-gnu",
    "deploy": ["/path/to/server-bundle"],
    "executor_bin": "numactl -N 0 -m 0 ./catalyst-executor",
    "env": {"LD_LIBRARY_PATH": "."},
}

K = O.shape[0]  # one wire per logical observable

CPU = qp.Controller(
    name="cpu-controller",
    device=qp.device("lightning.qubit", wires=K),
    remote=True,
    executor_options={**SERVER, "port": 8810},
    init_args={"config": "dev=mlx5_1;gid=3;cpu_pin=4;rt=1"},
)

######################################################################
# Next, the coprocessor. There is one subtlety here: our compiled decoder is a *host* function that
# drives the GPU itself, rather than a persistent GPU kernel that waits for messages. So we declare
# the coprocessor with ``hardware="cpu"``, even though all of the decoding happens on the GPU ---
# ``hardware`` describes the end of the RDMA link that receives each message, and here that's the
# server's CPU, which hands the syndrome to the GPU for decoding. The GPU to use is set by
# ``device=`` in the decoder's configuration (``device=0`` by default).

DECODER = qp.Coprocessor(
    name="relay-bp",
    coprocessor_fn=relay_bp,
    hardware="cpu",
    remote=True,
    endpoint=qp.Endpoint("192.168.1.2", 7760),
    executor_options={**SERVER, "port": 8813},
    init_args={"config": "dev=mlx5_1;gid=3;cpu_pin=6;rt=1"},
)

dev = qp.Backline(controller=CPU, coprocessors=[DECODER], transport="rdma")

######################################################################
# And that's all we need to do to ship the decoder. Since the coprocessor runs on another machine,
# Catalyst adds our library to the coprocessor's executor deployment: it is copied to the server
# alongside ``catalyst-executor``, and loaded there as a plugin when the program starts.
#
# Nothing else is needed on the server: no Python environment, no PyTorch, and no Triton compiler.
#
# Decoding on the backline
# ------------------------
#
# We can now write the quantum program. As in the :doc:`hierarchical decoding demo
# <backline-ising-decoders>`, we will use detectors sampled with Stim rather than measured on a
# simulator: 288 physical qubits are well beyond what any state-vector simulator can handle! The
# controller's device therefore only needs one wire per logical qubit, standing in for the 12
# logical qubits of the gross code.
#
# For each shot, the QNode
#
# 1. packs the 1,728 detectors into 216 bytes, eight detectors per byte;
# 2. sends them to the decoder with `qp.backline.decode
#    <https://docs.pennylane.ai/en/latest/code/api/pennylane.backline.decode.html>`__, which
#    returns the 2-byte reply (``bitpack=False`` sends and receives the bytes as they are); and
# 3. applies an :math:`X` correction to each logical wire whose bit is set.

from jax import numpy as jnp


@qp.qnode(dev)
def decode_shot(dets):
    """Decode one shot, and apply the predicted logical corrections."""
    packed = jnp.packbits(dets, bitorder="little")        # uint8[216]
    reply = qp.backline.decode(packed, bitpack=False)     # uint8[2]

    for k in range(K):
        if (reply[k // 8] >> (k % 8)) & 1:
            qp.X(k)

    return [qp.expval(qp.Z(k)) for k in range(K)]


######################################################################
# Since :math:`\langle Z_k \rangle = -1` exactly when we corrected logical qubit :math:`k`, the
# quantity :math:`(1 - \langle Z_k \rangle)/2` recovers the predicted corrections. Let's compile a
# program that decodes a batch of sampled shots, and compute the logical error rate --- the fraction
# of shots where *any* of the 12 predicted corrections differs from the observable flips recorded by
# Stim. With ``autograph=True``, the Python ``for`` loop below (and the ``if`` statement in the QNode
# above) are converted into compiled control flow.

import numpy as np

sampler = circuit.compile_detector_sampler(seed=42)
dets, obs = sampler.sample(1000, separate_observables=True)
dets, obs = dets.astype(np.uint8), obs.astype(np.uint8)


@qp.qjit(capture=True, autograph=True)
def decode_shots(dets):
    flips = jnp.zeros((dets.shape[0], K), dtype=int)

    for i in range(dets.shape[0]):
        z = jnp.stack(decode_shot(dets[i]))
        flips[i] = jnp.round((1 - z) / 2).astype(int)

    return flips


predicted = np.asarray(decode_shots(dets))
print(f"Logical error rate (Relay-BP on Backline): {np.mean(np.any(predicted != obs, axis=1)):.4f}")

######################################################################
# .. rst-class:: sphx-glr-script-out
#
#   .. code-block:: none
#
#       Logical error rate (Relay-BP on Backline): [to be filled in from the MI300X run]
#
# ..
#     TODO: replace the placeholder above with the output of the GPU run, and add a comparison with
#     Pelagos called directly from Python on the same shots (the corrections should be identical).
#
# How do we know it's right?
# --------------------------
#
# A compiled replay of someone else's decoder should be treated with healthy suspicion. How do we
# know the engine decodes *exactly* like Pelagos? The answer is that we can check this without a GPU
# at all.
#
# The engine talks to the GPU through a small backend interface (allocate, copy, fill, launch). Swap
# in a backend that keeps all of its buffers in host memory and runs each launch with Triton's CPU
# interpreter, and the same engine --- the same program, the same control flow, the same flag
# reads --- runs Pelagos's kernels on the CPU, where we can compare every correction against
# Pelagos itself:
#
# .. rst-class:: docstable
#
#   +--------------------------------------------------------+--------------------------------------+
#   | Check                                                  | Result                               |
#   +========================================================+======================================+
#   | Engine vs Pelagos, :math:`[[18, 4, 3]]` code capacity  | 9/9 recorded and 12/12 new syndromes |
#   |                                                        | identical, bit for bit               |
#   +--------------------------------------------------------+--------------------------------------+
#   | Engine vs Pelagos, :math:`[[72, 12, 6]]` code capacity | 5/5 syndromes identical              |
#   +--------------------------------------------------------+--------------------------------------+
#   | Engine built from a recording in which no kernel ran,  | 14/14 syndromes identical            |
#   | vs a fresh Pelagos decoder                             |                                      |
#   +--------------------------------------------------------+--------------------------------------+
#   | GPU build of the gross-code decoder, against a         | 30,704/30,704 launches match the     |
#   | stand-in driver that records every launch              | recording (kernel, grid, block, and  |
#   |                                                        | every argument)                      |
#   +--------------------------------------------------------+--------------------------------------+
#   | ``qjit`` program, memcpy transport, compiled engine,   | 4/4 corrections identical to Pelagos |
#   | corrections applied in the circuit                     |                                      |
#   +--------------------------------------------------------+--------------------------------------+
#
# The third row is what lets us record without running a single kernel: a decoder built from a
# "dry" recording decodes identically to one built from a real recording.
#
# Performance considerations
# --------------------------
#
# The compiled engine keeps Pelagos's control flow, so its cost per decode follows the same
# structure: five kernel launches and a one-byte flag read per iteration, plus a 4-byte round trip
# per relay leg. What it removes is everything *around* the kernels --- the Python interpreter, the
# GIL, tensor allocation, and autotuning --- from every decoding round.
#
# How much does Python cost on its own? As a point of comparison, we measured the round trip of a
# trivial decoder (one that just returns a byte) over Backline's in-process memcpy transport, once
# as a compiled C function and once as a Python function called from the coprocessor:
#
# .. rst-class:: docstable
#
#   +---------------------------------+-----------+
#   | Coprocessing function           | Per round |
#   +=================================+===========+
#   | Compiled C function             | 1.2 µs    |
#   +---------------------------------+-----------+
#   | Python function (via the C API) | 15.5 µs   |
#   +---------------------------------+-----------+
#
# That's around 14 µs of overhead per round before the decoder has done any work --- larger than the
# entire FPGA-to-GPU round trip we measured in our :doc:`first Backline demo <backline>`.
#
# ..
#     TODO: add the decode latency of the compiled gross-code decoder on the MI300X (median and
#     p99), and Pelagos's own single-shot time from Python on the same GPU, for comparison.
#
# A few things to keep in mind when using the compiled decoder:
#
# - **One shot per message.** Pelagos is at its fastest when decoding large batches of shots.
#   The engine currently replays single-shot decodes only, which is what real-time decoding needs
#   (each correction is needed as soon as possible), but leaves batch throughput on the table.
#
# - **Launch overhead.** Each iteration launches five small kernels. Capturing each iteration in a
#   GPU graph is a natural next step to reduce launch overhead.
#
# - **Re-export when Pelagos changes.** The library embeds the recorded program. Since ``export``
#   re-checks the structure of the loop every time, an incompatible change shows up as an export
#   error, not as a wrong correction.
#
# Conclusion
# ----------
#
# QEC decoders are increasingly developed as complete software packages, with their own GPU kernels,
# data structures, and control loops. In this demo, we took one of them, Pelagos, and turned it into
# a Backline coprocessing function, without modifying a single kernel:
#
# - we described a circuit-level decoding problem for the gross code with Stim and Pelagos;
# - we compiled Pelagos's Relay-BP decoder into a native library from Python, on a CPU;
# - Catalyst shipped the library to a remote GPU server, where it runs without Python; and
# - we wrote the quantum program that decodes each shot and applies the corrections, all within a
#   single ``qjit``-compiled workflow.
#
# The same approach applies to any decoder whose device work is a sequence of GPU kernel launches
# driven from Python: record what it does, compile what it launches, and let Backline handle the
# rest.
#
# To continue exploring heterogeneous quantum compilation and execution, check out the following
# resources:
#
# - Explore how to go from local prototyping to low-latency FPGA-GPU communication in our
#   :doc:`first Backline demo <backline>`, and how to bring an ONNX model to Backline with
#   :doc:`NVIDIA Ising and PyMatching <backline-ising-decoders>`.
#
# - Learn more about the codes we decoded today with :doc:`tutorial_qldpc_codes`.
#
# - Read the `Backline technical documentation <https://docs.pennylane.ai/en/latest/code/qp_backline.html>`__.
#
# - Read the `Backline technical paper <https://arxiv.org/abs/2609.09270>`__ for a technical
#   overview of the infrastructure and its performance.
#
# - Check out `Pelagos on GitHub <https://github.com/quantum-supercomputing/pelagos>`__.
#
# References
# ----------
#
# .. [#diamantopoulos2026pelagos] D. Diamantopoulos, L. von Niederhäusern, A. S. Anghel, A. Kayi,
#    R. Pletka, T. Müller, N. Papandreou, J. Blanusa, S. Sarafijanovic, N. M. Gonzalez, F. Abel,
#    C. Hagleitner, S. Seelam, B. Johnson, R. Haas, and H. Pozidis. "Pêlagos: An Adaptive GPU
#    Decoder for Error Correction Simulations of Quantum LDPC Codes." *2026 IEEE International
#    Conference on Quantum Software (QSW)*, pp. 152--165 (2026).
#
# .. [#bravyi2024high] Sergey Bravyi, Andrew W. Cross, Jay M. Gambetta, Dmitri Maslov, Patrick
#    Rall, and Theodore J. Yoder. "High-threshold and low-overhead fault-tolerant quantum memory."
#    *Nature* (2024). `doi:10.1038/s41586-024-07107-7 <https://doi.org/10.1038/s41586-024-07107-7>`__.
#
# .. [#muller2025improved] Tristan Müller, Thomas Alexander, Michael E. Beverland, Markus Bühler,
#    Blake R. Johnson, Thilo Maurer, and Drew Vandeth. "Improved belief propagation is sufficient
#    for real-time decoding of quantum memory." *arXiv preprint arXiv:2506.01779* (2025).
#
# *Instinct is a trademark of Advanced Micro Devices, Inc. NVIDIA is a trademark and/or registered
# trademark of NVIDIA Corporation in the U.S. and other countries.*
#
