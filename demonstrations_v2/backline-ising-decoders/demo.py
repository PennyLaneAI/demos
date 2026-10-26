r"""
Hierarchical QEC decoding with Backline, NVIDIA Ising, and PyMatching
=====================================================================

To achieve fault-tolerant quantum computing (FTQC), quantum error correction (QEC) decoders must
process streams of physical measurement syndromes and compute logical corrections in real time —
often within sub-microsecond latency windows. This "decoding wall" requires tight co-processing
across heterogeneous hardware: QPUs for quantum state execution, FPGAs for ultra-low latency signal
handling, GPUs for AI inference or parallel processing, and CPUs for complex graph algorithms.

In our previous :doc:`demo <backline>`, we saw how PennyLane, Catalyst, and Backline
allow you to take a QEC application and push it through levels of
complexity and optimization entirely from Python — scaling from local prototyping with CPUs, to
low-latency remote hardware execution with FPGAs, GPUs, and Triton.

But what if you want to Bring Your Own Decoder™? QEC decoding is an active area of research, with a
wealth of decoder tooling available across a range of codes and architectures. In this demo, we'll
use Backline to combine two existing decoders into a single decoding pipeline:

- **NVIDIA Ising**, a neural network *pre-decoder*, which we'll run on an AMD GPU; and
- **PyMatching**, a minimum-weight perfect matching *global* decoder, which we'll run in Python on the CPU
  controller.

Hierarchical decoding
---------------------

In **hierarchical decoding**, the decoding workload is split into multiple stages, each running on
the hardware that suits it best. Here, we use two stages, both working with *detectors*. A detector
compares consecutive measurements of the same stabilizer, and is 0 when they agree. A detector with
value 1 is called a *detection event*, and signals that an error occurred nearby.

The first stage is a pre-decoder from `NVIDIA Ising Decoding
<https://github.com/NVIDIA/ising-decoding>`__ [#chamberland2026fast]_, a family of 3D convolutional
neural networks (CNNs) for surface codes. You can train these models yourself by following the
instructions in the repository, or download trained models from `Hugging Face
<https://huggingface.co/nvidia/Ising-Decoder-SurfaceCode-1-Fast>`__. When given the detectors from
several rounds of syndrome measurement, the pre-decoder returns

- a predicted logical correction, and
- the *residual* detectors, with most of the detection events removed.

The pre-decoder only removes errors that it can identify from their local neighborhood, so a global
decoder must then decode the residual detectors. For surface codes, a common choice is `PyMatching
<https://github.com/oscarhiggott/PyMatching>`__, a Python and C++ library that implements a
minimum-weight perfect matching (MWPM) decoder known as Sparse Blossom [#higgott2025sparse]_. The
final logical correction combines the corrections predicted by both stages.

Why combine an AI pre-decoder with a traditional global matcher? The run time of matching decoders
grows with the number of detection events, which increases with the physical error rate and the
code size. However, 3D Convolutional Neural Networks (CNNs) excel at processing these dense, local
noise patterns massively in parallel on GPUs.

By using NVIDIA Ising as a first pass, we can strip away roughly 90% of the local defects. This
leaves behind a sparse residual syndrome graph that a CPU- or FPGA-based global decoder (like
PyMatching) can solve.

We won't go into detail about how these decoding algorithms work (see `The PennyLane Guide to FTQC
<https://pennylane.ai/topics/fault-tolerant-quantum-computing>`__ to learn more). Instead, we'll
focus on how you can use both of them --- and other decoders! --- with PennyLane and Backline.

Exporting the pre-decoder to ONNX
---------------------------------

Backline can run models saved in the `ONNX format <https://github.com/onnx/onnx>`__, an open
standard for machine learning models that PyTorch (and many other frameworks) can export to. In
this section, we use PyTorch and NVIDIA's Ising repository to produce three files:

- ``predecoder.onnx``: the pre-decoder model,
- ``circuit.dem``: the error model of the surface code circuit, used by PyMatching, and
- ``shots.npz``: sampled detectors to decode, along with the correct logical outcome for each shot.

This step doesn't involve PennyLane, and only needs to be run once. If you'd like to skip
straight to Backline, jump to :ref:`backline-ising-building-the-backline`.

First, we make the utilities in the Ising repository importable, and describe the experiment we want
to decode: a :math:`d=5` surface code, with :math:`T=5` rounds of syndrome measurement, followed by
a measurement of the logical qubit in the :math:`X` basis (a *memory experiment*). NVIDIA's
repository reads these settings from a configuration file, which we load and modify:
"""

import sys

import numpy as np
import torch

# A local clone of github.com/NVIDIA/ising-decoding at commit 33acb15, with the
# packages from its requirements_public_inference.txt installed.
ISING_REPO = "/path/to/ising-decoding"
sys.path.insert(0, f"{ISING_REPO}/code")

from omegaconf import OmegaConf
from workflows.config_validator import apply_public_defaults_and_model, validate_public_config
from evaluation.logical_error_rate import PreDecoderMemoryEvalModule, _build_stab_maps
from export.safetensors_utils import load_safetensors

cfg = OmegaConf.load(f"{ISING_REPO}/conf/config_public.yaml")
cfg.distance = 5
cfg.n_rounds = 5
cfg.workflow.task = "inference"
cfg = apply_public_defaults_and_model(cfg, validate_public_config(cfg))
cfg.test.meas_basis_test = "X"
cfg.test.num_samples = 10000  # the number of shots to sample later

######################################################################
# Next, we load the trained weights of the `Ising-Decoder-SurfaceCode-1-Fast
# <https://huggingface.co/nvidia/Ising-Decoder-SurfaceCode-1-Fast>`__ model, downloaded from Hugging
# Face. The checkpoint stores them in half precision, so we convert them to single precision.
#
# On its own, the CNN works with a 3D grid of stabilizer values. NVIDIA's
# ``PreDecoderMemoryEvalModule`` wraps the CNN so that it instead takes each shot's detectors as a
# flat array, and returns the logical correction and residual detectors in the same flat format.
# This wrapped module is the complete pre-decoder that we will run on the GPU.

model, _ = load_safetensors(
    f"{ISING_REPO}/models/ising_decoder_surface_code_1_fast_r9_v1.0.77_fp16.safetensors",
    device="cpu",
)
model = model.float().eval()

# Where each stabilizer sits on the d x d grid of the surface code.
maps = _build_stab_maps(cfg.distance, cfg.data.code_rotation)
pipeline = PreDecoderMemoryEvalModule(model, cfg, maps, torch.device("cpu")).eval()

######################################################################
# We can now export the pre-decoder with ``torch.onnx.export``. The exporter traces the model
# using an example input, so we provide a random array of detectors with the right shape: one
# detector per stabilizer (:math:`d^2-1 = 24`) per round (:math:`T = 5`), for 120 in total. The
# remaining options are the same ones that NVIDIA's repository uses for `its own ONNX export
# <https://github.com/NVIDIA/ising-decoding/blob/33acb15/code/evaluation/logical_error_rate.py>`__.

num_dets = cfg.test.n_rounds * (cfg.distance**2 - 1)
example_dets = torch.randint(0, 2, (1, num_dets), dtype=torch.uint8)

torch.onnx.export(
    pipeline,
    example_dets,
    "predecoder.onnx",
    input_names=["dets"],
    output_names=["L_and_residual_dets"],
    # The first axis is the number of shots, which may vary between calls.
    dynamic_axes={"dets": {0: "batch"}, "L_and_residual_dets": {0: "batch"}},
    opset_version=18,
    external_data=False,
    do_constant_folding=True,
    dynamo=False,
)

######################################################################
# The result is a single ONNX model that maps an array of shape ``uint8[batch, 120]`` to an array
# of shape ``uint8[batch, 121]``. For each shot, the first byte of the output is the pre-decoder's
# logical correction, and the remaining 120 bytes are the residual detectors.
#
# To test the decoders, we also need shots to decode. We sample them by
# simulating the memory experiment  with `Stim <https://github.com/quantumlib/stim>`__. Each sampled
# shot contains the 120 detectors, followed by the value of the logical observable --- whether
# errors flipped the logical qubit. We save the detectors
# and observables to ``shots.npz``, and the circuit's detector error model, which describes the
# probability of each error and the detectors it triggers, to ``circuit.dem``:

from data.factory import DatapipeFactory

dataset = DatapipeFactory.create_datapipe_inference(cfg)
sampler = dataset.circ.stim_circuit.compile_detector_sampler(seed=42)
samples = sampler.sample(cfg.test.num_samples, append_observables=True).astype(np.uint8)
np.savez_compressed("shots.npz", dets=samples[:, :-1], obs=samples[:, -1])

dem = dataset.circ.stim_circuit.detector_error_model(
    decompose_errors=True, approximate_disjoint_errors=True
)
dem.to_file("circuit.dem")

######################################################################
# ``shots.npz`` now holds two arrays, both sampled from the same Stim circuit that ``circuit.dem``
# describes:
#
# - ``dets``: one row of 120 detector bytes per shot representing the syndrome measurements, and
# - ``obs``: the expected logical observable for each shot.
#
# .. _backline-ising-building-the-backline:
#
# Building the backline
# ---------------------
#
# With the three files in hand, we can turn to PennyLane and Backline. We start by loading the
# ONNX model as a coprocessing function using `qp.backline.onnx_decoder
# <https://docs.pennylane.ai/en/latest/code/api/pennylane.backline.onnx_decoder.html>`__:

import pennylane as qp

predecoder_fn = qp.backline.onnx_decoder("predecoder.onnx")

######################################################################
# .. note::
#
#     ``onnx_decoder`` runs the model with `ONNX Runtime <https://onnxruntime.ai>`__, so the same
#     program can run on different hardware.
#
#     By default, it uses the GPU supported by your installed
#     ONNX Runtime package --- for example, AMD GPUs with ``onnxruntime-migraphx``, or NVIDIA GPUs
#     with ``onnxruntime-gpu`` --- and falls back to the CPU if there is none.
#
#     To choose explicitly,
#     pass the ``provider`` argument, for example ``provider="cpu"``. Currently, the coprocessor
#     running the model must be in the same process as the controller.
#
# Next, we create the backline. As in our :doc:`previous demo <backline>`, a backline consists of a
# **controller**, which executes the quantum program on a quantum device (here, the
# ``lightning.qubit`` simulator) and sends data off for processing, and one or more
# **coprocessors**, which process that data.
#
# In this example, we will define a local CPU controller and a local GPU coprocessor.

qdev = qp.device("lightning.qubit", wires=1)

CPU = qp.Controller(device=qdev)
GPU = qp.Coprocessor(hardware="gpu", coprocessor_fn=predecoder_fn)
dev = qp.Backline(controller=CPU, coprocessors=[GPU], transport="memcpy")

######################################################################
# Note that we didn't need to specify the size of the messages exchanged between the controller and
# the coprocessor (120 bytes sent, and 121 bytes received, per shot); Backline reads them from the
# input and output shapes of the ONNX model.
#
# For the second stage, we load the detector error model with Stim, and build a PyMatching decoder
# from it:

import pymatching
import stim

error_model = stim.DetectorErrorModel.from_file("circuit.dem")
matcher = pymatching.Matching.from_detector_error_model(error_model)

######################################################################
# You may have noticed that we have not registered the PyMatching outer decoder
# as a coprocessing function; since this is a Python function, we will simply
# use a :func:`~catalyst.pure_callback` to call this function directly from the controller.
# However, if your outer decoder **is** a compiled function, this can be registered to
# a local or remote coprocessor if desired.

from catalyst import pure_callback


@pure_callback
def match(residual) -> int:
    """PyMatching's predicted logical correction for one shot's residual detectors."""
    return np.asarray(matcher.decode(np.asarray(residual))[0], dtype=np.int64)


######################################################################
# .. note::
#
#     PyMatching's core is written in C++, but it doesn't ship a library with a C interface that
#     Backline can call directly. To run PyMatching on a coprocessor instead, you could:
#
#     1. Write a small C++ wrapper that exposes a function to decode one shot's detectors with
#        PyMatching;
#
#     2. Compile the wrapper, together with the PyMatching C++ source code, into a shared library;
#        and
#
#     3. Register the shared library as a coprocessing function with ``qp.CoprocessorFunction``,
#        or call it from the controller with ``qp.runtime_call``.
#
# Decoding on the backline
# ------------------------
#
# We can now write the QNode that decodes a single shot. The figure below shows how data flows
# between the controller and the coprocessor:
#
# .. figure:: ../demonstrations_v2/backline-ising-decoders/dataflow.png
#     :align: center
#     :width: 80%
#
#     Decoding one shot. Each square is one byte; only a few of the 120 detectors are shown.
#
# The QNode takes the shot's 120 detectors, and
#
# 1. sends them to the GPU coprocessor with ``qp.backline.decode``, which returns the pre-decoder's
#    121-byte output (``bitpack=False`` sends one byte per detector, matching the model's input);
#
# 2. runs PyMatching on the residual detectors, and combines the two predicted logical corrections
#    (a logical correction applied twice cancels out, so we take their XOR); and
#
# 3. applies the resulting logical correction.
#
# In this demo, the detectors come from the Stim samples rather than from measurements on a quantum
# device, so the controller's device only needs a single qubit, standing in for the logical qubit.
# In a full execution, the QNode would also contain the surface code circuit whose measurements
# produce the detectors.

from jax import numpy as jnp


@qp.qnode(dev)
def decode_shot(dets):
    """Decode one shot and apply the predicted logical correction."""
    # Stage 1: pre-decode on the GPU coprocessor.
    predecoded = qp.backline.decode(dets, bitpack=False)
    pre_flip, residual = predecoded[0], predecoded[1:]

    # Stage 2: decode the residual detectors with PyMatching on the controller.
    flip = pre_flip.astype(jnp.int64) ^ match(residual)

    if flip == 1:
        qp.X(0)

    return qp.expval(qp.Z(0))


######################################################################
# Finally, we compile a program that decodes every shot. Since :math:`\langle Z\rangle` is
# :math:`+1` if no correction was applied and :math:`-1` if corrected, the quantity
# :math:`(1-\langle Z\rangle)/2` recovers the predicted logical correction (:math:`0` or :math:`1`)
# for each shot. With ``autograph=True``, we can use regular Python ``for`` loops and ``if``
# statements, which are converted into compiled control flow.

shots = np.load("shots.npz")
dets, obs = shots["dets"], shots["obs"]


@qp.qjit(capture=True, autograph=True)
def decode_shots(dets):
    flips = jnp.zeros(dets.shape[0], dtype=int)

    for i in range(dets.shape[0]):
        flips[i] = jnp.round((1 - decode_shot(dets[i])) / 2).astype(int)

    return flips


predicted = np.asarray(decode_shots(dets))
print(predicted)

######################################################################
# .. rst-class:: sphx-glr-script-out
#
#   .. code-block:: none
#
#       [0 0 1 ... 0 0 0]
#
# A shot is decoded correctly if the predicted logical correction matches the logical observable
# from the simulation. The *logical error rate* is the fraction of shots where they differ:

print(f"Logical error rate (Ising + PyMatching): {np.mean(predicted != obs):.4f}")

######################################################################
# .. rst-class:: sphx-glr-script-out
#
#   .. code-block:: none
#
#       Logical error rate (Ising + PyMatching): 0.0035
#
# How does this compare with using PyMatching alone, without pre-decoding? To check, we only need to
# modify the QNode so that it passes the detectors straight to PyMatching:


@qp.qnode(dev)
def decode_shot_pymatching(dets):
    """Decode one shot with PyMatching only, and apply the predicted logical correction."""
    flip = match(dets)

    if flip == 1:
        qp.X(0)

    return qp.expval(qp.Z(0))


@qp.qjit(capture=True, autograph=True)
def decode_shots_pymatching(dets):
    flips = jnp.zeros(dets.shape[0], dtype=int)

    for i in range(dets.shape[0]):
        flips[i] = jnp.round((1 - decode_shot_pymatching(dets[i])) / 2).astype(int)

    return flips


predicted_pymatching = np.asarray(decode_shots_pymatching(dets))
print(f"Logical error rate (PyMatching only):    {np.mean(predicted_pymatching != obs):.4f}")

######################################################################
# .. rst-class:: sphx-glr-script-out
#
#   .. code-block:: none
#
#       Logical error rate (PyMatching only):    0.0050
#
# Pre-decoding with NVIDIA Ising lowers the logical error rate, from 0.0050 with PyMatching alone
# to 0.0035.
#
# Performance
# -----------
#
# How fast is the pre-decoder on Backline? We timed it on an AMD Instinct™ MI210 GPU (ROCm 7.2.4,
# MIGraphX 2.15, ONNX Runtime 1.27.1) and on an AMD EPYC™ 9275F CPU with 8 threads, using the
# ``predecoder.onnx`` and ``shots.npz`` files from above, in two ways:
#
# - **Through Backline**: a compiled program sends each shot from the controller, through the
#   transport, to the coprocessor running the pre-decoder, and back, as ``decode_shot`` does, but
#   without PyMatching. We report the average time per shot over 5,000 shots.
#
# - **With ONNX Runtime's Python API**: the same model, without Backline, either one shot per call
#   or in batches of 64 and 1,024 shots per call. We report the median time per call, with the 99th
#   percentile (p99) in parentheses, and, for batches, the average time per shot.
#
# .. rst-class:: docstable
#
#   +---------------------------------------+-----------------------+-----------------------+--------------+
#   | Measurement                           | CPU (8 threads)       | MI210 GPU             | GPU speed-up |
#   +=======================================+=======================+=======================+==============+
#   | One shot through Backline (average)   | 1,457 µs              | 433 µs                | 3.4x         |
#   +---------------------------------------+-----------------------+-----------------------+--------------+
#   | One shot, ONNX Runtime                | 1,400 µs (1,431 µs)   | 422 µs (433 µs)       | 3.3x         |
#   +---------------------------------------+-----------------------+-----------------------+--------------+
#   | 64 shots per call, ONNX Runtime       | 83.5 ms (83.9 ms)     | 2.79 ms (2.80 ms)     | 30x          |
#   |                                       |                       |                       |              |
#   | (average per shot)                    | (1,305 µs per shot)   | (43.5 µs per shot)    |              |
#   +---------------------------------------+-----------------------+-----------------------+--------------+
#   | 1,024 shots per call, ONNX Runtime    | 1,340 ms (1,347 ms)   | 19.6 ms (19.8 ms)     | 69x          |
#   |                                       |                       |                       |              |
#   | (average per shot)                    | (1,310 µs per shot)   | (19.1 µs per shot)    |              |
#   +---------------------------------------+-----------------------+-----------------------+--------------+
#
# These times vary by a few percent from run to run. A few observations:
#
# - **Backline adds little to the pre-decoder's run time.** On the GPU, a shot through Backline
#   takes about as long as a call to ONNX Runtime from Python (433 µs and 422 µs), even though it
#   also crosses the transport. On the CPU it is about 4% slower (1,457 µs and 1,400 µs), mostly
#   because Backline's coprocessor runs ONNX Runtime with its threads' spinning turned off, so that
#   they do not compete with the transport's threads.
#
# - **The CPU is compute bound.** Batching barely changes its time per shot (from 1,400 µs to
#   1,310 µs), and neither do more threads: 8 threads were about 10% faster than 1, and all 22 cores
#   available to us were slower than 8.
#
# - **For a single shot, the GPU is mostly idle.** MIGraphX's profiler shows that almost all of the
#   single-shot time is spent running the model's kernels, not copying data or launching them. Most
#   of it goes to the four 3D convolutions, which, with only one shot, are too small to occupy the
#   MI210's 104 compute units. A batch of 64 shots fills more of the GPU, and takes only 6.6 times as
#   long as one shot.
#
# - **Batches help throughput, not latency.** When decoding a running quantum program, each shot's
#   detectors arrive one at a time, and what matters is how soon each correction is ready: about
#   0.43 ms per shot on the GPU. Batching shots would make each shot wait for the rest of its batch.
#   Batches are useful when many logical qubits produce detectors at the same time (64 logical qubits
#   could be pre-decoded together in 2.8 ms), or when decoding offline, as we did for the logical
#   error rates above. ``onnx_decoder`` currently sends one shot per message, so through Backline the
#   batch size is always 1.
#
# - **Both devices give consistent times with 8 threads.** The p99 is within about 3% of the median
#   on both. On the CPU, using all 22 cores instead made the p99 erratic, as high as 4.4 ms.
#
# Note that the times above leave out start-up. Each QNode call that uses the coprocessor costs about 7 ms to
# set up, on both devices, so ``decode_shots``, which makes one QNode call per shot, takes about 7 ms
# per shot. To decode many shots quickly, decode them within a single QNode call, as our timings do.
# On the GPU, MIGraphX also compiles the model each time a program starts, which takes a few
# seconds. Setting the ``ORT_MIGRAPHX_MODEL_CACHE_PATH`` environment variable to an existing
# directory makes ONNX Runtime save the compiled model there and reuse it.
#
# To run these benchmarks and reproduce these measurements, see the `onnx_demo_performance.py <tk>`__ in the Backline repository.
#
# Conclusion
# ----------
#
# Historically, orchestrating this level of heterogeneous execution required writing extensive,
# low-level language bindings and custom memory management boilerplate.
#
# With Backline, no decoder is out of reach. Whether you are bringing custom PyTorch or
# JAX neural decoders, compiled C++ libraries, or highly optimized Triton and CUDA kernels, you
# can seamlessly drop them into your pipeline directly from Python. Backline handles the underlying
# orchestration, memory transfers, and latency optimizations automatically.
#
# To continue exploring heterogeneous quantum compilation and execution, check out the following
# resources:
#
# - Explore how you can use Backline to go from local-prototyping all the way to low-latency FPGA-GPU
#   communications with our :doc:`previous demo <backline>`.
#
# - Read the `Backline technical documentation <https://docs.pennylane.ai/en/latest/code/qp_backline.html>`__.
#
# - View additional Backline demos in the `Backline repository on GitHub
#   <https://github.com/PennyLaneAI/backline/blob/main/demos/>`__, including CPU-CPU interactions
#   over RDMA, and explicit usage of `runtime_call
#   <https://docs.pennylane.ai/en/latest/code/api/pennylane.runtime_call.html>`__ to execute
#   coprocessor functions.
#
# - Read the `Backline technical paper <https://arxiv.org/abs/2609.09270>`__ to get a technical
#   overview of the infrastructure and performance.
#
# - Check out the `PennyLane blog post
#   <https://pennylane.ai/blog/2026/09/real-time-classical-processing-with-backline-amd>`__ to
#   learn more about Backline.
#
# References
# ----------
#
# .. [#chamberland2026fast] Christopher Chamberland, Jan Olle, Muyuan Li, Scott Thornton, and Igor
#    Baratta. "Fast and accurate AI-based pre-decoders for surface codes." *arXiv preprint
#    arXiv:2604.12841* (2026).
#
# .. [#higgott2025sparse] Oscar Higgott and Craig Gidney. "Sparse blossom: correcting a million
#    errors per core second with minimum-weight matching." *Quantum 9 (2025): 1600*.
#
# *Instinct and EPYC are trademarks of Advanced Micro Devices, Inc.*
