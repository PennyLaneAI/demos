
r"""
How to use PennyLane and Backline for low-latency quantum error correction on GPUs and FPGAs
============================================================================================

Quantum is never purely quantum. Large-scale systems built for quantum computing, sensing, and
networking rely heavily on robust classical processing, from processing of applications to
low-level hardware control. Not only that, but tight communication and feedback between quantum
processors and classical accelerators are essential to identify and correct errors --- a process
known as `quantum error correction
<https://pennylane.ai/topics/fault-tolerant-quantum-computing>`__ (QEC) --- before fragile physical
information degrades.

In this demo, we will take a quantum error correction application and push it through increasing
levels of complexity and optimization, entirely from Python — scaling from local prototyping with CPUs, to
low-latency remote hardware execution with FPGAs, GPUs, and Triton.

.. figure:: ../_static/demonstration_assets/backline-for-low-latency-qec-on-gpus-and-fpgas/architecture.png
    :align: center
    :width: 50%

Getting started
---------------

`Backline <tk>`__ is an open platform for compilation and low-latency execution --- built within
PennyLane and `Catalyst <https://docs.pennylane.ai/projects/catalyst>`__ --- to dynamically connect
quantum workloads to the right classical engine (such as CPUs, GPUs, FPGAs, and ASICs). It allows
you to write high-level algorithmic logic in Python, and easily jump through abstraction layers to
write custom, highly optimized low-level hardware kernels.

To showcase this, we'll take an example of a workflow that requires low-latency execution ---
`quantum error correction <https://pennylane.ai/topics/fault-tolerant-quantum-computing>`__ --- and
begin prototyping immediately with CPUs, before progressively including consumer-grade and
enterprise GPUs and FPGAs — all from the same software environment.

To start with, we need to install the latest versions of PennyLane and Catalyst. We will need to
install these from source; we can follow the build instructions available in the `Catalyst
documentation <https://github.com/PennyLaneAI/backline-for-low-latency-qec-on-gpus-and-fpgas/blob/main/INSTALL.md>`__. To execute all of the demos
demo, including GPUs and FPGAs examples, you will need to make sure you have the required hardware
and software, including:

- A server with an AMD Instinct™ GPU, ROCm 6 or newer, and an RDMA NIC;
- An AMD Xilinx™ VPK120 FPGA board.

If you would like to simply prototype using local CPUs (the first example in this demo), the above
hardware requirements are not necessary.

Now that you have Backline installed, let's get started!

Backlines, coprocessors, and controllers
----------------------------------------

There are three main components to Backline:

- **Controllers**: This is the classical hardware (such as a CPU or FPGA) that controls the QPU
  (a quantum hardware or simulator :external+backline:func:`~pennylane.device`), receives quantum measurement
  results, and initiates data transfers with other hardware devices (*coprocessors*). For
  example, it might perform QEC syndrome measurements on the QPU, and send these to a coprocessor
  for decoding.

- **Coprocessors**: These are hardware devices (such as CPUs, GPUs, or FPGAs) that receive
  information from a controller for processing. They run specific **coprocessing functions**,
  potentially as a persistent kernel.

- **Backline**: A representation of the complete hardware infrastructure supporting the
  quantum-classical program. The backline includes a controller, one or more coprocessors, and a
  transport method. A backline object is given directly to a QNode in place of a traditional
  QNode :external+backline:func:`~pennylane.device`.

In the examples below, we'll see how to create our controller, coprocessors, and backline, and how
to use these to execute PennyLane programs.

Local CPU-to-CPU with a pre-compiled kernel
-------------------------------------------

First, we’ll start with a local CPU-CPU interaction — a great place to begin prototyping, as it
comes with significantly fewer specialized hardware barriers. The best thing is, this workflow
easily extends to remote workflows on specialized hardware (such as GPUs and FPGAs) with minimal
changes.

For the first example, we will encode a logical circuit using the :doc:`Steane code
<tutorial_bp_catalyst>` (:math:`[[7, 1, 3]]`), a specific type of Calderbank-Shor-Steane (CSS) code
based on the classical :math:`[7, 4, 3]` `Hamming code
<https://en.wikipedia.org/wiki/Hamming_code>`__. The Steane code encodes one logical qubit using 7
physical qubits, and has the ability to correct arbitrary single qubit errors.

Conveniently, we can leverage PennyLane to simply define a **logical circuit**, and point Catalyst
and Backline to existing optimization passes (for quantum error correction encoding) and
pre-compiled libraries (for decoding), to allow the error correction to occur automatically within
the stack.

We'll start by registering the Steane decoder. We will point to a pre-compiled library that comes
with Catalyst, but you can point this to any pre-compiled decoding library you wish to use.
"""

import os
from pathlib import Path

import pennylane as qp

STEANE_LIB_CPU = str(
    Path(os.environ.get("CATALYST_ROOT", "~/catalyst")).expanduser()
        / "runtime/build/lib"
        / "libsteane_coprocessor_cpu.so"
)

steane_decode = qp.CoprocessorFunction("steane_coprocessor", STEANE_LIB_CPU)


######################################################################
# Note the use of the :external+backline:class:`~pennylane.CoprocessorFunction`. This allows us to register
# a coprocessing function that will be run on a coprocessor.
#
# Next, we can create our two CPUs: the controller, and the coprocessor. The controller will run
# quantum instructions on the PennyLane Lightning simulator, while the coprocessor will be
# performing the decoding.

qdev = qp.device("lightning.qubit", wires=3)
CPU1 = qp.Controller(device=qdev)
CPU2 = qp.Coprocessor(coprocessor_fn=steane_decode)

######################################################################
# And that's it — we can create our backline (using memcpy as our transport mechanism, although
# `RDMA
# <https://github.com/PennyLaneAI/backline-for-low-latency-qec-on-gpus-and-fpgas/blob/main/demos/demo_1a_local_cpu_to_local_cpu_rdma.py>`__
# is another option), register it to our QNode, and run our workflow. Let's create a logical GHZ
# state:

dev = qp.Backline(controller=CPU1, coprocessors=[CPU2], transport="memcpy", qec_code="steane")

@qp.qjit(capture=True)
@qp.set_shots(10)
@qp.qnode(dev, mcm_method="one-shot")
def ghz():
    qp.Hadamard(0)
    qp.CNOT([0, 1])
    qp.CNOT([1, 2])
    return qp.sample([qp.measure(0), qp.measure(1), qp.measure(2)])

print("samples:", ghz())

######################################################################
# .. rst-class:: sphx-glr-script-out
#
#   .. code-block:: none
#
#     samples: [[0 0 0]
#      [1 1 1]
#      [1 1 1]
#      [0 0 0]
#      [1 1 1]
#      [1 1 1]
#      [1 1 1]
#      [0 0 0]
#      [0 0 0]
#      [0 0 0]]
#
# Note that, as we define ``qec_code="steane"``, QEC encoding and decoding will be automatically
# applied; the former during MLIR optimization, and the latter on our coprocessor.
#
# Remote CPU-to-GPU with a Python defined kernel
# ----------------------------------------------
# Next, we'll consider a *remote* CPU-to-GPU interaction using a Python-defined QEC decoding kernel,
# written using Triton. In this example, the controller and coprocessor are no longer local, but on
# a remote server. Furthermore, we'll show how you can program the GPU coprocessor directly from
# Python using the `Triton <https://triton-lang.org/main/index.html>`__ library for writing
# optimized GPU kernels.
#
# To further increase the complexity, this time we will upgrade our QEC encoding to use
# a :doc:`qLDPC code <tutorial_qldpc_codes>`, which leverage high connectivity between qubits to
# drastically reduce qubit overheads. In particular, we will use the :math:`[[13, 1, 3]]`
# Hypergraph Product code, a well-known family of qLDPC codes that uses 13 physical wires for
# encoding, and a single auxiliary wire to extract syndromes. To *decode* the qLDPC code, we will
# use a belief propagation decoder --- an iterative message-passing algorithm used to decode errors
# by working on the `Tanner graph <https://en.wikipedia.org/wiki/Tanner_graph>`__ of the code.
#
# To start with, we define our belief propagation decoder. We do so using two parity check
# matrices:

import numpy as np
import pennylane as qp

Hx = np.array([[1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
               [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0],
               [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
               [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1],
               [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1]])

Hz = np.array([[1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
               [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
               [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0],
               [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1],
               [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1]])


######################################################################
# Next, we make use of the :external+backline:func:`~pennylane.backline.css_bp_decoder` function to easily compile a
# CSS code's Tanner graph into a GPU-compatible belief propagation decoder function.

bp_decoder = qp.backline.css_bp_decoder(Hx, Hz, postprocess="osd", num_iters=10, platform="hip:gfx90a:64")

######################################################################
# .. note::
#
#     Behind the scenes, this function utilizes ``@triton.jit`` to compile an optimized GPU kernel ---
#     feel free to look under-the-hood at the `source code
#     <https://github.com/PennyLaneAI/pennylane/blob/main/pennylane/backline-for-low-latency-qec-on-gpus-and-fpgas/functions.py#L134>`__ to
#     see how Triton is being used. Later in this demo, we will also show you how to compile your
#     own Triton function for Backline coprocessing.
#
# With the decoder defined, we can now create our controller and coprocessor. To start, we'll define
# our server configuration, representing our remote server carrying an AMD GPU and an RDMA capable
# NIC (the server configuration will need to be updated as per your specific server details):

SERVER = {
    'host':"192.168.3.14",  # The address of the remote server
    'user': 'username',  # The SSH account of the user on the remote machine
    'triple': "x86_64-unknown-linux-gnu",  # The LLVM target triple for compilation
    'sudo': True,
    'deploy': str(Path(os.environ.get("RDMA_DEV_ROOT"))
                  / "cross-build-artifacts"
                  / "threadripper-bundle"),
    'executor_bin': "numactl -N 0 -m 0 ./catalyst-executor",
    'env': {"LD_LIBRARY_PATH": "."}
}

######################################################################
# Here, ``deploy`` and ``executor_bin`` specify the target workspace directory (including artifacts)
# and the starting command, respectively. Next, we define a remote controller on the server. As in
# the previous example, the controller is in charge of executing the quantum instructions, the only
# difference is that now it is on a remote server rather than on the same local machine where the
# quantum-classical workflow is defined.

qdev = qp.device("lightning.qubit", wires=N + 1)

CPU = qp.Controller(
    name="cpu-controller",
    device=qdev,
    remote=True,
    executor_options={**SERVER, "port": 8810},
    init_args={"config": "dev=mlx5_1;gid=3;cpu_pin=4;rt=1"}
)


######################################################################
# ``executor_options`` specifies the remote machine and how it is reached, and ``init_args``
# specifies the backend-specific initialization arguments, which are forwarded to the transport
# backend. For more details, see the :external+backline:class:`~pennylane.Controller` documentation.
#
# We can now define a remote GPU coprocessor on the server, and specify the Python-defined belief
# propagation decoding function it will be executing:

GPU = qp.Coprocessor(
    name="gpu-coproc",
    coprocessor_fn=bp_decoder,  # our Python-defined BP decoder
    remote=True,
    endpoint=qp.Endpoint("192.168.1.2", 7760),
    executor_options={**SERVER, "port": 8813},
    hardware="gpu",
    init_args={"config": "dev=mlx5_1;gid=3;cpu_pin=4;rt=1;gpu=0"}
)

######################################################################
# Here, the endpoint argument specifies the RDMA connection between
# the controller and the coprocessor (which is distinct from the ``SERVER`` connection).
# We now have all the pieces to define our backline!

dev = qp.Backline(controller=CPU, coprocessors=[GPU], transport="rdma")

######################################################################
# Since we did not specify any automatic quantum error correction, we need to explicitly perform
# encoding and decoding. As a result, we will:
#
# - **Manually encode our logical circuit** (in effect, defining
#   a **physical** circuit to be compiled and executed on the backline); and
# - **Manually measure the syndromes, decode, and apply correction**.
#
# Luckily, PennyLane and Catalyst make this easy!

N = 13  # data wires
AUX = N  # single reused auxiliary wire

encoder = {
    0: [6, 9, 11],
    1: [7, 9, 10, 11, 12],
    2: [8, 10, 12],
    3: [6, 11],
    4: [7, 11, 12],
    5: [8, 12],
}

@qp.qjit(capture=True)
@qp.set_shots(1)
@qp.qnode(dev, mcm_method="one-shot")
def encoded_decoded_circuit(error_qubit, error_kind):
    # ========= Encoded logical circuit =========
    # encode a logical 0 state
    for pivot, targets in encoder.items():
        qp.Hadamard(wires=pivot)

        for t in targets:
            qp.CNOT(wires=[pivot, t])

    # encode a logical X gate
    for w in [6, 7, 8]:
        qp.X(wires=w)

    # encode a logical Z gate
    for w in [2, 5, 8]:
        qp.Z(wires=w)

    # encode a logical H gate
    for w in range(N):
        qp.Hadamard(wires=w)
    for a, b in [(1, 3), (2, 6), (5, 7), (10, 11)]:
        qp.SWAP(wires=[a, b])

    # QEC decoding using our belief propagation decoder
    correction_rounds()

    return (qp.expval(mean_stabilizer(Hz, qp.Z)), qp.expval(mean_stabilizer(Hx, qp.X)))


######################################################################
# We won't work through the details of the encoding here, but check out
# :doc:`tutorial_qldpc_codes` if you would like to learn more.
#
# You may notice a few functions that have not yet been defined, such as
# ``correction_rounds`` and ``mean_stabilizer``. Let's define them now.
#
# First up, ``correction_rounds``. This is a for loop over wires that:
#
# 1. Injects single qubit errors onto each data wire;
# 2. Performs measurements in order to extract the syndrome;
# 3. Calls the coprocessor to perform QEC decoding (via the
#    :external+backline:func:`~pennylane.backline.decode` function);
# 4. Applies the corrections to the quantum device.
#
# .. note::
#
#     :external+backline:func:`~pennylane.backline.decode` makes a series of runtime calls directly on the
#     coprocessor. The ability to manually perform runtime calls is also available via
#     :external+backline:func:`~pennylane.runtime_call`. If you are curious to see the internal runtime calls, `see
#     the corresponding demo in the Backline repository
#     <https://github.com/PennyLaneAI/backline-for-low-latency-qec-on-gpus-and-fpgas/blob/main/demos/demo_2a_remote_cpu_to_remote_gpu_triton_runtime_calls.py>`__.
#     This demo expands out ``decode`` into explicit ``get_session``, ``stage_payload``, ``post``, and ``collect`` calls.


@qp.for_loop(0, N, 1)
def correction_rounds(error_qubit, error_kind):
    # ========= Inject errors =========
    # Apply I/X/Y/Z to one chosen data wire.
    qp.cond(error_kind == 1, qp.X)(wires=error_qubit)
    qp.cond(error_kind == 2, qp.Y)(wires=error_qubit)
    qp.cond(error_kind == 3, qp.Z)(wires=error_qubit)

    # ========= Decoding =========
    z_syndrome, x_syndrome = extract_syndromes()

    correction_z = qp.backline.decode(x_syndrome, decoder_id=0)
    correction_x = qp.backline.decode(z_syndrome, decoder_id=1)

    # Apply X to every data qubit whose correction bit is set
    for q in range(N):
        qp.cond(correction_x[q], qp.X)(wires=q)

    # Apply Z to every data qubit whose correction bit is set
    for q in range(N):
        qp.cond(correction_z[q], qp.Z)(wires=q)


def extract_syndromes():
    z_syndrome = []

    for row in Hz:
        for q in np.flatnonzero(row):
            qp.CNOT(wires=[int(q), AUX])

        z_syndrome += qp.measure(AUX, reset=True)

    x_syndrome = []

    for row in Hx:
        qp.Hadamard(wires=AUX)

        for q in np.flatnonzero(row):
            qp.CNOT(wires=[AUX, int(q)])

        qp.Hadamard(wires=AUX)
        x_syndrome += qp.measure(AUX, reset=True)

    return z_syndrome, x_syndrome


######################################################################
# Next, ``mean_stabilizers``, which returns a linear combination of :math:`X` and :math:`Z`
# stabilizers from the parity check matrices. By returning the expectation value of these operators
# from the QNode, we are able to check if the quantum state remains within the valid code space;
# acting as a high-level diagnostic metric of "how healthy" the error-corrected quantum state is.
#
# In a perfect quantum error-correcting code with zero errors, the expectation value of every
# single stabilizer is exactly 1. If errors occur, some of those stabilizers will flip
# to -1, reducing the overall expectation value. So if the QNode is correctly error-encoded,
# the returned value should be 1.

def mean_stabilizer(checks, pauli):
    return qp.dot(
        [1 / len(checks)] * len(checks),
        [qp.prod(*(pauli(wires=int(q)) for q in np.flatnonzero(row))) for row in checks],
    )

######################################################################
# We now have all the pieces in place to run our explicit QEC encoding and decoding example on the
# backline:

for error_kind, error_name in enumerate(["I", "X", "Y", "Z"]):
    print(error_name, encoded_decoded_circuit(error_kind))


######################################################################
# .. rst-class:: sphx-glr-script-out
#
#   .. code-block:: none
#
#       scp: Connection closed
#       [scp] the SFTP backend is declining. Retrying it with the legacy protocol (scp -O), which hosts without an SFTP subsystem need
#       [cpu-controller] catalyst-executor: loaded /home/catalyst-exec/librt_transport.so
#       [cpu-controller] catalyst-executor: loaded /home/catalyst-exec/librt_capi.so
#       [cpu-controller] catalyst-executor: loaded /home/catalyst-exec/liblightning_qubit_catalyst.so
#       [cpu-controller] Listening on 127.0.0.1:8810
#       [cpu-controller] [127.0.0.1:8810] executor ready, waiting for connections
#       scp: Connection closed
#       [scp] the SFTP backend is declining. Retrying it with the legacy protocol (scp -O), which hosts without an SFTP subsystem need
#       [gpu-coproc] catalyst-executor: loaded /home/catalyst-exec/librt_transport.so
#       [gpu-coproc] catalyst-executor: loaded /home/catalyst-exec/librt_capi.so
#       [gpu-coproc] catalyst-executor: loaded /home/catalyst-exec/librdma_triton_decoder.so
#       [gpu-coproc] Listening on 127.0.0.1:8813
#       [gpu-coproc] [127.0.0.1:8813] executor ready, waiting for connections
#       [gpu-coproc] [127.0.0.1:8813] accepted connection from 127.0.0.1:56120 on pid 1546394, waiting for next connection
#       [cpu-controller] [127.0.0.1:8810] accepted connection from 127.0.0.1:56154 on pid 1546399, waiting for next connection
#       I (Array(1., dtype=float64), Array(1., dtype=float64))
#       X (Array(1., dtype=float64), Array(1., dtype=float64))
#       Y (Array(1., dtype=float64), Array(1., dtype=float64))
#       Z (Array(1., dtype=float64), Array(1., dtype=float64))
#       JIT session error: FD-transport disconnected
#       JIT session error: disconnecting
#       JIT session error: FD-transport disconnected
#       JIT session error: disconnecting
#
# Remote GPU-to-FPGA with a pre-compiled kernel
# ---------------------------------------------
#
# We can now scale the exact same logic to a remote FPGA-to-GPU environment. For simplicity, let's
# return to compiling a logical circuit with PennyLane, encoding the Steane code with Catalyst, and
# using a pre-compiled decoder function.
#
# First, we create the FPGA controller. Here, it is a Xilinx VPK120 board:

FPGA_SERVER = {
    'host': "192.168.3.15",
    'user': "username",  # user on the FPGA server
    'triple': "aarch64-unknown-linux-gnu",
    'sudo': True,
    'deploy': str(Path(os.environ.get("RDMA_DEV_ROOT")) / "cross-build-artifacts" / "vpk-bundle"),
    'env': {
        "XMM_SQ_TYPE": "PL", "XMM_RQ_TYPE": "PL", "XMM_CQ_TYPE": "PL",
        "XMM_APP_MAX_QP": "4", "XMM_APP_RQ_SGE": "1", "XMM_SQ_DEPTH": "64",
        "XMM_RQ_DEPTH": "128", "HWHS_RTT_WARMUP": "0"
    }
}

FPGA = qp.Controller(
    name="fpga-controller",
    remote=True,
    hardware="fpga",
    executor_options={**FPGA_SERVER, "port": 8811},
    init_args={"config": "dev=xib_0;gid=1;sq_mem=bram;data_mem=bram;reply_mem=bram;reply_poll=hw;stride_log2=6"}
)

######################################################################
# Note that since we haven't provided a quantum device, the controller will default to using
# ``null.qubit``, a convenient dummy device that performs no quantum processing (but simply accepts
# quantum operations and returns 0). We can use the same remote GPU coprocessor we set up earlier,
# but replace ``coprocessor_fn`` with ``'gpu_steane_launcher'`` (a convenient alias for
# Catalyst's precompiled Steane decoder, which is already loaded and available for use).

GPU = qp.Coprocessor(
    name="gpu-coproc",
    coprocessor_fn="gpu_steane_launcher",
    remote=True,
    endpoint=qp.Endpoint("192.168.1.2", 7760),
    executor_options={**SERVER, "port": 8813},
    hardware="gpu",
    init_args={"config": "dev=mlx5_1;gid=3;cpu_pin=4;rt=1;gpu=0"}
)

######################################################################
# Defining the backline, this time we provide our FPGA as the
# controller:

dev = qp.Backline(controller=FPGA, coprocessors=[GPU], transport="rdma", qec_code="steane")

######################################################################
# Our backline infrastructure looks as follows:
#
# .. figure:: ../_static/demonstration_assets/backline-for-low-latency-qec-on-gpus-and-fpgas/server-setup.png
#     :align: center
#     :width: 50%
#
# We can now create and execute our logical quantum circuit:

@qp.qjit(capture=True)
@qp.set_shots(1000)
@qp.qnode(dev, mcm_method="one-shot")
def ghz():
    qp.Hadamard(0)
    qp.CNOT([0, 1])
    qp.CNOT([1, 2])
    return qp.sample([qp.measure(0), qp.measure(1), qp.measure(2)])

print("samples:", ghz())


######################################################################
# .. rst-class:: sphx-glr-script-out
#
#   .. code-block:: none
#
#       [fpga-controller] catalyst-executor: loaded /home/petalinux/catalyst-exec/librt_transport.so
#       [fpga-controller] catalyst-executor: loaded /home/petalinux/catalyst-exec/librt_capi.so
#       [fpga-controller] catalyst-executor: loaded /home/petalinux/catalyst-exec/librtd_null_qubit.so
#       [fpga-controller] Listening on 127.0.0.1:7811
#       [fpga-controller] [127.0.0.1:7811] executor ready, waiting for connections
#       scp: Connection closed
#       [gpu-coproc] catalyst-executor: loaded /home/catalyst-exec/librt_transport.so
#       [gpu-coproc] catalyst-executor: loaded /home/catalyst-exec/librt_capi.so
#       [gpu-coproc] Listening on 127.0.0.1:7813
#       [gpu-coproc] [127.0.0.1:7813#3133915] Accepted connection
#       [fpga-controller] [127.0.0.1:7811] accepted connection from 127.0.0.1:39924 on pid 1209, waiting for next connection
#       [fpga-controller]
#       [fpga-controller]=== engine RTT (n=16000, 0 warmup dropped, hardware handshake) ===
#       [fpga-controller]   min          4600 ns
#       [fpga-controller]   p50          4855 ns
#       [fpga-controller]   p95          5095 ns
#       [fpga-controller]   p99          5570 ns
#       [fpga-controller]   p99.9        5815 ns
#       [fpga-controller]   max          9660 ns
#       [fpga-controller]   mean         4931 ns
#       samples: [[0 0 0]
#        [0 0 0]
#        [0 0 0]
#        ...
#        [0 0 0]
#        [0 0 0]
#        [0 0 0]]
#       JIT session error: FD-transport disconnected
#       JIT session error: disconnecting
#       JIT session error: FD-transport disconnected
#       JIT session error: disconnecting
#
# During execution, backline is managing the following communication pathways:
#
# .. figure:: ../_static/demonstration_assets/backline-for-low-latency-qec-on-gpus-and-fpgas/communications.png
#     :align: center
#     :width: 50%
#
# Bring your own Triton decoder
# -----------------------------
#
# With a simple modification to the above remote GPU-FPGA example, we can replace the pre-compiled
# Steane decoder with a custom Triton decoder --- all written from Python.
#
# Since we are using the Steane code, a relatively simple QEC code that uses only 7 wires to encode
# a logical qubit, we only have 64 possible error patterns to track. Due to this low number, one
# decoding strategy is to simply pre-calculate every solution and store them in a lookup table.
#
# With 3 :math:`X` and 3 :math:`Z` stabilizers, there are :math:`2^3=8` possible syndromes for both
# :math:`X` and :math:`Z`. We can create an highly efficient Triton function that maps each
# three‑bit syndrome to a weight‑1 error.

import triton
import triton.language as tl

def steane_lookup(syndrome: tl.uint64):
    # Encode the Steane syndrome in 3 bits by using a bitwise mask.
    s = syndrome & 0x7

    # If the syndrome is 0, return 0
    # Otherwise, return the correction as a bitmask.
    return tl.where(s != 0, 1 << (s - 1), 0)


######################################################################
# We can use the provided :external+backline:func:`~pennylane.backline.triton_decoder` function to
# compile this for our target system using ``triton.jit``, and then it is simply a matter of
# providing the compiled ``steane_triton_decoder`` as our coprocessing function when defining the
# GPU coprocessor:

steane_triton_decoder = qp.backline.triton_decoder(
    (steane_lookup,),
    platform="hip:gfx90a:64",
)

GPU = qp.Coprocessor(
    name="gpu-coproc",
    coprocessor_fn=steane_triton_decoder,
    remote=True,
    endpoint=qp.Endpoint("192.168.1.2", 7760),
    executor_options={**SERVER, "port": 8813},
    hardware="gpu",
    init_args={"config": "dev=mlx5_1;gid=3;cpu_pin=4;rt=1;gpu=0"}
)


dev = qp.Backline(controller=FPGA, coprocessors=[GPU], transport="rdma", qec_code="steane")

@qp.qjit(capture=True)
@qp.set_shots(1000)
@qp.qnode(dev, mcm_method="one-shot")
def ghz():
    qp.Hadamard(0)
    qp.CNOT([0, 1])
    qp.CNOT([1, 2])
    return qp.sample([qp.measure(0), qp.measure(1), qp.measure(2)])

print("samples:", ghz())

######################################################################
# .. rst-class:: sphx-glr-script-out
#
#   .. code-block:: none
#
#       tk
#       [fpga-controller] catalyst-executor: loaded /home/petalinux/catalyst-exec/librt_transport.so
#       [fpga-controller] catalyst-executor: loaded /home/petalinux/catalyst-exec/librt_capi.so
#       [fpga-controller] catalyst-executor: loaded /home/petalinux/catalyst-exec/librtd_null_qubit.so
#       [fpga-controller] Listening on 127.0.0.1:7811
#       [fpga-controller] [127.0.0.1:7811] executor ready, waiting for connections
#       scp: Connection closed
#       [gpu-coproc] catalyst-executor: loaded /home/catalyst-exec/librt_transport.so
#       [gpu-coproc] catalyst-executor: loaded /home/catalyst-exec/librt_capi.so
#       [gpu-coproc] Listening on 127.0.0.1:7813
#       [gpu-coproc] [127.0.0.1:7813#3133915] Accepted connection
#       [fpga-controller] [127.0.0.1:7811] accepted connection from 127.0.0.1:39924 on pid 1209, waiting for next connection
#       [fpga-controller]
#       [fpga-controller]=== engine RTT (n=16000, 0 warmup dropped, hardware handshake) ===
#       [fpga-controller]   min          4600 ns
#       [fpga-controller]   p50          4855 ns
#       [fpga-controller]   p95          5095 ns
#       [fpga-controller]   p99          5570 ns
#       [fpga-controller]   p99.9        5815 ns
#       [fpga-controller]   max          9660 ns
#       [fpga-controller]   mean         4931 ns
#       samples: [[0 0 0]
#        [0 0 0]
#        [0 0 0]
#        ...
#        [0 0 0]
#        [0 0 0]
#        [0 0 0]]
#       JIT session error: FD-transport disconnected
#       JIT session error: disconnecting
#       JIT session error: FD-transport disconnected
#       JIT session error: disconnecting
#
# Conclusion
# ----------
#
# In this demo, we demonstrated how PennyLane, Backline, and Catalyst eliminate the bottleneck of
# transitioning from local quantum error correction (QEC) prototyping to production-grade hardware
# deployment. By keeping the entire workflow within Python, you can dynamically connect quantum
# workloads to classical engines --- such as CPUs, GPUs, and FPGAs --- without rewriting control
# logic into low-level languages. Start with simple CPU-CPU prototypes with explicit quantum
# encoding and decoding directly in Python, and rapidly scale up to remote classical accelerators,
# optimized MLIR passes, and pre-compiled GPU decoder kernels.
#
# Furthermore, the integration with Triton allows for the rapid creation of highly optimized, custom
# hardware kernels **all from Python**.
#
# To continue exploring heterogeneous quantum compilation and execution, check out the following
# resources:
#
# - View additional Backline demos in the `Backline repository on GitHub
#   <https://github.com/PennyLaneAI/backline/blob/main/demos/>`__, including CPU-CPU interactions
#   over RDMA, and explicit usage of :external+backline:func:`~pennylane.runtime_call` to execute
#   coprocessor functions.
#
# - Read the `Backline technical paper <tk>`__ to get a technical overview of the infrastructure and
#   performance.
#
# - Check out the `PennyLane blog post <tk>`__ to learn more about Backline.






