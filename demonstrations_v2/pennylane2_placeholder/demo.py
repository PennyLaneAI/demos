r"""
Simulating near-infrared spectroscopy with PennyLane2
=====================================================

.. note::

    This demo is only compatible with ``pennylane2``, which can be installed with
    ``pip install pennylane2`` in your Python environment.

    TODO: Link to the PennyLane2 documentation here (URLs TBD). This demo is not executed when
    the website is built. The outputs shown were generated with the development versions of
    PennyLane (0.46.0-dev159) and Catalyst (0.16.0-dev147); update them to the open beta release
    versions.

    TODO: Publish the HC\ :sub:`2`\ N\ :sub:`3` data files and add their download links (URLs TBD)
    to the start of the "Loading in concrete data" section.

PennyLane2 takes quantum algorithms from inspiration to implementation. You write an algorithm
with high-level, state-of-the-art building blocks, get resource estimates from coarse information
alone, and then compile and analyze the full algorithm, at scale, in seconds. Compilation and
resource analysis are two sides of the same coin: the pipeline that compiles a program is also
what tells you what it costs.

We show this with a valuable application: near-infrared spectroscopy (NIRS), which measures how
molecules absorb light through overtones and combinations of their vibrations. These spectra are a
fingerprint for chemical detection, and computing them accurately is a natural application for
quantum computers [#Loaiza2025]_.

In this demo, we define the NIRS algorithm for the azidoacetylene molecule
(HC\ :sub:`2`\ N\ :sub:`3`) with only abstract inputs and shape information, and get resource
estimates in seconds. Then we replace the coarse data with actual data, compile the algorithm with
billions of operations, and analyze its resources along the way.

PennyLane2 runs every program through a single compilation pipeline, powered by
`Catalyst <https://docs.pennylane.ai/projects/catalyst>`__. That is what makes estimating,
compiling and analyzing algorithms at this scale fast.
"""

######################################################################
# The algorithm
# -------------
#
# The NIRS spectrum is obtained from the time-correlation function
# :math:`C(t) = \langle \psi | e^{-iHt} | \psi \rangle`, where :math:`H` is the vibrational
# Hamiltonian of the molecule and :math:`|\psi\rangle` is the state created when the molecule
# absorbs light: the dipole operator applied to the vibrational ground state, projected onto the
# near-infrared part of the spectrum [#Loaiza2025]_. The spectrum is the Fourier transform of
# :math:`C(t)`, damped to account for the width of the absorption peaks.
#
# A quantum computer estimates :math:`C(t)` with a Hadamard test: a control qubit decides whether
# the time evolution is applied, and the expectation values of :math:`X` and :math:`Y` on that
# qubit give the real and imaginary parts of :math:`C(t)`. This is repeated at the evolution times
# :math:`t_k = k\tau` with :math:`k = 1, \dots, k_{\max}`. Following [#Malpathak2026]_, we use
# :math:`\tau = 250` atomic units and :math:`k_{\max} = 300`, enough to resolve the spectrum to
# about 10 cm\ :sup:`-1`. The longest evolution, :math:`t = k_{\max}\tau`, needs the largest
# circuit, so that is the one we build. For a detailed walkthrough of spectroscopy algorithms of
# this kind, see the :doc:`X-ray Absorption Spectroscopy (XAS) <demos/tutorial_xas>` demo.
#
# Also following [#Malpathak2026]_, we write :math:`H` in the Christiansen form, in which each
# vibrational mode is encoded in a few qubits, one per vibrational state (modal) of that mode. We
# split :math:`H` into fragments with Christiansen greedy fragmentation (CGF). Each fragment is a
# basis rotation followed by a diagonal operator, so its time evolution is cheap to implement. We
# approximate :math:`e^{-iHt}` with a second-order product formula of these fragments, with
# :math:`r = 2` Trotter steps per time step :math:`\tau`. [#Malpathak2026]_ picks :math:`r` per
# molecule from a Trotter error estimate, and finds :math:`r = 2` for CH\ :sub:`4`; we use the same
# value for HC\ :sub:`2`\ N\ :sub:`3`, which that work does not cover.
#
# Estimating resources with coarse information
# --------------------------------------------
#
# How many resources does this algorithm need for HC\ :sub:`2`\ N\ :sub:`3`? With PennyLane2, we
# can answer this before computing a single Hamiltonian coefficient. Operators accept abstract
# inputs, which have a shape but no values, and :func:`~pennylane.estimate` projects the resources
# of a circuit onto a gate set without running it.
#
# HC\ :sub:`2`\ N\ :sub:`3` has 12 vibrational modes. With 4 modals per mode, the system register
# has 48 qubits. Its Hamiltonian is split into 5 fragments, and its initial state is a
# superposition of 19 basis states. A :class:`~pennylane.CGFHamiltonian` is defined by its core
# and leaf tensors, and we only give their shapes, with :data:`~pennylane.typing.Float`.
#
# The state preparation needs to know which basis states make up the initial state at compile
# time, because the circuit that prepares them depends on their bit patterns. We don't have them
# yet, so we pick random ones. They affect the state preparation, but not the time evolution,
# which dominates the cost.

import random

import pennylane as qp


def abstract_molecule(num_modes, num_modals=4, num_fragments=5, num_basis_states=19):
    num_qubits = num_modes * num_modals
    hamiltonian = qp.CGFHamiltonian(
        core_tensors=qp.typing.Float[num_fragments, num_modes, num_modes, num_modals, num_modals],
        leaf_tensors=qp.typing.Float[num_fragments, num_modes, num_modals, num_modals],
        nuc_constant=qp.typing.Float,
    )
    state = {
        "coefficients": qp.typing.Float[num_basis_states],
        "indices": tuple(sorted(random.Random(0).sample(range(2**num_qubits), num_basis_states))),
        "num_qubits": num_qubits,
    }
    return hamiltonian, state


hamiltonian, state = abstract_molecule(num_modes=12)

######################################################################
# Building the circuit
# ^^^^^^^^^^^^^^^^^^^^
#
# The circuit uses three high-level operations from PennyLane2:
#
# * :class:`~pennylane.SumOfSlatersPrep` prepares :math:`|\psi\rangle`, a sparse superposition of
#   basis states.
# * :class:`~pennylane.TrotterCGF` applies the Trotterized time evolution under the CGF
#   Hamiltonian, and :func:`~pennylane.ctrl` controls it on the Hadamard-test qubit.
# * :class:`~pennylane.PhaseGradientStatePrep` prepares a phase gradient state, which lets us
#   implement many rotations with cheap arithmetic instead of costly rotation synthesis. See the
#   :doc:`phase gradient demo <demos/efficient_rotations_with_phase_gradient_states>` for how
#   this works.
#
# Besides the Hadamard-test qubit and the system register, each operation needs some auxiliary
# wires. :meth:`~pennylane.SumOfSlatersPrep.required_register_sizes` tells us how many the state
# preparation needs, and the phase gradient registers need as many qubits as there are bits of
# precision in each rotation angle. :func:`~pennylane.registers` then assigns wires to every
# register.

epsilon = 1e-10  # accuracy of each rotation
sos_registers = ("enumeration_wires", "identification_wires", "qrom_work_wires", "mcx_cache_wires")


def make_registers(state):
    indices, num_qubits = state["indices"], state["num_qubits"]
    precision = qp.math.ceil_log2(1 / epsilon)
    sos_sizes = qp.SumOfSlatersPrep.required_register_sizes(indices, num_qubits)
    return qp.registers(
        {
            "hadamard": 1,
            "system": num_qubits,
            "angle_wires": precision,
            "phase_grad_wires": precision,
            **{name: sos_sizes[name] for name in sos_registers},
            "work_wires": max(precision - 1, qp.math.ceil_log2(len(indices)) - 2),
        }
    )


######################################################################
# Next, we create two decomposition rules that use the phase gradient state on our registers.
# The first, from
# :func:`~pennylane.transforms.decompositions.make_selectpaulirot_to_phase_gradient_decomp`,
# implements the multiplexed rotations (:class:`~pennylane.SelectPauliRot`) in the state
# preparation with arithmetic on the phase gradient register. The second, from
# :func:`~pennylane.transforms.decompositions.make_phase_gradient_distillation_decomp`, prepares
# the phase gradient state itself. We name the first rule so that we can select it when
# compiling.

decomps = qp.transforms.decompositions


def make_rules(registers):
    pg_wires = {k: registers[k] for k in ("angle_wires", "phase_grad_wires", "work_wires")}
    spr_rule = decomps.make_selectpaulirot_to_phase_gradient_decomp(**pg_wires)
    spr_rule.name = "nirs_spr_phase_gradient"
    pgsp_rule = decomps.make_phase_gradient_distillation_decomp(
        pg_wires["angle_wires"], pg_wires["work_wires"]
    )
    return {qp.SelectPauliRot: spr_rule, qp.PhaseGradientStatePrep: pgsp_rule}


######################################################################
# Finally, the circuit itself. It runs on ``null.qubit``
# (:class:`~pennylane.devices.NullQubit`), a device that performs no simulation at all: our
# circuits have well over a hundred qubits, far beyond what we could simulate classically, and
# we only want to estimate, compile and analyze them.


def make_circuit(hamiltonian, state, registers, k_max):
    dev = qp.device("null.qubit", wires=sum(len(wires) for wires in registers.values()))

    @qp.qnode(dev)
    def circuit():
        qp.PhaseGradientStatePrep(registers["phase_grad_wires"])
        qp.SumOfSlatersPrep(
            state["coefficients"],
            wires=registers["system"],
            indices=state["indices"],
            **{name: registers[name] for name in sos_registers},
        )
        qp.ctrl(
            qp.TrotterCGF(
                evolution_time=k_max * 250.0,  # tau = 250 atomic units
                num_trotter_steps=2 * k_max,
                hamiltonian=hamiltonian,
                wires=registers["system"],
                double_phase=True,
            ),
            control=registers["hadamard"],
        )
        return qp.expval(qp.X(registers["hadamard"])), qp.expval(qp.Y(registers["hadamard"]))

    return circuit


######################################################################
# We build the circuit for the longest evolution time, :math:`k_{\max} = 300`. Only the register
# sizes matter for the estimate, so even the wire labels can be abstract, with
# :func:`~pennylane.core.abstractify`.


def abstract_registers(state):
    return {name: qp.core.abstractify(wires) for name, wires in make_registers(state).items()}


registers = abstract_registers(state)
rules = make_rules(registers)
circuit = make_circuit(hamiltonian, state, registers, k_max=300)

######################################################################
# Estimating resources
# ^^^^^^^^^^^^^^^^^^^^
#
# We estimate the resources for two gate sets. The first is made of Clifford gates, :math:`T`
# gates and :math:`R_Z` rotations, the standard way to count the cost of fault-tolerant
# algorithms. The second is made of Pauli product rotations (PPRs) and measurements, the
# instructions of `Pauli-based computation
# <https://pennylane.ai/compilation/pauli-based-computation>`__ (PBC). PBC suits surface-code
# architectures based on lattice surgery, where multi-qubit Pauli measurements are the native
# operations [#Litinski2019]_. In both, we keep the :math:`R_Z` rotations by arbitrary angles
# explicit, since they are synthesized from the other gates later, during compilation.
#
# The weights in the gate sets tell the decomposition how much each gate costs, and
# :func:`~pennylane.decompose` selects our phase gradient rule for the multiplexed rotations. Our
# custom rules are registered locally, with :func:`~pennylane.decomposition.local_decomps`, while
# estimating.

clifford = {
    "X": 1, "Y": 1, "Z": 1, "S": 10, "Hadamard": 10, "SX": 10, "Adjoint(S)": 10, "Adjoint(SX)": 10,
    "CNOT": 50, "CY": 50, "CZ": 50, "GlobalPhase": 0, "Identity": 0,
}  # fmt: skip
measurements = {
    "PauliMeasure": 1, "PauliMeasureMP": 1, "MidMeasure": 1, "MidMeasureMP": 1,
    "MidCircuitMeasure": 1,
}  # fmt: skip
gate_set = {"T": 100, "Adjoint(T)": 100, "RZ": 5000} | clifford | measurements
pbc_gate_set = {"PPR_8": 100, "PPR_4": 10, "PPR_2": 1, "GlobalPhase": 0, "RZ": 5000} | {
    name: 2 for name in measurements
}
decompose = qp.decompose(
    gate_set=gate_set, fixed_decomps={qp.SelectPauliRot: "nirs_spr_phase_gradient"}
)


def estimate(circuit, rules, target):
    compiled = qp.qjit(decompose(circuit), target="mlir_opt", enable_aot_compilation=False)
    with qp.decomposition.local_decomps():
        for op, rule in rules.items():
            qp.add_decomps(op, rule)
        return qp.estimate(compiled, target=target)()


import time

for label, target in (("Clifford+T+RZ", gate_set), ("PBC+RZ", pbc_gate_set)):
    start = time.perf_counter()
    specs = estimate(circuit, rules, target)
    print(f"{label}, estimated in {time.perf_counter() - start:.0f} s")
    print(specs.resources)

######################################################################
# .. rst-class:: sphx-glr-script-out
#
#   .. code-block:: none
#
#     Clifford+T+RZ, estimated in 4 s
#     Quantum operations:
#     - Total: 2.468E+7
#       - Adjoint(S): 6.920E+5
#       - Adjoint(T): 924
#       - CNOT: 1.254E+7
#       - CZ: 462
#       - GlobalPhase: 1.383E+6
#       - Hadamard: 4.150E+6
#       - PauliMeasure: 1,817
#       - PauliX: 6,571
#       - PauliZ: 57,723
#       - RZ: 5.157E+6
#       - S: 6.915E+5
#       - T: 924
#     Measurement processes:
#     - expval(PauliX): 1
#     - expval(PauliY): 1
#     Total wires: 163
#     Circuit Depth: Not computed
#     PBC+RZ, estimated in 3 s
#     Quantum operations:
#     - Total: 5.612E+7
#       - GlobalPhase: 1.260E+7
#       - PPR_2: 64,558
#       - PPR_4: 3.830E+7
#       - PPR_8: 1,848
#       - PauliMeasure: 1,817
#       - RZ: 5.157E+6
#     Measurement processes:
#     - expval(PauliX): 1
#     - expval(PauliY): 1
#     Total wires: 163
#     Circuit Depth: Not computed

######################################################################
# In seconds, and without any data, we know that HC\ :sub:`2`\ N\ :sub:`3` needs about 160 logical
# qubits and :math:`5 \times 10^6` :math:`R_Z` rotations, which dominate the cost: at our
# accuracy, each rotation takes about 50 :math:`T` gates, or :math:`\pi/8` PPRs
# [#Malpathak2026]_. The exact number of auxiliary qubits depends on the basis states of the
# initial state, which we only know once we load the data.
#
# Since the estimate only needs shapes, we can just as quickly ask how the cost grows with the
# size of the molecule. We repeat the estimate for molecules with more and more vibrational modes,
# keeping everything else fixed.

import matplotlib.pyplot as plt

mode_counts = [3, 6, 9, 12, 15]
num_qubits, num_rotations = [], []
for num_modes in mode_counts:
    hamiltonian, state = abstract_molecule(num_modes)
    registers = abstract_registers(state)
    circuit = make_circuit(hamiltonian, state, registers, k_max=300)
    resources = estimate(circuit, make_rules(registers), gate_set).resources
    num_qubits.append(resources.num_wires)
    num_rotations.append(resources.quantum_operations["RZ"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
ax1.plot(mode_counts, [n / 1e6 for n in num_rotations], "o-", color="#2a78d6", linewidth=2, markersize=8)
ax1.set(xlabel="Vibrational modes", title="$R_Z$ rotations (millions)", ylim=(0, None))
ax2.plot(mode_counts, num_qubits, "o-", color="#2a78d6", linewidth=2, markersize=8)
ax2.set(xlabel="Vibrational modes", title="Logical qubits", ylim=(0, None))
for ax in (ax1, ax2):
    ax.set_xticks(mode_counts)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
plt.show()

######################################################################
# .. figure:: ../_static/demonstration_assets/pennylane2_placeholder/estimate_scaling.png
#     :align: center
#     :width: 90%
#     :alt: Estimated R_Z rotations and logical qubits against the number of vibrational modes
#
# The number of rotations grows roughly quadratically with the number of modes, as the two-mode
# terms of the Hamiltonian do, while the number of qubits, most of them auxiliary, grows only
# linearly.
#
# Loading in concrete data
# ------------------------
#
# Now we replace the coarse information with actual data. The CGF Hamiltonian and initial state of
# HC\ :sub:`2`\ N\ :sub:`3` were computed classically and stored as NumPy arrays. TODO: download
# links for the data (URLs TBD). Place them in a ``nirs_data`` folder next to this demo.

import numpy as np


def load_molecule(name):
    ham_data = np.load(f"nirs_data/{name}/cgf_regrouped.npz")
    state_data = np.load(f"nirs_data/{name}/vscf_state.npz")

    hamiltonian = qp.CGFHamiltonian(
        core_tensors=ham_data["core_tensors"],
        leaf_tensors=ham_data["leaf_tensors"],
        nuc_constant=ham_data["nuc_constant"].item(),
    )
    state = {
        "coefficients": state_data["coefficients"],
        "indices": tuple(int(i) for i in state_data["indices"]),
        "num_qubits": state_data["num_qubits"].item(),
    }
    return hamiltonian, state


hamiltonian, state = load_molecule("hc2n3")

print(hamiltonian)
print(f"{len(state['indices'])} basis states on {state['num_qubits']} qubits")

######################################################################
# .. rst-class:: sphx-glr-script-out
#
#   .. code-block:: none
#
#     CGFHamiltonian(core_tensors=tensor(shape=(5, 12, 12, 4, 4)), leaf_tensors=tensor(shape=(5, 12, 4, 4)), nuc_constant=array(0.01672215))
#     19 basis states on 48 qubits

######################################################################
# The shapes match the ones we estimated with. The circuit is built with the same functions as
# before, now with concrete data and concrete wires.

registers = make_registers(state)
rules = make_rules(registers)
circuit = make_circuit(hamiltonian, state, registers, k_max=300)

######################################################################
# Compilation and resource analysis
# ---------------------------------
#
# To compile the circuit, we chain compilation passes into a
# :class:`~pennylane.CompilePipeline`. Here we target PBC:
#
# #. The ``decompose`` pass from before breaks the circuit down into Clifford gates, :math:`T`
#    gates and :math:`R_Z` rotations.
# #. :func:`~pennylane.transforms.gridsynth` approximates each :math:`R_Z` rotation to accuracy
#    :math:`\epsilon` with :math:`\pi/8` and Clifford PPRs. We use it because its cost per
#    rotation is close to the one assumed in [#Malpathak2026]_.
# #. :func:`~pennylane.transforms.to_ppr` converts the remaining gates into PPRs, and
#    :func:`~pennylane.transforms.ppr_to_ppm` replaces every PPR with Pauli product measurements
#    (PPMs).
#
# One more pass is needed in between. The decomposition keeps a function for every rule it
# considered, even the ones it did not use, and some of those contain controlled :math:`R_Z`
# rotations, which ``gridsynth`` cannot handle. Catalyst's ``symbol-dce`` (dead code elimination)
# pass removes these unused functions first.

pipeline = qp.CompilePipeline(
    decompose,
    qp.transform(pass_name="symbol-dce"),
    qp.transforms.gridsynth(epsilon=epsilon, method="mixed", ppr_basis=True),
    qp.transforms.to_ppr,
    qp.transforms.ppr_to_ppm,
)

######################################################################
# We apply the pipeline, compile with :func:`~pennylane.qjit`, and analyze the resources of the
# circuit after every pass with :func:`~pennylane.analyze`. Since we only analyze the circuit and
# never run it, we ask ``qjit`` to stop once the compilation passes are done
# (``target="mlir_opt"``) instead of producing an executable.


def analyze(circuit, rules, level):
    compiled = qp.qjit(pipeline(circuit), target="mlir_opt", enable_aot_compilation=False)
    with qp.decomposition.local_decomps():
        for op, rule in rules.items():
            qp.add_decomps(op, rule)
        return qp.analyze(compiled, level=level)()


start = time.perf_counter()
specs = analyze(circuit, rules, level="all")
print(f"Compiled and analyzed in {time.perf_counter() - start:.0f} s\n")

######################################################################
# The full report lists every gate type, and every PPR and PPM by its weight, the number of
# qubits it acts on. To keep things short, we summarize it for each stage of the pipeline.

print(f"{'Stage':<22}{'Operations':>15}{'RZ':>11}{'PPRs':>15}{'PPMs':>15}{'Magic states':>15}")
for stage, resources in specs.resources.items():
    ops = resources.quantum_operations
    pprs = sum(count for name, count in ops.items() if name.startswith("PPR"))
    ppms = sum(count for name, count in ops.items() if name.startswith("PPM"))
    print(
        f"{stage:<22}{resources.total_quantum_operations:>15,}{ops.get('RZ', 0):>11,}"
        f"{pprs:>15,}{ppms:>15,}{ops.get('pbc.fabricate', 0):>15,}"
    )

######################################################################
# .. rst-class:: sphx-glr-script-out
#
#   .. code-block:: none
#
#     Compiled and analyzed in 22 s
#
#     Stage                      Operations         RZ           PPRs           PPMs   Magic states
#     Before MLIR Passes                  3          0              0              0              0
#     graph-decomposition        24,692,137  5,156,547              0          5,999              0
#     symbol-dce                 24,692,137  5,156,547              0          5,999              0
#     gridsynth                 289,434,462          0    264,742,325          5,999              0
#     to-ppr                    340,959,867          0    316,267,733          5,999              0
#     ppr-to-ppm              1,238,079,028          0    316,267,732    632,390,732    264,734,429
#     Device Preprocessing    1,238,079,028          0    316,267,732    632,390,732    264,734,429

######################################################################
# The three high-level operations become millions of gates, and finally over a billion
# instructions. The decomposition produces 5,156,547 :math:`R_Z` rotations, one more than we
# estimated from shapes alone; the difference is in the state preparation, which now uses the
# actual basis states. Each rotation then becomes about 50 :math:`\pi/8` PPRs. In the last
# column, each :math:`\pi/8` rotation consumes a magic state, the most expensive resource on a
# fault-tolerant quantum computer.
#
# Compilation at scale
# ^^^^^^^^^^^^^^^^^^^^
#
# The full spectrum needs :math:`k_{\max} = 300`, but how does compilation cope with longer or
# shorter evolutions? We compile the circuit for an increasing number of time steps, and time it.

k_maxes = [1, 10, 100, 300, 1000]
num_operations, compile_times = [], []
for k_max in k_maxes:
    circuit = make_circuit(hamiltonian, state, registers, k_max=k_max)
    start = time.perf_counter()
    specs = analyze(circuit, rules, level="user")
    compile_times.append(time.perf_counter() - start)
    num_operations.append(specs.resources.total_quantum_operations)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
ax1.plot(k_maxes, num_operations, "o-", color="#2a78d6", linewidth=2, markersize=8)
ax1.set(xscale="log", yscale="log", xlabel="$k_{\\max}$", title="Compiled operations")
ax2.plot(k_maxes, compile_times, "o-", color="#2a78d6", linewidth=2, markersize=8)
ax2.set(xscale="log", xlabel="$k_{\\max}$", title="Compile time (s)", ylim=(0, 1.2 * max(compile_times)))
for ax in (ax1, ax2):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
plt.show()

######################################################################
# .. figure:: ../_static/demonstration_assets/pennylane2_placeholder/compile_scaling.png
#     :align: center
#     :width: 90%
#     :alt: Compiled operations grow linearly with k_max while compile time stays flat
#
# The number of operations grows a thousandfold, to over four billion, but the compile time
# stays flat. Catalyst keeps repeated structure, such as the Trotter steps, as loops in the
# compiled program instead of unrolling it, so the size of the problem barely affects how long
# it takes to compile.
#
# Conclusion
# ----------
#
# We implemented a state-of-the-art quantum algorithm for near-infrared spectroscopy, starting
# from the research papers. With abstract inputs, we estimated its resources for
# HC\ :sub:`2`\ N\ :sub:`3`, and for larger molecules, before using any data. With the actual data,
# the same circuit compiled to over a billion Pauli-based instructions, with its resources
# analyzed after every pass. With PennyLane2, the algorithm is a few high-level operations,
# compilation and resource analysis are one pipeline, and problems with billions of operations
# compile in seconds.
#
# To learn more, see the PennyLane2 documentation, including the guide for users
# `coming from PennyLane Classic
# <https://docs.pennylane.ai/en/latest/introduction/coming_from_classic.html>`__, and the
# :doc:`resource estimation demo for spectroscopy
# <demos/tutorial_resource_estimation_spectroscopy>`.
#
# References
# ----------
#
# .. [#Loaiza2025]
#
#     I. Loaiza, S. Fomichev, D. Motlagh, P. A. M. Casares, D. Honciuc Menendez, S. Shum,
#     A. Delgado, J. M. Arrazola,
#     "Simulating near-infrared spectroscopy on a quantum computer for enhanced chemical
#     detection". `arXiv:2504.10602 <https://arxiv.org/abs/2504.10602>`__ (2025).
#
# .. [#Litinski2019]
#
#     D. Litinski,
#     "A Game of Surface Codes: Large-Scale Quantum Computing with Lattice Surgery".
#     `Quantum 3, 128 (2019) <https://doi.org/10.22331/q-2019-03-05-128>`__.
#
# .. [#Malpathak2026]
#
#     S. Malpathak, S. D. Kallullathil, I. Loaiza, S. Fomichev, J. M. Arrazola, A. F. Izmaylov,
#     "Trotter simulation of vibrational Hamiltonians on a quantum computer".
#     `J. Chem. Theory Comput. 22, 95 (2026) <https://doi.org/10.1021/acs.jctc.5c01600>`__.
#
