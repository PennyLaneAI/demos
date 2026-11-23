r"""
Simulating near-infrared spectroscopy with PennyLane2
=====================================================

.. note::

    TODO: This demo only works with PennyLane2 installed. Link to the PennyLane2 installation
    instructions and documentation here (URLs TBD). This demo is not executed when the website
    is built. The outputs shown were generated with the development versions of PennyLane
    (0.46.0-dev159) and Catalyst (0.16.0-dev147); update them to the open beta release versions.

    TODO: Publish the H\ :sub:`2`\ S and HC\ :sub:`2`\ N\ :sub:`3` data files and add their
    download links (URLs TBD) to the start of the "Simulating real molecules" section.

Near-infrared spectroscopy (NIRS) measures how molecules absorb light through overtones and
combinations of their vibrations. These spectra are a fingerprint for chemical detection, and
computing them accurately is a natural application for fault-tolerant quantum computers
[#Loaiza2025]_.

In this demo, we go from inspiration to implementation with PennyLane2: we take the NIRS
algorithm from the research literature and write it with a handful of high-level building blocks.
Before touching any data, we estimate the resources it needs for a molecule of a given size. Then
we load real molecules and compile the algorithm into instructions for a fault-tolerant quantum
computer, scaling up to billions of operations, compiled in seconds.

PennyLane2 runs every program through a single compilation pipeline, powered by
`Catalyst <https://docs.pennylane.ai/projects/catalyst>`__ and designed for fault-tolerant
quantum computing (FTQC). That is what makes estimating, compiling and analyzing algorithms at
this scale fast.
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
# value throughout.
#
# Estimating resources with abstract inputs
# -----------------------------------------
#
# How many resources would this algorithm need for a molecule of a given size? With PennyLane2,
# we can answer this before computing a single Hamiltonian coefficient. Operators accept
# abstract inputs, which have a shape but no values, and :func:`~pennylane.estimate` projects the
# resources of a circuit onto a gate set without running it.
#
# Consider a hypothetical molecule with 10 vibrational modes and 4 modals per mode, so 40 system
# qubits, whose Hamiltonian is split into 5 fragments, and an initial state that is a superposition
# of 16 basis states. A :class:`~pennylane.CGFHamiltonian` is defined by its core and leaf
# tensors, and we only give their shapes with :data:`~pennylane.typing.Float`. The positions of
# the basis states in the initial state must be known at compile time, so we use placeholders.

import pennylane as qp

num_modes, num_modals, num_fragments, num_basis_states = 10, 4, 5, 16

hamiltonian = qp.CGFHamiltonian(
    core_tensors=qp.typing.Float[num_fragments, num_modes, num_modes, num_modals, num_modals],
    leaf_tensors=qp.typing.Float[num_fragments, num_modes, num_modals, num_modals],
    nuc_constant=qp.typing.Float,
)
state = {
    "coefficients": qp.typing.Float[num_basis_states],
    "indices": tuple(range(num_basis_states)),  # placeholder positions
    "num_qubits": num_modes * num_modals,
}

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
# (:class:`~pennylane.devices.NullQubit`), a device that performs no simulation at all: the
# circuits in this demo have well over a hundred qubits, far beyond what we could simulate
# classically, and we only want to estimate, compile and analyze them.


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

registers = make_registers(state)
registers = {name: qp.core.abstractify(wires) for name, wires in registers.items()}
rules = make_rules(registers)
circuit = make_circuit(hamiltonian, state, registers, k_max=300)

######################################################################
# Estimating resources
# ^^^^^^^^^^^^^^^^^^^^
#
# We estimate the resources in terms of Clifford gates, :math:`T` gates and :math:`R_Z`
# rotations, a common gate set for fault-tolerant algorithms. The gate set also tells the
# decomposition how much each gate costs, and :func:`~pennylane.decompose` selects our phase
# gradient rule for the multiplexed rotations. Our custom rules are registered locally, with
# :func:`~pennylane.decomposition.local_decomps`, while estimating.

clifford = {
    "X": 1, "Y": 1, "Z": 1, "S": 10, "Hadamard": 10, "SX": 10, "Adjoint(S)": 10, "Adjoint(SX)": 10,
    "CNOT": 50, "CY": 50, "CZ": 50, "GlobalPhase": 0, "Identity": 0,
}  # fmt: skip
measurements = {
    "PauliMeasure": 1, "PauliMeasureMP": 1, "MidMeasure": 1, "MidMeasureMP": 1,
    "MidCircuitMeasure": 1,
}  # fmt: skip
gate_set = {"T": 100, "Adjoint(T)": 100, "RZ": 5000} | clifford | measurements
decompose = qp.decompose(
    gate_set=gate_set, fixed_decomps={qp.SelectPauliRot: "nirs_spr_phase_gradient"}
)


def estimate(circuit, rules):
    compiled = qp.qjit(decompose(circuit), target="mlir_opt", enable_aot_compilation=False)
    with qp.decomposition.local_decomps():
        for op, rule in rules.items():
            qp.add_decomps(op, rule)
        return qp.estimate(compiled, target=gate_set)()


import time

start = time.perf_counter()
print(estimate(circuit, rules))
print(f"Estimated in {time.perf_counter() - start:.0f} s")

######################################################################
# .. rst-class:: sphx-glr-script-out
#
#   .. code-block:: none
#
#     Device: null.qubit
#     Device wires: 152
#     Shots: Shots(total=None)
#     Level: top
#
#     Quantum operations:
#     - Total: 1.821E+7
#       - Adjoint(S): 5.766E+5
#       - Adjoint(T): 720
#       - CNOT: 8.766E+6
#       - CZ: 360
#       - GlobalPhase: 1.152E+6
#       - Hadamard: 3.458E+6
#       - PauliMeasure: 1,549
#       - PauliX: 4,060
#       - PauliZ: 48,077
#       - RZ: 3.625E+6
#       - S: 5.762E+5
#       - T: 720
#     Measurement processes:
#     - expval(PauliX): 1
#     - expval(PauliY): 1
#     Total wires: 152
#     Circuit Depth: Not computed
#     Estimated in 4 s

######################################################################
# In a few seconds, and without any data, we know that this molecule needs about 150 logical
# qubits and :math:`3.6 \times 10^6` :math:`R_Z` rotations, which dominate the cost: at our
# accuracy, each rotation takes about 50 :math:`T` gates [#Malpathak2026]_. To estimate a
# different molecule, change the shapes; to explore a different algorithm, change the circuit.
#
# Simulating real molecules
# -------------------------
#
# Now we move from estimates to a concrete circuit, compiled all the way down for a
# fault-tolerant quantum computer. The CGF Hamiltonians and initial states that we use were
# computed classically and stored as NumPy arrays. TODO: download links for the data (URLs TBD).
# Place them in a ``nirs_data`` folder next to this demo.
#
# We load them for hydrogen sulfide (H\ :sub:`2`\ S) first.

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


hamiltonian, state = load_molecule("h2s")

print(hamiltonian)
print(f"{len(state['indices'])} basis states on {state['num_qubits']} qubits")

######################################################################
# .. rst-class:: sphx-glr-script-out
#
#   .. code-block:: none
#
#     CGFHamiltonian(core_tensors=tensor(shape=(6, 3, 3, 4, 4)), leaf_tensors=tensor(shape=(6, 3, 4, 4)), nuc_constant=array(0.0037294))
#     22 basis states on 12 qubits

######################################################################
# The Hamiltonian has six fragments acting on the three vibrational modes of H\ :sub:`2`\ S, with
# four modals per mode, so the system register needs 12 qubits. The circuit is built with the
# same functions as before, now with concrete data. We start small, with a single time step
# (:math:`k_{\max} = 1`).

registers = make_registers(state)
rules = make_rules(registers)
circuit = make_circuit(hamiltonian, state, registers, k_max=1)

######################################################################
# Compiling for a fault-tolerant quantum computer
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We compile the circuit to `Pauli-based computation
# <https://pennylane.ai/compilation/pauli-based-computation>`__ (PBC), in which a circuit is a
# sequence of Pauli product rotations (PPRs) and Pauli product measurements (PPMs). PBC is a
# natural fit for surface-code architectures based on lattice surgery, where multi-qubit Pauli
# measurements are the native operations [#Litinski2019]_. In PennyLane2, getting there is a
# matter of chaining compilation passes into a :class:`~pennylane.CompilePipeline`:
#
# #. The ``decompose`` pass from before breaks the circuit down into Clifford gates, :math:`T`
#    gates and :math:`R_Z` rotations.
# #. :func:`~pennylane.transforms.gridsynth` approximates each :math:`R_Z` rotation to accuracy
#    :math:`\epsilon` with :math:`\pi/8` and Clifford PPRs.
# #. :func:`~pennylane.transforms.to_ppr` converts the remaining gates into PPRs, and
#    :func:`~pennylane.transforms.ppr_to_ppm` replaces every PPR with PPMs.
#
# The PBC passes alone are not enough: :func:`~pennylane.transforms.ppr_to_ppm` only knows how to
# implement Clifford and :math:`\pi/8` rotations, the latter by consuming a magic state. An
# :math:`R_Z` rotation by an arbitrary angle has no such implementation, so ``gridsynth`` first
# approximates it with rotations that do. Alternatively, these rotations could be implemented
# with the phase gradient state, as we do for the multiplexed rotations. We use ``gridsynth``
# because its cost per rotation is close to the one assumed in [#Malpathak2026]_.

pipeline = qp.CompilePipeline(
    decompose,
    qp.transform(pass_name="symbol-dce"),  # remove unused decomposition rules
    qp.transforms.gridsynth(epsilon=epsilon, method="mixed", ppr_basis=True),
    qp.transforms.to_ppr,
    qp.transforms.ppr_to_ppm,
)

######################################################################
# We apply the pipeline, compile with :func:`~pennylane.qjit`, and inspect the circuit after
# every pass with :func:`~pennylane.analyze`. Since we only analyze the circuit and never run it,
# we ask ``qjit`` to stop once the compilation passes are done (``target="mlir_opt"``) instead of
# producing an executable.


def analyze(circuit, rules, level):
    compiled = qp.qjit(pipeline(circuit), target="mlir_opt", enable_aot_compilation=False)
    with qp.decomposition.local_decomps():
        for op, rule in rules.items():
            qp.add_decomps(op, rule)
        return qp.analyze(compiled, level=level)()


specs = analyze(circuit, rules, level="all")

######################################################################
# The full report lists every gate type, and every PPR and PPM by its weight, the number of
# qubits it acts on. To keep things short, we summarize it for each stage of the pipeline.

print(f"{'Stage':<28}{'Operations':>12}{'PPRs':>10}{'PPMs':>10}{'Magic states':>14}")
for stage, resources in specs.resources.items():
    ops = resources.quantum_operations
    pprs = sum(count for name, count in ops.items() if name.startswith("PPR"))
    ppms = sum(count for name, count in ops.items() if name.startswith("PPM"))
    magic_states = ops.get("pbc.fabricate", 0)
    total = resources.total_quantum_operations
    print(f"{stage:<28}{total:>12,}{pprs:>10,}{ppms:>10,}{magic_states:>14,}")

######################################################################
# .. rst-class:: sphx-glr-script-out
#
#   .. code-block:: none
#
#     Stage                         Operations      PPRs      PPMs  Magic states
#     Before MLIR Passes                     3         0         0             0
#     graph-decomposition               43,227         0     4,961             0
#     symbol-dce                        43,227         0     4,961             0
#     gridsynth                        132,100    88,873     4,961             0
#     to-ppr                           202,696   159,472     4,961             0
#     ppr-to-ppm                       589,855   159,471   300,308        91,813
#     Device Preprocessing             589,855   159,471   300,308        91,813

######################################################################
# The three high-level operations become tens of thousands of Clifford gates, :math:`T` gates,
# :math:`R_Z` rotations and measurements, and finally a sequence of PPMs. The number that matters
# most is the last column: the magic states that each non-Clifford :math:`\pi/8` rotation
# consumes. Magic states are the most expensive resource on a fault-tolerant quantum computer.
#
# Scaling up
# ^^^^^^^^^^
#
# H\ :sub:`2`\ S with a single time step is a small example. We now move to azidoacetylene
# (HC\ :sub:`2`\ N\ :sub:`3`), a molecule with 12 vibrational modes, and compile the longest
# circuit for an increasing number of time steps, up to beyond the :math:`k_{\max} = 300` that a
# full spectrum needs. This time, we only look at the end of the pipeline, and we time the
# compilation.

hamiltonian, state = load_molecule("hc2n3")
registers = make_registers(state)
rules = make_rules(registers)

print(f"{'k_max':>6}{'Logical qubits':>16}{'Operations':>12}{'Magic states':>14}{'Time':>7}")
for k_max in (1, 10, 100, 300, 1000):
    circuit = make_circuit(hamiltonian, state, registers, k_max=k_max)
    start = time.perf_counter()
    specs = analyze(circuit, rules, level="user")
    elapsed = time.perf_counter() - start
    ops = specs.resources.quantum_operations
    total = specs.resources.total_quantum_operations
    print(
        f"{k_max:>6}{specs.num_device_wires:>16}{total:>12.2e}"
        f"{ops['pbc.fabricate']:>14.2e}{elapsed:>6.0f}s"
    )

######################################################################
# .. rst-class:: sphx-glr-script-out
#
#   .. code-block:: none
#
#      k_max  Logical qubits  Operations  Magic states   Time
#          1             176    4.60e+06      9.54e+05    20s
#         10             176    4.17e+07      8.89e+06    20s
#        100             176    4.13e+08      8.83e+07    20s
#        300             176    1.24e+09      2.65e+08    21s
#       1000             176    4.13e+09      8.82e+08    21s

######################################################################
# The number of operations grows with the number of time steps, to over four billion, but the
# compilation time does not. Catalyst keeps repeated structure, such as the Trotter steps, as
# loops in the compiled program instead of unrolling it, so the full-scale circuit compiles to
# fault-tolerant instructions in seconds.
#
# Conclusion
# ----------
#
# We implemented a state-of-the-art quantum algorithm for near-infrared spectroscopy, starting
# from the research papers. With abstract inputs, we estimated its resources for a molecule of a
# given size before using any data. With real data, we compiled it into Pauli-based instructions
# for a fault-tolerant quantum computer. With PennyLane2, the algorithm is a few high-level
# operations, compilation is a pipeline of passes, and a full-scale problem with billions of
# operations compiles and analyzes in seconds.
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
