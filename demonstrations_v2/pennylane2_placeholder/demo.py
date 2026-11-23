r"""
Simulating near-infrared spectroscopy with PennyLane2
=====================================================

.. note::

    TODO: This demo only works with PennyLane2 installed. Link to the PennyLane2 installation
    instructions and documentation here (URLs TBD). This demo is not executed when the website
    is built. The outputs shown were generated with the development versions of PennyLane
    (0.46.0-dev159) and Catalyst (0.16.0-dev147); update them to the open beta release versions.

Near-infrared spectroscopy (NIRS) measures how molecules absorb light through overtones and
combinations of their vibrations. These spectra are a fingerprint for chemical detection, and
computing them accurately is a natural application for fault-tolerant quantum computers
[#Loaiza2025]_.

In this demo, we go from inspiration to implementation with PennyLane2: we take the NIRS
algorithm from the research literature, write it with a handful of high-level building blocks,
and compile it into instructions for a fault-tolerant quantum computer. Then we scale it up to
a full-sized problem with over a billion operations, compiled in seconds.

PennyLane2 runs every program through a single compilation pipeline, powered by
`Catalyst <https://docs.pennylane.ai/projects/catalyst>`__ and designed for fault-tolerant
quantum computing (FTQC). That is what makes compiling and analyzing algorithms at this scale
fast.
"""

######################################################################
# The algorithm
# -------------
#
# The NIRS spectrum is the Fourier transform of the time-correlation function
# :math:`\langle \psi | e^{-iHt} | \psi \rangle`, where :math:`H` is the vibrational Hamiltonian
# of the molecule and :math:`|\psi\rangle` is the state created when the molecule absorbs light.
# The quantum algorithm estimates this function with a Hadamard test: a control qubit decides
# whether the time evolution is applied, and measuring :math:`X` and :math:`Y` on that qubit
# gives the real and imaginary parts of the correlation function. Repeating this for
# evolution times :math:`t_k = k\tau` with :math:`k = 1, \dots, k_{\max}` yields the
# spectrum. For a detailed walkthrough of spectroscopy algorithms of this kind, see the
# :doc:`X-ray Absorption Spectroscopy (XAS) <demos/tutorial_xas>` demo.
#
# Loading the Hamiltonian and initial state
# -----------------------------------------
#
# Following [#Malpathak2026]_, we write the vibrational Hamiltonian in the Christiansen form and
# factorize it into a sum of fragments with Christiansen greedy fragmentation (CGF). Each
# fragment can be implemented efficiently with rotations and Pauli :math:`ZZ` interactions. The
# initial state :math:`|\psi\rangle` is the dipole operator applied to the vibrational ground
# state, restricted to the near-infrared window. It is a superposition of a small number of
# computational basis states.
#
# Both were computed classically and stored as NumPy arrays. We load them for hydrogen sulfide
# (H\ :sub:`2`\ S) first, and wrap the Hamiltonian in a :class:`~pennylane.CGFHamiltonian`.

import numpy as np
import pennylane as qp


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
# four states per mode, so the system register needs 12 qubits.
#
# Building the circuit
# --------------------
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
# Registers
# ^^^^^^^^^
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


registers = make_registers(state)

######################################################################
# Decomposition rules
# ^^^^^^^^^^^^^^^^^^^
#
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
    spr_rule.name = "spr_phase_gradient"
    pgsp_rule = decomps.make_phase_gradient_distillation_decomp(
        pg_wires["angle_wires"], pg_wires["work_wires"]
    )
    return {qp.SelectPauliRot: spr_rule, qp.PhaseGradientStatePrep: pgsp_rule}


rules = make_rules(registers)

######################################################################
# The circuit
# ^^^^^^^^^^^
#
# Finally, the circuit itself. It has 130 qubits, far beyond what we could simulate classically,
# and we only want to compile and analyze it. So we run it on ``null.qubit``
# (:class:`~pennylane.devices.NullQubit`), a device that performs no simulation at all.
#
# The algorithm runs one circuit for each evolution time :math:`t_k`, and the longest one,
# :math:`t = k_{\max}\tau`, is the most expensive. Since we only analyze that circuit, we build
# it directly. We start small, with a single time step (:math:`k_{\max} = 1`) for
# H\ :sub:`2`\ S.


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


circuit = make_circuit(hamiltonian, state, registers, k_max=1)

######################################################################
# Compiling for a fault-tolerant quantum computer
# -----------------------------------------------
#
# We compile the circuit to `Pauli-based computation
# <https://pennylane.ai/compilation/pauli-based-computation>`__ (PBC), in which a circuit is a
# sequence of Pauli product rotations (PPRs) and Pauli product measurements (PPMs). PBC is a
# natural fit for surface-code architectures based on lattice surgery, where multi-qubit Pauli
# measurements are the native operations [#Litinski2019]_. In PennyLane2, getting there is a
# matter of chaining compilation passes into a :class:`~pennylane.CompilePipeline`:
#
# #. :func:`~pennylane.decompose` breaks the circuit down into Clifford gates, :math:`T` gates and
#    :math:`R_Z` rotations, using our phase gradient rule for the multiplexed rotations.
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
# because its cost per rotation is close to the one assumed in [#Loaiza2025]_.
#
# The gate set tells the decomposition which gates to target, and how much each one costs.

clifford = {"X": 1, "Y": 1, "Z": 1, "S": 10, "Adjoint(S)": 10, "Hadamard": 10}
clifford |= {"CNOT": 50, "CZ": 50, "GlobalPhase": 0}
measurements = {"MidMeasure": 1, "PauliMeasure": 1}
gate_set = {"T": 100, "Adjoint(T)": 100, "RZ": 5000} | clifford | measurements

pipeline = qp.CompilePipeline(
    qp.decompose(gate_set=gate_set, fixed_decomps={qp.SelectPauliRot: "spr_phase_gradient"}),
    qp.transform(pass_name="symbol-dce"),  # remove unused decomposition rules
    qp.transforms.gridsynth(epsilon=epsilon, method="mixed", ppr_basis=True),
    qp.transforms.to_ppr,
    qp.transforms.ppr_to_ppm,
)

######################################################################
# We apply the pipeline, compile with :func:`~pennylane.qjit`, and inspect the circuit after
# every pass with :func:`~pennylane.analyze`. Since we only analyze the circuit and never run it,
# we ask ``qjit`` to stop once the compilation passes are done (``target="mlir_opt"``) instead of
# producing an executable. Our custom rules are registered locally while compiling.

from pennylane.decomposition.decomposition_rule import local_decomps


def analyze(circuit, rules, level):
    compiled = qp.qjit(pipeline(circuit), target="mlir_opt", enable_aot_compilation=False)
    with local_decomps():
        for op, rule in rules.items():
            qp.add_decomps(op, rule)
        return qp.analyze(compiled, level=level)()


print(analyze(circuit, rules, level="all"))

######################################################################
# .. rst-class:: sphx-glr-script-out
#
#   .. code-block:: none
#
#     Device: null.qubit
#     Device wires: 130
#     Shots: Shots(total=None)
#     Levels:
#     - 0: Before MLIR Passes
#     - 1: After graph-decomposition
#     - 2: After symbol-dce
#     - 3: After gridsynth
#     - 4: After to-ppr
#     - 5: After ppr-to-ppm
#     - 6: After Device Preprocessing
#
#     ↓Metric             Level→ |        0 |        1 |        2 |        3 |        4 |        5 |        6
#     -------------------------------------------------------------------------------------------------------
#     Quantum operations:        |
#     - Total                    |        3 |   43,227 |   43,227 | 1.321E+5 | 2.027E+5 | 5.899E+5 | 5.899E+5
#       - 1C(TrotterCGF)         |        1 |        0 |        0 |        0 |        0 |        0 |        0
#       - PhaseGradientStatePrep |        1 |        0 |        0 |        0 |        0 |        0 |        0
#       - SumOfSlatersPrep       |        1 |        0 |        0 |        0 |        0 |        0 |        0
#       - Adjoint(S)             |        0 |    1,583 |    1,583 |    1,583 |        0 |        0 |        0
#       - Adjoint(T)             |        0 |    1,473 |    1,473 |    1,473 |        0 |        0 |        0
#       - CNOT                   |        0 |   10,527 |   10,527 |   10,527 |        0 |        0 |        0
#       - CZ                     |        0 |      737 |      737 |      737 |        0 |        0 |        0
#       - GlobalPhase            |        0 |    1,585 |    1,585 |    3,316 |   38,263 |   38,263 |   38,263
#       - Hadamard               |        0 |    6,564 |    6,564 |    6,564 |        0 |        0 |        0
#       - PPM-w1                 |        0 |    2,029 |    2,029 |    2,029 |    2,029 | 1.497E+5 | 1.497E+5
#       - PPM-w10                |        0 |      105 |      105 |      105 |      105 |      105 |      105
#       - PPM-w11                |        0 |      105 |      105 |      105 |      105 |      105 |      105
#       - PPM-w12                |        0 |      105 |      105 |      105 |      105 |      105 |      105
#       - PPM-w13                |        0 |      105 |      105 |      105 |      105 |      105 |      105
#       - PPM-w14                |        0 |       76 |       76 |       76 |       76 |       76 |       76
#       - PPM-w15                |        0 |       76 |       76 |       76 |       76 |       76 |       76
#       - PPM-w16                |        0 |       76 |       76 |       76 |       76 |       76 |       76
#       - PPM-w17                |        0 |       76 |       76 |       76 |       76 |       76 |       76
#       - PPM-w18                |        0 |       76 |       76 |       76 |       76 |       76 |       76
#       - PPM-w19                |        0 |       76 |       76 |       76 |       76 |       76 |       76
#       - PPM-w2                 |        0 |      105 |      105 |      105 |      105 | 1.365E+5 | 1.365E+5
#       - PPM-w20                |        0 |       76 |       76 |       76 |       76 |       76 |       76
#       - PPM-w21                |        0 |       76 |       76 |       76 |       76 |       76 |       76
#       - PPM-w22                |        0 |       76 |       76 |       76 |       76 |       76 |       76
#       - PPM-w23                |        0 |       76 |       76 |       76 |       76 |       76 |       76
#       - PPM-w24                |        0 |       76 |       76 |       76 |       76 |       76 |       76
#       - PPM-w25                |        0 |       76 |       76 |       76 |       76 |       76 |       76
#       - PPM-w26                |        0 |       76 |       76 |       76 |       76 |       76 |       76
#       - PPM-w27                |        0 |       76 |       76 |       76 |       76 |       76 |       76
#       - PPM-w28                |        0 |       76 |       76 |       76 |       76 |       76 |       76
#       - PPM-w29                |        0 |       76 |       76 |       76 |       76 |       76 |       76
#       - PPM-w3                 |        0 |      105 |      105 |      105 |      105 |   11,368 |   11,368
#       - PPM-w30                |        0 |       76 |       76 |       76 |       76 |       76 |       76
#       - PPM-w31                |        0 |       76 |       76 |       76 |       76 |       76 |       76
#       - PPM-w32                |        0 |       76 |       76 |       76 |       76 |       76 |       76
#       - PPM-w33                |        0 |       76 |       76 |       76 |       76 |       76 |       76
#       - PPM-w34                |        0 |       76 |       76 |       76 |       76 |       76 |       76
#       - PPM-w35                |        0 |       76 |       76 |       76 |       76 |       76 |       76
#       - PPM-w4                 |        0 |      105 |      105 |      105 |      105 |      105 |      105
#       - PPM-w5                 |        0 |      105 |      105 |      105 |      105 |      105 |      105
#       - PPM-w6                 |        0 |      105 |      105 |      105 |      105 |      105 |      105
#       - PPM-w7                 |        0 |      105 |      105 |      105 |      105 |      105 |      105
#       - PPM-w8                 |        0 |      105 |      105 |      105 |      105 |      105 |      105
#       - PPM-w9                 |        0 |      105 |      105 |      105 |      105 |      105 |      105
#       - PauliX                 |        0 |   11,575 |   11,575 |   11,575 |        0 |        0 |        0
#       - PauliZ                 |        0 |      221 |      221 |      221 |        0 |        0 |        0
#       - RZ                     |        0 |    1,731 |    1,731 |        0 |        0 |        0 |        0
#       - S                      |        0 |      797 |      797 |      797 |        0 |        0 |        0
#       - T                      |        0 |    1,473 |    1,473 |    1,473 |        0 |        0 |        0
#       - PPR-pi/2-w1            |        0 |        0 |        0 |        2 |   11,797 | 1.482E+5 | 1.482E+5
#       - PPR-pi/4-w1            |        0 |        0 |        0 |        3 |   44,599 |        0 |        0
#       - PPR-pi/8-w1            |        0 |        0 |        0 |   88,868 |   91,813 |        0 |        0
#       - PPR-pi/4-w2            |        0 |        0 |        0 |        0 |   11,263 |        0 |        0
#       - PPR-pi/2-w2            |        0 |        0 |        0 |        0 |        0 |   11,263 |   11,263
#       - pbc.fabricate          |        0 |        0 |        0 |        0 |        0 |   91,813 |   91,813
#     Measurement processes:     |
#     - expval(PauliX)           |        1 |        1 |        1 |        1 |        1 |        1 |        1
#     - expval(PauliY)           |        1 |        1 |        1 |        1 |        1 |        1 |        1
#     Total wires                |      130 |      130 |      130 |      130 |      130 | 1.478E+5 | 1.478E+5
#     PBC Depths:                |
#     - Any commuting depth      |        - |    2,133 |    2,133 |   91,004 | 1.392E+5 | 3.789E+5 | 3.789E+5
#     - Qubit disjoint depth     |        - |    2,133 |    2,133 |   91,004 | 1.506E+5 | 3.903E+5 | 3.903E+5

######################################################################
# Each column is one stage of the pipeline. The three high-level operations become about 43,000
# Clifford gates, :math:`T` gates, :math:`R_Z` rotations and measurements, and finally a sequence
# of PPMs. The number that matters most is ``pbc.fabricate``: the magic states that each
# non-Clifford :math:`\pi/8` rotation consumes. Magic states are the most expensive resource on
# a fault-tolerant quantum computer.
#
# Scaling up
# ----------
#
# H\ :sub:`2`\ S with a single time step is a small example. A useful spectrum needs a larger
# molecule and the full set of evolution times. We now build the longest circuit,
# :math:`k_{\max} = 300`, for azidoacetylene (HC\ :sub:`2`\ N\ :sub:`3`), a molecule with 12
# vibrational modes. We call the same functions with a different molecule and
# :math:`k_{\max}`. This time, we only look at the end of the pipeline and time the compilation.

import time

hamiltonian, state = load_molecule("hc2n3")
registers = make_registers(state)
rules = make_rules(registers)
circuit = make_circuit(hamiltonian, state, registers, k_max=300)

start = time.perf_counter()
specs = analyze(circuit, rules, level="user")
elapsed = time.perf_counter() - start

resources = specs.resources
print(f"Logical qubits: {specs.num_device_wires}")
print(f"Operations: {resources.total_quantum_operations:.3e}")
print(f"Magic states: {resources.quantum_operations['pbc.fabricate']:.3e}")
print(f"Compiled and analyzed in {elapsed:.0f} s")

######################################################################
# .. rst-class:: sphx-glr-script-out
#
#   .. code-block:: none
#
#     Logical qubits: 176
#     Operations: 1.238e+09
#     Magic states: 2.647e+08
#     Compiled and analyzed in 19 s

######################################################################
# The full-scale circuit has over a billion operations, and PennyLane2 compiles it down to
# fault-tolerant instructions in seconds. Catalyst keeps repeated structure, such as the 600
# Trotter steps, as loops in the compiled program instead of unrolling it, so the compilation
# time does not grow with the number of operations.
#
# Conclusion
# ----------
#
# We implemented a state-of-the-art quantum algorithm for near-infrared spectroscopy, starting
# from the research papers, and compiled it into Pauli-based instructions for a fault-tolerant
# quantum computer. With PennyLane2, the algorithm is a few high-level operations, compilation
# is a pipeline of passes, and a billion-operation, full-scale problem compiles and analyzes in
# seconds.
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
