r"""
Simulating near-infrared spectroscopy with PennyLane2
=====================================================

Near-infrared spectroscopy (NIRS) probes overtones and combination bands of molecular
vibrations. In this demo, we build the quantum algorithm that computes an NIRS spectrum for
hydrogen sulfide (H\ :sub:`2`\ S), following [#nirs_paper]_, and compile it all the way down to a
fault-tolerant instruction set with PennyLane2 and Catalyst.

The algorithm estimates the time-correlation function
:math:`\langle \psi | e^{-iHt} | \psi \rangle` of a vibrational Hamiltonian :math:`H` with a
Hadamard test. The spectrum is then the Fourier transform of these estimates over many evolution
times :math:`t`.

.. note::

    This demo is not executed when the website is built. The outputs shown were generated with
    the development versions of PennyLane (0.46.0-dev154) and Catalyst (0.16.0-dev143).
"""

######################################################################
# Loading the Hamiltonian and initial state
# -----------------------------------------
#
# We use a pre-computed vibrational Hamiltonian for H\ :sub:`2`\ S in the Christiansen form,
# factorized into fragments with Christiansen greedy fragmentation (CGF) [#cgf_paper]_, together
# with the initial state :math:`|\psi\rangle`: the dipole operator applied to the vibrational
# ground state, restricted to the near-infrared window.

import numpy as np
import pennylane as qp

ham_data = np.load("nirs_data/cgf_regrouped.npz")
state_data = np.load("nirs_data/vscf_state.npz")

hamiltonian = qp.CGFHamiltonian(
    core_tensors=ham_data["core_tensors"],
    leaf_tensors=ham_data["leaf_tensors"],
    nuc_constant=ham_data["nuc_constant"].item(),
)

indices = tuple(int(i) for i in state_data["indices"])
coefficients = state_data["coefficients"]
num_qubits = state_data["num_qubits"].item()

print(hamiltonian)
print(f"{len(indices)} basis states on {num_qubits} qubits")

######################################################################
# .. rst-class:: sphx-glr-script-out
#
#   .. code-block:: none
#
#     CGFHamiltonian(core_tensors=tensor(shape=(6, 3, 3, 4, 4)), leaf_tensors=tensor(shape=(6, 3, 4, 4)), nuc_constant=array(0.0037294))
#     22 basis states on 12 qubits

######################################################################
# The Hamiltonian has six fragments acting on three vibrational modes, each with four modals,
# so the system register needs 12 qubits. The initial state is a superposition of only 22
# computational basis states.
#
# Setting up the registers
# ------------------------
#
# Besides the system, the circuit needs a control qubit for the Hadamard test, wires for the
# state preparation, and wires for implementing rotations with a phase gradient state. We
# synthesize rotations to an accuracy of :math:`\epsilon = 10^{-10}`.

epsilon = 1e-10
precision = qp.math.ceil_log2(1 / epsilon)
num_work_wires = max(precision - 1, qp.math.ceil_log2(len(indices)) - 2)

register_sizes_sos = qp.SumOfSlatersPrep.required_register_sizes(indices, num_qubits)
register_sizes_sos.pop("wires")

register_sizes = {
    "hadamard": 1,
    "system": num_qubits,
    "angle_wires": precision,
    "phase_grad_wires": precision,
    **register_sizes_sos,
    "work_wires": num_work_wires,
}
registers = qp.registers(register_sizes)
total_wires = sum(register_sizes.values())

print(register_sizes)
print("Total wires:", total_wires)

######################################################################
# .. rst-class:: sphx-glr-script-out
#
#   .. code-block:: none
#
#     {'hadamard': 1, 'system': 12, 'angle_wires': 34, 'phase_grad_wires': 34, 'enumeration_wires': 5, 'identification_wires': 0, 'qrom_work_wires': 4, 'mcx_cache_wires': 7, 'work_wires': 33}
#     Total wires: 130

######################################################################
# Building the circuit
# --------------------
#
# The circuit prepares a phase gradient state, prepares :math:`|\psi\rangle` with
# :class:`~pennylane.SumOfSlatersPrep`, and applies the controlled Trotterized time evolution
# :class:`~pennylane.TrotterCGF`. The expectation values of :math:`X` and :math:`Y` on the
# control qubit give the real and imaginary parts of the time-correlation function. We build
# the shortest circuit, with evolution time :math:`t = 250` atomic units and two Trotter steps.

evolution_time = 250.0
num_trotter_steps = 2

dev = qp.device("null.qubit", wires=total_wires)

sos_work_wires = {
    k: registers[k]
    for k in ("enumeration_wires", "qrom_work_wires", "identification_wires", "mcx_cache_wires")
}


@qp.qnode(dev)
def circuit():
    qp.PhaseGradientStatePrep(registers["phase_grad_wires"])
    qp.SumOfSlatersPrep(coefficients, wires=registers["system"], indices=indices, **sos_work_wires)
    qp.ctrl(
        qp.TrotterCGF(
            evolution_time=evolution_time,
            num_trotter_steps=num_trotter_steps,
            hamiltonian=hamiltonian,
            wires=registers["system"],
            double_phase=True,
        ),
        control=registers["hadamard"],
    )
    return qp.expval(qp.X(registers["hadamard"])), qp.expval(qp.Y(registers["hadamard"]))


######################################################################
# Compiling to Pauli-based computation
# ------------------------------------
#
# Multiplexed rotations in the state preparation are implemented with the phase gradient
# technique, and the phase gradient state itself is prepared by distillation. Both rules act on
# the registers defined above, so we create them here and register them locally.

from pennylane.decomposition.decomposition_rule import local_decomps

pg_registers = {k: registers[k] for k in ("angle_wires", "phase_grad_wires", "work_wires")}

spr_rule = qp.transforms.decompositions.make_selectpaulirot_to_phase_gradient_decomp(
    **pg_registers
)
spr_rule.name = "nirs_spr_phase_gradient"
pgsp_rule = qp.transforms.decompositions.make_phase_gradient_distillation_decomp(
    pg_registers["angle_wires"], pg_registers["work_wires"]
)

######################################################################
# The compilation pipeline decomposes the circuit into Clifford + T + RZ gates, synthesizes the
# RZ rotations with ``gridsynth``, and converts the result into Pauli product rotations (PPRs) and
# Pauli product measurements (PPMs).

CLIFFORD = {
    "X": 1, "Y": 1, "Z": 1, "S": 10, "Hadamard": 10, "SX": 10, "Adjoint(S)": 10,
    "Adjoint(SX)": 10, "CNOT": 50, "CY": 50, "CZ": 50, "GlobalPhase": 0, "Identity": 0,
}
MEASUREMENTS = {
    "PauliMeasure": 1, "PauliMeasureMP": 1, "MidMeasure": 1, "MidMeasureMP": 1,
    "MidCircuitMeasure": 1,
}
gate_set = {"T": 100, "Adjoint(T)": 100, "RZ": 5000} | CLIFFORD | MEASUREMENTS

pipeline = qp.CompilePipeline(
    qp.decompose(gate_set=gate_set, fixed_decomps={qp.SelectPauliRot: spr_rule.name}),
    qp.transform(pass_name="symbol-dce"),
    qp.transforms.gridsynth(epsilon=epsilon, method="mixed", ppr_basis=True),
    qp.transforms.to_ppr,
    qp.transforms.ppr_to_ppm,
)

compiled = qp.qjit(pipeline(circuit), capture=True, target="mlir_opt", enable_aot_compilation=False)

######################################################################
# We can now inspect the resources at every stage of the pipeline with :func:`~pennylane.analyze`.

with local_decomps():
    qp.add_decomps(qp.PhaseGradientStatePrep, pgsp_rule)
    qp.add_decomps(qp.SelectPauliRot, spr_rule)
    print(qp.analyze(compiled, level="all")())

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
#     - 1: graph-decomposition
#     - 2: symbol-dce
#     - 3: gridsynth
#     - 4: to-ppr
#     - 5: ppr-to-ppm
#     - 6: Device Preprocessing
#
#     ↓Metric             Level→ |        0 |        1 |        2 |        3 |        4 |        5 |        6
#     -------------------------------------------------------------------------------------------------------
#     Quantum operations:        |
#     - Total                    |        3 |   38,861 |   38,861 | 1.277E+5 | 1.986E+5 | 5.861E+5 | 5.861E+5
#       - C(TrotterCGF)          |        1 |        0 |        0 |        0 |        0 |        0 |        0
#       - PhaseGradientStatePrep |        1 |        0 |        0 |        0 |        0 |        0 |        0
#       - SumOfSlatersPrep       |        1 |        0 |        0 |        0 |        0 |        0 |        0
#       - Adjoint(S)             |        0 |    1,575 |    1,575 |    1,575 |        0 |        0 |        0
#       - Adjoint(T)             |        0 |    1,457 |    1,457 |    1,457 |        0 |        0 |        0
#       - CNOT                   |        0 |   12,997 |   12,997 |   12,997 |        0 |        0 |        0
#       - CZ                     |        0 |      729 |      729 |      729 |        0 |        0 |        0
#       - GlobalPhase            |        0 |        1 |        1 |    1,732 |   36,845 |   36,845 |   36,845
#       - Hadamard               |        0 |    4,172 |    4,172 |    4,172 |        0 |        0 |        0
#       - PPM-w1                 |        0 |    2,013 |    2,013 |    2,013 |    2,013 | 1.499E+5 | 1.499E+5
#       - PauliX                 |        0 |   11,816 |   11,816 |   11,816 |        0 |        0 |        0
#       - PauliZ                 |        0 |      116 |      116 |      116 |        0 |        0 |        0
#       - RZ                     |        0 |    1,731 |    1,731 |        0 |        0 |        0 |        0
#       - S                      |        0 |      797 |      797 |      797 |        0 |        0 |        0
#       - T                      |        0 |    1,457 |    1,457 |    1,457 |        0 |        0 |        0
#       - PPR-pi/2-w1            |        0 |        0 |        0 |        2 |   11,933 | 1.461E+5 | 1.461E+5
#       - PPR-pi/4-w1            |        0 |        0 |        0 |        3 |   42,339 |        0 |        0
#       - PPR-pi/8-w1            |        0 |        0 |        0 |   88,868 |   91,781 |        0 |        0
#       - PPR-pi/4-w2            |        0 |        0 |        0 |        0 |   13,725 |        0 |        0
#       - PPM-w2                 |        0 |        0 |        0 |        0 |        0 | 1.341E+5 | 1.341E+5
#       - PPM-w3                 |        0 |        0 |        0 |        0 |        0 |   13,725 |   13,725
#       - PPR-pi/2-w2            |        0 |        0 |        0 |        0 |        0 |   13,725 |   13,725
#       - pbc.fabricate          |        0 |        0 |        0 |        0 |        0 |   91,781 |   91,781
#     Measurement processes:     |
#     - expval(PauliX)           |        1 |        1 |        1 |        1 |        1 |        1 |        1
#     - expval(PauliY)           |        1 |        1 |        1 |        1 |        1 |        1 |        1
#     Total wires                |      130 |      130 |      130 |      130 |      130 | 1.480E+5 | 1.480E+5
#     PBC Depths:                |
#     - Any commuting depth      |        - |    2,117 |    2,117 |   90,988 | 1.350E+5 | 3.755E+5 | 3.755E+5
#     - Qubit disjoint depth     |        - |    2,117 |    2,117 |   90,988 | 1.491E+5 | 3.896E+5 | 3.896E+5

######################################################################
# The three high-level operations become about 39,000 Clifford + T + RZ gates. After rotation
# synthesis and conversion, the circuit needs 91,781 non-Clifford :math:`\pi/8` rotations, each
# implemented with a magic state (``pbc.fabricate``).
#
# Conclusion
# ----------
#
# TODO: summary and outlook.
#
# References
# ----------
#
# .. [#nirs_paper]
#
#     TODO: reference for arXiv:2504.10602.
#
# .. [#cgf_paper]
#
#     TODO: reference for arXiv:2508.11865.
#
