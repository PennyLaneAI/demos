r"""
Quantifying resource requirements for vibronic dynamics simulation
==================================================================

Standard quantum chemistry often relies on the adiabatic Born-Oppenheimer approximation, assuming nuclei move on a single potential energy surface (PES)
that is well-separated from others. But in the world of photochemistry, where molecules absorb light and can undergo bond dissociation, this approximation
breaks down. The timescales of electronic and nuclear motion become comparable, giving rise to vibronic coupling. This coupling drives
critical processes like photosynthesis and solar cell efficiency. Accurately simulating this requires a Hamiltonian that treats the dynamics
of electrons and nuclei simultaneously.

However, modeling such dynamics is computationally demanding. Before we attempt to run such complex algorithms on future fault-tolerant hardware,
we need to answer a fundamental question: Is this algorithm actually feasible?

.. figure:: ../_static/demo_thumbnails/opengraph_demo_thumbnails/pennylane-demo-resource-estimation-vibronic-open-graph.png
    :align: center
    :width: 60%
    :target: javascript:void(0)

This demo is based on our paper `Quantum Algorithm for Vibronic Dynamics: Case Study on Singlet Fission Solar Cell Design
<https://arxiv.org/abs/2411.13669>`__ by Motlagh et al. [#Motlagh2025]_, and will present how to estimate the resource requirements of a vibronic Hamiltonian simulation
from the ground up. An efficient treatment of the potential energy will be presented. With the modular building blocks provided by PennyLane's resource
:mod:`~.pennylane.estimator`, we can perform a feasibility analysis, determining exactly what resources are required to simulate
these complex non-adiabatic processes on quantum hardware.


The Vibronic Hamiltonian
------------------------
To assess feasibility, we must first define the system. We use the vibronic coupling Hamiltonian,
which describes a set of :math:`N` electronic states interacting with :math:`M` vibrational modes.

Unlike standard electronic structure problems, this model mixes discrete electronic levels with continuous
vibrational motion, and the Hamiltonian takes the form:

.. math::

    H = T + V,

where :math:`T` represents the kinetic energy of the nuclei, and is diagonal in the electronic subspace while :math:`V`
contains off-diagonal electronic couplings.

To simulate the time evolution :math:`U(t) = e^{-iHt}`, we employ a product formula (Trotterization) approach as outlined in
Motlagh et al. [#Motlagh2025]_. A key challenge in Trotterization is decomposing the potential energy matrix, :math:`V`, in the Hamiltonian efficiently.
Here, we use the fragmentation scheme proposed in the reference, grouping the terms :math:`\ket{j} \bra{i} \otimes V_{ji}` such
that they differ by a fixed bitstring :math:`m`. This grouping method results in :math:`N` different fragments, which can be viewed as
blocks of the potential energy matrix as shown in Figure 1. Each of these fragments can then be block-diagonalized by using only
Clifford gates and implemented as a sequence of evolutions controlled by the corresponding electronic states.

.. figure:: ../_static/demonstration_assets/vibronic_re/vibronic_fragments.png
    :align: center
    :width: 80%
    :target: javascript:void(0)

    Figure 1: (a) Fragmentation of the potential matrix :math:`V` into blocks :math:`H_m`. Note that :math:`H_0` captures the diagonal elements while :math:`H_{1,2,3}` represent specific off-diagonal couplings.
    (b) Example of off-diagonal fragments being block-diagonalized using Clifford gates to simplify quantum circuit implementation.

Defining the Hamiltonian
^^^^^^^^^^^^^^^^^^^^^^^^
With the algorithmic approach defined, the next step is to instantiate the system we wish to benchmark.

We don't need access to the Hamiltonian coefficients to derive a reliable baseline; we only need structural parameters. By defining the number of modes,
electronic states, and grid size, we can map out the cost topology of a real system without needing to generate and store the full Hamiltonian. Of course,
access to the full Hamiltonian coefficients would allow us to further optimize costs by leveraging the commutativity of electronic parts.

As a case study, we select (NO)$_4$-Anth, a molecule created by introducing four N-oxyl radical fragments to
anthracene. Proposed for its theoretically record-breaking singlet fission speed [#Pradhan2022]_, we use this molecule to define our simulation parameters:
"""

num_modes = 19  # Number of vibrational modes
num_states = 5  # Number of electronic states
k_grid = 4  # Number of qubits per mode (discretization)
taylor_degree = 2  # Truncate to Quadratic Vibronic Coupling

#################################################################################
# In our model, we truncate the interaction terms for potential energy fragments to linear and quadratic terms only.
# This approximation, known as the Quadratic Vibronic Coupling (QVC) model, captures the dominant physical effects while simplifying the potential
# energy circuits significantly.
#
# Constructing Circuits for A Single Time Step
# ------------------------------------------
# The next step is to define what the circuit will look like for evolving a single time step.
# Based on the fragmentation scheme defined above, a single Trotter step is composed of two distinct types of
# quantum circuits interleaved together:
#
# 1.  **Kinetic Energy Fragment:** This implements the nuclear kinetic energy. Since the kinetic operator depends
#     on momentum :math:`P`, this circuit uses the `Quantum Fourier Transform (QFT) <https://pennylane.ai/qml/demos/tutorial_qft>`_
#     to switch to the momentum basis, applies a phase rotation, and then switches back.
# 2.  **Potential Energy Fragments:** These implement the interaction terms. For each fragment, the algorithm
#     loads coefficients using a `QROM (Quantum Read-Only Memory) <https://pennylane.ai/qml/demos/tutorial_intro_qrom>`_,
#     computes the vibrational monomial product using quantum arithmetic, and applies a `phase gradient <https://pennylane.ai/compilation/phase-gradient/>`_.
#
#
# .. figure:: ../_static/demonstration_assets/vibronic_re/ke_circuit.png
#     :align: center
#     :width: 80%
#     :target: javascript:void(0)
#
#     Figure 2: Circuit for implementing the exponential of kinetic energy fragment.
#
# The implementation of the kinetic energy fragment is illustrated in Figure 2. Let's see how this visual blueprint of the circuit translates into PennyLane code:
#
import pennylane.estimator as qre

def kinetic_circuit(mode_wires, phase_wires, scratch_wires, coeff_wires):
    ops = []
    num_phase_wires = len(phase_wires)
    k_grid = len(mode_wires[0])
    for mode in range(num_modes):
        ops.append(qre.AQFT(order=1, num_wires=k_grid, wires=mode_wires[mode]))

    for i in range(num_modes):
        ops.append(
            qre.OutOfPlaceSquare(
                register_size=k_grid, wires=scratch_wires + mode_wires[i]
            )
        )

        for j in range(2 * k_grid):
            ctrl_wire = [scratch_wires[j]]
            target_wires = (
                coeff_wires[: len(phase_wires) - j]
                + phase_wires[: len(phase_wires) - j]
            )
            ops.append(
                qre.Controlled(
                    qre.SemiAdder(max_register_size=num_phase_wires - j),
                    num_ctrl_wires=1,
                    num_zero_ctrl=0,
                    wires=target_wires + ctrl_wire,
                )
            )

        ops.append(
            qre.Adjoint(
                qre.OutOfPlaceSquare(
                    register_size=k_grid, wires=scratch_wires + mode_wires[i]
                )
            )
        )

    for mode in range(num_modes):
        ops.append(
            qre.Adjoint(
                qre.AQFT(order=1, num_wires=k_grid, wires=mode_wires[mode])
            )
        )

    return qre.Prod(ops)


######################################################################
# Similarly, we can define the structure for the potential energy fragments. For a QVC model truncated to quadratic terms,
# each fragment will implement a monomial with a maximum degree of 2.
#
# .. math::
#
#     V_{ji} = \sum_{r} c_{r} Q_{r} + \sum_{r} \tilde{c}_{r} Q_{r}^2,
#
# where :math:`c_r` and :math:`\tilde{c}_{r}` are the linear and quadratic coupling coefficients respectively. The exponential
# of each term is implemented by the modular circuit shown in Figure 3.
#
# .. figure:: ../_static/demonstration_assets/vibronic_re/circuit_diagram.png
#     :align: center
#     :width: 80%
#     :target: javascript:void(0)
#
#     Figure 3: Circuit for implementing the exponential of a monomial term in the potential energy fragment.
#
# The circuit executes the following steps:
#
# 1. Load the coefficients from QROM controlled by the electronic state.
# 2. Compute the monomial product by using a square operation for quadratic terms.
# 3. Accumulate the phase by decomposing the multiplication into a series of controlled adders that adds the product directly onto the phase gradient register.
# 4. Uncompute the intermediate steps to clean up ancilla wires.
#
# We can define this circuit using two different segments, let's first define the linear term circuit:


def linear_circuit(num_states, elec_wires, phase_wires, coeff_wires, scratch_wires):
    ops = []
    ops.append(
        qre.QROM(
            num_bitstrings=num_states,
            size_bitstring=len(phase_wires),
            restored=False,
            wires=elec_wires + coeff_wires,
        )
    )

    for i in range(k_grid):
        ctrl_wire = [scratch_wires[i]]
        target_wires = (
            coeff_wires[: len(phase_wires) - i] + phase_wires[: len(phase_wires) - i]
        )
        ops.append(
            qre.Controlled(
                qre.SemiAdder(max_register_size=len(phase_wires) - i),
                num_ctrl_wires=1,
                num_zero_ctrl=0,
                wires=target_wires + ctrl_wire,
            )
        )

    ops.append(
        qre.Adjoint(
            qre.QROM(
                num_bitstrings=num_states,
                size_bitstring=len(phase_wires),
                restored=False,
                wires=elec_wires + coeff_wires,
            )
        )
    )
    return qre.Prod(ops)

#################################################################################
# Next, we can define the quadratic term circuit:

def quadratic_circuit(
    num_states, elec_wires, phase_wires, coeff_wires, mode_wires, scratch_wires
):
    ops = []
    k_grid = len(mode_wires)
    ops.append(
        qre.QROM(
            num_bitstrings=num_states,
            size_bitstring=len(phase_wires),
            restored=False,
            wires=elec_wires + coeff_wires,
        )
    )

    qre.OutOfPlaceSquare(register_size=k_grid, wires=mode_wires + scratch_wires)
    for i in range(2 * k_grid):
        ctrl_wire = [scratch_wires[i]]
        target_wires = (
            coeff_wires[: len(phase_wires) - i] + phase_wires[: len(phase_wires) - i]
        )
        ops.append(
            qre.Controlled(
                qre.SemiAdder(max_register_size=len(phase_wires) - i),
                num_ctrl_wires=1,
                num_zero_ctrl=0,
                wires=target_wires + ctrl_wire,
            )
        )

    ops.append(
        qre.Adjoint(
            qre.OutOfPlaceSquare(
                register_size=k_grid, wires=mode_wires + scratch_wires
            )
        )
    )
    ops.append(
        qre.Adjoint(
            qre.QROM(
                num_bitstrings=num_states,
                size_bitstring=len(phase_wires),
                restored=False,
                wires=elec_wires + coeff_wires,
            )
        )
    )
    return qre.Prod(ops)


#################################################################################
# Finally, to ensure precise resource estimation, we explicitly label our wire registers. This avoids ambiguity
# about which circuit operations are mapped to which quantum registers, so that the resource estimator captures the
# full width of the circuit.

def get_wire_labels(num_modes, num_states, k_grid, phase_prec):
    """Generates the wire map for the full system."""
    num_elec_qubits = (num_states - 1).bit_length()
    elec_wires = [
        f"e_{i}" for i in range(num_elec_qubits)
    ]  # Electronic State Register

    phase_wires = [
        f"pg_{i}" for i in range(phase_prec)
    ]  # Resource State For Phase Gradients
    coeff_wires = [f"c_{i}" for i in range(phase_prec)]  # Coefficient Register

    mode_wires = []
    for m in range(num_modes):
        mode_wires.append([f"m{m}_{w}" for w in range(k_grid)])  # Mode m Register

    scratch_wires = [
        f"s_{i}" for i in range(2 * k_grid)
    ]  # Scratch Space for Arithmetic

    return elec_wires, phase_wires, coeff_wires, mode_wires, scratch_wires


#################################################################################
# We now define the full circuit by assembling our fragments into
# a second-order `Suzuki-Trotter expansion <https://docs.pennylane.ai/en/stable/code/api/pennylane.estimator.templates.TrotterProduct.html>`_.
#
# To construct the expansion, we loop over all fragments: the kinetic energy fragment is added once, while the
# number of potential energy fragments is determined by the size of the electronic register.
# This ensures that every block of the Hamiltonian identified in our fragmentation
# scheme (Figure 1) is accounted for in the simulation.
#
# Additionally, it is necessary to prepare the phase gradient register. The size of this register
# is dictated by the desired precision for phase rotations in both the
# potential and kinetic energy steps, directly influencing the simulation's accuracy.
#

import math
def circuit(num_modes, num_states, k_grid, taylor_degree, num_steps, phase_grad_prec=1e-6):
    fragments = []
    phase_grad_wires = int(math.ceil(math.log2(1 / phase_grad_prec)))
    elec_wires, phase_wires, coeff_wires, mode_wires, scratch_wires = (
        get_wire_labels(num_modes, num_states, k_grid, phase_grad_wires)
    )
    qre.PhaseGradient(num_wires=len(phase_wires), wires=phase_wires) # Prepare Phase Gradient State
    kinetic_fragment = kinetic_circuit(
        mode_wires, phase_wires, scratch_wires, coeff_wires
    )
    fragments.append(kinetic_fragment) # Add Kinetic Energy Fragment

    num_fragments = 2**len(elec_wires)
    for i in range(num_fragments): # Loop over all potential energy fragments
        frag_op = []
        for mode in range(num_modes):
            if taylor_degree >= 1:
                frag_op.append(
                    linear_circuit(
                        num_states, elec_wires, phase_wires, coeff_wires, scratch_wires
                    )
                )
            if taylor_degree >= 2:
                frag_op.append(
                    quadratic_circuit(
                        num_states,
                        elec_wires,
                        phase_wires,
                        coeff_wires,
                        mode_wires[mode],
                        scratch_wires,
                    )
                )
        fragments.append(qre.Prod(frag_op))

    qre.TrotterProduct(first_order_expansion=fragments, num_steps=num_steps, order=2)


#################################################################################
# Finally, we can estimate the resource requirements for this full circuit using the `estimate <https://docs.pennylane.ai/en/stable/code/api/pennylane.estimator.estimate.estimate.html>`_
# function. To ensure the simulation reaches a total time of 100 fs with sufficient accuracy, we set the number of
# Trotter steps to 500. This value is chosen based on benchmark data from Motlagh et al.,
# where this step count was shown to maintain reliable accuracy for a 100 fs simulation of the (NO)$_4$-Anth system.

print(qre.estimate(circuit)(num_modes, num_states, k_grid, taylor_degree, num_steps=500))

#################################################################################
# We observe that while our results follow the same scaling and orders of magnitude reported in Motlagh et al. [#Motlagh2025]_,
# the Toffoli gate count is slightly higher.
# The higher gate counts in this estimation occur because we assume a dense Hamiltonian, whereas the reference work
# leverages system-specific sparsity by only implementing non-zero coupling terms. These numbers can therefore be viewed as
# a reliable upper bound for the cost of simulating (NO)$_4$-Anth dynamics.
# Notably, even this upper bound demonstrates that the resource requirements for such a complex vibronic system are remarkably
# low, suggesting that these non-adiabatic simulations are highly viable for early-generation fault-tolerant quantum computers.
#
#
# Conclusions
# -----------
# By constructing the full simulation workflow for vibronic dynamics from the ground up, we have gained critical insights into
# the resource requirements for simulating these complex processes on future quantum hardware.
# This approach allows us to break down complex algorithms into manageable building blocks, providing a transparent view of how
# algorithmic choices directly impact gate counts and qubit overhead.
# This demonstration highlights the power of modular quantum algorithm design, enabling researchers to build, analyze, and optimize
# complex workflows for next-generation quantum simulations.
# Ultimately, this framework is designed to be an open sandbox for researchers. We encourage you to use the building blocks
# provided here to explore alternative algorithms tailored to your specific research interests.
#
# References
# ----------
#
# .. [#Motlagh2025]
#
#      D. Motlagh et al., "Quantum Algorithm for Vibronic Dynamics: Case Study on Singlet Fission Solar Cell Design".
#      `arXiv preprint arXiv:2411.13669 (2025) <https://arxiv.org/abs/2411.13669>`__.
#
# .. [#Pradhan2022]
#
#     E. Pradhan et al., "Design of the Smallest Intramolecular Singlet Fission Chromophore with the Fastest Singlet Fission".
#     `Journal of Physical Chemistry Letters 13.48 (2022) <https://pubs.acs.org/doi/10.1021/acs.jpclett.2c03131>`__.
#