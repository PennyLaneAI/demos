r"""How to estimate the resource requirements of quantum algorithms
===============================================================

Fault-tolerant quantum computers are on their way. However, how do we ensure that useful algorithms can
actually run on them? An algorithm is hardly helpful when it cannot be executed.

This is a major challenge in quantum algorithm development, especially since we are often working at
scales where simulation is no longer feasible. We therefore need to analyze our algorithms
through **resource estimation**: getting an idea of how many resources an algorithm requires, such
as logical qubits and gates. In turn, this gives us an indication of how long the algorithm will take
to execute on a given quantum hardware architecture, or if it will even fit in memory to begin with.

PennyLane is here to make that process easy, with our new resource estimation module.
The :mod:`estimator <pennylane.estimator>` module leverages the latest resource estimates,
decompositions, and compilation techniques from the literature, and is designed to do so as
quickly as possible.

.. figure:: ../_static/demo_thumbnails/opengraph_demo_thumbnails/pennylane-demo-how-to-resource-estimation-open-graph.png
    :align: center
    :width: 70%
    :target: javascript:void(0)

In this demo, we will estimate the quantum resources necessary for a simple Hamiltonian simulation workflow:
evolving the quantum state of a honeycomb lattice of spins under the
:func:`Kitaev Hamiltonian <pennylane.spin.kitaev>`.
"""

######################################################################
# Estimating the Resources of existing PennyLane workflows
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Let's say you've already written your workflow as a PennyLane circuit.
# To estimate the resources of PennyLane workflows, you can simply invoke
# :func:`qre.estimate <pennylane.estimator.estimate.estimate>`
# directly on the QNode.
#
# We demonstrate this with a :math:`25 \times 25` honeycomb lattice of spins.  
# Here, we generate the Hamiltonian ourselves, using the
# :func:`qp.spin.kitaev <pennylane.spin.kitaev>` function,
# and group the Hamiltonian terms into qubit-wise
# commuting groups:
#

import pennylane as qp
import numpy as np
import time

n_cells = [25, 25]
kx, ky, kz = (0.5, 0.6, 0.7)

t1 = time.time()
flat_hamiltonian = qp.spin.kitaev(n_cells, coupling=np.array([kx, ky, kz]))
flat_hamiltonian.compute_grouping()  # compute the qubit-wise commuting groups!

groups = []
for group_indices in flat_hamiltonian.grouping_indices:
    grouped_term = qp.sum(*(flat_hamiltonian.operands[index] for index in group_indices))
    groups.append(grouped_term)

grouped_hamiltonian = qp.sum(*groups)
t2 = time.time()
t_generation = t2 - t1

######################################################################
# Here we define our circuit for Hamiltonian simulation.
#

num_steps = 10
order = 6

@qp.qnode(qp.device("default.qubit"))
def executable_circuit(hamiltonian, num_steps, order):
    for wire in hamiltonian.wires: # uniform superposition over all basis states
        qp.Hadamard(wire)
    qp.TrotterProduct(hamiltonian, time=1.0, n=num_steps, order=order)
    return qp.state()

######################################################################
# Now, let’s import our quantum resource :mod:`estimator <pennylane.estimator>`.

import pennylane.estimator as qre

######################################################################
# Just call :func:`qre.estimate <pennylane.estimator.estimate.estimate>`
# to generate the resource estimates:
#

t1 = time.time()
resources_exec = qre.estimate(executable_circuit)(grouped_hamiltonian, num_steps, order)
t2 = time.time()

print(f"Processing time: {(t2 - t1):.3g} seconds")
print(resources_exec)

######################################################################
# Fast estimation with less information
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

######################################################################
# What if we wanted to estimate the quantum resources necessary to evolve the quantum state of a
# :math:`100 \times 100` honeycomb lattice of spins under the Kitaev Hamiltonian?
#
# **That's 20,000 spins!**
#
# Generating such Hamiltonians becomes a bottleneck quickly as the system size increases.
# However, :mod:`estimator <pennylane.estimator>`
# doesn't require detailed descriptions of Hamiltonians
# for estimation; instead, we can define 
# `resource Hamiltonians <https://docs.pennylane.ai/en/stable/code/qml_estimator.html#resource-hamiltonians>`__
# which capture the resources required for Hamiltonian simulation
# without the need to compute costly Hamiltonians.
#
# In the particular case of the Kitaev Hamiltonian on a honeycomb
# lattice, we can directly compute some important quantities about the
# Hamiltonian. Based on the number of lattice cells :math:`n` we can
# determine the number of qubits :math:`n_{q}` the Hamiltonian acts on,
# the number of :math:`XX`-type interactions :math:`n_{XX}`,
# the number of :math:`YY`-type interactions :math:`n_{YY}`,
# and the number of :math:`ZZ`-type interactions :math:`n_{ZZ}`.
#
# .. math::
#   n_{q} &= 2 n^{2}, \\
#   n_{XX} &= n^{2}, \\
#   n_{YY} = n_{ZZ} &= n (n - 1), \\

n_cell = 100

def pauli_quantities(n_cell):
    n_q = 2 * n_cell**2
    n_xx = n_cell**2
    n_yy = n_cell * (n_cell - 1)
    n_zz = n_yy
    return n_q, n_xx, n_yy, n_zz

n_q, n_xx, n_yy, n_zz = pauli_quantities(n_cell)

######################################################################
# We can capture this information in a compact representation
# using the
# :class:`qre.PauliHamiltonian <pennylane.estimator.compact_hamiltonian.PauliHamiltonian>`:

pauli_word_distribution = {"XX": n_xx, "YY": n_yy, "ZZ": n_zz}

kitaev_H = qre.PauliHamiltonian(
    num_qubits=n_q,
    pauli_terms=pauli_word_distribution,
)

######################################################################
# Similarly, we can then use existing resource
# `operators <https://docs.pennylane.ai/en/stable/code/qml_estimator.html#id1>`__ and
# `templates <https://docs.pennylane.ai/en/stable/code/qml_estimator.html#resource-templates>`__
# from the :mod:`estimator <pennylane.estimator>` module to express our circuit.
# These
# :class:`ResourceOperator <pennylane.estimator.resource_operator.ResourceOperator>`
# classes, like the ``PauliHamiltonian`` above, are designed to require minimal information
# --- avoiding costly compute ---
# while still providing trustworthy estimates.
#

def circuit(hamiltonian, num_steps, order):
    qre.UniformStatePrep(num_states=2**n_q)  # uniform superposition over all basis states
    qre.TrotterPauli(hamiltonian, num_steps, order)

######################################################################
# This circuit is purely for resource estimation and it **cannot be executed for simulation**.
# That's what makes resource estimation so fast ⚡.
#
# The cost of an algorithm is typically quantified by the number of logical qubits required and the
# number of gates used.
# While different hardware may natively support different gatesets,
# the default gateset used by :mod:`estimator <pennylane.estimator>` is:
# ``{'Hadamard', 'S', 'CNOT', 'T', 'Toffoli', 'X', 'Y', 'Z'}``.
#
# We now have a representation of our workflow using resource
# operators and a resource Hamiltonian. As before, we simply call
# :func:`qre.estimate <pennylane.estimator.estimate.estimate>`
# to estimate the resources:
#

t1 = time.time()
res = qre.estimate(circuit)(kitaev_H, num_steps, order)
t2 = time.time()

print(f"Processing time: {t2 - t1:.3g} seconds\n")
print(res)

######################################################################
# Our resource estimate was generated in the blink of an eye 👁️.
#
# We can also analyze the resources of an individual
# :class:`ResourceOperator <pennylane.estimator.resource_operator.ResourceOperator>`.
# This can be helpful in determining which operators in a workflow demand the most resources.
#
# For example, let's consider the resource estimates of
# ``qre.TrotterPauli``, and see how it changes as we provide additional
# information:

resources_without_grouping = qre.estimate(qre.TrotterPauli(kitaev_H, num_steps, order))

######################################################################
# Providing additional information can help to produce more accurate resource estimates.
# In the case of our
# :class:`qre.PauliHamiltonian <pennylane.estimator.compact_hamiltonian.PauliHamiltonian>`,
# we can split the terms into groups of commuting terms:

commuting_groups = [{"XX": n_xx}, {"YY": n_yy}, {"ZZ": n_zz}]

kitaev_H_with_grouping = qre.PauliHamiltonian(
    num_qubits=n_q,
    pauli_terms=commuting_groups,
)

resources_with_grouping = qre.estimate(
    qre.TrotterPauli(kitaev_H_with_grouping, num_steps, order)
)

######################################################################
# Let’s see how the cost of ``qre.TrotterPauli`` differs in these two cases:

# Just compare T gates:
t_count_1 = resources_without_grouping.gate_counts["T"]
t_count_2 = resources_with_grouping.gate_counts["T"]
reduction = abs((t_count_2 - t_count_1) / t_count_1)
print("--- Without grouping ---", f"\n T gate count: {t_count_1:.3E}\n")
print("--- With grouping ---", f"\n T gate count: {t_count_2:.3E}\n")
print(f"Difference: {100*reduction:.1f}% reduction")

######################################################################
# By splitting our terms into groups, we’ve managed to reduce the ``T`` gate count of our
# Trotterization by over 30 percent!
#

######################################################################
# Changing gatesets and precision
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

######################################################################
# Here are the resources for our entire circuit using the updated Hamiltonian:

res = qre.estimate(circuit)(kitaev_H_with_grouping, num_steps, order)
print(f"{res}")

######################################################################
# We can also configure the gateset to obtain resource estimates at various levels of abstraction.
# Here, we configure a high-level gateset which adds gate types such as rotations, and a low
# level-gateset limited to just ``Hadamard``, ``CNOT``, ``S``, and ``T`` gates.
#
# We can see how the resources manifest at these different levels.
#

highlvl_gateset = {
    "RX","RY","RZ",
    "Toffoli",
    "X","Y","Z",
    "Adjoint(S)","Adjoint(T)",
    "Hadamard","S","CNOT","T",
}

highlvl_res = qre.estimate(
    circuit,
    gate_set=highlvl_gateset,
)(kitaev_H_with_grouping, num_steps, order)

print(f"High-level resources:\n{highlvl_res}\n")

######################################################################
#

lowlvl_gateset = {"Hadamard", "S", "CNOT", "T"}

lowlvl_res = qre.estimate(
    circuit,
    gate_set=lowlvl_gateset,
)(kitaev_H_with_grouping, num_steps, order)

print(f"Low-level resources:\n{lowlvl_res}")

######################################################################
# When decomposing our algorithms to a particular gateset, it is often the case that we only have some
# approximate decomposition of a building-block into the target gateset. For example, approximate
# state loading to some precision, or rotation synthesis within some precision of the rotation angle.
#
# These approximate decompositions are accurate within some error threshold; tuning this error
# threshold impacts the required resources. We can set and tune these errors using a
# resource configuration: :class:`ResourceConfig <pennylane.estimator.resource_config.ResourceConfig>`.
#
# Notice that a more precise estimate requires more ``T`` gates!
#

custom_rc = qre.ResourceConfig()  # generate a resource configuration

rz_precisions = custom_rc.resource_op_precisions[qre.RZ]
print(f"Default setting: {rz_precisions}\n")

custom_rc.set_precision(qre.RZ, 1e-15) # customize precision

res = qre.estimate(
    circuit,
    gate_set=lowlvl_gateset,
    config=custom_rc, # provide our custom configuration
)(kitaev_H_with_grouping, num_steps, order)

# Just compare T gates:
print("--- Lower precision (1e-9) ---", f"\n T counts: {lowlvl_res.gate_counts['T']:.3E}")
print("\n--- Higher precision (1e-15) ---", f"\n T counts: {res.gate_counts['T']:.3E}")

######################################################################
# The :mod:`estimator <pennylane.estimator>` module also provides functionality for
# writing custom decompositions and custom resource operators.
# To find out how, check out our documentation for
# :class:`ResourceConfig <pennylane.estimator.resource_config.ResourceConfig>`
# and :class:`ResourceOperator <pennylane.estimator.resource_operator.ResourceOperator>`!

######################################################################
# Tailored resource estimates for your needs
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We can combine all of the features we have seen so far to determine
# the cost of Trotterized time evolution of the Kitaev Hamiltonian
# in our preferred setting:
#

t1 = time.time()

kitaev_hamiltonian = kitaev_H_with_grouping  # use compact Hamiltonian with grouping

custom_gateset = lowlvl_gateset # use the low-level gateset

custom_config = qre.ResourceConfig()
custom_config.set_precision(qre.RZ, precision=1e-12) # set higher precision

resources = qre.estimate(
    circuit,
    gate_set = custom_gateset,
    config = custom_config
)(kitaev_hamiltonian, num_steps, order)

t2 = time.time()
print(f"Processing time: {t2 - t1:.3g} seconds\n")
print(resources)

######################################################################
# Comparing estimates: Full vs. Resource workflows
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We've shown that you can estimate your workflow's resources
# using both typical PennyLane circuits, and circuits written with
# :class:`ResourceOperator <pennylane.estimator.resource_operator.ResourceOperator>`
# classes.
#
# Now, we'll demonstrate that the resource estimates are consistent across both of these cases.
#
# Let's return to a :math:`25 \times 25` unit honeycomb lattice of spins.  
# We'll use :mod:`estimator <pennylane.estimator>` to make sure everything matches.
#

t1 = time.time()
n_cell = 25
n_q, n_xx, n_yy, n_zz = pauli_quantities(n_cell)

commuting_groups = [{"XX": n_xx}, {"YY": n_yy}, {"ZZ": n_zz}]

compact_hamiltonian = qre.PauliHamiltonian(
    num_qubits = n_q,
    pauli_terms = commuting_groups,
)
t2 = time.time()
t_estimation = t2 - t1

######################################################################
# The resulting data can be easily compared for a sanity check.
#

print(f"Processing time for Hamiltonian generation: {(t_generation):.3g} seconds")
print("Total number of terms:", len(flat_hamiltonian.operands))
print("Total number of qubits:", len(flat_hamiltonian.wires), "\n")

print(f"Processing time for Hamiltonian estimation: {(t_estimation):.3g} seconds")
print("Total number of terms:", compact_hamiltonian.num_terms)
print("Total number of qubits:", compact_hamiltonian.num_qubits)

######################################################################
# Notice how much faster it was to prepare
# the resource Hamiltonian for estimation!
#
# Here's the resource estimate from our earlier *execution* circuit.
#

print(resources_exec)

######################################################################
# Let's validate the results by comparing with our resource *estimation* circuit.
#

t1 = time.time()
resources_est = qre.estimate(circuit)(compact_hamiltonian, num_steps, order)
t2 = time.time()

print(f"Processing time: {(t2 - t1):.3g} seconds")
print(resources_est)

######################################################################
# The numbers check out!
#

######################################################################
# Your turn!
# ~~~~~~~~~~
#
# Now that you’ve seen how powerful PennyLane’s
# quantum resource :mod:`estimator <pennylane.estimator>` is,
# go try it out yourself!
# See how convenient it is to estimate the resources of powerful algorithms like
# `DQI <https://pennylane.ai/qml/demos/tutorial_dqi>`__,
# `XAS <https://pennylane.ai/qml/demos/tutorial_xas>`__,
# or `Shor's algorithm <https://pennylane.ai/qml/demos/tutorial_shors_algorithm_catalyst>`__.
# :mod:`estimator <pennylane.estimator>` includes a host of resource operators, templates, and Hamiltonians
# which serve to make resource estimation a breeze for a wide variety of algorithms.
# If you're also interested in exact resource tracking of programs compiled for *execution*,
# take a look at :func:`pennylane.specs`, which provides
# resource information throughout compilation for programs compiled using :func:`pennylane.qjit`.
#
#
