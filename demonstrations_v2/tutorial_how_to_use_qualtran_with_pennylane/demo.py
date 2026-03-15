r"""How to use Qualtran with PennyLane
======================================

Get ready to expand your quantum programming toolkit! PennyLane and
`Qualtran <https://qualtran.readthedocs.io/en/latest/>`_ integrate their best features, enabling you
to build quantum circuits with a mix of
`Qualtran Bloqs <https://qualtran.readthedocs.io/en/latest/bloqs/index.html#bloqs-library>`_
and PennyLane operations. You can then analyze or simulate the circuits with tools
from both libraries. For those unfamiliar, Qualtran provides a set of abstractions for
representing quantum programs, a library of quantum algorithms called Bloqs, and a set of tools for
visualizing and analyzing these programs.

.. figure:: ../_static/demo_thumbnails/opengraph_demo_thumbnails/pennylane-demo-qualtran-integration-open-graph.png
    :align: center
    :width: 70%
    :target: javascript:void(0)

In this demo, we show you how easy it is to use PennyLane and Qualtran together. This integration
allows you to:

* Simulate Qualtran Bloqs: Convert Qualtran programs to PennyLane operations and simulate to confirm
    their outputs or try new algorithms in PennyLane circuits with the many components in the
    `Qualtran Bloqs library <https://qualtran.readthedocs.io/en/latest/bloqs/index.html#bloqs-library>`_.
* Analyze PennyLane circuits with Qualtran: Leverage Qualtran's advanced analysis tools like
    `drawing call graphs <https://qualtran.readthedocs.io/en/latest/reference/qualtran/drawing/show_call_graph.html>`_,
    `counting qubits <https://qualtran.readthedocs.io/en/latest/resource_counting/qubit_counts.html>`_,
    and `counting gates <https://qualtran.readthedocs.io/en/latest/reference/qualtran/drawing/show_counts_sigma.html>`_.

We'll start by focusing on the first key capability: simulating Qualtran ``Bloqs`` using
PennyLane ``QNodes``.
"""

######################################################################
# Simulating Qualtran Bloqs
# -------------------------
#
# With barely any work, you can drop Qualtran Bloqs into executable PennyLane circuits!
# You only need one class: :class:`~pennylane.FromBloq` which wraps a Qualtran ``Bloq`` as a
# PennyLane operation. It faithfully converts any Bloq into an
# :class:`~pennylane.operation.Operation` containing the usual methods and attributes such as
# :func:`~pennylane.operation.Operation.decomposition` and
# :func:`~pennylane.operation.Operation.matrix`.
#
# Let's see how it works! In the following example, we wrap a Qualtran ``XGate`` instance using
# :class:`~pennylane.FromBloq`. Qualtran Bloqs don't always apply to specific wires and
# may not have wires defined. But PennyLane operations do apply to specific wires, so we need to
# provide qubit information to :class:`~pennylane.FromBloq` via the ``wires`` argument.

import pennylane as qml
qml.capture.enable()
from qualtran.bloqs.basic_gates import XGate

bloq_as_op = qml.FromBloq(XGate(), wires=0)
print(bloq_as_op)

######################################################################
# We can see that the output is a :class:`~.pennylane.io.FromBloq` instance, whose properties are
# the same as PennyLane's :class:`~pennylane.PauliX` operator.
#
# When wire requirements are complicated, you can use the :func:`~.pennylane.bloq_registers` 
# helper function to generate the input values for ``wires``:

print(qml.bloq_registers(XGate()))

######################################################################
# This will create register names in accordance to the Bloq's signature. Here, the function created
# one ``'q'`` register with a single qubit as required by ``XGate``. You will see an additional
# example in the next section.
#
# Expanding PennyLane circuits with Qualtran Bloqs
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Let's look at a more complicated example! Qualtran has a special type of addition operation, known
# as Galois Field Addition or `GF2Addition <https://qualtran.readthedocs.io/en/latest/bloqs/gf_arithmetic/gf2_addition.html#gf2addition>`_,
# that is not implemented in PennyLane. You can think of it
# as a binary addition that doesn't allow carrying. Let's use the combined force of PennyLane and
# Qualtran to bring ``GF2Addition`` to life!
# We will use it below to calculate the result of :math:`5 + 10` in binary.

from qualtran.bloqs.gf_arithmetic import GF2Addition

arithmetic_bloq = GF2Addition(4)
wires = qml.bloq_registers(arithmetic_bloq) # This gives us 2 registers, 'x' and 'y'
print(wires)

######################################################################
five = [0, 1, 0, 1] # 5 in binary
ten = [1, 0, 1, 0] # 10 in binary

@qml.set_shots(1)
@qml.qnode(qml.device('default.qubit'))
def circuit():
    # Prepare the input registers for 5 and 10
    qml.BasisState(five + ten, wires=wires['x']+wires['y'])
    # Sum the two registers
    qml.FromBloq(arithmetic_bloq, wires=wires['x']+wires['y'])
    # Measure the output binary string
    a = [qml.measure(i) for i in range(len(wires['x']+wires['y']))]
    return qml.sample(a)

# Simulate the circuit and process binary output to integer
binary_string = "".join([str(bit) for bit in circuit()[0]])
print("GF2Addition of 5 + 10 =", int(binary_string[len(wires['x']):],2))

######################################################################
# Wow! We used a Qualtran Bloq as a PennyLane template without any additional
# work. With Qualtran's expansive library of quantum algorithms, you can now build a greater
# variety of circuits using a combination of PennyLane templates and Qualtran Bloqs.
#
# Analyzing PennyLane Circuits with Qualtran
# ------------------------------------------
#
# Now, we'll show you how to convert PennyLane objects to Qualtran Bloqs using the
# :func:`~pennylane.to_bloq` function
# and how to analyze them with Qualtran tools
# like `drawing call graphs <https://qualtran.readthedocs.io/en/latest/reference/qualtran/drawing/show_call_graph.html>`_,
# `counting qubits <https://qualtran.readthedocs.io/en/latest/resource_counting/qubit_counts.html>`_,
# and `counting gates <https://qualtran.readthedocs.io/en/latest/reference/qualtran/drawing/show_counts_sigma.html>`_.
# For brevity, we'll 
# mainly cover how operators get converted to Bloqs in our examples, but functions of operators and
# ``QNodes`` work just the same.
#
# There are three main options for the conversion. We'll briefly introduce them here but go into
# greater detail in subsequent sections:
#
# * Smart defaults: In this option, PennyLane chooses what Qualtran Bloq to translate to. If an option exists,
#     we'll give you a Qualtran Bloq that is highly similar to the PennyLane operator.
#
# * Custom mapping: Want something different from the smart default? Don't worry, you can customize
#     what Bloq you want your operator to map to. This makes it easy to refine the finer details of
#     your algorithm.
#
# * Wrapping: Think of this as an analogue of :class:`~pennylane.FromBloq`. It faithfully converts any operator, ``QNode``, or
#     Qfunc into a Bloq. The output is a :class:`~pennylane.io.ToBloq` instance.
#
# These options are all accessible through the :func:`~pennylane.to_bloq` function. In the
# following sections, we'll explore how we can wield this powerful function to get all the
# functionality introduced above.
#
# Smart defaults
# ~~~~~~~~~~~~~~
#
# By default, ``qml.to_bloq`` tries its best to translate PennyLane objects to Qualtran-native
# objects. This makes certain Qualtran functionalities, such as gate counting and
# `generalizers <https://qualtran.readthedocs.io/en/latest/reference/qualtran/resource_counting/generalizers.html>`_,
# work more seamlessly. In the following example, PennyLane's :class:`~pennylane.PauliX` operator is
# mapped directly to Qualtran's ``XGate``.

op_as_bloq = qml.to_bloq(qml.X(0))
print(op_as_bloq)

######################################################################
# Note that not all PennyLane operators are as straightforward to map as the
# :class:`~pennylane.PauliX` operator. For example, PennyLane's
# :class:`~pennylane.QuantumPhaseEstimation` could be mapped to a variety of Qualtran quantum phase
# estimation (QPE) Bloqs, such as
# `TextbookQPE <https://qualtran.readthedocs.io/en/latest/bloqs/phase_estimation/text_book_qpe.html>`_ or
# `QubitizationQPE <https://qualtran.readthedocs.io/en/latest/bloqs/phase_estimation/qubitization_qpe.html>`_. 
# In cases where the mapping is ambiguous, we get the smart default:

unitary = qml.RY(phi=0.3, wires=[0])
estimation_wires = [1, 2, 3]
op = qml.QuantumPhaseEstimation(unitary=unitary, estimation_wires=estimation_wires)
qpe_bloq = qml.to_bloq(op)
print(qpe_bloq)

######################################################################
# We can use Qualtran's
# `show_call_graph <https://qualtran.readthedocs.io/en/latest/reference/qualtran/drawing/show_call_graph.html>`_
# to investigate how :class:`~pennylane.QuantumPhaseEstimation` is mapped. This tool lets you visualize the
# full stack of a quantum circuit and analyze what causes specific algorithms and gates to be called
# and how often.
#
# .. code-block:: python
#     
#     from qualtran.drawing import show_call_graph
#     show_call_graph(qpe_bloq, max_depth=1)
#
# .. figure:: ../_static/demonstration_assets/how_to_use_qualtran_with_pennylane/smart_default.svg
#     :align: center
#     :width: 50%

######################################################################
# Here, the smart default is to call a Qualtran
# `TextbookQPE <https://qualtran.readthedocs.io/en/latest/bloqs/phase_estimation/text_book_qpe.html>`_
# that uses
# `RectangularWindowState <https://qualtran.readthedocs.io/en/latest/bloqs/phase_estimation/text_book_qpe.html#rectangularwindowstate>`_ 
# to prepare a state. But what if we wanted to use a different Bloq for our state preparation? In
# this case, we turn to custom mappings.
#  
# Custom mapping
# ~~~~~~~~~~~~~~
# To use another state preparation routine in our QPE workflow, e.g., `LPResourceState <https://qualtran.readthedocs.io/en/latest/bloqs/phase_estimation/lp_resource_state.html#lpresourcestate>`_
# rather than `RectangularWindowState <https://qualtran.readthedocs.io/en/latest/bloqs/phase_estimation/text_book_qpe.html#rectangularwindowstate>`_,
# we can override the smart default with a custom map. As shown below, custom maps are defined with
# a dictionary where keys are the original operations and values are the new operations we want to
# map those to.

from qualtran.bloqs.phase_estimation import LPResourceState
from qualtran.bloqs.phase_estimation.text_book_qpe import TextbookQPE

custom_map = {
    op: TextbookQPE(
        unitary=qml.to_bloq(qml.RY(phi=0.3, wires=[0])), 
        ctrl_state_prep=LPResourceState(3)
    )
}

qpe_bloq = qml.to_bloq(op, custom_mapping=custom_map)
print(qpe_bloq)

######################################################################
# Below, we see that ``LPResourceState`` has replaced ``RectangularWindowState`` as defined in the
# custom map:
#
# .. code-block:: python
#
#     show_call_graph(qpe_bloq, max_depth=1)
#
# .. figure:: ../_static/demonstration_assets/how_to_use_qualtran_with_pennylane/lpresource.svg
#     :align: center
#     :width: 50%

######################################################################
#
# Wrapping
# ~~~~~~~~
# When a PennyLane object does not have a mapping -- a direct Qualtran equivalent
# or smart default -- the circuit is wrapped as a :class:`~pennylane.io.ToBloq` object.

def circ():
    qml.X(0)
    qml.X(1)

qfunc_as_bloq = qml.to_bloq(circ)
print(type(qfunc_as_bloq))

######################################################################
# Wrapping a PennyLane quantum function or operator as a ``ToBloq`` is similar to wrapping a
# Qualtran Bloq as a ``FromBloq``. A wrapped PennyLane object acts like a Bloq: it can be analyzed
# using the language of Qualtran to simulate algorithms, estimate resource requirements, draw
# diagrams, and more. 
#
# We can choose to wrap our PennyLane :class:`~pennylane.QuantumPhaseEstimation` simply by setting
# ``map_ops`` to ``False``. This wraps the operators as a ``ToBloq`` object and keeps the original
# PennyLane object information.

unitary = qml.RY(phi=0.3, wires=[0])
estimation_wires = [1, 2, 3]
op = qml.QuantumPhaseEstimation(unitary=unitary, estimation_wires=estimation_wires)
wrapped_qpe_bloq = qml.to_bloq(op, map_ops=False)
print(wrapped_qpe_bloq)

######################################################################
# The call graph below shows how the wrapped version of :class:`~pennylane.QuantumPhaseEstimation`
# preserves the PennyLane definition of the operator:
#
# .. code-block:: python
#
#     show_call_graph(wrapped_qpe_bloq, max_depth=1)
#
# .. figure:: ../_static/demonstration_assets/how_to_use_qualtran_with_pennylane/wrapped_qpe_bloq.svg
#     :align: center
#     :width: 50%

######################################################################
#
# Let's see how mapping and wrapping affect our resource count estimates.

_, mapped_sigma = qpe_bloq.call_graph()
_, wrapped_sigma = wrapped_qpe_bloq.call_graph()
print("--- Mapped counts ---")
for gate, count in mapped_sigma.items():
    print(f"{gate}: {count}") 
print("\n--- Wrapped counts ---")
for gate, count in wrapped_sigma.items():
    print(f"{gate}: {count}")

######################################################################
# Here, we can clearly see that the resource counts for the two methods are distinctly different.
# This is because the underlying implementations for the two QPE operators differ.
#
# When Qualtran computes the resource counts for a ``ToBloq``, it first checks if there is a call
# graph defined. If it is defined, Qualtran uses that call graph to compute the resource count
# estimates. If it is not defined, Qualtran uses the PennyLane decomposition or the PennyLane 
# :mod:`~.pennylane.estimator` module to compute the resource count estimates.
#
# Since computing the PennyLane decompositions is expensive, many PennyLane templates such as QPE
# have call graphs defined when wrapped as a ``ToBloq`` object. By defining these call graphs,
# you can now efficiently compute resource count estimates for circuits that may require thousands
# of qubits and trillions of gates.
#
# There is one caveat to note: for performance reasons, the call graphs sometimes differ from the
# actual decomposition. This means you may get more optimal, but still accurate, counts than what
# the original PennyLane decomposition might have prescribed. As we continue to develop both the 
# PennyLane-Qualtran integration and PennyLane itself, these call graphs will evolve as well.

######################################################################
# Conclusion
# ----------
# In this demo, we took a look at how to use PennyLane with Qualtran. We encourage you to play
# around with the integration and try swapping out PennyLane operators with Qualtran Bloqs and
# running it on a real quantum device! Or convert an existing PennyLane workflow into Qualtran and
# use Qualtran tools to analyze the circuit. If you'd like to see more examples of Qualtran and
# PennyLane or would like to know further technical details, be sure to check out the docs for
# :func:`~.pennylane.to_bloq` and :class:`~.pennylane.FromBloq`!
#
