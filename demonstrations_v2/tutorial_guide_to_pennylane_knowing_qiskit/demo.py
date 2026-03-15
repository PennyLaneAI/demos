r"""Your guide to PennyLane if you know Qiskit
===================================================

Greetings, fellow quantum programmers 👋! Isn’t it such a wonderful world we live in, with so many
different quantum software development kits (SDKs) at our fingertips? I remember the days of (messy)
personalized code bases that took hours and hours to develop just *to be able* to start researching
👴. Nowadays, those tools are there for us to access for free, being developed and maintained by
`savvy open-source software developers <https://docs.pennylane.ai/en/stable/development/guide/contributing.html>`__ around the clock. Again, what a wonderful world!

When it comes to quantum programming SDKs, `PennyLane <https://pennylane.ai/install/>`__ and `Qiskit <https://github.com/Qiskit/qiskit>`__ (``v1.0`` and ``<=v0.46``) are two
of the most widely-used by the community. PennyLane has a few staples that make it so:

- **Hardware-agnostic**: PennyLane has no opinions on what hardware or simulator backends you want
  to use for your research. You can program an emporium of real hardware and simulator backends all
  from the easy-to-use PennyLane API. This includes IBM’s hardware with the `PennyLane-Qiskit
  plugin <https://docs.pennylane.ai/projects/qiskit/en/stable/>`__.
   
- **Everything is differentiable**: A quantum circuit in PennyLane is designed to behave like a
  differentiable function, unlocking quantum differentiable programming and allowing to integrate
  seamlessly with your favourite machine learning frameworks.
   
- **Community-focused**: Let’s face it, you’re going to get stuck at some point when you’re
  researching or learning. That’s why we have a mandate to make our `documentation <https://docs.pennylane.ai/en/stable/index.html>`__ easy to navigate,
  dedicated teams for creating :doc:`new demonstrations </demonstrations>` when we release new features, and an active
  `discussion forum <https://discuss.pennylane.ai/>`__ to answer your questions.

.. figure:: ../_static/demo_thumbnails/opengraph_demo_thumbnails/OGthumbnail_guide_to_pennylane_knowing_qiskit.png
    :align: center
    :width: 70%
   
"""

######################################################################
# In this demo, we're going to demonstrate to you the fundamentals of PennyLane and all that makes it awesome, with
# the idea in mind that you're coming from Qiskit. If you want to follow along on your computer, all you’ll
# need to do is install the `PennyLane-Qiskit plugin <https://docs.pennylane.ai/projects/qiskit/en/stable/>`__:
#
# .. code-block::
#
#   pip install -U pennylane-qiskit
#
# Now, let’s get started.
#

######################################################################
# PennyLane 🤝 Qiskit
# -------------------
#
# With the first stable release of Qiskit in February 2024 — `Qiskit 1.0 <https://github.com/Qiskit/qiskit>`__ —
# breaking changes were to be expected. Although learning a new language can be hard, PennyLane has very
# simple tools that will help via the PennyLane-Qiskit plugin. This is your gateway to the land of PennyLane
# that even lets you keep your existing Qiskit work, and you don't even have to know a ton about how PennyLane
# works to use it.
#
# There are two functions you need to know about:
#
# - :func:`~pennylane.from_qiskit`: converts an entire Qiskit ``QuantumCircuit`` — the whole thing — into
#   PennyLane. It will faithfully convert Qiskit-side measurements (even mid-circuit measurements) or
#   you can append PennyLane-side measurements directly to it.
#
# - :func:`~pennylane.from_qiskit_op`: converts a ``SparsePauliOp`` in Qiskit 1.0 to the equivalent operator in
#   PennyLane.
#
#
# These two functions give you all that you need to access PennyLane’s features and user interface
# starting from the Qiskit side. As an example, let’s say you have the following code in Qiskit that
# prepares a Bell state.
#

from qiskit import QuantumCircuit

def qiskit_circuit():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    return qc

qc = qiskit_circuit()

######################################################################
# To convert this circuit to PennyLane, you just use ``qml.from_qiskit``:
#

import pennylane as qml
qml.capture.enable()
import matplotlib.pyplot as plt

pl_func = qml.from_qiskit(qc)
qml.draw_mpl(pl_func, style="pennylane")()
plt.show()

######################################################################
# Want to measure some expectation values of Pauli operators, as well? Use ``qml.from_qiskit_op`` to
# convert a ``SparsePauliOp`` into PennyLane’s equivalent operator.
#

from qiskit.quantum_info import SparsePauliOp

qiskit_pauli_op = SparsePauliOp("XY")
pl_pauli_op = qml.from_qiskit_op(qiskit_pauli_op)

######################################################################
# Then, you can *append* the expectation value measurement — done with ``qml.expval`` — to the PennyLane
# circuit when you create it with ``qml.from_qiskit``:
#

pl_func = qml.from_qiskit(qc, measurements=[qml.expval(pl_pauli_op)])
qml.draw_mpl(pl_func, style='pennylane')()
plt.show()

######################################################################
# And just like that, you’re in PennyLaneLand! ... *It's in your ears and in your eyes!* 🎶 . Now you might be asking: “What is ``pl_func`` and how
# could I use it further?” To answer those questions, we need to get to know PennyLane a little better.
#

######################################################################
# Get to Know PennyLane 🌞
# ------------------------
#
# Let’s go back to that Qiskit circuit (``qc``) that we created `earlier <#pennylane-qiskit>`_. If we want to execute that
# circuit in Qiskit and get some results, we can do this:
#

from qiskit.primitives import StatevectorSampler

qc.measure_all()

sampler = StatevectorSampler()

job_sampler = sampler.run([qc], shots=1024)
result_sampler = job_sampler.result()[0].data.meas.get_counts()

print(result_sampler)

######################################################################
# When we use ``qml.from_qiskit`` on our Qiskit circuit, this is equivalent to creating this function
# in PennyLane.
#

def pl_func():
    """
    Equivalent to doing:
    pl_func = qml.from_qiskit(qc, measurements=qml.counts(wires=[0, 1]))
    """
    qml.Hadamard(0)
    qml.CNOT([0, 1])
    return qml.counts(wires=[0, 1])

######################################################################
# .. note ::
#
#    Qubits in PennyLane are called *wires*. Why? As was mentioned, PennyLane is a hardware-agnostic
#    framework, and “wires” is a hardware-agnostic term for quantum degrees of freedom that
#    quantum computers can be based on.
#
#

######################################################################
# A function like ``pl_func`` is called a **quantum function**. A quantum function in PennyLane just
# contains quantum gates and (optionally) returns a measurement. Measurements in PennyLane are quite
# different than in Qiskit 1.0 — we'll touch on how measurements work in PennyLane shortly. But, in our
# case, :func:`qml.counts(wires=[0, 1]) <pennylane.counts>` is the measurement, which counts the number
# of times each basis state is sampled.
#
# If we actually want to execute the circuit and see the result of our measurement, we need to define
# what the circuit runs on, just like how we defined a ``StatevectorSampler`` instance in Qiskit
# (a new `V2 primitive <https://pennylane.ai/qml/glossary/what-are-qiskit-primitives/>`__). PennyLane’s
# way of doing this is simple: (1) define a device with :func:`qml.device <pennylane.device>` and (2) pair 
# the device with the quantum function with :class:`~pennylane.QNode`.
#

dev = qml.device("default.qubit")
pl_circuit = qml.set_shots(qml.QNode(pl_func, dev), shots = 1024)

print(pl_circuit())

######################################################################
# Now that we have the full picture of how a circuit gets created and executed in PennyLane, let’s take
# a step back and summarize what’s going on. 
# 
# The first thing you’ll notice is that PennyLane’s primitives are Pythonic and array-like; quantum circuits 
# are *functions*, returning measurements that behave like `NumPy arrays <https://numpy.org/doc/stable/reference/generated/numpy.array.html>`__. The function ``pl_circuit`` is 
# called a *quantum node* (QNode), which is the union of two things:
#
# - **A quantum function that contains quantum instructions**. This is ``pl_func``, which just contains
#   quantum operations (gates) and returns a measurement. In this case, ``qml.counts(wires=1)`` is
#   the measurement, which counts the number of times each basis state is sampled and returns a dictionary
#   whose values are NumPy arrays.
#
# - **A device** (e.g., ``qml.device("default.qubit")``). PennyLane has `many devices you can choose from <https://pennylane.ai/plugins/#built-in-devices>`__, 
#   but ``"default.qubit"`` is our battle-tested Python state vector simulator.
#
#
# As for measurements in PennyLane, they are quite different from Qiskit's V2 primitives. 
# :doc:`PennyLane's measurement API <introduction/measurements>` comprises ergonomic functions that a QNode 
# can return, including
# 
# - :func:`~pennylane.state`: returns the quantum state vector,
#
# - :func:`~pennylane.probs`: returns the probability distribution of the quantum state, and
# 
# - :func:`~pennylane.expval`: returns the expectation value of a provided observable.
#
# All of this allows for a QNode to be called like a regular Python function, executing on the device
# you specified and returning the measurement you asked for — as simple as that 🌈.
#
# Alternatively, wrapping a quantum
# function with :class:`qml.QNode <pennylane.QNode>` is the same as *decorating* it with :func:`@qml.qnode(dev) <pennylane.qnode>`:
#

@qml.set_shots(1024)
@qml.qnode(dev)
def pl_circuit():
    """
    Equivalent to doing:
    pl_circuit = qml.QNode(qml.from_qiskit(qc, measurements=qml.counts(wires=[0, 1])), dev)
    """
    qml.Hadamard(0)
    qml.CNOT([0, 1])
    return qml.counts(wires=[0, 1])

######################################################################
# This is a minor point, but both approaches work.
#
# What's great about converting your work in Qiskit to PennyLane is that now you have access to all
# of `PennyLane's plugins <https://pennylane.ai/plugins/>`__, meaning you can run your Qiskit circuit on more than just IBM hardware! All
# you need to do is install the plugin of interest and change the name of the device in ``qml.device``.
#

######################################################################
# Further resources 📓
# --------------------
#
# There’s so much more to learn about what’s possible in PennyLane, and if you’re coming from Qiskit,
# you’re in good hands. The PennyLane-Qiskit plugin is your personal chaperone to the PennyLane
# ecosystem. You can dive deeper into what’s possible with the PennyLane-Qiskit plugin by visiting the
# `plugin homepage <https://docs.pennylane.ai/projects/qiskit/en/stable/>`__ or by checking out our
# :doc:`how-to guide for using Qiskit 1.0 with PennyLane <demos/how_to_use_qiskit1_with_pennylane>`.
#
# Another great thing about the PennyLane ecosystem is that we have an `emporium of up-to-date
# demos <https://pennylane.ai/search/?contentType=DEMO&sort=publication_date>`__ maintained by the
# same people that develop PennyLane. If you’re just starting out, I recommend reading our :doc:`qubit
# rotation tutorial <demos/tutorial_qubit_rotation>`.
#
# Now that you’ve used PennyLane, every road in the wonderful world of quantum programming SDKs is
# open with no set speed limits 🏎️. If you have any questions about the PennyLane-Qiskit plugin,
# PennyLane, or even Qiskit, drop us a question on the `PennyLane Discussion
# Forum <https://discuss.pennylane.ai>`__ and we’ll promptly respond. You can also keep exploring our website,
# `pennylane.ai <https://pennylane.ai>`__, to see the latest and greatest PennyLane features, demos,
# and blogs, receive the `monthly Xanadu newsletter <https://xanadu.us17.list-manage.com/subscribe?u=725f07a1d1a4337416c3129fd&id=294b062630>`__, or follow us on `LinkedIn <https://www.linkedin.com/company/pennylaneai/>`__ or `X
# (formerly Twitter) <https://twitter.com/PennyLaneAI>`__ to stay updated!
#
