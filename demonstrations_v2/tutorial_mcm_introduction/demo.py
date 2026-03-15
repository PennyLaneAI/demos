r"""Introduction to mid-circuit measurements
============================================

Mid-circuit measurements are an important building block in quantum algorithms
and quantum error correction, and with :doc:`measurement-based quantum computing
<demos/tutorial_mbqc>`, they even power a complete quantum computing paradigm.
In this tutorial, we will dive into the basics of mid-circuit measurements with
PennyLane. You will learn about

- basic measurement processes in quantum mechanics,

- the impact of a measurement on one- and two-qubit systems,

- postselection and qubit reset, and

- dynamic quantum circuits powered by conditional operations.

.. figure:: ../_static/demonstration_assets/mcm_introduction/socialthumbnail_mcm_introduction.png
    :align: center
    :width: 50%

We also have dedicated learning material if you want to know :doc:`how to collect
statistics of mid-circuit measurements <demos/tutorial_how_to_collect_mcm_stats>` or
:doc:`how to create dynamic circuits with mid-circuit measurements
<demos/tutorial_how_to_create_dynamic_mcm_circuits>`.
"""

######################################################################
# Measurements in quantum mechanics
# ---------------------------------
#
# Measurements are the subject of important questions in quantum mechanics:
# What is a measurement? How does it affect the measured system? And how can we
# describe a measurement process mathematically?
# Given how fundamental those question are, there is a plethora of learning
# resources on this topic, from textbooks [#mike_n_ike]_ and (video) lectures
# [#feynman]_, [#zwiebach]_ to interactive studying material like
# `the PennyLane Codebook </codebook/>`__
# or Ref. [#van_der_Sar]_.
# Furthermore, discussing measurements quickly
# leads to questions about the interpretation of quantum mechanics and philosophical,
# if not metaphysical, discourse.
# For these reasons, we will not aim at discussing those deep question in great detail,
# but focus on understanding the basics that will help us understand measurements
# performed within quantum circuits, i.e., mid-circuit measurements, and how to realize
# them in PennyLane.
#
# Bear with us, we will briefly look at a mathematicaly definition for
# measurements but then turn to practical examples and hands-on calculations.
#
# Mathematical description
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# We will go with the following definition: A measurement :math:`M` is a process that
# maps a valid quantum state :math:`\rho` to a classical probabilistic mixture
#
# .. math:: M[\rho]=\sum_{i=1}^n p_i \rho_i
#
# of post-measurement quantum states :math:`\rho_i` that are specified by :math:`M.`
# Here, :math:`n` is the number of possible measurement outcomes and :math:`p_i`
# is the probability to measure the outcome :math:`i` associated to :math:`\rho_i,`
# given the input state :math:`\rho.` For a qubit in the
# :math:`|+\rangle=(|0\rangle + |1\rangle)/\sqrt{2}` state measured in the :math:`Z` basis, we find
#
# .. math::
#
#     M[|+\rangle\langle +|]=\frac{1}{2}|0\rangle\langle 0|+\frac{1}{2}|1\rangle\langle 1|,
#
# because the probability to measure :math:`0` or :math:`1` is :math:`50\%` each. We
# will explore this example in more detail below.
#
# The expression above describes the probabilistic mixture after the quantum mechanical
# measurement if we do *not* record the measurement outcome.
# If we do record the measurement outcome and only keep those samples
# that match a specific postselection rule,
# we no longer have a probabilistic mixture, but find the state :math:`\rho_i` for
# the filtered outcome :math:`i.`
#
# For the rest of this tutorial, we will restrict ourselves to standard measurements
# commonly found in mid-circuit measurements, using so-called projective measurements.
# In this setting, the measurement comes with one projector :math:`\Pi_i` per
# measurement outcome, and all projectors sum to the identity.
# The post-measurement states are given by
#
# .. math:: \rho_i = \frac{\Pi_i \rho \Pi_i}{\operatorname{tr}[\Pi_i \rho]}
#
# and the probabilities are dictated by the Born rule,
# :math:`p_i=\operatorname{tr}[\Pi_i \rho].`
# This means that if we do not record the measurement outcome, the system simply
# ends up in the state
#
# .. math:: M[\rho] = \sum_{i=1}^n \Pi_i \rho \Pi_i.
#
# To understand this abstract description better, let's look at three simple examples;
# measuring a single qubit, measuring a Bell state, and resetting qubits.
#
# Measuring a single qubit
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# Consider a single qubit in the state :math:`|+\rangle,` i.e., in the equal
# superposition of :math:`|0\rangle` and :math:`|1\rangle.`
# To get started, let's first implement this state in PennyLane and compute some
# expectation values that will be insightful later on. We follow these steps:
#
# - Import PennyLane and define a device using :func:`~.pennylane.device`.
#   The built-in ``"default.qubit"`` Python statevector simulator will suffice for our
#   purposes, however PennyLane provides a wide array of additional
#   `high-performance and hardware devices <https://pennylane.ai/plugins/>`__.
#
# - Write a quantum function that first creates the :math:`|+\rangle` state using
#   a :class:`~.pennylane.Hadamard` gate, and then measures the expectation values
#   :math:`\langle X\rangle` and :math:`\langle Z\rangle` in this state, using
#   :func:`~.pennylane.expval`.
#
# - Specify the device on which the quantum function should be executed,
#   using the :func:`~.pennylane.qnode` decorator.
#
# - Run the quantum node and show the computed expectation values!
#
# If you'd like more guidance on any of these steps, also have a look at
# :doc:`our tutorial on qubit rotation <demos/tutorial_qubit_rotation>` explaining
# them in detail.
#

import pennylane as qml
qml.capture.enable()

dev = qml.device("default.qubit")

@qml.qnode(dev)
def before():
    qml.Hadamard(0)  # Create |+> state
    return qml.expval(qml.X(0)), qml.expval(qml.Z(0))

b = before()
print(f"Expectation values before any measurement: {b[0]:.1f}, {b[1]:.1f}")

######################################################################
# The result is not surprising: :math:`|+\rangle` is the eigenstate of :math:`X` for
# the eigenvalue :math:`+1,` and it has the well-known expectation value
# :math:`\langle +|Z|+\rangle=0.`
#
# Now we bring in a mid-circuit measurement in the computational, or Pauli-:math:`Z,`
# basis. It comes with the projections :math:`\Pi_i=|i\rangle\langle i|,`
# :math:`i\in\{0, 1\},` onto the computational basis states.
# If we execute the measurement process but do not record the outcome, we find the state
#
# .. math::
#
#     M[\rho]
#     &= \Pi_0 \rho \Pi_0 + \Pi_1\rho \Pi_1\\
#     &= |0\rangle\langle 0|+\rangle\langle +|0\rangle\langle 0|
#     \ +\ |1\rangle\langle 1|+\rangle\langle +|1\rangle\langle 1|\\
#     &= \frac{1}{2}\mathbb{I}.
#
# where we used the overlaps :math:`\langle +|i\rangle=1/\sqrt{2}` and the decomposition
# :math:`\mathbb{I} = |0\rangle\langle 0| + |1\rangle\langle 1|` of the identity.
# This means that the measurement sends the qubit from a pure state into a mixed state,
# i.e., it not only affects the state but even the *class* of states it is in. And this
# is *because* we did not even record the measurement outcome!
#
# Let's look at this example in PennyLane. We repeat the steps from above but
# additionally include a mid-circuit measurement, calling :func:`~.pennylane.measure`
# on the qubit ``0``. Note that we just perform the measurement and do not assign any
# variable to its outcome.
#

@qml.qnode(dev)
def after():
    qml.Hadamard(0)  # Create |+> state
    qml.measure(0)  # Measure without recording the outcome
    return qml.expval(qml.X(0)), qml.expval(qml.Z(0))

a = after()
print(f"Expectation value after the measurement:  {a[0]:.1f}, {a[1]:.1f}")

######################################################################
# The measurement moved the qubit from the :math:`|+\rangle` eigenstate of
# the Pauli-:math:`X` operator into a mixed state with expectation value zero for all
# Pauli operators, explaining the values we just observed.
#
# Now if we filter for one measurement outcome, say :math:`0,` we find the state
#
# .. math::
#
#    M[\rho]=\rho_0
#    =\frac{|0\rangle\langle 0|+\rangle\langle +|0\rangle\langle 0|}{\operatorname{tr}[|0\rangle\langle 0|+\rangle\langle +|]}
#    =|0\rangle\langle 0|,
#
# that is, the qubit is in a new, pure state. In PennyLane, we can postselect on the case
# where we measured a :math:`0` using the ``postselect`` keyword argument of
# ``qml.measure``:
#

@qml.qnode(dev)
def after():
    qml.Hadamard(0)  # Create |+> state
    qml.measure(0, postselect=0)  # Measure and only accept 0 as outcome
    return qml.expval(qml.X(0)), qml.expval(qml.Z(0))

a = after()
print(f"Expectation value after the postselected measurement:  {a[0]:.1f}, {a[1]:.1f}")

######################################################################
# As expected, we find the that the measured, postselected qubit is in the
# :math:`|0\rangle` eigenstate of the Pauli-:math:`Z` operator with eigenvalue
# :math:`+1,` yielding :math:`\langle X\rangle=0` and :math:`\langle Z\rangle=1.` For
# ``postselect=1``, we would have obtained the :math:`|1\rangle` eigenstate of :math:`Z`
# with eigenvalue :math:`-1,` instead.
#
# Measuring a Bell pair
# ~~~~~~~~~~~~~~~~~~~~~
# Next, we consider a pair of qubits, entangled in a Bell state:
#
# .. math:: |\phi\rangle = \frac{1}{\sqrt{2}} (|00\rangle + |11\rangle).
#
# This is a pure state with density matrix
#
# .. math::
#
#     |\phi\rangle\langle \phi | = \frac{1}{2}\left(|00\rangle\langle 00|
#     + |00\rangle\langle 11| + |11\rangle\langle 00| + |11\rangle\langle 11|\right).
#
# We will again measure only the first qubit.
# We code this circuit up similar to the one above, using an additional ``CNOT`` gate
# to create the Bell state. We also include optional hyperparameters such as
# ``postselect`` as keyword arguments to our quantum function and pass them on to
# ``qml.measure``. Note that we can't complete the quantum function yet, because we still
# need to discuss what to return from it!
#

def bell_pair_preparation(**kwargs):
    qml.Hadamard(0)
    qml.CNOT([0, 1])  # Create a Bell pair
    qml.measure(0, **kwargs)  # Measure first qubit, using keyword arguments

######################################################################
# Without recording the outcome, i.e., ``postselect=None``, we obtain the state
#
# .. math::
#
#     M[\rho] = \frac{1}{2}\left(|00\rangle\langle 00| + |11\rangle\langle 11|\right),
#
# which again could be described by a classical mixture as well. If we instead postselect
# on measuring, say, a :math:`1,` we find :math:`M[\rho] = |11\rangle\langle 11|.`
#
# There are two striking differences between whether we record the measurement outcome
# or not: the state of the qubits changes from a mixed to a pure state, as witnessed
# by the state's *purity*; and its entanglement changes, too, as witnessed by the
# *von Neumann entanglement entropy*. We can compute both quantities easily in PennyLane,
# using :func:`~.pennylane.purity` and :func:`~.pennylane.vn_entropy`, respectively.
# And those will be the return types to complete our quantum function:
#

@qml.qnode(dev)
def bell_pair(postselect):
    bell_pair_preparation(postselect=postselect)
    return qml.purity([0, 1]), qml.vn_entropy(0)

######################################################################
# So let's compare the purities and von Neumann entropies of the Bell state
# after measurement:
#

without_ps = bell_pair(None)
with_ps = bell_pair(1)
print(f"                     | without ps | with ps ")
print(f"Purity               |     {without_ps[0]:.1f}    |   {with_ps[0]:.1f}")
print(f"Entanglement entropy |     {without_ps[1]:.2f}   |  {with_ps[1]:.1f}")

######################################################################
# We indeed see a change in the purity and entanglement entropy based on postselection.
#
# Qubit reset
# ~~~~~~~~~~~
#
# Another commonly used feature with mid-circuit measurements is to reset the measured
# qubit, i.e., if we measured a :math:`1,` we flip it back into to the :math:`|0\rangle`
# state with a Pauli :math:`X` operation. If there is just one qubit, this is the same
# as if we never measured it but reset it directly to the initial state :math:`|0\rangle,`
# as long as we do not use the measurement outcome for anything.
# For the Bell pair example from above, resetting the measured qubit means that
# we flip the first bit if it is a :math:`1.` Alternatively, we can trace out the
# first qubit and re-initialize it in the state :math:`|0\rangle.` Denoting the reset
# step explicitly as :math:`R,` this leads to the post-measurement state
#
# .. math::
#
#     R[M[\rho]]
#     &=|0\rangle\langle 0|\otimes \operatorname{tr}_1[M[\rho]]\\
#     &=|0\rangle\langle 0|\otimes
#     \left[\frac{1}{2}\left(|0\rangle\langle 0| + |1\rangle\langle 1|\right)\right]\\
#     &= |0\rangle\langle 0|\otimes \frac{1}{2}\mathbb{I}.
#
# We see that the qubits are no longer entangled, even if we do not postselect.
# Let's compute some exemplary expectation values in this state with PennyLane.
# We recycle the state preparation subroutine from above, to which we
# can simply pass the keyword argument ``reset`` to activate the qubit reset:
#

@qml.qnode(dev)
def bell_pair_with_reset(reset):
    bell_pair_preparation(reset=reset)
    return qml.expval(qml.Z(0)), qml.expval(qml.Z(1)), qml.expval(qml.Z(0) @ qml.Z(1))

no_reset = bell_pair_with_reset(reset=False)
reset = bell_pair_with_reset(reset=True)

print(f"              | <Z₀> | <Z₁> | <Z₀Z₁> ")
print(f"Without reset |  {no_reset[0]:.1f} |  {no_reset[1]:.1f} |   {no_reset[2]:.1f}")
print(f"With reset    |  {reset[0]:.1f} |  {reset[1]:.1f} |   {reset[2]:.1f}")

######################################################################
# Resetting the qubit changed the expectation values of the local observable :math:`Z_0`
# and the global observable :math:`Z_0Z_1.`
#
# Dynamically controlling a quantum circuit
# -----------------------------------------
# So far we've only talked about mid-circuit measurements that directly affect the state of
# qubits, about postselection, and about qubit reset as an additional step after
# performing the measurement. However, the outcomes of a measurement can not only be used
# to decide whether or not to discard a circuit execution. More importantly, as
# mid-circuit measurements are performed while the quantum circuit is up and running, their
# outcomes can be used to modify the circuit structure itself *dynamically*.
#
# This technique is widely used to improve quantum algorithms or to trade off classical
# and quantum computing resources. It also is an elementary building block for quantum
# error correction, as the corrections need to happen while the circuit is running.
#
# Here we look at a simple yet instructive example subroutine called a *T-gadget*,
# a technique related to :doc:`quantum teleportation <demos/tutorial_teleportation>`.
#
# T-gadget in PennyLane
# ~~~~~~~~~~~~~~~~~~~~~
# In fault-tolerant quantum computing, a standard way to describe a quantum circuit is to
# separate Clifford gates (which map Pauli operators to Pauli operators) from
# :class:`~.pennylane.T` gates. Clifford gates, including :class:`~.pennylane.X`,
# :class:`~.pennylane.Hadamard`, :class:`~.pennylane.S`, and
# :class:`~.pennylane.CNOT`, alone can not express arbitrary quantum circuits, but it's
# enough to add the ``T`` gate to this set [#gottesman]_!
#
# Applying a ``T`` gate on an error-corrected quantum computer is usually hard.
# A *T-gadget* [#zhou]_ allows us to replace a ``T`` gate by Clifford gates, provided
# we have an auxiliary qubit in the right initial state, a so-called magic state.
# The gadget then consists of the following steps:
#
# - Prepare an auxiliary qubit in a magic state
#   :math:`(|0\rangle + e^{i\pi/4} |1\rangle)/\sqrt{2},` for example using :doc:`magic
#   state distillation <demos/tutorial_magic_state_distillation>`;
#
# - Entangle the auxiliary and target qubit with a ``CNOT``;
#
# - Measure the auxiliary qubit with ``measure`` and record the outcome;
#
# - If the measurement outcome was :math:`1,` apply an ``S`` gate to the target qubit.
#   The conditional is realized with :func:`~.pennylane.cond`.
#

import numpy as np

magic_state = np.array([1, np.exp(1j * np.pi / 4)]) / np.sqrt(2)

def t_gadget(wire, aux_wire):
    qml.StatePrep(magic_state, aux_wire)
    qml.CNOT([wire, aux_wire])
    mcm = qml.measure(aux_wire, reset=True)  # Resetting disentangles aux qubit
    qml.cond(mcm, qml.S)(wire)  # Apply qml.S(wire) if mcm was 1

######################################################################
# We will not derive why this works (see, e.g., [#zhou]_ instead), but
# illustrate that this gadget implements a ``T`` gate by combining it with an adjoint
# ``T†`` gate and looking at the resulting action on the
# eigenstates of ``X``. For this, we
#
# - prepare a :math:`|+\rangle` or :math:`|-\rangle` state, chosen by an input;
#
# - apply the T-gadget from above;
#
# - apply ``T†``, using :func:`~.pennylane.adjoint`;
#
# - return the expectation value :math:`\langle X_0\rangle.`
#

@qml.qnode(dev)
def test_t_gadget(init_state):
    qml.Hadamard(0)  # Create |+> state
    if init_state == "-":
        qml.Z(0)  # Flip to |-> state

    t_gadget(0, 1)  # Apply T-gadget
    qml.adjoint(qml.T)(0)  # Apply T^† to undo the gadget

    return qml.expval(qml.X(0))

print(f"<X₀> with initial state |+>: {test_t_gadget('+'):4.1f}")
print(f"<X₀> with initial state |->: {test_t_gadget('-'):4.1f}")

######################################################################
# The T-gadget indeed performs a ``T`` gate, which is being reversed by
# ``T†``. As a result, the expectation values match those of the initial
# states :math:`|\pm\rangle.`
#
# How can we understand the above circuit intuitively? We did not postselect
# the measurement outcome, but we did record (and use) it to modify the
# circuit structure. For a single measurement, or shot, this would have
# led to exactly *one* of the events "measure :math:`0,` do not apply ``S``"
# or "measure :math:`1,` apply ``S``", with equal probability for either one.
# The state on wire ``0`` is :math:`T|\pm\rangle` in either case!
#
# For scenarios in which the different events lead to *distinct* states,
# one has to pay attention to whether a single shot or a collection of
# shots is used, and to the computed measurement statistics.
#
# Conclusion
# ----------
#
# This concludes our introduction to mid-circuit measurements. We saw how
# quantum mechanical measurements affect qubit systems and how postselection
# affects the state after measurement and validated the theoretical examples
# with short PennyLane examples. Then we looked into dynamic circuits powered
# by operations conditioned on mid-circuit measurements.
#
# For more detailed material also check out the dedicated how-tos
# on :doc:`mid-circuit measurement statistics <demos/tutorial_how_to_collect_mcm_stats>`
# and :doc:`dynamic circuits <demos/tutorial_how_to_create_dynamic_mcm_circuits>`,
# as well as the `measurements quickstart page
# <https://docs.pennylane.ai/en/stable/introduction/measurements.html#mid-circuit-measurements-and-conditional-operations>`_
# and the documentation of :func:`~.pennylane.measure`.
#
# Happy measuring!
#
# References
# ----------
# .. [#mike_n_ike]
#
#     Michael Nielsen, Isaac Chuang
#     "Quantum computation and quantum information", Cambridge university press,
#     `Book website <http://www.michaelnielsen.org/qcqi/>`__, 2010.
#
# .. [#feynman]
#
#     Richard P. Feynman
#     "Feynman lectures on physics", volume 3,
#     `open access at Caltech <https://www.feynmanlectures.caltech.edu/III_toc.html>`__, 1963.
#
# .. [#zwiebach]
#
#     Barton Zwiebach
#     "Quantum Physics II",
#     `MIT OpenCourseWare <https://ocw.mit.edu/courses/8-05-quantum-physics-ii-fall-2013/>`__, 2013.
#
# .. [#van_der_Sar]
#
#     Toeno van der Sar, Gary Steele
#     "Open Quantum Sensing and Measurement",
#     `open access at TUDelft <https://interactivetextbooks.tudelft.nl/qsm/src/index.html>`__, 2023.
#
# .. [#gottesman]
#
#     Daniel Gottesman
#     "Theory of fault-tolerant quantum computation", Physical Review A, **57**, 127,
#     `open acces at Caltech <https://authors.library.caltech.edu/3850/1/GOTpra98.pdf>`__, 1998.
#
# .. [#zhou]
#
#     Xinlan Zhou, Debbie W. Leung, Isaac L. Chuang
#     "Methodology for quantum logic gate constructions", Physical Review A, **62**, 052316,
#     `arXiv quant-ph/0002039 <https://arxiv.org/abs/quant-ph/0002039>`__, 2000
#
