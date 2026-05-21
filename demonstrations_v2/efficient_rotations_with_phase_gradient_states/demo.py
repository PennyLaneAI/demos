r"""
Efficient Rotations with Phase Gradient States
==============================================

Efficiency, efficiency, efficiency. Could this be more than just words, words, words? 


"""
import pennylane as qp
import numpy as np
import matplotlib.pyplot as plt
from pennylane.labs.transforms import select_pauli_rot_phase_gradient
import pennylane.estimator as qre

###############################################################################
# Quantifying Efficiency
# ----------------------
# As we work toward practical quantum computation, we simultaneously (and necessarily) strive to ensure each building block can operate as efficiently as possible. This principle is embedded in the goals of algorithm advancement, especially as the field works toward achieving `fault tolerance <https://pennylane.ai/topics/fault-tolerant-quantum-computing>`_. Fault-tolerant quantum devices will inevitably need to integrate some kind of `quantum error correction (QEC) <https://pennylane.ai/codebook/quantum-error-correction>`_ strategy into their architecture, which very quickly becomes a conversation about cost. From this discussion has emerged conclusions that divide known gates into two categories: fault-tolerant and not-fault-tolerant. Those that fall into the fault-tolerant categories encourage good neighbourly qualities, meaning that if a single qubit in a set of qubits passed through the gate accumulates an error, it will not induce subsequent errors on the other qubits in the set. Since the error is therefore contained, it is much easier to identify the impacted qubit and correct the issue (ex. apply an X gate to reverse an unwanted bit flip). Members of the `Clifford gates <https://pennylane.ai/qml/demos/tutorial_clifford_circuit_simulations>`_ family are prominent examples of fault-tolerant gates. 
#
# Non-fault-tolerant gates, however, do not have the same courtesy. When a qubit incurs an error as a result of an interaction with one of these gates, it has the potential to cascade through other physical qubits in the system. As a result, errors become much more difficult (if not impossible) to correct. Though it is tempting to say that the use of these gates should just be avoided in favour of fault tolerance, doing so would eliminate the possibility of achieving universiality in a gate set. To attempt to work around this issue, non-Clifford gates can be broken down into an equivalent series of `T-gates <https://pennylane.ai/qml/demos/tutorial_achieving_universality_with_the_clifford_hierarchy>`_, which represent the rotation :math:`|T\rangle=\frac{1}{\sqrt{2}}(|0\rangle+\exp{\i\pi/4})|1\rangle`. While these (like arbitrary rotations) are not fault tolerant, they can be "protected" via a process called `magic state distillation <https://pennylane.ai/qml/demos/tutorial_magic_state_distillation>`_. Perhaps this all sounds a bit far fetched now that `magic <https://pennylane.ai/qml/demos/tutorial_magic_states>`_ has come into play, but the idea can be summarized as follows: non-Clifford gates tend to not have inherent fault tolerance, so they need to be built out of components on which errors can be easily detected such that any error ridden qubits can be thrown away before the entire system is impacted. 
# 
# TGATE GRAPHIC HERE (ANIMATION?)
#
# The point? It takes **a lot** of T-gates to do this. Consider, as we will for the rest of this demo, an arbitrary rotation. To execute an 11.71:math:`^\circ` rotation, for example, a cascade of 45:math:`^\circ` rotations in the form of T-gates will need to be applied in tandem with Clifford gates to achieve the desired outcome. This can quickly grow to the scale of 100s of T-gates and beyond. This is resource intensive in itself, but when one remembers that the process of magic state distillation involves throwing away any T-gate that accumulates an error, the total cost can become astronomical. So, though it is common in classical algorithms to discuss efficiency in terms of time, quantum algorithms, as they stand, tend to have efficiency benchmarks expressed in terms of T-gate count. These quanities seem to not be completely divorced, however, since high distillation costs equate to a dominant bottleneck courtesy of :math:`|T\rangle` state production [#Gidney2018]. So, do we ensure our algorithms are as efficient as possible? For now, by striving to minimize the amount of T-gates used by (also known as the T-count of) an algorithm.
#
# Efficient Rotations?
# --------------------
# As established, arbitrary rotations are highly inefficient due to being not-fault-tolerant and typically demanding a high T-count. Unfortunately, arbirary rotations are foundational to some of the most important algorithmic tools we have, such as the `quantum fourier transform (QFT)<https://pennylane.ai/qml/demos/tutorial_qft>`_, `quantum read-only memory (QROM) <https://pennylane.ai/qml/demos/tutorial_intro_qrom>`_, and `trotterization<https://pennylane.ai/challenges/a_simple_trotterization>`_. 
#
# MORE HERE OF COURSE
#
# The Phase Gradient State
# ------------------------
# What if, instead of having to apply an individual, expensive rotation to each bit to achieve the desired outcome, we could simply add a pre-defined phase to each qubit as needed?
#
# `Phase gradient <https://pennylane.ai/compilation/phase-gradient>`_ states are a type of catalytic resource state that can be applied to applications that care about efficient rotations. For clarity, a state is "catalytic" if it is unchanged by a process that it takes part in. To (aptly) borrow the language of chemistry, it exists to catalyze (assist) a specific process (operation). The phase gradient state can be interpreted as a catalyst for phase shifts, invoking phase accumulation on other qubits without sacrificing its own properties. Essentially, the phase gradient state acts once on a dedicated register to encode a set of spatially dependent angles that can be invoked to computational states via `quantum addition <https://pennylane.ai/qml/demos/tutorial_how_to_use_quantum_arithmetic_operators>`_ when needed. 
#
# .. math::
#    |\nabla_b\rangle=\otimes_{j=1}^b\frac{1}{\sqrt{2}}(|0\rangle+\exp{-i\frac{2\pi}{2^j}}|1\rangle).
#
# Here, :math:`b` is the total number of qubits stored in the gradient register and :math:`j` is the index of a specific qubit within the register.
# References
# ----------
# .. [#Gidney2018] C. Gidney, "Halving the cost of quantum addition," *Quantum*, vol. 2, p. 74, Jun. 2018. https://doi.org/10.22331/q-2018-06-18-74