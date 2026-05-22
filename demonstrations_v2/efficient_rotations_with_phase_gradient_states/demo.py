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
# Non-fault-tolerant gates, however, do not have the same courtesy. When a qubit incurs an error as a result of an interaction with one of these gates, it has the potential to cascade through other physical qubits in the system. As a result, errors become much more difficult (if not impossible) to correct. Though it is tempting to say that the use of these gates should just be avoided in favour of fault tolerance, doing so would eliminate the possibility of achieving universiality in a gate set. To attempt to work around this issue, non-Clifford gates can be broken down into an equivalent series of `T-gates <https://pennylane.ai/qml/demos/tutorial_achieving_universality_with_the_clifford_hierarchy>`_, which represent the rotation :math:`|T\rangle=\frac{1}{\sqrt{2}}(|0\rangle+\exp{-i\pi/4})|1\rangle`. While these (like arbitrary rotations) are not fault tolerant, they can be "protected" via a process called `magic state distillation <https://pennylane.ai/qml/demos/tutorial_magic_state_distillation>`_. Perhaps this all sounds a bit far fetched now that `magic <https://pennylane.ai/qml/demos/tutorial_magic_states>`_ has come into play, but the idea can be summarized as follows: non-Clifford gates tend to not have inherent fault tolerance, so they need to be built out of components on which errors can be easily detected such that any error ridden qubits can be thrown away before the entire system is impacted. 
# 
# TGATE GRAPHIC HERE (ANIMATION?)
#
# The point? It takes **a lot** of T-gates to do this. Consider, as we will for the rest of this demo, an arbitrary rotation. To execute an 11.71° rotation, for example, a cascade of 45° rotations in the form of T-gates will need to be applied in tandem with Clifford gates to achieve the desired outcome. This can quickly grow to the scale of 100s of T-gates and beyond. This is resource intensive in itself, but when one remembers that the process of magic state distillation involves throwing away any T-gate that accumulates an error, the total cost can become astronomical. So, though it is common in classical algorithms to discuss efficiency in terms of time, quantum algorithms, as they stand, tend to have efficiency benchmarks expressed in terms of T-gate count. These quanities seem to not be completely divorced, however, since high distillation costs equate to a dominant bottleneck courtesy of :math:`|T\rangle` state production [#Gidney2018]_. So, do we ensure our algorithms are as efficient as possible? For now, by striving to minimize the amount of T-gates used by (also known as the T-count of) an algorithm.
#
# Efficient Rotations?
# --------------------
# As established, arbitrary rotations are expensive. Unfortunately, they are also foundational to some of the most important tools we have in quantum algorithms and computing, such as the `quantum fourier transform (QFT) <https://pennylane.ai/qml/demos/tutorial_qft>`_, `quantum read-only memory (QROM) <https://pennylane.ai/qml/demos/tutorial_intro_qrom>`_, and `trotterization <https://pennylane.ai/challenges/a_simple_trotterization>`_. As a result, optimizing these operations in the fault-tolerant picture is very important. There have been a diverse array of optimization strategies proposed that all boil down to reducing the number of T-gates required to carry out a required rotation. 
#
# Let us take the example of binary state loading, which is typically carried out through some method of phase manipulation. For QROM, for example, an :math:`N`-bit binary string can be represented by :math:`n=log_2(N)` qubits in superposition that conditionally undergo a phase shift invoked by a set of gates. To ensure fault tolerange, these rotations should be carried out using distilled T-gates, causing the system to scale as :math:`\mathcal{O}(2Nlog_2(1/\epsilon))`, where :math:`\epsilon` is the desired precision. Yikes.
#
# A quick resource estimation in PennyLane for a single pass of this procedure can be carried out using PennyLane and the qre.estimate() tool. Note that, for a :math: n qubit system, SelectPauliRot (a multiplexer) will apply a total of :math:`R=2^n` rotations to address each possible binary state. Thus, for a 3 qubit system, 8 rotations will be applied.
# CIRCUIT DIAGRAM

prec = 0.1 #Desired Accuracy
n = 3 #Control Qubits
np.random.seed(35)
angles = np.random.rand(2**n)*2*np.pi

#Define Circuit Wires
control_wires = list(range(n))
target_wire = n

dev = qp.device("default.qubit")

@qp.qnode(dev)
def circuit_baseline():
  for wire in control_wires:
    qp.Hadamard(wire)
  qp.SelectPauliRot(angles, control_wires, target_wire, rot_axis="Y")
  return qp.probs(target_wire)

print(qre.estimate(circuit_baseline)())

###############################################################################
# From this estimate, we can see that even a simple, single pass of a rotation multiplexer requires over 300 T-gates. Considering the cost of distillation and the fact that useful applications of this procedure will need to handle many more qubits and many more rotations, it is easy to see that the cost will grow very quickly. We need a hero!
#
# The Phase Gradient State
# ------------------------
# What if, instead of having to apply an individual, expensive rotation to each bit to achieve the desired outcome, we could simply add a pre-defined phase to each qubit as needed?
#
# `Phase gradient <https://pennylane.ai/compilation/phase-gradient>`_ states are a type of catalytic resource state that can be applied to applications that care about efficient rotations. For clarity, a state is "catalytic" if it is unchanged by a process that it plays a role in facilitating. To (aptly) borrow the language of chemistry, it exists to catalyze (assist) a specific process (operation). The phase gradient state can be interpreted as a catalyst for phase shifts, invoking phase accumulation on other qubits without sacrificing its own properties. Essentially, the phase gradient state acts once on a dedicated register to encode a set of spatially dependent angles that can be invoked to computational states via `quantum addition <https://pennylane.ai/qml/demos/tutorial_how_to_use_quantum_arithmetic_operators>`_ when needed. The state can be represented as the conditional phase operator
#
# .. math::
#    |\nabla_b\rangle=\otimes_{j=1}^b\frac{1}{\sqrt{2}}(|0\rangle+e^{-i\frac{2\pi}{2^j}}|1\rangle).
#
# Here, :math:`b` is the total number of qubits stored in the gradient register and :math:`j` is the index of a specific qubit within the register. 
#
# The phase gradient state can basically acts as a conditional rotation operator in itself. What is special, however, is that this state can be prepared once, stored in a register, then *added* to a data register (like the control register we discuessed previously) to induce the desired phase shift via `phase kickback <https://pennylane.ai/codebook/quantum-phase-estimation/catch-the-phase>`_. One can take the term "gradient" literally here; the phase gradient state essentially stores spatially dependent phases that can be applied to input data as a function of qubit position. There are two main takeaways from this, the first being that the phase gradient states only needs to be generated once since its catalytic nature leaves it unchanged after it interacts with the data register, and the second being that the phase shifts are applied by an addition operation rather than multiplication. As a result, the phase gradient method's T-gate count scales by :math:`\mathcal{O}(4(N+b))`. 
#
# ADDITION ANIMATION?
#
# Now, we can implement the same procedure as above, replacing the individual rotations carried out by the Pauli rotation operator with a SemiAdder() facilitated application of the phase gradient state. This can be done easily using PennyLane's lab transform `select_pauli_rot_phase_gradient() <https://github.com/PennyLaneAI/pennylane/pull/8738>`_, which will identify Pauli rotation gates and replace them with the necessary phase gradient steps.
#
# PHASE GRAD CIRCUIT DIAGRAM

#Define Auxillary Wires for Phase Gradient Transformation
b = int(np.ceil(np.log2(1/prec))) #Gradient Register Size

#Define Auxillary Wires for Phase Gradient Transformation
angle_wires = list(range(n+1,n+1+b))
gradient_wires = list(range(n+1+b,n+1+2*b))
work_wires = list(range(n+1+2*b,n+1+2*b+(3*b)))

dev2 = qp.device("default.qubit")
@qp.qnode(dev2)
@select_pauli_rot_phase_gradient(angle_wires,gradient_wires,work_wires)
def circuit_phase_grad():
  for wire in control_wires:
    qp.Hadamard(wire)
  qp.SelectPauliRot(angles, control_wires, target_wire, rot_axis="Y")
  return qp.probs(target_wire)

print(qre.estimate(circuit_phase_grad)())

###############################################################################
# Note that a standard approximation takes 1 Toffoli gate = 4 T-gates, meaning this implementation requires an estimated 44 T-gates. So, for the same operation with the same goal, this operation has now reduced in T-gate cost by an order of magnitude. Not too shabby! 
#
# The magnitude of this efficiency increase scales exponentially. The plot below shows how drastically the resource requirements scale in the first case with even a small increase in system size.
#
# .. figure:: ../demonstrations_v2/efficient_rotations_with_phase_gradient_states/QubitsvsTgatesNotLog.png
#    :align: center
#    :width: 80%
# 
# Sizing Up Other Optimizations
# --------------------------------
# The importance of carrying out highly efficient rotations has been known to the industry for some time. As a result, the phase gradient approach is not the only gate-synthesis strategy available in the quantum mechanic's toolkit. There are certainly nuances between the applications and each implementation could be demostrated independently, but a full exploration will not be included here.  
#
# 
#
# Conclusion
# ----------
# Understanding of the theory behind phase gradient states and the role they can play in carrying out efficient rotations opens the door to several compilation tools in PennyLane. The `phase gradient page in the compilation hub <https://pennylane.ai/compilation/phase-gradient>`_ is the best place to go to learn more. The tools available can be applied to a plethora of applications, give them a try the next time you are experimenting with `quantum phase estimation <https://pennylane.ai/qml/demos/tutorial_qpe>`_, for example!
#
# +----------------------------+--------------------+-----------------------------------------------------+
# | Algorithm                  | Setup Cost         | Gate Cost Per Rotation                              |
# +============================+====================+=====================================================+
# | GridSynth                  | 0                  | :math:`3\log_2(1/\epsilon)` [#Ross2016]_            | 
# +----------------------------+--------------------+-----------------------------------------------------+
# | Kliuchnikov                | 0                  | :math:`2\log(1/\epsilon)` [#Kliuchnikov2015]_       | 
# +----------------------------+--------------------+-----------------------------------------------------+
# | Phase Gradient             | :math:`4\log_2(1/\ | 4 [#Gidney2018]_                                    |
# |                            | epsilon)`          |                                                     | 
# +----------------------------+--------------------+-----------------------------------------------------+
# | Repeat Until Success (RUS) | 0                  | :math:`2.4\log_2(1/\epsilon)-3.28` [#Paetznick2014]_| 
# +----------------------------+--------------------+-----------------------------------------------------+
# | Solovay-Kitaev             | 0                  | :math:`\log^{3.97}(1/\epsilon)` [#Dawson2006]_      | 
# +----------------------------+--------------------+-----------------------------------------------------+
#
# .. _references:
#
# References
# ----------
# .. [#Gidney2018] C. Gidney, "Halving the cost of quantum addition," *Quantum*, vol. 2, p. 74, Jun. 2018. https://doi.org/10.22331/q-2018-06-18-74
# .. [#Ross2016] Neil J. Ross and Peter Selinger. "Optimal ancilla-free Clifford+T approximation of z-rotations." *Quantum Information and Computation*, vol. 16, no. 11-12, 2016, pp. 901–953. arXiv:1403.2975 [quant-ph]
# .. [#Paetznick2014] Adam Paetznick and Krysta M. Svore. "Repeat-Until-Success: Non-deterministic decomposition of single-qubit unitaries." *Quantum Information & Computation*, vol. 14, no. 15-16, 2014, pp. 1277–1301. arXiv:1311.1074 [quant-ph]
# .. [#Dawson2006] Christopher M. Dawson and Michael A. Nielsen. "The Solovay-Kitaev algorithm." *Quantum Information and Computation*, vol. 6, no. 1, 2006, pp. 81–95. arXiv:quant-ph/0505030
# .. [#Kliuchnikov2015] Vadym Kliuchnikov, Alex Bocharov, Martin Roetteler, and Jon Yard. "A Framework for Approximating Qubit Unitaries." *arXiv*, 2015. arXiv:1510.03888 [quant-ph]