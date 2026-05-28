r"""
Efficient Rotations with Phase Gradient States
==============================================

Efficiency, efficiency, efficiency. Is it possible for this to become more than just words, words, words? 

The field of quantum algorithms has long been focused on ensuring operations can be executed on quantum hardware as efficiently as possible. Even when integrating error correction and techniques toward fault-tolerant quantum computing, the question of carrying out the processes we want to execute as effectively as possible using as few resources as possible remains central. The answer is likely not universal; different operations will require different strategies to maintain efficacy and ensure efficiency. This demo will focus on one specific and important example: the **phase gradient state**, a gate synthesis strategy that can be applied to reduce the dominant expense of arbitrary rotation operations. If you care about efficient rotations, you should care about this!

.. admonition:: Quantifying Efficiency
   :class: note

   Broadly, quantum gates can be divided into two categories: `fault-tolerant <https://pennylane.ai/topics/fault-tolerant-quantum-computing>`_ and non-fault-tolerant. 
   
   |

   Fault-tolerant gates encourage good neighbourly qualities. If a single qubit accumulates an error when interacting with a gate from this set, it will not induce subsequent errors on the other qubits in the system. Since the error is therefore contained, it is much easier to identify the impacted qubit and correct the issue (ex. apply an X gate to reverse an unwanted bit flip). Members of the `Clifford gates <https://pennylane.ai/qml/demos/tutorial_clifford_circuit_simulations>`_ category are prominent examples of fault-tolerant gates. 

   |

   Non-fault-tolerant gates, however, do not have the same courtesy. When one of these gates invokes an error, has the potential to cascade through the entire system. As a result, errors become much more difficult (if not impossible) to correct. Though it is tempting to say the use of these gates should just be avoided, doing so would eliminate the possibility of achieving `universality <https://pennylane.ai/compilation/clifford-t-gate-set>`_ in a gate set. 

   |

   To work around this issue, non-Clifford gates can be `decomposed <https://docs.pennylane.ai/en/stable/code/api/pennylane.transforms.decompose.html>`_ into an equivalent series of `T-gates <https://pennylane.ai/qml/demos/tutorial_achieving_universality_with_the_clifford_hierarchy>`_, which represent a fixed :math:`\frac{\pi}{4}` rotation, and auxiliary Clifford gates. While these (like arbitrary rotations) are not fault tolerant, they can be "protected" via a process called `magic state distillation <https://pennylane.ai/qml/demos/tutorial_magic_state_distillation>`_. Perhaps this all sounds a bit far-fetched now that `magic <https://pennylane.ai/qml/demos/tutorial_magic_states>`_ has come into play, but the idea can be summarized as follows: non-Clifford gates tend to not have inherent fault tolerance, so the operations they represent should ideally be built out of components on which errors can be easily detected such that any error-ridden qubits can be thrown away before the entire system is impacted. 
 
   .. figure:: ../demonstrations_v2/efficient_rotations_with_phase_gradient_states/GateDecompBig.gif
        :align: center
        :width: 300px

   The point? It takes **a lot** of T-gates to decompose non-fault-tolerant gates. Consider, as we will for the rest of this demo, an arbitrary rotation. To execute an 11.71° rotation, for example, a cascade of 45° rotations in the form of T-gates will need to be applied in tandem with Clifford gates to achieve the desired outcome. This can quickly grow to the scale of 100s of T-gates and beyond. This is resource intensive in itself, but when one remembers that the process of magic state distillation involves throwing away any T-gate that accumulates an error, the total cost can become astronomical. 
   
   |

   So, though it is common in classical algorithms to discuss efficiency in terms of time, quantum algorithms, as they stand, tend to have efficiency benchmarks expressed in terms of T-gate count. These quantities seem to not be completely divorced, however, since high distillation costs equate to a dominant bottleneck courtesy of :math:`|T\rangle` state production [#Gidney2018]_. So, how do we ensure our algorithms are as efficient as possible? For now, by striving to minimize the number of T-gates used by (also known as the T-count of) an algorithm.

Efficient Rotations?
--------------------
Arbitrary rotations are expensive. Unfortunately, they are also foundational to some of the most important tools we have in quantum algorithms and computing, such as the `quantum Fourier transform (QFT) <https://pennylane.ai/qml/demos/tutorial_qft>`_, `quantum phase estimation <https://pennylane.ai/demos/tutorial_qpe/>`_, and the `trotterization algorithm <https://pennylane.ai/challenges/a_simple_trotterization>`_. As a result, optimizing these operations in the fault-tolerant picture is very important. There have been a diverse array of optimization strategies proposed that all boil down to reducing the number of T-gates required to carry out a rotation. 

To emphasize the benefit of T-gate optimization, let us take the example of binary state loading, which is typically carried out through some method of phase manipulation. Here, an :math:`N`-bit binary string can be represented by :math:`n=\log_2(N)` qubits in superposition that conditionally undergo a rotation invoked by a set of gates. To ensure fault tolerance, these rotations should be carried out using distilled T-gates, causing the system to scale as :math:`\mathcal{O}(2^n\log_2(1/\epsilon))`, where :math:`\epsilon` is the desired precision, when taking a naïve approach. Yikes.

A quick resource estimation for a single pass of this procedure can be carried out using PennyLane and the qre.estimate() tool. Note that, for an :math:`n`-qubit system, SelectPauliRot (a multiplexer) will apply a total of :math:`R=2^n` rotations to address each possible binary state. Thus, for a 3-qubit system, 8 rotations will be applied in a single pass. The T-count estimate will is made assuming a Repeat-UntiL-Success (RUS) procedure [#Paetznick2014], which is the naïve approach taken by the estimation function. 

"""
import pennylane as qp
import numpy as np
import matplotlib.pyplot as plt
from pennylane.labs.transforms import select_pauli_rot_phase_gradient
import pennylane.estimator as qre

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
# From this estimate, we can see that even a simple, single pass of a rotation multiplexer for a small set of arbitrary angles requires over 300 T-gates. Considering the cost of distillation and the fact that useful applications of this procedure will need to handle many more qubits and many more rotations, it is easy to see that the cost will grow very quickly. We need a hero!
#
# The Phase Gradient State
# ------------------------
# What if, instead of having to apply an individual, expensive rotation to each bit to achieve the desired outcome, we could simply add a pre-defined phase to each qubit as needed?
#
# `Phase gradient states <https://pennylane.ai/compilation/phase-gradient>`_ are a type of catalytic resource state that can be used to facilitate rotations additively. For clarity, a state is "catalytic" if it is unchanged by a process that it plays a role in facilitating. To (aptly) borrow from the language of chemistry, it exists to catalyze (assist) a specific process (operation). The phase gradient state can be interpreted as a catalyst for phase shifts, invoking phase accumulation on other qubits without sacrificing its own properties. The state can be represented as
#
# .. math::
#    |\nabla_b\rangle=\frac{1}{\sqrt{B}}\sum_{k=0}^{B-1}e^{-2\pi i \frac{k}{B}}|k\rangle
#
# or, in product state form,
#
# .. math::
#    |\nabla_b\rangle=\otimes_{j=1}^b\frac{1}{\sqrt{2}}(|0\rangle+e^{-i\frac{2\pi}{2^j}}|1\rangle).
#
# Here, :math:`B=2^b` is the total number of possible states in the superposition, :math:`b=\log_2(1/\epsilon)` is the total number of qubits stored in the gradient register, and :math:`j` is the index of a specific qubit within the register. 
#
# The phase gradient state can be interpreted as acting as a pre-defined plane of stored angles that can be accessed and invoked on a target state when desired. This state can be prepared once, stored in an auxiliary register, and reused. To induce a phase shift, the data register (storing an integer value) simply needs to be added to the gradient register, which is done most commonly (and most efficiently) using a `SemiAdder() <https://docs.pennylane.ai/en/stable/code/api/pennylane.SemiAdder.html>`_ step. This addition can be interpreted as a "push" invoked by the data register on the phase gradient register. Via quantum addition, the gradient register is shifted by an amount equivalent to the binary weight of each data qubit that was added to it. Since the data state remains "stationary", the two states will now be *out of phase* by an amount equivalent to the shift experienced by the register. Since phase is relative, it can be said without issue that the data register has accumulated a phase equivalent to this shift. This process is referred to as `phase kickback <https://pennylane.ai/qml/demos/tutorial_phase_kickback>`_. As mentioned, this phase and the positional shift that invoked it are completely relative, so the properties of the gradient register are globally unchanged, solidifying it as a catalytic resource. Thus, the phase gradient state essentially stores spatially dependent phases that can be applied to encode data as a function of qubit position. 
#
# .. figure:: ../demonstrations_v2/efficient_rotations_with_phase_gradient_states/PhaseKickbackAnimation.gif
#      :align: center
#      :width: 500px
#
# To summarize, the phase gradient rotation algorithm can be itemized as follows,
#
# 1. A phase gradient state is encoded onto a register composed of :math:`b` qubits.
# 2. A semi-adder operation is performed between a data register and the gradient register.
# 3. The phase gradient register shifts proportionally to the weight of the data qubit added to it.
# 4. The shift in the gradient register causes the data register to accumulate a relative phase via phase kickback.
# 5. Since position shifts are relative and do not alter structure, the catalytic phase gradient state remains unchanged and can be reused as desired.
#
# This approach, as a whole, has two major optimization benefits. First, the phase gradient state only needs to be generated *once* since its catalytic nature leaves it unchanged after it interacts with the data register, meaning it will have a one-time, upfront preparation T-gate cost that is never repeated. Second, the phase shifts are applied by an addition operation rather than multiplication, meaning the phase gradient method's T-gate count scales as :math:`\mathcal{O}(2^n+\log_2(1/\epsilon))` when memory costs (here being `QROM <https://pennylane.ai/demos/tutorial_intro_qrom>`_) are considered. 
#
# Now, we can implement the same procedure as above, replacing the individual rotations carried out by the Pauli rotation operator with a `SemiAdder() <https://docs.pennylane.ai/en/stable/code/api/pennylane.SemiAdder.html>`_ facilitated application of the phase gradient state. This can be done easily using PennyLane's lab transform `select_pauli_rot_phase_gradient() <https://github.com/PennyLaneAI/pennylane/pull/8738>`_, which will identify Pauli rotation gates and replace them with the necessary phase gradient steps.
#

prec = 0.1 #Desired Accuracy
b = int(np.ceil(np.log2(1/prec))) #Gradient Register Size

#Define Auxiliary Wires for Phase Gradient Transformation
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
# Note that there are different ways to translate Toffoli gates into T-gate counts, but here we will take Gidney's approximation of 1 Toffoli gate = 4 T-gates [#Gidney2018]_, meaning this implementation requires an estimated 44 T-gates. So, for the same task with the same goal, this operation has now reduced in T-gate cost by an order of magnitude. Not too shabby! 
#
# The scale of this efficiency becomes increasingly clear as the size of the data register (and, therefore, the size of the system as a whole) increases. The plot below shows how drastically the resource requirements scale in the first case with even a small increase in system size.
#
# .. figure:: ../demonstrations_v2/efficient_rotations_with_phase_gradient_states/QubitsvsTgatesNotLog.png
#    :align: center
#    :width: 80%
# 
# Though the phase gradient approach requires an investment of resources to prepare the gradient register, the additive nature of the loading procedure results in a very slow accumulation of resources. The benefit of this quickly outweighs the lack of setup cost intrinsic to alternative approaches, justifying the benefit of phase gradient states in rotation-based algorithms.
#
# Sizing Up Other Optimizations
# --------------------------------
# The importance of efficiently decomposing arbitrary rotations has been known to the industry for some time. As a result, the phase gradient approach is not the only gate-synthesis strategy available in the quantum algorithmic toolkit. There are certainly nuances between the ideal applications of each algorithm, but a full exploration will not be included here. Instead, comparing the per-rotation gate cost reveals the relative cost of each strategy, which must be weighed against the nuance of the specific application.
#
# +-----------------------------------+------------------------------+-----------------------------------------------------+
# |             Algorithm             |        Setup Cost (T)        |             Gate Cost Per Rotation (T)              |
# +===================================+==============================+=====================================================+
# |          Solovay-Kitaev           |              0               |   :math:`\log^{3.97}(1/\epsilon)` [#Dawson2006]_    |
# +-----------------------------------+------------------------------+-----------------------------------------------------+
# |             GridSynth             |              0               |      :math:`3\log_2(1/\epsilon)` [#Ross2016]_       |
# +-----------------------------------+------------------------------+-----------------------------------------------------+
# |    Repeat Until Success (RUS)     |              0               |   :math:`2.4\log_2(1/\epsilon)` [#Paetznick2014]_   |
# +-----------------------------------+------------------------------+-----------------------------------------------------+
# |            Kliuchnikov            |              0               |    :math:`2\log(1/\epsilon)` [#Kliuchnikov2015]_    |
# +-----------------------------------+------------------------------+-----------------------------------------------------+
# |  Single Qubit Gate Approximation  |              0               |  :math:`0.56\log_2(1/\epsilon)` [#Kliuchnikov2022]_ |
# +-----------------------------------+------------------------------+-----------------------------------------------------+
# |          Phase Gradient           |  :math:`\log_2(1/\epsilon)`  |                  1 [#Gidney2018]_                   |
# +-----------------------------------+------------------------------+-----------------------------------------------------+
#
#
# Conclusion
# ----------
# Phase gradient states are becoming more and more present in state-of-the-art algorithms, such as `dynamic simulations <https://arxiv.org/html/2601.16264v1>`_. When it comes to developing useful, s for the future's quantum computers, it is crucial to keep in mind what resources will be required so that compatibility with hardware remains reasonable. As emphasized, the additive nature of using phase gradient states for this application greatly reduces the T-count of applying arbitrary rotations. This, among many other advancements, continues to point toward the possibility of physical fault-tolerant architecture. Understanding of the theory behind phase gradient states and the role they can play in carrying out efficient rotations opens the door to several compilation tools in PennyLane. The `phase gradient page in the compilation hub <https://pennylane.ai/compilation/phase-gradient>`_ is the best place to go to learn more. The tools available can be applied to a plethora of applications, give them a try the next time you are experimenting with `quantum phase estimation <https://pennylane.ai/qml/demos/tutorial_qpe>`_, for example!
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
# .. [#Kliuchnikov2022] Kliuchnikov, V., Lauter, K., Minko, R., Paetznick, A., & Petit, C. (2022). Shorter quantum circuits via single-qubit gate approximation. Quantum, 7, 1208. https://doi.org/10.22331/q-2023-12-18-1208