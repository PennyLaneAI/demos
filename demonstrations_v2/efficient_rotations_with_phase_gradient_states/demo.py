r"""
Efficient Rotations with Phase Gradient States
==============================================

If you care about efficient rotations, you should care about phase gradient states.

The field of quantum algorithms has long been focused on ensuring operations can be executed on quantum hardware as efficiently as possible. Especially when integrating error correction and techniques toward fault-tolerant quantum computing, carrying out the processes we want to execute as effectively as possible using as few resources as possible is of central importance. It is unlikely that the best way to do this is universal; different operations will require different implementation strategies to concurrently maintain functionality and efficiency. This demo will focus on one specific and important example: the **phase gradient state**, a gate synthesis strategy that can be applied to reduce the dominant expense of arbitrary rotation operations. 

Efficient Rotations?
--------------------
A As it stands, quantum algorithmic efficiency tends to be quantified by the number of T gates required to execute the `non-Clifford <https://pennylane.ai/qml/demos/tutorial_clifford_circuit_simulations>`_ operations included in a system. Since each T gate represents a fixed :math:`\frac{\pi}{4}` rotation, generating arbitrary angles that deviate from this angle become increasingly resource intensive the further it varies. As it stands, arbitrary rotations are very expensive. Unfortunately, they are also foundational to some of the most important tools we have in quantum algorithms and computing, such as the `quantum Fourier transform (QFT) <https://pennylane.ai/qml/demos/tutorial_qft>`_, `quantum phase estimation <https://pennylane.ai/demos/tutorial_qpe/>`_, and the `Trotterization algorithm <https://pennylane.ai/challenges/a_simple_trotterization>`_. As a result, optimizing these operations in the fault-tolerant picture is very important.The field of quantum algorithms is continuously working to reduce the number of T gates required to carry out these rotations, especially since it seems T gate generation is the dominant bottleneck in execution [#Gidney2018]_.

To emphasize the benefit of T gate optimization, let us take the example of time evolution simulation on a computational grid. Here, a wavefunction discretized to :math:`N=2^n` grid points, where :math:`n` is the number of qubits required to represent the dimensions of the grid, can be effectively evolved in time via the application of position-dependent phase shifts represented as controlled rotations. In the naïve approach, where each point on the grid is treated independently, the system's gate count will scale as :math:`\mathcal{O}(2^n\log_2(1/\epsilon))`, where :math:`\epsilon` is the desired precision. Yikes.

A quick resource estimation for a single pass of this procedure can be carried out using PennyLane and the `estimator tool <https://docs.pennylane.ai/en/stable/code/api/pennylane.estimator.estimate.estimate.html>`_. Note that, for an :math:`n`-qubit system, ``SelectPauliRot()`` (a type of multiplexed rotation) will apply a total of :math:`R=2^n` rotations to address each possible binary state. Thus, for a 3-qubit system, 8 rotations will be applied in a single pass. The T count estimate here is made assuming a Repeat-Until-Success (RUS) synthesis strategy [#Paetznick2014]_, which is the default approach taken by PennyLane's estimation function. 

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
data_wires = list(range(n))
target_wire = n

dev = qp.device("default.qubit")

@qp.qnode(dev)
def circuit_baseline():
  for wire in data_wires:
    qp.Hadamard(wire)
  qp.SelectPauliRot(angles, data_wires, target_wire, rot_axis="Y")
  return qp.probs(target_wire)

print(qre.estimate(circuit_baseline)())
###############################################################################
# From this estimate, we can see that even a single, simple pass of a rotation multiplexer requires over 300 T gates for a small set of randomly-chosen angles. Considering the cost of T gates and the fact that useful applications of this procedure will need to handle many more qubits and many more rotations, it is easy to see that the cost will grow very quickly. We need a hero!
#
# The Phase Gradient State
# ------------------------
# What if, instead of having to apply an individual, expensive rotation for each bit value to achieve the desired outcome, we could simply add a pre-defined phase to each qubit as needed?
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
# Here, :math:`b=\log_2(1/\epsilon)` is the total number of qubits stored in the phase gradient register, :math:`B=2^b` is the total number of possible states in the superposition between all qubits in the gradient register, and :math:`j` is the index of a specific qubit within the register. 
#
# The phase gradient state can be interpreted as acting as a pre-defined plane of stored angles that can be accessed and invoked on a target state when desired. This state can be prepared once, stored in an auxiliary register, and reused. To induce a phase shift, the data register (storing an integer value) simply needs to be added to the gradient register, which is done most commonly using a `SemiAdder() <https://docs.pennylane.ai/en/stable/code/api/pennylane.SemiAdder.html>`_ step.
#
# .. math::
#    \begin{aligned}
#    |\Psi\rangle|\nabla_b\rangle &= \alpha|0\rangle|\nabla_b\rangle+\beta|1\rangle|\nabla_b\rangle \\
#    C(Add_n)|\Psi\rangle|\nabla_b\rangle &= \alpha|0\rangle|\nabla_b\rangle+\beta|1\rangle Add_k |\nabla_b\rangle.
#    \end{aligned}
#
# .. figure:: ../demonstrations_v2/efficient_rotations_with_phase_gradient_states/PhaseShiftCircuitDiagram.png
#        :align: center
#        :width: 900px
#        *Equivalent circuits for executing a phase shift, in which the phase shift operator can be replaced with an addition step between a data register and a phase gradient register*
#
# This addition is basically a "push" invoked by the data register on the phase gradient register. Via quantum addition, the gradient register is shifted by an amount equivalent to the binary weight of each data qubit that is added to it. Since the data state remains "stationary", the two states will be *out of phase* by an amount equivalent to the shift experienced by the register following the addition operation. 
#
# .. figure:: ../demonstrations_v2/efficient_rotations_with_phase_gradient_states/PhaseKickback.gif
#      :align: center
#      :width: 500px
#
#      *Phase kickback can be imagined as a change in the relative phase between the data register and the phase gradient register. As depicted, a controlled addition between the two registers will result in the positional displacement of the phase gradient state which, in turn, causes the phase difference that can be associated with either state. Even though the gradient register shifts, the data register can "pick up" the relative phase difference*.
#
# Since phase is relative, it can be said without issue that the data register has accumulated a phase equivalent to this shift. This process is referred to as `phase kickback <https://pennylane.ai/qml/demos/tutorial_phase_kickback>`_. Again thanks to the relative nature of this shift, the properties of the gradient register are globally unchanged, solidifying it as a catalytic resource. Thus, the phase gradient state essentially stores spatially dependent phases that can be applied to encode phase as a function of qubit position. 
#
# .. math::
#    \begin{aligned}
#    C(Add_n)|\Psi\rangle|\nabla_b\rangle &= \alpha|0\rangle|\nabla_b\rangle+\beta|1\rangle e^{-\frac{2\pi i k}{B}} |\nabla_b\rangle \\
#                                         &= (\alpha|0\rangle+\beta e^{-\frac{2\pi i k}{B}} |1\rangle)|\nabla_b\rangle.
#    \end{aligned}
#
# .. admonition:: Phase Gradient Rotation Algorithm
#    :class: note
#
#    1. A phase gradient state is encoded onto a register composed of :math:`b` qubits.
#    2. A semi-adder operation is performed between a data register and the gradient register.
#    3. The phase gradient register shifts proportionally to the weight of the data qubit added to it.
#    4. The shift in the gradient register causes the data register to accumulate a relative phase via phase kickback.
#    5. Since position shifts are relative and do not alter structure, the catalytic phase gradient state remains unchanged and can be reused as desired.
#
# This approach, as a whole, has two major optimization benefits. First, the phase gradient state only needs to be generated *once* since its catalytic nature leaves it unchanged after it interacts with the data register. This means it will have a one-time, upfront preparation T gate cost that is never repeated. Second, the phase shifts are applied by an addition operation rather than multiplication, meaning the phase gradient method's T count scales as :math:`\mathcal{O}(2^n+\log_2(1/\epsilon))` when memory costs are considered. 
#
# Now, we can implement the same procedure as above, replacing the individual rotations carried out by the Pauli rotation operator with a `SemiAdder() <https://docs.pennylane.ai/en/stable/code/api/pennylane.SemiAdder.html>`_-facilitated application of the phase gradient state. This can be done easily using PennyLane's transform `select_pauli_rot_phase_gradient() <https://docs.pennylane.ai/en/stable/code/api/api/pennylane.labs.transforms.select_pauli_rot_phase_gradient.html>`_, which will identify Pauli rotation gates and replace them with the necessary phase gradient steps.
#

prec = 0.1 #Desired Accuracy
b = int(np.ceil(np.log2(1/prec))) #Gradient Register Size

#Define Auxiliary Wires for Phase Gradient Transformation
angle_wires = list(range(n+1,n+1+b))
gradient_wires = list(range(n+1+b,n+1+2*b))
work_wires = list(range(n+1+2*b,n+1+2*b+(3*b)))

@qp.qnode(dev)
@select_pauli_rot_phase_gradient(angle_wires,gradient_wires,work_wires)
def circuit_phase_grad():
  for wire in data_wires:
    qp.Hadamard(wire)
  qp.SelectPauliRot(angles, data_wires, target_wire, rot_axis="Y")
  return qp.probs(target_wire)

print(qre.estimate(circuit_phase_grad)())

###############################################################################
# Note that there are different ways to translate Toffoli gates into T gate counts, but here we will take Gidney's approximation of 1 Toffoli gate = 4 T gates [#Gidney2018]_, meaning this implementation requires an estimated 44 T gates. So, for the same task with the same goal, this operation has now reduced in T gate cost by an order of magnitude. Not too shabby! 
#
# The scale of these savings becomes increasingly clear as the size of the data register that stores the input state (and, therefore, the size of the system as a whole) increases. The plot below shows how drastically the resource requirements scale in the first case with even a small increase in system size.
#
# .. figure:: ../demonstrations_v2/efficient_rotations_with_phase_gradient_states/t_gate_comparison.png
#    :align: center
#    :width: 80%
# 
# Though the phase gradient approach requires an investment of resources to prepare the gradient register, the additive nature of the rotation procedure results in a very slow accumulation of resources. It is, of course, important to acknowledge that the implementation of the phase gradient state requires additional qubits to be added to the system. Though this contributes to resource cost, the comparative cost of generating a usable T gate (typically on the order of :math:`\mathcal{O}(10)`) still outweighs this investment in the long run. Thus, even though this approach requires an upfront investment of resources, it pays off in the long-run as additional per-rotation costs accumulate much more slowly than in alternative methods.
#
# Sizing Up Other Optimizations
# --------------------------------
# The importance of cheaply decomposing arbitrary rotations has been known to the industry for some time. As a result, the phase gradient approach is not the only gate-synthesis strategy available in the quantum algorithmic toolkit. There are many nuances involved in asserting the ideal applications of each algorithm, but a full exploration will not be included here. Instead, comparing the per-rotation gate cost reveals the relative cost of each strategy, which must be weighed against the nuance of the specific application.
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
# Phase gradient states are becoming more and more present in state-of-the-art algorithms, such as `dynamic simulations <https://arxiv.org/html/2601.16264v1>`_. When it comes to developing useful applications for the future's quantum computers, it is crucial to keep in mind what resources will be required so that compatibility with hardware and physical systems remains reasonable. As emphasized, the additive nature of using phase gradient states for this application greatly reduces the T count of applying arbitrary rotations. Understanding of the theory behind phase gradient states and the role they can play in carrying out efficient rotations opens the door to several compilation tools in PennyLane. The `phase gradient page in PennyLane's compilation hub <https://pennylane.ai/compilation/phase-gradient>`_ is the best place to go to learn more. The tools available can be applied to a plethora of applications. Give them a try the next time you are experimenting with `quantum phase estimation <https://pennylane.ai/qml/demos/tutorial_qpe>`_, for example!
#
# .. _references:
#
# References
# ----------
# .. [#Gidney2018] C.\ Gidney, "Halving the cost of quantum addition," *Quantum*, vol. 2, p. 74, Jun. 2018. `doi: 10.22331/q-2018-06-18-74 <https://quantum-journal.org/papers/q-2018-06-18-74/>`_`.
#
# .. [#Ross2016] N.\ J. Ross and P. Selinger, "Optimal ancilla-free Clifford+T approximation of z-rotations," *Quantum Inf. Comput.*, vol. 16, no. 11-12, pp. 901–953, 2016, `arXiv: 1403.2975 <https://arxiv.org/abs/1403.2975>`_.
#
# .. [#Paetznick2014] A.\ Paetznick and K. M. Svore, "Repeat-Until-Success: Non-deterministic decomposition of single-qubit unitaries," *Quantum Inf. Comput.*, vol. 14, no. 15-16, pp. 1277–1301, 2014, `arXiv: 1311.1074 <https://arxiv.org/pdf/1311.1074>`_.
#
# .. [#Dawson2006] C.\ M. Dawson and M. A. Nielsen, "The Solovay-Kitaev algorithm," *Quantum Inf. Comput.*, vol. 6, no. 1, pp. 81–95, 2006, `arXiv: quant-ph/0505030 <https://arxiv.org/pdf/quant-ph/0505030>`_.
#
# .. [#Kliuchnikov2015] V.\ Kliuchnikov, A. Bocharov, M. Roetteler, and J. Yard, "A Framework for Approximating Qubit Unitaries," 2015, `arXiv: 1510.03888 [quant-ph] <https://arxiv.org/pdf/1510.03888>`_.
#
# .. [#Kliuchnikov2022] V.\ Kliuchnikov, K. Lauter, R. Minko, A. Paetznick, and C. Petit, "Shorter quantum circuits via single-qubit gate approximation," 2022, `arXiv: 2203.10064 <https://arxiv.org/abs/2203.10064>`_
