r"""
Trotterization
==============

Whether we like it or not, time is always moving forward. Even more frustrating, things tend to change with time, meaning we cannot neglect time evolution as we set out to model realistic systems. To execute this successfully, though, tends to be incredibly computationally intense. In the case of particle systems, which constitute a lot of interesting, cutting-edge simulation work, the exponential nature of the system's scaling (i.e. for :math:`n` particles, there are :math:`2^n` possible configurations, meaning the Hamiltonian energy matrix of the system used in the time evolution operator :math:`U(t)=e^{-iHt}` would be :math:`2^n \times 2^n`) very quickly renders classical, sequential simulation impossible, even for incredibly short time scales. 

Luckily, quantum computers have shown promise in addressing this issue since, instead of having to represent each possible state contained in the problem's Hilbert space, qubits can themselves act as analogues for the particles that make up the system, enabling polynomial scaling with particle count rather than exponential scaling. `Maybe Feynman was onto something <https://s2.smu.edu/~mitch/class/5395/papers/feynman-quantum-1981.pdf>`_! Carrying out time evolution on these more-efficient systems, however, is not easy, meaning Hamiltonian simulation techniques such as **Trotterization** need to be implemented to achieve legible and realistic results. This demo will explore how we can do exactly that!

The Commutation Problem
-----------------------

For a completely isolated free particle experiencing no potential energy, the Hamiltonian of the system can be simply defined in terms of kinetic energy and applied to the system via a unitary gate representing :math:`U_{free}(t)=e^{-iHt}=e^{-iTt}`, where :math:`T` is the kinetic energy term. Sticking to this idealized case would, of course, be useless. With each additional term added to the Hamiltonian, though, the feasibility of the time evolution operator :math:`e^{-iHt}` being executable diminishes. This brings to mind a supposedly simple solution, why not just split the Hamiltonian into pieces that are executable? If we take the representation :math:`H=H_1+H_2+...+H_n`, we could naïvely say that our time evolution operator becomes :math:`e^{-i(H_1+H_2+...+H_n)t}=e^{-iH_1t}e^{-iH_2t}...e^{-iH_nt}`. In our naïveté, however, we would neglect to consider the commutation relations of the split Hamiltonian components. If certain fragments of the Hamiltonian do not commute (:math:`[H_i\neq H_j]`), we cannot exponentiate it in its entirety without deviating from our expected outcome.

.. admonition:: Recall
   :class: note

   For commuting matrices, where :math:`AB=BA` and it is implied that A and B are completely independent,

   |

   :math:`(A+B)^2 = A^2+2AB+B^2`.

   |

   For non-commuting matrices, where :math:`AB \neq BA` and it is implied that A and B have some level of dependency,

   |

   :math:`(A+B)^2 = A^2+AB+BA+B^2`

   |

   So, the Taylor expansion :math:`e^{A+B}=I+(A+B)+\frac{1}{2}(A+B)^2+...` differs in the two cases!

As we know, most physical systems require a combination of non-commuting operators to be fully described. Thus, splitting the Hamiltonian this way in a non-commuting scenario would not produce an accurate picture the time evolution of the system being simulated. To imagine this more physically, consider what it would mean to exponentiate two dependent properties in this way. Taking :math:`A` and :math:`B` to be position and momentum, our naïve approach would lead us to blindly exponentiate as :math:`e^{iHt}=e^{iAt}e^{iBt}`. However, applying these operators sequentially in this way assumes that the system is accurately described by iterating the position *then* iterating the momentum, one after the other. This completely ignores the nuance of how the two properties influence each other, thus rendering the simulation inaccurate. Rats!

Luckily for us, the problem of exponentiating non-commuting formulas is not new. Even outside of the time-evolution picture, a productive amount of ink has been spilled over how to deal with this issue. The `Lie-Trotter product formula <https://en.wikipedia.org/wiki/Lie_product_formula>`_ is a main result of this effort. The theorem states that for arbitrary :math:`m \times m` matrices :math:`A` and :math:`B`

.. math::
   e^{A+B}=\lim_{r \to \infty}(e^{A/r}e^{B/r})^r.

Turning back to the Hamiltonian picture, if we take a large, finite :math:`r`, letting :math:`r` represent the number of time steps taken in the evolution, we can approximate the Lie-Trotter formula to

.. math::
   e^{-iHt}=(\prod_j e^{-iH_j t/r})^r.

So, by slicing the total time the simulation is trying to emulate, the dependency shared by non-commuting properties can be integrated via alternating applications of each operator. To turn back to our earlier example, instead of taking one complete position step and one complete momentum step, we are now alternating small, partial steps in position and momentum, approimating simultaneity to the best of our ability. The example of position and momentum also raises the question of operator bases, since clearly these two time evolution operators are defined in Fourier-conjugate bases. Indeed, if the non-commuting operators do not exist in the same basis, a basis change step (such as a `quantum Fourier transform (QFT) <https://pennylane.ai/demos/tutorial_qft>`_) will need to be added between each operator application. 

.. figure:: ../demonstrations_v2/exploring_trotterization/IterativeFitIllustration.png
   :align: center
   :width: 700px

Implementing the Trotter Method
-------------------------------
The form of the Lie-Trotter approximation implies a clear order of operations that must be executed to simulate the time evolution of a non-commuting system. 

As implied, the first step of carrying out Hamiltonian simulation using Trotterization is splitting the Hamiltonian. To jump a few steps ahead, implementing the Trotterized time evolution operators will require the construction of equivalent unitary operators using universal gate sets. As a rule of thumb, feasible algorithms should strive for a minimal gate count and minimized opportunity for error, meaning we should try, where possible, to reduce the number of operators required and, therefore, the number of Hamiltonian fragments. To achieve this, the Hamiltonian should be split only where mathematically necessary, meaning commuting terms should always be grouped together to the greatest extent possible. Take, for example, a Hamiltonian containing the terms :math:`Z_1Z_3`, :math:`Z_2Z_4`, and :math:`X_1X_2`, where :math:`Z_i` and :math:`X_j` are Pauli operators. The first and second terms will always commute since they consist completely of Z operators, but neither of the first two operators commute with the third operator. Therefore, the most optimal splitting would yield :math:`H_1=Z_1Z_3+Z_2Z_4` and :math:`H_2=X_1X_2`. This is not always so obvious in complex physical systems, so thorough analysis should be carried out to ensure optimal operator pairing. 

Once the Hamiltonian is appropriately split, we need to ensure that the fragmented Hamiltonian we are working with can be implemented on realistic hardware in terms of accessible gates. This tends to require some kind of `transformation <https://pennylane.ai/demos/tutorial_mapping>`_ to take place. Recalling the relevance of this method in simulating particle systems, one relevant example is the mapping of Fermionic states to qubit states via the `Jordan-Wigner transformation <https://docs.pennylane.ai/en/stable/code/api/pennylane.jordan_wigner.html>`_, which maps creation and annihilation operators to Pauli operators. For the purposes of the following simple demonstration, we will assume our Hamiltonian was originally defined in the Pauli basis and forgo the transformation step, defining :math:`H=\alpha X + \beta Z`, where :math:`\alpha` and :math:`\beta` are arbitrary coefficients. 

To carry out the split-time method, the number of time steps :math:`r` must be defined, ideally achieving a high level of accuracy while remaining within reasonable computational resource bounds. From there, the circuit should simply alternate between the time evolution operator unitary gates for the desired number of steps. Since our Hamiltonian is completely in the Pauli basis, we will not need to perform a basis change step in our Trotter circuit. However, recall that exponentiated Pauli operators are represented by rotation gates, where, letting :math:`\sigma_i` be a Pauli operator,

.. math::
   R_{\sigma_i}(\theta)=e^{-i\frac{\theta}{2}\sigma_i}.

So, taking :math:`H_1=\alpha X` and :math:`H_2=\beta Z`

.. math::
   U(t)=e^{-i \alpha X t}e^{-i \beta Z t}=R_X(2\alpha t)R_Z(2 \beta t).

This is easily translatable to code.
"""

import pennylane as qp
import numpy as np
from scipy.linalg import expm
import pennylane.estimator as qre
from pennylane.transforms.rz_phase_gradient import rz_phase_gradient

#Define System Parameters
coeffs = [0.2, 1.3] #Define Hamiltonian coefficients
observables = [qp.PauliX(0), qp.PauliZ(0)] #Define observables
t = 10 #Define time in seconds
R = [10, 20, 30, 40, 50, 100, 200, 400] #Trotter steps

dev = qp.device("lightning.qubit", wires=1)

#Define Trotterization Function
@qp.qnode(dev)
def TrotterStepper(t,r,coeffs):
    del_t = t/r

    #Apply the rotation r times
    for i in range(r):
        U_A = qp.RX(2*coeffs[0]*del_t, wires=0)
        U_B = qp.RZ(2*coeffs[1]*del_t, wires=0)

    return [qp.expval(qp.PauliX(0)), qp.expval(qp.PauliY(0)), qp.expval(qp.PauliZ(0))]
###############################################################################
# Trotter Error
# -------------
# Understanding how Hamiltonian simulations deviate from theoretically expected system behaviour is a field of study in itself. Intuitively, we expect the size of the time step being takento be a major contributor to the error, since time steps that are too large tend to obscure system behaviour. A nuanced exploration of exact Trotter error tends to begin expansion via the `Baker-Campbell-Hausdorff formula <https://en.wikipedia.org/wiki/Baker%E2%80%93Campbell%E2%80%93Hausdorff_formula>`_ [#Su2020]_, which describes the true expansion of a non-commuting operator pair as
#
# .. math::
#     e^{-iBt}e^{iAt}=e^{it(A+B)-\frac{t^2}{2}[B,A]+i\frac{t^3}{12}[B[B,A]]-...}.
#
# As is familiar when handling series expansions, the degree to which the BCH formula is truncated in the system representation dictates the amount of error to expect in the Hamiltonian. Comparing the previous definition of the approximated Trotter formula to this expansion expression, we can see that we are only concerned with the first term and, therefore, our error is dominated by the term :math:`-\frac{t^2}{2}[B,A]` (also known as the First-Order Trotter error). Taking :math:`t=\Delta t` for each time step, we can reason out that the error is proportional to :math:`\frac{t^2}{r}`. This result aligns with our earlier intuition, implying that error will increase with length of time and decrease with number of steps. This, of course, neglects to consider the upper bound on the number of time steps that can be handled computationally, which will be examined briefly later.
#
# In the simple example implemented in this demo, an observed Trotter error can be achieved by calculating time evolution of the system analytically and carrying out a comparison. This would, of course, be unfeasible for large, complicated (in other words, useful) models or long time scales, but it is sufficient for this example.
#

#Approximate evolution for Trotter error calculation
H = qp.Hamiltonian(coeffs, observables)

#Pauli matrix representations
X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])
Y = np.array([[0, -1j], [1j, 0]])

H = coeffs[0]*X + coeffs[1]*Z
U = expm(-1j*H*t)

state_0 = np.array([1,0])
state_1 = np.array([0,1])

evolved_H = U @ state_0

exact_exp_X = np.real(evolved_H.conj() @ X @ evolved_H)
exact_exp_Y = np.real(evolved_H.conj() @ Y @ evolved_H)
exact_exp_Z = np.real(evolved_H.conj() @ Z @ evolved_H)
###############################################################################
# By keeping the simulation duration fixed and varying the number of steps taken, the error at each resolution can be obtained. 

print(f"{'r':>5} | {'X error':>8} | {'Y error':>8} | {'Z error':>8} | {'Total error':>11}")
print("-" * 51)

for r in R:
    result = TrotterStepper(t,r,coeffs)
    X_error = abs(result[0]-exact_exp_X)
    Y_error = abs(result[1]-exact_exp_Y)
    Z_error = abs(result[2]-exact_exp_Z)
    total_error = np.sqrt(X_error**2+Y_error**2+Z_error**2)
    print(f"{r:>5} | {X_error:>8.5f} | {Y_error:>8.5f} | {Z_error:>8.5f} | {total_error:>11.5f}")
###############################################################################
# As expected, the simulation error decreases with increasing time steps! 
# 
# Gate Synthesis Considerations
# -----------------------------
# In fault tolerant architectures, arbitrary rotations must be decomposed into a series of `Clifford+T gates <https://pennylane.ai/compilation/clifford-t-gate-set>`_ via some method of `gate synthesis <https://pennylane.ai/compilation/two-qubit-synthesis>`_. The PennyLane `estimate() <https://docs.pennylane.ai/en/stable/code/api/pennylane.estimator.estimate.estimate.html>`_ tool can be used to determine how many gates are used to implement the synthesized rotation and takes a Repeat Until Success (RUS) approach, which tends to cost :math:`\mathcal{O}(2.4\log_2(1/\epsilon))` T gates per rotation [#Paetznick2014]_. An alternative approach is the `phase gradient method <https://pennylane.ai/demos/efficient_rotations_with_phase_gradient_states>`_ (you can check out the linked demo if you need to brush up on T counts and quantifying efficiency, too), which takes advantage of a static register that holds spatially dependent phase values which can be added to a target state as needed. This method has a total cost of :math:`\mathcal{O}(4\log_2(1/\epsilon)+4N)`, where :math:`N` is the number of rotations. 
#
# When we discuss selecting a gate synthesis method, our initial instinct might be to minimize the T count in favour of implementability and efficiency. From this perspective, it seems obvious to always select the method with the lowest cost, here being the phase gradient approach. Using PennyLane's `rz_phase_gradient() <https://docs.pennylane.ai/en/stable/code/api/pennylane.transforms.rz_phase_gradient.html>`_ transformation, the expensive :math:`R_z(\phi)` rotations implemented in the above Trotterization can be translated into phase gradient additions and compared to the naïve approach. Since this method only transforms :math:`R_z` operations, we can perform our desired :math:`R_x` rotations using an :math:`R_z` rotation sandwiched between two Hadamard gates. For example, a step count of :math:`r=200` will be taken in the resource estimation step.

#Convert to phase gradient approach
prec = 0.05
b = int(np.ceil(np.log2(1/prec)))

angle_wires = list(range(1,1+b))
gradient_wires = list(range(1+b,1+2*b))
work_wires = list(range(1+2*b,1+3*b))

dev2 = qp.device("lightning.qubit", wires=(1+3*b))

@qp.qnode(dev2)
@rz_phase_gradient(angle_wires,gradient_wires,work_wires)
def TrotterStepperPG(t,r,coeffs):
    del_t = t/r

    #Apply the rotation r times
    for i in range(r):
        #Perform X rotation in terms of RZ gates using basis change
        qp.Hadamard(wires=0)
        qp.RZ(2*coeffs[0]*del_t, wires=0)
        qp.Hadamard(wires=0)
        
        U_B = qp.RZ(2*coeffs[1]*del_t, wires=0)

    return [qp.expval(qp.PauliX(0)), qp.expval(qp.PauliY(0)), qp.expval(qp.PauliZ(0))]


#Resource estimation
test_r = 200
Trotter_resources = qre.estimate(TrotterStepper)(t,test_r,coeffs) #Repeat Until Success!
Trotter_resources_PG = qre.estimate(TrotterStepperPG)(t,test_r,coeffs)
print("Repeat Until Success")
print(Trotter_resources)
print("Phase Gradient")
print(Trotter_resources_PG)
###############################################################################
# As expected, the T count is much lower in the phase gradient method, even when adding in the Toffoli gate decomposition taking 1 Toffoli = 4 T gates [#Gidney2018]. This, however, is not a definitive affirmation that the phase gradient method is the ideal gate synthesis strategy in this case. Let us compare the Trotter error in the two methods before we jump to any conclusions.

print("Repeat Until Success")
print(f"{'r':>5} | {'X error':>8} | {'Y error':>8} | {'Z error':>8} | {'Total error':>11}")
print("-" * 51)

for r in R:
    result = TrotterStepper(t,r,coeffs)
    X_error = abs(result[0]-exact_exp_X)
    Y_error = abs(result[1]-exact_exp_Y)
    Z_error = abs(result[2]-exact_exp_Z)
    total_error = np.sqrt(X_error**2+Y_error**2+Z_error**2)
    print(f"{r:>5} | {X_error:>8.5f} | {Y_error:>8.5f} | {Z_error:>8.5f} | {total_error:>11.5f}")

print("Phase Gradient")
print(f"{'r':>5} | {'X error':>8} | {'Y error':>8} | {'Z error':>8} | {'Total error':>11}")
print("-" * 51)

for r in R:
    resultPG = TrotterStepperPG(t,r,coeffs)
    X_error_PG = abs(resultPG[0]-exact_exp_X)
    Y_error_PG = abs(resultPG[1]-exact_exp_Y)
    Z_error_PG = abs(resultPG[2]-exact_exp_Z)
    total_error_PG = np.sqrt(X_error_PG**2+Y_error_PG**2+Z_error_PG**2)
    print(f"{r:>5} | {X_error_PG:>8.5f} | {Y_error_PG:>8.5f} | {Z_error_PG:>8.5f} | {total_error_PG:>11.5f}")
#############################################################################################################
# Ah ha! A tradeoff has made itself clear! In the phase gradient implementation, the Trotter error is universally higher than in the RUS case.  What is also (maybe even more) interesting is that, after a certain :math:`r` threshold is reached, the phase gradient system no longer evolves. This is an example of `underflow <https://en.wikipedia.org/wiki/Arithmetic_underflow>`_ in `quantum arithmetic <https://pennylane.ai/demos/tutorial_how_to_use_quantum_arithmetic_operators>`_, where, put very simply, the simulation has reached a computational limit and is stuck rounding to the same value each pass. So, when we choose which techniques to use for gate synthesis, we must consider the needs of our system in tandem with the cost of our system. Sometimes an investment is necessary!
#
# Fast-Forwarding
# ---------------
# As demonstrated, implementing quantum simulation on hardware requires that a series of gates are implemented for each time step, meaning the depth of the circuit grows notably with increasing time. If the depth of a simulation circuit maintains proportionality to the length of the time interval being simulated, for example, it runs the (very real) risk of exceeding the `coherence time <https://en.wikipedia.org/wiki/Quantum_decoherence>`_ of the system. Ideally, running a simulation for time :math:`t` would require a complexity less than :math:`\mathcal{O}(t)` or, in other words, a complexity that is sublinear in :math:`t`. This, in theory, can be achieved by employing a **fast-forwarding** technique, which is an umbrella term that refers to strategies used to reduce the depth of a simulation circuit below the time step threshold, essentially carrying out a time evolution simulation in less time than the system evolves for.
#
# The specific fast-forwarding technique that should be used dominantly depends on the structure of the Hamiltonian being simulated. If your system has an analytical solution, your life is easy. Fast-forwarding in the analytical case can be carried out by computing the unitary transformation that diagonalizes the system's Hamiltonian and executing exactly using quantum gates. Since the entire evolution of the system is known, the phase angles used in the time-evolution operator (as shown above) can be altered to obtain different time lengths, essentially carrying out simulation with :math:`\mathcal{O}(1)` complexity. This is an incredibly ideal result that applies to a select few, generally not very interesting scenarios, such as a fixed pendulum. 
#
# When more complexity is added and the analytical solution of the system is no longer obtainable, we encounter the *no-fast-forwarding theorem* [#Childs2010]_.
#
# .. admonition:: The No-Fast-Forwarding Theorem, Put Simply
#    :class: tip
#    The time evolution of a generic physical system, often described by a general sparse Hamiltonian, cannot be carried out in sublinear time since there does not exist a generic method for fast-forwarding Hamiltonian simulations.
#
# In the general case, then, we can either accept the resource requirements are at least :math:`\mathcal{O}(t)` or employ an approximate method of fast-forwarding at the cost of some error. One such approximation method is called **Variational Fast Forwarding (VFF)**, in which feedback is used to approximately diagonalize the general Hamiltonian [#Cirstoiu2020]_ before carrying out the fast-forwarding algorithm. In the case of Trotterization, this implies that the unitary describing the Trotter step :math:`e^{-iH\Delta t}` is diagonalized via a gradient descent optimization that targets the diagonalized form. When this is adequately approximated (i.e. an error threshold has been reached), the diagonal unitary :math:`D` can be applied to the system as :math:`WDW^\dagger\approx e^{-iH\Delta t}`, where :math:`W` is an operator required to carry out necessary basis changes. Fast-forwarding can be carried out, again, by altering the phase angles of the operator and executing :math:`WD^N W^\dagger`, where :math:`N` is the number of applications [#Cirstoiu2020]_.
#
# ILLUSTRATION HERE? (Fast forwarding circuit diagram?)
#
# Higher-Order Trotterizations
# ----------------------------
#
# Conclusion
# ----------
#
# .. _references:
#
# References
# ----------
# .. [#Su2020] Y.\ Su, "A Theory of Trotter Error," presented at the Simons Institute for the Theory of Computing, UC Berkeley, Berkeley, CA, USA, Apr. 2020. [Online]. Available: https://simons.berkeley.edu/sites/default/files/docs/15639/trottererrortheorysimons.pdf
#
# .. [#Paetznick2014] A.\ Paetznick and K. M. Svore, "Repeat-Until-Success: Non-deterministic decomposition of single-qubit unitaries," *Quantum Inf. Comput.*, vol. 14, no. 15-16, pp. 1277–1301, Nov. 2014, arXiv: 1311.1074 [quant-ph].
#
# .. [#Childs2010] A.\ M. Childs and R. Kothari, "Limitations on the simulation of non-sparse Hamiltonians," *Quantum Inf. Comput.*, vol. 10, no. 7, pp. 669–684, Jul. 2010, arXiv: `0908.4398 <https://arxiv.org/abs/0908.4398>`_ [quant-ph].
#
# .. [#Cirstoiu2020] C.\ Cîrstoiu, Z. Holmes, J. Iosue, L. Cincio, P. J. Coles, and A. Sornborger, "Variational fast forwarding for quantum simulation beyond the coherence time," *npj Quantum Inf.*, vol. 6, no. 1, p. 82, Sep. 2020, arXiv: `1910.04292 <https://arxiv.org/abs/1910.04292>`_ [quant-ph].
#
# # .. [#Gidney2018] C.\ Gidney, "Halving the cost of quantum addition," *Quantum*, vol. 2, p. 74, Jun. 2018. `doi: 10.22331/q-2018-06-18-74 <https://quantum-journal.org/papers/q-2018-06-18-74/>`_`.
