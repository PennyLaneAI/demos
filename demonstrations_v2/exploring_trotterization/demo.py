r"""
Trotterization
==============

Whether we like it or not, time is always moving forward. 

Even more frustrating, things tend to change with time, meaning we cannot neglect time evolution as we set out to model realistic systems, which tends to be costly. Take, for example, a system of :math:`n` particles, in which there are :math:`2^n` possible configurations. In this case, the Hamiltonian energy matrix of the system used in the time evolution operator :math:`U(t)=e^{-iHt}` would be :math:`2^n \times 2^n`. That's a lot to deal with!

Luckily, quantum computers have shown promise in addressing this issue. Instead of having to represent each possible state contained in the problem's Hilbert space, qubits can themselves act as analogues for the particles that make up the system. This has the potential to enable polynomial scaling with particle count rather than exponential scaling. `Maybe Feynman was onto something <https://s2.smu.edu/~mitch/class/5395/papers/feynman-quantum-1981.pdf>`_! 

To make this feasible, tools have been put forward to aid in the task of `Hamiltonian simulation <https://pennylane.ai/topics/hamiltonian-simulation>`_. This demo will focus on **Trotterization**, a simulation methods that implements time evolution by segmenting and iteratively walking a Hamiltonian forward in time. Together, we will learn why Trotterization is seen as an essential tool in quantum computing by exploring its ability to handle complex, non-commuting Hamiltonian terms and implementing a simple Trotterization on a Pauli Hamiltonian. 

Time is of the essence!

The Commutation Problem
-----------------------
For a completely isolated free particle experiencing no external potential, the system's Hamiltonian can be simply defined in terms of kinetic energy and applied to the system via a unitary gate representing the time evolution operator :math:`U_{free}(t)=e^{-iHt}=e^{-iTt}`, where :math:`T` is the kinetic energy term. If this is the only scenario we can deal with, however, our capabilities would be incredibly limited. 

As a Hamiltonian becomes more complex (e.g., additional terms are added), the feasibility of the time evolution operator :math:`e^{-iHt}` being executable within reasonable computational resource bounds diminishes. If complexity is the issue, why not just split the Hamiltonian into smaller pieces that *are* computationally executable? If we take the representation :math:`H=H_1+H_2+...+H_n`, we could naïvely say that our time evolution operator becomes :math:`e^{-i(H_1+H_2+...+H_n)t}=e^{-iH_1t}e^{-iH_2t}...e^{-iH_nt}`. In our naïveté, however, we would neglect to consider the commutation relations of the split Hamiltonian components. If certain fragments of the Hamiltonian do not commute (:math:`[H_i,H_j] \neq 0`), we cannot exponentiate it in its entirety without deviating from our expected outcome.

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

As we know, most physical systems require a combination of non-commuting operators to be fully described. If we took :math:`A` and :math:`B` to be our non-commuting operators, our naïve approach would lead us to blindly exponentiate as :math:`e^{-iHt}=e^{-iAt}e^{-iBt}`. Applying these operators sequentially assumes that the system is accurately described by iterating the position *then* iterating the momentum, one after the other. This completely ignores how they influence each other, thus rendering the simulation inaccurate. Rats!

Luckily for us, the problem of exponentiating non-commuting equations is not new. The `Lie-Trotter product formula <https://en.wikipedia.org/wiki/Lie_product_formula>`_ is a main result of this effort. The theorem states that for arbitrary :math:`m \times m` matrices :math:`A` and :math:`B`

.. math::
   e^{A+B}=\lim_{r \to \infty}(e^{A/r}e^{B/r})^r.

Turning back to the Hamiltonian picture, if we take a large, finite :math:`r`, letting :math:`r` represent the number of time steps taken in the desired time evolution, we can approximate the Lie-Trotter formula to the first order Trotterization expression

.. math::
   e^{-iHt}=\left(\prod_j e^{-iH_j t/r}\right)^r.

So, by slicing the total time the simulation is trying to emulate, the dependency shared by non-commuting properties can be integrated via alternating applications of each operator within each time step. So, instead of taking one complete :math:`A` step and one complete :math:`B` step, we are now alternating small, partial steps in :math:`A` and :math:`B`, approximating simultaneity to the best of our ability. 

.. figure:: ../demonstrations_v2/exploring_trotterization/IterativeFitIllustration.png
   :align: center
   :width: 700px

By splitting the Hamiltonian into fragments :math:`H_j`, we are generating a new *effective* Hamiltonian that becomes an approximation of the initial Hamiltonian (a true perspective is that :math:`e^{-i(A_B)t}\approx e^{-iAt}e^{-iBt}`). Carrying out the above Trotterization allows us to carry out an *exact* simulation of an *approximate* representation!

Returning to the example of position and momentum also raises the question of dealing with operators that exist in different bases. Indeed, if the non-commuting operators do not exist in the same basis, a basis change step (such as a `quantum Fourier transform (QFT) <https://pennylane.ai/demos/tutorial_qft>`_) will need to be added between each operator application. This increases the resources required to carry out a step, but further opens the door to simulating realistic systems.

Implementing the Trotter Method
-------------------------------
The form of the Lie-Trotter approximation implies a clear order of operations that must be executed to simulate the time evolution of a non-commuting system. 

The first step of carrying out Hamiltonian simulation using Trotterization is splitting the Hamiltonian. As a rule of thumb, feasible algorithms should strive for a minimal gate count and minimized opportunity for error, meaning we should try to minimize the number of operators used. To achieve this, the Hamiltonian should be split only where mathematically necessary, meaning commuting terms should always be grouped together to the greatest extent possible. 

Take, for example, a Hamiltonian containing the terms :math:`Z_1Z_3`, :math:`Z_2Z_4`, and :math:`X_1X_2`, where :math:`Z_i` and :math:`X_j` are Pauli operators. The first and second terms will always commute since they consist completely of Z operators, but neither of the first two operators commute with the third operator. Therefore, the most optimal splitting would yield :math:`H_1=Z_1Z_3+Z_2Z_4` and :math:`H_2=X_1X_2`. This is not always so obvious in complex physical systems, so thorough analysis should be carried out to ensure optimal operator pairing. Tools such as :func:`~qp.pauli.group_observables` are very helpful in this case!

Once the Hamiltonian is appropriately split, we need to ensure that the fragments we are working with can actually be implemented using realistic quantum devices. This tends to require some kind of :doc:`transformation <demos/tutorial_mapping>` between the native representation of the system and a computationally compatible representation. Recalling the relevance of this method in simulating particle systems, one relevant example is the mapping of Fermionic states to qubit states via the :func:`~qp.jordan_wigner` transformation. For the purposes of the following simple demonstration, we will assume our Hamiltonian was originally defined in the Pauli basis and forgo the transformation step. 

For this demo, let our Hamiltonian be given by

.. math::
   H=\alpha X + \beta Z, 
   
where :math:`\alpha` and :math:`\beta` are arbitrary coefficients. 

Finally, the number of time steps :math:`r` must be defined with the goal of achieving a high level of accuracy while remaining within reasonable computational resource bounds. From there, the circuit should simply alternate between the time evolution operator unitary gates for the desired number of steps. Since our Hamiltonian is completely in the Pauli basis, we will not need to perform a basis change step in our Trotter circuit. Recall that exponentiated Pauli operators are represented by rotation gates, where, letting :math:`\sigma_i` be a Pauli operator,

.. math::
   R_{\sigma_i}(\theta)=e^{-i\frac{\theta}{2}\sigma_i}.

So, taking :math:`H_1=\alpha X` and :math:`H_2=\beta Z`

.. math::
   U(t)=e^{-i \beta Z t}e^{-i \alpha X t}=R_Z(2\beta t)R_X(2 \alpha t).

This is easily translatable to code.
"""

import pennylane as qp
import numpy as np
from scipy.linalg import expm
import pennylane.estimator as qre
from pennylane.transforms.rz_phase_gradient import rz_phase_gradient

#Define System Parameters
coeffs = [0.2, 1.3] #Define Hamiltonian coefficients [alpha, beta]
observables = [qp.PauliX(0), qp.PauliZ(0)] #Define observables
t = 10
R = [10, 20, 30, 40, 50, 100, 200, 400] #Trotter steps

dev = qp.device("lightning.qubit", wires=1)

#Define Trotterization Function
@qp.qnode(dev)
def TrotterStepper(t,r,coeffs):
    del_t = t/r

    #Apply the rotation r times
    for i in range(r):
        qp.RX(2*coeffs[0]*del_t, wires=0)
        qp.RZ(2*coeffs[1]*del_t, wires=0)

    return [qp.expval(qp.PauliX(0)), qp.expval(qp.PauliY(0)), qp.expval(qp.PauliZ(0))]
###############################################################################
# Luckily for us, PennyLane has the tools to make this much simpler. Using :func:`~qp.Trotterize`, the exact same procedure can be carried out on the target Hamiltonian.
#
def first_order_expansion(time, theta, phi, wires):
    qp.RX(2*time*theta, wires = 0)
    qp.RZ(2*time*phi, wires = 0)

@qp.qnode(dev)
def BuiltInTrotter(time, theta, phi, num_trotter_steps):
    qp.trotterize(
        first_order_expansion, 
        n = num_trotter_steps,
        order = 1
    )(time, theta, phi, wires=['a','b'])
    return [qp.expval(qp.PauliX(0)), qp.expval(qp.PauliY(0)), qp.expval(qp.PauliZ(0))]

#Compare results for 50 Trotter steps
print(TrotterStepper(t, R[4], coeffs))
print(BuiltInTrotter(t, coeffs[0], coeffs[1], R[4]))

###############################################################################
# That's a good match! This procedure is advantageous for dealing with larger Hamiltonians or working Trotterization into a larger PennyLane workflow. For the purposes of this demonstration, we will continue using our DIY solution, but the choice is yours!
#
# Trotter Error
# -------------
# Understanding how Hamiltonian simulations deviate from theoretically expected system behaviour is a field of study in itself. Intuitively, we expect the size of the time step to be a major contributor to the error, since time steps that are too large tend to obscure system behaviour. A nuanced exploration of exact Trotter error typically begins with consideration of the `Baker-Campbell-Hausdorff formula <https://en.wikipedia.org/wiki/Baker%E2%80%93Campbell%E2%80%93Hausdorff_formula>`_ [#Su2020]_, which describes the true expansion of an exponentiated non-commuting operator pair as
#
# .. math::
#     e^{-iBt}e^{-iAt}=e^{-it(A+B)-\frac{t^2}{2}[B,A]+i\frac{t^3}{12}[B,[B,A]]-...}.
#
# As is familiar when handling series expansions, the degree to which the BCH formula is truncated in the system representation dictates the amount of error to expect in the Hamiltonian. Comparing the previous definition of the approximated Trotter formula to this expansion expression, we can see that we are only concerned with the first term and, therefore, our error is dominated by the second-order term :math:`-\frac{t^2}{2}[B,A]`. Taking :math:`t=\Delta t=\frac{t}{r}` for each time step, we can reason that, in this case, the error is proportional to :math:`\frac{t^2}{r}` after the operator is applied :math:`r` times. This result aligns with our intuition, implying that error will increase with length of time and decrease with number of steps. This, of course, neglects to consider the upper bound on the number of time steps that can be handled computationally, which will be examined briefly later.
#
# In the simple example implemented in this demo, an observed Trotter error can be achieved by calculating time evolution of the system analytically and carrying out a comparison. This would, of course, be unfeasible for large, complicated (in other words, useful) models or long time scales, but it is sufficient for this example. `Methods for investigating Trotter error in non-analytical cases <https://simons.berkeley.edu/sites/default/files/docs/15639/trottererrortheorysimons.pdf>`_ are a continually developing field of study.
#

#Approximate evolution for Trotter error calculation
#Pauli matrix representations
X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])
Y = np.array([[0, -1j], [1j, 0]])

H = coeffs[0]*X + coeffs[1]*Z
U = expm(-1j*H*t)

state_0 = np.array([1,0])

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
# For feasible implementation on quantum systems, arbitrary rotation gates (or any non-Clifford gate in general) must be decomposed into a series of `Clifford+T gates <https://pennylane.ai/compilation/clifford-t-gate-set>`_ via some method of `gate synthesis <https://pennylane.ai/compilation/two-qubit-synthesis>`_. The PennyLane `estimate() <https://docs.pennylane.ai/en/stable/code/api/pennylane.estimator.estimate.estimate.html>`_ tool can be implemented to approximate how many gates are used to implement the synthesized rotation in a naïve approach. Alternatively, the `multiplexed phase gradient method <https://pennylane.ai/demos/efficient_rotations_with_phase_gradient_states>`_, which takes advantage of a static register that holds spatially dependent phase values which can be *added* to a target state as needed, can be used for synthesis. This method has a cost of :math:`\mathcal{O}\log_2(1/\epsilon)+\frac{4R}{N}\log_2(1/\epsilon)`, where :math:`N` is the number of states, :math:`\epsilon` is the target accuracy, and :math:`R` is the number of rotations.
#
# When we discuss selecting a gate synthesis method, our initial instinct might be to minimize the T count in favour of implementability and efficiency. From this perspective, it seems obvious to always select the method with a low cost, such as the phase gradient approach. Using PennyLane's `rz_phase_gradient() <https://docs.pennylane.ai/en/stable/code/api/pennylane.transforms.rz_phase_gradient.html>`_ transformation, the expensive :math:`R_z(\phi)` rotations implemented in the above Trotterization can be translated into phase gradient additions and compared to the naïve approach. Since this method only transforms :math:`R_z` operations, we can perform our desired :math:`R_x` rotations using an :math:`R_z` rotation sandwiched between two Hadamard gates. For this example, a step count of :math:`r=200` will be taken in the resource estimation step.

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
# As expected, the T count is much lower in the phase gradient method, even when adding in the Toffoli gate decomposition taking 1 Toffoli = 4 T gates [#Gidney2018]_. This, however, is not a definitive affirmation that the phase gradient method is the ideal gate synthesis strategy in this case. Let us compare the Trotter error in the two methods before we jump to any conclusions.

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
# Ah ha! A tradeoff has made itself clear! In the phase gradient implementation, the Trotter error is universally higher.  What is also (maybe even more) interesting is that, after a certain :math:`r` threshold is reached, the phase gradient system no longer evolves. This is an example of `underflow <https://en.wikipedia.org/wiki/Arithmetic_underflow>`_ in `quantum arithmetic <https://pennylane.ai/demos/tutorial_how_to_use_quantum_arithmetic_operators>`_, where, put very simply, the simulation has reached a computational limit and is stuck rounding to the same value each pass. Since the phase gradient method relies on quantum addition, it may not be the right tool here if our main goal is to reduce error. So, when we choose which techniques to use for gate synthesis, we must consider the needs of our system in tandem with the cost of our system. Sometimes an investment is necessary!
#
# Another thing to consider is the potential **fast-forwardability** of the system. As shown, carrying out quantum simulation on hardware requires a series of gates to be implemented for each time step, meaning the depth of the circuit grows notably with increasing time. If the depth of a simulation circuit maintains proportionality to the length of the time interval being simulated, for example, it runs the (very real) risk of exceeding the `coherence time <https://en.wikipedia.org/wiki/Quantum_decoherence>`_ of the system. Ideally, running a simulation for time :math:`t` would require a complexity less than :math:`\mathcal{O}(t)` or, in other words, a complexity that is sublinear in :math:`t`. This, in theory, can be achieved by employing a fast-forwarding technique, in which the phase angles used in the time evolution operator can be strategically altered to, put simply, cover more time in a simple step to achieve sub-linear complexity. This technique requires the Hamiltonian have a known analytical solution (or, in other words, that it is "exactly integrable"), which is often not the case.
#
# Higher-Order Trotterizations
# ----------------------------
# To this point we have taken the first-order Trotterization approach, in which the root equation is truncated to the first order term. While this approach is sufficient for simple, short time scale problems (ex. systems with weak interaction), ignoring higher order terms and, therefore, reducing precision can result in ignorance of important system characteristics and unnecessarily high error that requires increased resources to account for [#Childs2021]_. Thus, the use of higher-order Trotterizations is often necessary to achieve realistic simulation.
#
# Higher-order Trotter products and the associated Trotter error is a complex and developing field of study. A simple, baseline approach to achieving high-order Trotterizations is introducing symmetries into the system that move the simulation closer to reality. Revisiting the analogy used above, the first-order Trotterization alternates between two non-commuting operators and incrementally steps each term in alternating time slices, resulting in an approximate interaction. In a second-order approach, one of the operator terms would be further divided into two half-steps to be applied before and after the other operator. In the case of our Hamiltonian, the second-order Trotter product, discovered simultaneously by Strang [#Strang1968]_ and Verlet [#Verlet1967]_, would be
# 
# .. math::
#    e^{-i\alpha Xt}e^{-i\beta Zt}=(e^{-i\alpha Xt/2r}e^{-i\beta Zt/r}e^{-i\alpha Xt/2r})
#
# which can be implemented as
#
# .. math::
#    S_2(t)=R_X(\alpha t)R_Z(2\beta t)R_X(\alpha t).
#
# A benefit of imposing a symmetric formula such as this is that it cancels out dominant error term discussed previously. The symmetry, essentially, does not allow for even powers to survive the operator application process, eliminating the dominant :math:`t^2` term discussed previously.
#
# Altering the TrotterStepper() function to a second-order Trotterization shows the impact this symmetry has on the Trotter error.

@qp.qnode(dev)
def TrotterStepperSO(t,r,coeffs):
    del_t = t/r

    #Apply the rotation r times
    for i in range(r):
        U_A_half = qp.RX(coeffs[0]*del_t, wires=0)
        U_B = qp.RZ(2*coeffs[1]*del_t, wires=0)
        U_A_half = qp.RX(coeffs[0]*del_t, wires=0)

    return [qp.expval(qp.PauliX(0)), qp.expval(qp.PauliY(0)), qp.expval(qp.PauliZ(0))]

print(f"{'r':>5} | {'X error':>8} | {'Y error':>8} | {'Z error':>8} | {'Total error':>11}")
print("-" * 51)

for r in R:
    resultSO = TrotterStepperSO(t,r,coeffs)
    X_errorSO = abs(resultSO[0]-exact_exp_X)
    Y_errorSO = abs(resultSO[1]-exact_exp_Y)
    Z_errorSO = abs(resultSO[2]-exact_exp_Z)
    total_errorSO = np.sqrt(X_errorSO**2+Y_errorSO**2+Z_errorSO**2)
    print(f"{r:>5} | {X_errorSO:>8.5f} | {Y_errorSO:>8.5f} | {Z_errorSO:>8.5f} | {total_errorSO:>11.5f}")
#############################################################################################################
# Comparing to the first-order results where, for example, the error when :math:`r=10` is approximately 20%, updating to the second-order Trotterization reduces this error to approximately 12%. Not too shabby! The following plot demonstrates the improved performance of the second-order Trotterization, in which low error rates are achieved for much fewer time steps. Thus, even though additional resources are required to implement the additional unitary, the reduction in required steps also implies improved cost.
#
# .. figure:: ../demonstrations_v2/exploring_trotterization/HigherOrderComp.png
#   :align: center
#   :width: 700px
#
# Beyond second-order, higher-order Trotterizations can be achieved via the nested application of the second-order Trotterization sequence. A well known example is the Suzuki five-step ladder [#Suzuki1990]_, which achieves a fourth-order implementation represented as
#
# .. math::
#    S_4(t)=S_2(s_1 t)S_2(s_1 t)S_2((1-4s_1)t)S_2(s_1 t)S_2(s_1 t).
#
# Where :math:`s_1=(4-4^{1/3})^{-1}` is the first-order suzuki constant. It should be noted that this is not the only method of achieving higher-order representations and that various methods exist with the goal of reducing Trotter error, minizing gate count, and achiving high accuracy. To further optimize, some approaches allow the selective application of high-order Trotter products to dominant terms in a simulation, allowing for resources to be allocated to only the instances they are justified. 
#
# Conclusion
# ----------
# There is a phenomenon that is renewed over and over again in which the field of mathematics is continually years ahead of physics (especially applied physics). The use of product formulas as tools for time evolution in Hamiltonian simulation is a beautiful example of this, in which a purely mathematical description of how to exponentiate non-commuting operators has become a defining method for time-evolution simulation in today's quantum pursuits. Understanding the basics of how Trotter products can be used and optimized for various applications of quantum simulation opens the door to not only increased utility but a heightened awareness of how the input of many fields is required to achieve viable outcomes. Keep calm and Trotter on!
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
# .. [#Gidney2018] C.\ Gidney, "Halving the cost of quantum addition," *Quantum*, vol. 2, p. 74, Jun. 2018. `doi: 10.22331/q-2018-06-18-74 <https://quantum-journal.org/papers/q-2018-06-18-74/>`_.
#
# .. [#Childs2021] A.\ M. Childs, Y. Su, M. C. Tran, N. Wiebe, and S. Zhu, "Theory of Trotter Error with Commutator Scaling," *Phys. Rev. X*, vol. 11, no. 1, Feb. 2021, doi: `10.1103/PhysRevX.11.011020 <https://doi.org/10.1103/PhysRevX.11.011020>`_.
#
# .. [#Strang1968] G.\ Strang, "On the Construction and Comparison of Difference Schemes," *SIAM Journal on Numerical Analysis*, vol. 5, no. 3, Sept. 1968, doi: `10.1137/0705041 <https://doi.org/10.1137/0705041>`_.
#
# .. [#Verlet1967] L.\ Verlet, "Computer "Experiments" on Classical Fluids. I. Thermodynamical Properties of Lennard-Jones Molecules," *Phys. Rev.*, vol. 159, no. 1, pp. 98-103, Jul. 1967, doi: `10.1103/PhysRev.159.98 <https://doi.org/10.1103/PhysRev.159.98>`_.
#
# .. [#Suzuki1990] M.\ Suzuki, "Fractal decomposition of exponential operators to many-body theories and Monte Carlo simulations," *Phys. Lett. A*, vol. 146, no. 6, pp. 319-323, Jun. 1990, doi:`10.1016/0375-9601(90)90962-N <https://doi.org/10.1016/0375-9601(90)90962-N>`_.