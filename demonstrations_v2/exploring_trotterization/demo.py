r"""
Trotterization
==============

Whether we like it or not, time is always moving forward. 

Even more frustrating, things tend to change with time, which we cannot neglect as we set out to model realistic systems. Take, for example, a system of :math:`n` particles, in which there are :math:`2^n` possible configurations. In this case, the Hamiltonian energy matrix of the system used in the time evolution operator :math:`U(t)=e^{-iHt}` would be :math:`2^n \times 2^n`. Things get unaffordable very quickly!

Luckily, quantum computers have shown promise in addressing this issue. Instead of having to represent each possible state contained in the problem's Hilbert space, qubits can themselves act as analogues for the particles that make up the system. This has the potential to enable polynomial scaling with particle count rather than exponential scaling. `Maybe Feynman was onto something <https://s2.smu.edu/~mitch/class/5395/papers/feynman-quantum-1981.pdf>`_! 

To make this feasible, tools have been put forward to aid in the task of `Hamiltonian simulation <https://pennylane.ai/topics/hamiltonian-simulation>`_. This demo will focus on **Trotterization**, a simulation methods that implements time evolution by segmenting and iteratively walking a Hamiltonian forward in time. `Recent developments <https://arxiv.org/abs/2606.30741>`_ have been pushing this method further into the spotlight. Together, we will explore why Trotterization is an essential tool for quantum algorithms, work through a simple implementation to various orders, and determine what trade-offs may be necessary in the fault tolerant picture. 

Let's get started, time is of the essence!

The Commutation Problem
-----------------------
For a completely isolated free particle experiencing no external potential, the system's Hamiltonian can be simply defined in terms of kinetic energy and applied via a unitary gate representing the time evolution operator :math:`U_{free}(t)=e^{-iHt}=e^{-iTt}`, where :math:`T` is the kinetic energy term. If this is the only scenario we can deal with, however, our capabilities would be incredibly limited. As a Hamiltonian becomes more complex (e.g., additional terms are added), the chance that the time evolution operator :math:`e^{-iHt}` can be implemented with available gates (determined by a process of calculation and decomposiiton into a relevant gate set) and computational resources diminishes. 

If runnability is the issue, why not just split the Hamiltonian into smaller pieces that *are* executable and hardware compatible? If we take the representation :math:`H=H_1+H_2+...+H_n`, where :math:`H_n` is an implementable Hamiltonian fragment, we could naïvely say that our time evolution operator becomes :math:`e^{-i(H_1+H_2+...+H_n)t}=e^{-iH_1t}e^{-iH_2t}...e^{-iH_nt}`. In our naïveté, however, we would neglect to consider the commutation relations of the split Hamiltonian components. If certain fragments of the Hamiltonian do not commute (:math:`[H_i,H_j] \neq 0`), then this product would merely be an approximation, resulting in systemic error and inaccurate outputs. 

.. admonition:: Recall
   :class: note

   For commuting matrices, where :math:`AB=BA` and it is implied that :math:`A` and :math:`B` are completely independent,

   |

   .. math::
      (A+B)^2 = A^2+2AB+B^2.

   |

   For non-commuting matrices, where :math:`AB \neq BA` and it is implied that :math:`A` and :math:`B` have some level of dependency,

   |

   .. math::
      (A+B)^2 = A^2+AB+BA+B^2.

   |

   So, the Taylor expansion :math:`e^{A+B}=I+(A+B)+\frac{1}{2}(A+B)^2+...` differs in the two cases!

As we know, most physical systems require a combination of non-commuting operators to be fully described. If we took :math:`A` and :math:`B` to be our non-commuting operators, our naïve approach would lead us to blindly exponentiate as :math:`e^{-iHt}=e^{-iAt}e^{-iBt}`. Applying these operators sequentially assumes that the system is accurately described by iterating the position *then* iterating the momentum, one after the other. This completely ignores how they influence each other, thus rendering the simulation inaccurate. Rats!

Luckily for us, the problem of exponentiating non-commuting equations is not new. The `Lie-Trotter product formula <https://en.wikipedia.org/wiki/Lie_product_formula>`_ is a main result of this effort. The theorem states that for arbitrary :math:`m \times m` matrices :math:`A` and :math:`B`

.. math::
   e^{A+B}=\lim_{r \to \infty}(e^{A/r}e^{B/r})^r.

Turning back to the Hamiltonian picture, if we take a large, finite :math:`r`, letting :math:`r` represent the number of time steps taken in the desired time evolution, we can approximate the Lie-Trotter formula to the first order Trotterization expression

.. math::
   e^{-iHt}=\left(\prod_j e^{-iH_j t/r}\right)^r.

So, by slicing the total time the simulation is trying to emulate, the dependency shared by non-commuting properties can be integrated via alternating applications of each operator within each time step. Instead of taking one complete :math:`A` step and one complete :math:`B` step, we are now alternating small, partial steps in :math:`A` and :math:`B`, approximating simultaneity to the best of our ability. 

.. figure:: ../demonstrations_v2/exploring_trotterization/IterativeFitIllustration.png
   :align: center
   :width: 700px

By splitting the Hamiltonian into fragments :math:`H_j`, we are generating a new *effective* Hamiltonian that becomes an approximation of the initial Hamiltonian (a true perspective is that :math:`e^{-i(A_B)t}\approx e^{-iAt}e^{-iBt}`). Carrying out the above Trotterization allows us to carry out an *exact* simulation of an *approximate* representation!

Implementing the Trotter Method
-------------------------------
The form of the Lie-Trotter approximation implies a clear order of operations that must be executed to simulate the time evolution of a non-commuting system. 

The first step of carrying out Hamiltonian simulation using Trotterization is splitting the Hamiltonian. As a rule of thumb, feasible algorithms should strive for a minimal gate count and reduced opportunity for error, meaning we should try to minimize the number of operators required. To achieve this, the Hamiltonian should be split only where mathematically necessary, meaning commuting terms should always be grouped together to the greatest extent possible. 

Once the Hamiltonian is appropriately split, we need to ensure that the fragments we are working with can actually be implemented using realistic quantum devices. This tends to require some kind of :doc:`transformation <demos/tutorial_mapping>` between the native representation of the system and a computationally compatible representation. Recalling the relevance of this method in simulating particle systems, one relevant example is the mapping of Fermionic states to qubit states via the :func:`~qp.jordan_wigner` transformation. For the purposes of the following simple demonstration, we will assume our Hamiltonian was originally defined in the Pauli basis and forgo the transformation step. 

Let our Hamiltonian be given by

.. math::
   H=\alpha X + \beta Z, 
   
where :math:`\alpha` and :math:`\beta` are arbitrary coefficients. 

Recall that exponentiated Pauli operators are represented by rotation gates, where, letting :math:`\sigma_i` be a Pauli operator,

.. math::
   R_{\sigma_i}(\theta)=e^{-i\frac{\theta}{2}\sigma_i}.

So, taking :math:`H_1=\alpha X` and :math:`H_2=\beta Z`

.. math::
   U(t)=e^{-i \beta Z t}e^{-i \alpha X t}=R_Z(2\beta t)R_X(2 \alpha t).

Finally, the number of time steps :math:`r` must be defined with the goal of achieving a high level of accuracy while remaining within reasonable computational resource bounds. From there, the circuit should simply alternate between the time evolution operator unitary gates for the desired number of steps.

This is easily translated to code.
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
def TrotterStepper(t, coeffs, r):
    del_t = t/r

    #Apply the rotation r times
    for i in range(r):
        qp.RX(2*coeffs[0]*del_t, wires=0)
        qp.RZ(2*coeffs[1]*del_t, wires=0)

    return [qp.expval(qp.PauliX(0)), qp.expval(qp.PauliY(0)), qp.expval(qp.PauliZ(0))]

print(TrotterStepper(t, coeffs, R[4]))
###############################################################################
# Luckily for us, PennyLane has the tools to make this much simpler. Using :func:`~pennylane.Trotterize`, the exact same procedure can be carried out on the target Hamiltonian.
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
print(BuiltInTrotter(t, coeffs[0], coeffs[1], R[4]))

###############################################################################
# That's a good match! This procedure is advantageous for dealing with larger Hamiltonians or working Trotterization into a larger PennyLane workflow. For the purposes of this demonstration, we will continue using our DIY solution, but the choice is yours!
#
# For this simple Hamiltonian, fragmentation was handled by definition. In more complex scenarios, fragmentation must be carried out in its own step. Take, for example, a Hamiltonian containing the terms :math:`Z_1Z_3`, :math:`Z_2Z_4`, and :math:`X_1X_2`, where :math:`Z_i` and :math:`X_j` are Pauli operators. The first and second terms will always commute since they consist completely of Z operators, but neither of the first two operators commute with the third operator. Therefore, the most optimal splitting would yield :math:`H_1=Z_1Z_3+Z_2Z_4` and :math:`H_2=X_1X_2`. This is often not obvious in complex physical systems, so thorough analysis should be carried out to ensure optimal operator pairing. Tools such as :func:`~pennylane.pauli.group_observables` are very helpful in this case!
#
# Trotter Error
# -------------
# Understanding how Hamiltonian simulations deviate from theoretically expected system behaviour is a field of study in itself. Intuitively, we expect the size of the time step to be a major contributor to the error, since time steps that are too large tend to obscure system behaviour. A nuanced exploration of exact Trotter error typically begins with consideration of the `Baker-Campbell-Hausdorff formula <https://en.wikipedia.org/wiki/Baker%E2%80%93Campbell%E2%80%93Hausdorff_formula>`_ [#Su2020]_, which describes the true expansion of an exponentiated non-commuting operator pair as
#
# .. math::
#     e^{-iBt}e^{-iAt}=e^{-it(A+B)-\frac{t^2}{2}[B,A]+i\frac{t^3}{12}[B,[B,A]]-...}.
#
# As is familiar when handling series expansions, the degree to which the BCH formula is truncated in the system representation dictates the amount of error to expect in the Hamiltonian. Comparing the previous definition of the approximated Trotter formula to this expansion expression, we can see that we are only concerned with the first term and, therefore, our error is dominated by the second-order term :math:`-\frac{t^2}{2}[B,A]`. Taking :math:`t=\Delta t=\frac{t}{r}` for each time step, we can reason that, in this case, the error is proportional to :math:`\frac{t^2}{r}` after the operator is applied :math:`r` times.
#
# In the simple example implemented in this demo, an observed Trotter error can be achieved by calculating time evolution of the system analytically and carrying out a comparison. This would, of course, be unfeasible for large, complicated (in other words, useful) models or long time scales, but it is sufficient for this example. `Methods for investigating Trotter error <https://arxiv.org/pdf/2606.30738>`_ are a continually developing field of study.
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
    result = TrotterStepper(t,coeffs,r)
    X_error = abs(result[0]-exact_exp_X)
    Y_error = abs(result[1]-exact_exp_Y)
    Z_error = abs(result[2]-exact_exp_Z)
    total_error = np.sqrt(X_error**2+Y_error**2+Z_error**2)
    print(f"{r:>5} | {X_error:>8.5f} | {Y_error:>8.5f} | {Z_error:>8.5f} | {total_error:>11.5f}")
###############################################################################
# As expected, the simulation error decreases with increasing time steps! 
# 
# Higher-Order Trotterizations
# ----------------------------
# Not too long ago we mentioned that we were working with the BCH formula truncated to the first order term. In other words, we have been taking a first-order Trotterization approach. While this is sufficient for simple, short time scale problems (e.g., systems with weak interaction), ignoring higher order terms and, therefore, reducing precision can result in ignorance of important system characteristics and unnecessarily high error that requires increased resources to account for [#Childs2021]_. Thus, the use of higher-order Trotterizations is often necessary to achieve realistic simulation.
#
# A simple, baseline approach to achieving high-order Trotterizations is introducing symmetries into the system that move the simulation closer to reality. Revisiting the analogy used above, the first-order Trotterization alternates between two non-commuting operators and incrementally steps each term in alternating time slices. In a second-order approach, one of the operator terms would be further divided into two half-steps to be applied before and after the other operator. In the case of our Hamiltonian, the second-order Trotter product, discovered simultaneously by Strang [#Strang1968]_ and Verlet [#Verlet1967]_ in the 1960s, would be
# 
# .. math::
#    e^{-i\alpha Xt}e^{-i\beta Zt}=(e^{-i\alpha Xt/2r}e^{-i\beta Zt/r}e^{-i\alpha Xt/2r})
#
# which can be implemented as
#
# .. math::
#    S_2(t)=R_X(\alpha t)R_Z(2\beta t)R_X(\alpha t).
#
# A benefit of imposing a symmetric formula such as this is that it cancels out the dominant error term discussed previously. The symmetry, essentially, does not allow for even powers to survive the operator application process, eliminating the dominant :math:`t^2` term discussed previously.
#
# Altering the ``TrotterStepper()`` function to a second-order Trotterization shows the impact this symmetry has on the Trotter error.

@qp.qnode(dev)
def TrotterStepperSO(t,coeffs,r):
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
    resultSO = TrotterStepperSO(t,coeffs,r)
    X_errorSO = abs(resultSO[0]-exact_exp_X)
    Y_errorSO = abs(resultSO[1]-exact_exp_Y)
    Z_errorSO = abs(resultSO[2]-exact_exp_Z)
    total_errorSO = np.sqrt(X_errorSO**2+Y_errorSO**2+Z_errorSO**2)
    print(f"{r:>5} | {X_errorSO:>8.5f} | {Y_errorSO:>8.5f} | {Z_errorSO:>8.5f} | {total_errorSO:>11.5f}")
#############################################################################################################
# Comparing to the first-order results where, for example, the error when :math:`r=10` is approximately 20%, updating to the second-order Trotterization reduces this error to approximately 12%. Not too shabby! 
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
# Gate Synthesis Considerations
# -----------------------------
# Before we can officially add Trotterization to our list of capabilities, we must consider how our simulations will interface with eventual fault tolerant quantum hardware. Whether we like it or not, the resources provided by quantum hardware will be, in some way, limited. With this in mind, it is important that the resource requirements of the systems we work with are quantified and evaluated. One metric of interest is the number of `T-gates <https://pennylane.ai/blog/2025/01/optimizing-with-op-t-mize-dataset>`_ required to synthesize our operations.
#
# The PennyLane :func:`~pennylane.estimate` tool can be implemented to approximate the number of gates used to carry out a given process, taking a naïve approach. There are ample optimized methods that can be used to reduce resource requirements. Alternatively, the :doc:`multiplexed phase gradient method <demos/efficient_rotations_with_phase_gradient_states`, which takes advantage of a static register that holds spatially dependent phase values which can be *added* to a target state as needed, can be used for synthesis.
#
# When we select a gate synthesis method, our initial instinct might be to minimize the T count in favour of implementability and efficiency above all. From this perspective, it seems obvious to always select a low-cost method, such as the aforementioned phase gradient approach. Using PennyLane's :func:`~qp.rz_phase_gradient` transformation, the expensive :math:`R_z(\phi)` rotations implemented in the above Trotterization can be translated into phase gradient additions and compared to the naïve approach. Since this method only transforms :math:`R_z` operations, we can perform our desired :math:`R_x` rotations using an :math:`R_z` rotation sandwiched between two Hadamard gates. For this example, a step count of :math:`r=200` will be taken in the resource estimation step.

#Convert to phase gradient approach
prec = 0.05
b = int(np.ceil(np.log2(1/prec)))

angle_wires = list(range(1,1+b))
gradient_wires = list(range(1+b,1+2*b))
work_wires = list(range(1+2*b,1+3*b))

dev2 = qp.device("lightning.qubit", wires=(1+3*b))

@qp.qnode(dev2)
@rz_phase_gradient(angle_wires,gradient_wires,work_wires)
def TrotterStepperPG(t,coeffs,r):
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
Trotter_resources = qre.estimate(TrotterStepper)(t,coeffs,test_r) #Naïve approach
Trotter_resources_PG = qre.estimate(TrotterStepperPG)(t,coeffs,test_r) #Phase gradient approach
###############################################################################
# So, in the case of the naïve approach, the estimated requirements are:
print(Trotter_resources)
###############################################################################
# In the case of the phase gradient method, the estimated requirements are:
print(Trotter_resources_PG)
###############################################################################
# As expected, the T count is much lower in the phase gradient method, even when adding in the Toffoli gate decomposition taking 1 Toffoli = 4 T gates [#Gidney2018]_. Things are looking good, but let us compare the Trotter error in the two methods before we jump to any conclusions.
#
# Recalling our naïve approach accuracy,
print(f"{'r':>5} | {'X error':>8} | {'Y error':>8} | {'Z error':>8} | {'Total error':>11}")
print("-" * 51)

for r in R:
    result = TrotterStepper(t,coeffs,r)
    X_error = abs(result[0]-exact_exp_X)
    Y_error = abs(result[1]-exact_exp_Y)
    Z_error = abs(result[2]-exact_exp_Z)
    total_error = np.sqrt(X_error**2+Y_error**2+Z_error**2)
    print(f"{r:>5} | {X_error:>8.5f} | {Y_error:>8.5f} | {Z_error:>8.5f} | {total_error:>11.5f}")
###############################################################################
# Using the same comparison to the analytical solution, we can estimate the error in the phase gradient approach.

print(f"{'r':>5} | {'X error':>8} | {'Y error':>8} | {'Z error':>8} | {'Total error':>11}")
print("-" * 51)

for r in R:
    resultPG = TrotterStepperPG(t,coeffs,r)
    X_error_PG = abs(resultPG[0]-exact_exp_X)
    Y_error_PG = abs(resultPG[1]-exact_exp_Y)
    Z_error_PG = abs(resultPG[2]-exact_exp_Z)
    total_error_PG = np.sqrt(X_error_PG**2+Y_error_PG**2+Z_error_PG**2)
    print(f"{r:>5} | {X_error_PG:>8.5f} | {Y_error_PG:>8.5f} | {Z_error_PG:>8.5f} | {total_error_PG:>11.5f}")
#############################################################################################################
# Ah ha! A tradeoff has made itself clear! In the phase gradient implementation, the Trotter error is universally higher. What is also (maybe even more) interesting is that, after a certain :math:`r` threshold is reached, the phase gradient system no longer evolves. This is an example of `underflow <https://en.wikipedia.org/wiki/Arithmetic_underflow>`_ in `quantum arithmetic <https://pennylane.ai/demos/tutorial_how_to_use_quantum_arithmetic_operators>`_, where, put very simply, the simulation has  reached a computational limit and is stuck rounding to the same value each pass. Since the phase gradient method relies on quantum addition, it may not be the right tool here if our main goal is to reduce error. So, when we choose which techniques to use for gate synthesis, we must consider the needs of our system in tandem with the cost of our system. Sometimes an investment is necessary!
#
# Another thing to consider is the potential **fast-forwardability** of the system. As shown, carrying out quantum simulation on hardware requires a series of gates to be implemented for each time step, meaning the depth of the circuit grows notably with increasing time. If the depth of a simulation circuit maintains proportionality to the length of the time interval being simulated, for example, it runs the (very real) risk of exceeding the `coherence time <https://en.wikipedia.org/wiki/Quantum_decoherence>`_ of the system. Ideally, running a simulation for time :math:`t` would require a complexity less than :math:`\mathcal{O}(t)` or, in other words, a complexity that is sublinear in :math:`t`. This, in theory, can be achieved by employing a fast-forwarding technique, in which the phase angles used in the time evolution operator can be strategically altered to, put simply, cover more time in a simple step to achieve sub-linear complexity. This technique requires the Hamiltonian have a known analytical solution (or, in other words, that it is "exactly integrable"), which is often not the case.
#
# Conclusion
# ----------
# There is a phenomenon that is renewed over and over again in which the field of mathematics is continually years ahead of physics (especially applied physics). The use of product formulas as tools for time evolution in Hamiltonian simulation is a beautiful example of this, in which a purely mathematical description of how to exponentiate non-commuting operators has become a defining method for time-evolution simulation in today's quantum pursuits. Understanding the basics of how Trotter products can be used and optimized for various applications of quantum simulation opens the door to not only increased utility but a heightened awareness of how the input of many fields is required to achieve viable outcomes. If you are, wisely, looking to add Trotterization to your quantum skill set, check out our `Codebook chapter on product formulas <https://pennylane.ai/codebook/hamiltonian-simulation/trotterization>`_. Keep calm and Trotter on!
#
# .. _references:
#
# References
# ----------
# .. [#Su2020] Y.\ Su, "A Theory of Trotter Error," presented at the Simons Institute for the Theory of Computing, UC Berkeley, Berkeley, CA, USA, Apr. 2020. [Online]. Available: https://simons.berkeley.edu/sites/default/files/docs/15639/trottererrortheorysimons.pdf.
#
# .. [#Paetznick2014] A.\ Paetznick and K. M. Svore, "Repeat-Until-Success: Non-deterministic decomposition of single-qubit unitaries," *Quantum Inf. Comput.*, vol. 14, no. 15-16, pp. 1277–1301, Nov. 2014, doi: `10.48550/arXiv.1311.1074 <https://doi.org/10.48550/arXiv.1311.1074>`_.
#
# .. [#Childs2010] A.\ M. Childs and R. Kothari, "Limitations on the simulation of non-sparse Hamiltonians," *Quantum Inf. Comput.*, vol. 10, no. 7, pp. 669–684, Jul. 2010, arXiv: `10.48550/arXiv.0908.4398 <https://doi.org/10.48550/arXiv.0908.4398>`_.
#
# .. [#Cirstoiu2020] C.\ Cîrstoiu, Z. Holmes, J. Iosue, L. Cincio, P. J. Coles, and A. Sornborger, "Variational fast forwarding for quantum simulation beyond the coherence time," *npj Quantum Inf.*, vol. 6, no. 1, p. 82, Sep. 2020, arXiv: `10.48550/arXiv.1910.04292 <https://doi.org/10.48550/arXiv.1910.04292>`_ [quant-ph].
#
# .. [#Gidney2018] C.\ Gidney, "Halving the cost of quantum addition," *Quantum*, vol. 2, p. 74, Jun. 2018. `doi: 10.22331/q-2018-06-18-74 <https://doi.org/10.22331/q-2018-06-18-74>`_.
#
# .. [#Childs2021] A.\ M. Childs, Y. Su, M. C. Tran, N. Wiebe, and S. Zhu, "Theory of Trotter Error with Commutator Scaling," *Phys. Rev. X*, vol. 11, no. 1, Feb. 2021, doi: `10.1103/PhysRevX.11.011020 <https://doi.org/10.1103/PhysRevX.11.011020>`_.
#
# .. [#Strang1968] G.\ Strang, "On the Construction and Comparison of Difference Schemes," *SIAM Journal on Numerical Analysis*, vol. 5, no. 3, Sept. 1968, doi: `10.1137/0705041 <https://doi.org/10.1137/0705041>`_.
#
# .. [#Verlet1967] L.\ Verlet, "Computer "Experiments" on Classical Fluids. I. Thermodynamical Properties of Lennard-Jones Molecules," *Phys. Rev.*, vol. 159, no. 1, pp. 98-103, Jul. 1967, doi: `10.1103/PhysRev.159.98 <https://doi.org/10.1103/PhysRev.159.98>`_.
#
# .. [#Suzuki1990] M.\ Suzuki, "Fractal decomposition of exponential operators to many-body theories and Monte Carlo simulations," *Phys. Lett. A*, vol. 146, no. 6, pp. 319-323, Jun. 1990, doi:`10.1016/0375-9601(90)90962-N <https://doi.org/10.1016/0375-9601(90)90962-N>`_.