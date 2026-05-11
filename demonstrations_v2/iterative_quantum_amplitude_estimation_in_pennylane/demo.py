r"""
Implememting Iterative Quantum Amplitude Estimation in Pennylane
================================================================

Classical search algorithms are expensive. If, for example, you want to find a specific element in a list of length N, basic techniques require individual guesses to be evaluated sequentially. The resources required to carry this out scale with the size of the string being queried, implying :math:`\mathcal{O}(N)` efficiency and reducing implementation feasibility. Though Monte Carlo methods can be employed to alleviate some of this demand, resource requirements still scale intensely with complexity. Since quantum computing, in theory, enables efficient parallel computation, its utility in search techniques is clear. `Grover's algorithm <https://pennylane.ai/qml/demos/tutorial_grovers_algorithm/>`_ demonstrated the possibility of circumventing the efficiency challenges of classical search techniques by taking advantage of quantum information processing methods to search a data set for a specific entry via signal interference, eliminating the need to "guess and check" and improving efficiency to :math:`\mathcal{O}(\sqrt{N})`. Efficient and justifiable implementation of Grover's algorithm, however, requires the number of solutions to be known and that the input state exists in an equally weighted superposition, limiting its applications. 

In 2000, the quantum amplitude estimation (QAE) algorithm was put forward as a generalization of Grover's algorithm that seeks to estimate the probability that a certain state exists within a data set, if at all [#Brassard2000]_. In this generalization, a superposition state with uneven probabilities is operated on by the Grover operator :math:`2^n` times, where :math:`n` is the size of the evaluation register. The rotation and amplification induced by the Grover operator should, in theory, encode an amplitude in an evaluation register indicating the concentration of "good" states (a common term used to refer to the states that meet the success criteria of the search) in a data set, which is obtained through quantum processing via `quantum Fourier transform (QFT) <https://pennylane.ai/qml/demos/tutorial_qft/>`_ to carry out `quantum phase estimation (QPE) <https://pennylane.ai/qml/demos/tutorial_qp/>`_.

Though QAE is a major improvement on its predecessors, its reliance on QPE limits its practicality. To execute the required QFT, a large set of qubits (whose size depends on the desired precision) must be dedicated to the evaluation register that stores the information introduced by the Grover operator. This causes the required hardware width and depth to increase very quickly beyond current practicality. If we want to, therefore, employ QAE using near-term hardware, we need to find a way to take advantage of the benefits of a quantum search algorithm and extract meaningful results without performing the QPE measurements. 

In [#Grinko2021]_, an alternative is proposed: Iterative Quantum Amplitude Estimation (IQAE). In essence, IQAE is a quantum-classical hybrid algorithm that executes the Grover operator :math:`k` times with the goal of extracting an amplitude that corresponds to the probability of measuring a type of entry in a data set. The value of :math:`k` is determined and applied iteratively, hence the name, via classical methods. So, where QAE attempts to find the solution in one shot by interfering all elements of the input superposition state after the Grover operator has been applied and prior to measurement, IQAE measures sequentially directly after the application of the Grover operator to gather information and "zoom in" on the range the solution may exist in. Since no QPE is taking place, we no longer require the full evaluation register and, therefore, can reduce the width of the required circuitry into a range that is feasible for near-term hardware.

The goal of this demo is to introduce the IQAE algorithm and implement a simple example using Pennylane.
"""

######################################################################
import pennylane as qp
import numpy as np
import matplotlib.pyplot as plt
import math

######################################################################
# Initial State
# -------------
#
# IQAE, like QAE, is specifically focused on analyzing data sets composed of "good" and "bad" states that appear with unequal probability. In order to make this state searchable, each component must be assigned a marker that indicates which of the two categories it falls in. Taking :math:`|0\rangle` to be a "bad" marker and :math:`|1\rangle` to be a "good" marker following Boolean logic, the general IQAE state can be defined as
#
# .. math::
#    |\Psi_{IQAE}\rangle = \sqrt{1-a}|\psi_0\rangle|0\rangle+\sqrt{a}|\psi_1\rangle|1\rangle
#
# Where :math:`a` is the probability of measuring a "good" state, :math:`|\psi_0\rangle` is a "bad" state, and :math:`|\psi_1\rangle` is a "good" state.
#
# In this implementation, the goal of the IQAE algorithm will be to identify the probability that a given entry in a data set is a multiple of 8. When encoded in binary, multiples of 8 will always have 0 in the last three positions, which serves as a simple success criteria for the algorithm (ie. a "good" state will always have 0s in the 3 least significant positions). 
#
# To carry this search out, we will define an operator :math:`\mathcal{A}` that maps a set of input qubits onto the problem, meaning the structure of :math:`\mathcal{A}` will differ depending on the application. :math:`\mathcal{A}` should impose a unitary operation on the input states that invokes a superposition state that is identical to :math:`|\Psi_{IQAE}\rangle`. In this case, a randomly weighted superposition of all combinations of the input qubits should be generated and the final 3 qubits in each string should be checked for adherence to the success criteria (ie. are they all zero?) via a multi-controlled CNOT gate. If the logic gate is triggered, a single-qubit evaluation register will be flipped to :math:`|1\rangle`, indicating a "good" result. However, the goal will not be to identify all "good" results in one sweep. Instead, several iterations will be carried out in which the Grover operator is applied multiple times with the goal of extracting a probability amplitude with adequate accuracy by refining the interval within which the solution is likely to lie. To do this, each iteration of the IQAE algorithm will yield the following state:
#
# .. math::
#    \mathcal{Q}^k\mathcal{A}|0\rangle_n|0\rangle_{eval} = \cos((2k+1)\theta_a)|\psi_0\rangle_n|0\rangle_{eval}+\sin((2k+1)\theta_a)|\psi_1\rangle_n|1\rangle_{eval}
#
# Where :math:`n` is the number of qubits, :math:`\mathcal{Q}` is the Grover operator, :math:`\theta_a` is related to the angle that the state is rotated by Grover's operator (note :math:`a=sin^2(\theta_a)`), and :math:`k` is the number of times that the Grover operator is applied to the state in a single IQAE iteration. The specifications of this equation are covered thoroughly in [#Brassard2000]_, but the important result is that the probability of measuring a "good" state at the end of an iteration is given by:
#
# .. math::
#    \mathbb{P}(|1\rangle)=\sin^2((2k+1)\theta_a)
#
# From this, it is clear that the probability is correlated to the angle imposed by the Grover operator. This relationship can be made clearer by taking into consideration the fact that :math:`\mathcal{Q}` invokes a :math:`2\theta_a` rotation in a 2D space defined by the "good" and "bad" state axes each time it is applied. So, if we can figure out the angle that has been imposed on the state as a result of the operator application, we can obtain the probability of extracting a "good" state. Since we do not aspire to use QPE, our best bet is to use our iterative :math:`k` value combined with an measurement of the quantum circuit that outputs an intermediate amplitude guess. The above equation also shows that the size of :math:`k` determines the resolution of the search, with a large :math:`k` corresponding to a high probability oscillation frequency and, therefore, a high resolution. Thus, if an adequately sized :math:`k` is identified, a high accuracy estimation for the "good" state amplitude can be calculated [#Grinko2021]_.
#
######################################################################
# Defining The Input State and Operators
# --------------------------------------
#
# First, we can define the circuit specifications, generating a random list of probabilities that will be assigned as weights in the input state.

#Define system parameters
N = 10000 #Number of shots
num_qubits = 5
n = 2**num_qubits #Number of possible states

#Generate a random list of probabilities to be assigned to the initial state
random_vector = np.sqrt(np.random.rand(n))
distribution = random_vector/np.linalg.norm(random_vector)

#Define which indices should be checked for success criteria against the reference register
control_wires = [num_qubits-3,num_qubits-2,num_qubits-1,num_qubits]

###############################################################################
# As mentioned, the backbone of the quantum portion of the IQAE algorithm is the Grover operator :math:`\mathcal{Q}`, which aims to identify "good" states and introduce an identifiable phase flip and `amplitude amplification <https://pennylane.ai/qml/demos/tutorial_intro_amplitude_amplification/>`_. The basic structure of :math:`\mathcal{Q}` is 
#
# .. math::
#    \mathcal{Q}=-\mathcal{A}\mathcal{S}_0\mathcal{A}^{-1}\mathcal{S}_{\psi_1}
#
# In which :math:`\mathcal{S}_{\psi_1}` acts as the oracle and flips the phase of (marks) a "good" state and :math:`\mathcal{S}_0` flips everything except the :math:`|0\rangle` state (see figures 1 through 3 in the `amplitude amplification <https://pennylane.ai/qml/demos/tutorial_intro_amplitude_amplification/>`_ demo for an intuitive visualization). Since this is an uneven superposition, this operator that carries out this process needs to be defined rather than using Pennylane's built in GroverOperator() function, which assumes an even superposition. 
#
# First, :math:`\mathcal{A}` can be defined according to the following procedure: 
#
# 1. Generate :math:`n` qubits with amplitudes according to the previously generated random probability distribution using StatePrep.
# 2. Flip the state of the 3 final qubits in the string so that MultiControlledX() is triggered by a :math:`|111\rangle` state.
# 3. Implement MultiControlledX such that wire :math:`n+1` takes on the :math:`|1\rangle` state if the success criteria is met.
# 4. Flip the state of the 3 final qubits back to the original. 
#
# .. figure:: ../demonstrations_v2/iterative_quantum_amplitude_estimation_in_pennylane/A_Operator.png
#    :align: center
#    :width: 80%
#

#Define A operator
@qp.prod
def A(state):
  qp.StatePrep(state,wires=range(num_qubits)) #Randomly weighted superposition

  #Flip monitored qubits to so that MCX is triggered by |111>
  qp.PauliX(wires=num_qubits-3)
  qp.PauliX(wires=num_qubits-2)
  qp.PauliX(wires=num_qubits-1)

  qp.MultiControlledX(control_wires) #State is marked with |1> iff number is a multiple of 8

  #Flip back monitored bits to original state
  qp.PauliX(wires=num_qubits-3)
  qp.PauliX(wires=num_qubits-2)
  qp.PauliX(wires=num_qubits-1)

##############################################################################
# Since the "good" state is marked by a :math:`|1\rangle`, the :math:`\mathcal{S}_{\psi_1}` operator (also known as the oracle) can be constructed simply using a PauliZ gate, which will flip the phase of any state that has this marker and allow any state marked by :math:`|0\rangle` to pass unchanged. The :math:`\mathcal{S}_0` operator is analogous to a simple FlipSign() operation defined to act on the :math:`|0\rangle` state. When defining the operator, the global phase term can be neglected since it does not impact the final measurement.
#
# .. figure:: ../demonstrations_v2/iterative_quantum_amplitude_estimation_in_pennylane/Q_Operator.png
#    :align: center
#    :width: 80%
#

#Phase flip oracle
@qp.prod
def Oracle():
  qp.PauliZ(wires=num_qubits)

dev = qp.device("lightning.qubit", wires=num_qubits+1)

#Build the circuit using the Grover operator form Q=AS0A*Spsi
@qp.prod

def Q():
  Oracle()
  qp.adjoint(A(distribution))
  qp.FlipSign(0,wires=range(num_qubits))
  A(distribution)

##############################################################################
# These operators can be used to build the final, iterative circuit in which the number of Grover operator applications will vary.
#
# .. figure:: ../demonstrations_v2/iterative_quantum_amplitude_estimation_in_pennylane/Final_Circuit_Drawing.png
#    :align: center
#    :width: 80%
#
#
k_i = 0 #Begin with no Grover iterations
#Build the circuit Q^kA|0>n|0>
@qp.qnode(dev, shots=N)
def circuit(k_i):
  A(distribution)
  for i in range(k_i):
    Q()
  return qp.probs(wires=[num_qubits]) #Return the probability of measuring "good" and "bad" state

##############################################################################
# Digesting the FindNextK Function
# --------------------------------
#
# As shown, the iteration variable :math:`k` is directly tied to the total angle of the state since the Grover operator invokes a deterministic rotation each time it is applied. The :math:`sin^2(x)` function adds complexity to the probability calculations, so standard trigonometric identities can be employed to achieve:
#
# .. math::
#    \mathbb{P}(|1\rangle)=\frac{1-cos((4k+2)\theta_a)}{2}=\frac{1-cos(K_i\theta_a)}{2}
#
# Letting :math:`K_i=4k+2` be the frequency term.
#
# In [#Grinko2021]_, the authors define FindNextK() to iterate the number of times :math:`\mathcal{Q}` is implemented. The goal of FindNextK() is to identify the largest possible :math:`k` that adheres to what we will call the half-plane condition. The core principle of IQAE is the narrowing of a range of potential amplitudes to, eventually, hone in on an accurate estimate of the "good" state probability. To do this, each iteration of the algorithm must operate between an upper and lower bound that make up the function's confidence interval, which corresponds to the range of probabilities within which the final probability amplitude will exist. Since the calculation involves a :math:`\arcsin(x)` calculation, it is possible that this confidence interval could yield uninterpretable results if one angle falls in the upper half of the unit circle (ie. between 0 and :math:`\pi`) and the other falls in the lower half of the unit circle (ie. between :math:`\pi` and :math:`2\pi`) since information about the measurement's position on the probability curve would be lost. Think, for example, of measuring a probability of 40%. Without information on the range in which this measurement falls, it is impossible to know whether you are approaching a peak or a plateau in the sinusoidal probability curve. In addition, having both bounds of the confidence interval on the same helf-plane of the unit circle will result in unambiguous knowledge on which branch of :math:`\arcsin(x)` to use. Thus, valid results will should always fall in either the upper or lower half-plane of the unit circle.
#
# .. figure:: ../demonstrations_v2/iterative_quantum_amplitude_estimation_in_pennylane/Half_Plane_Illustration.png
#    :align: center
#    :width: 80%
#
#    Half-Plane Condition as Defined by [#Grinko2021]_.
#
# The FindNextK function validates this condition. The logic is as follows: 
#
# 1. For an initial guess :math:`k_i` yielding confidence interval :math:`[\theta_{min}^i,\theta_{max}]=[\theta_{lower}K_i,\theta_{upper}K_i]`, the function will return the current guess of :math:`k` if either both the upper and lower bounds are less than pi (ie. they fall in the upper half of the unit circle) or both the upper and lower bounds are greater than pi (ie. they fall in the lower half of the unit circle). 
# 2. If neither of these conditions are met (ie. the two bounds fall in different half-planes), the magnitude of the guess needs to be reduced.
#
# To carry out the actual comparison logic, however, some translation is required. First, the maximum possible value of :math:`k` must be defined in relation to the angles. [#Grinko2021]_ defines this value as
#
# .. math::
#    K_{max} = \lfloor \frac{\pi}{\theta_{max}-\theta_{min}} \rfloor
#
# Where :math:`K_{max}` can be interpreted as the maximum number of rotations that can be carried out before aliasing becomes an issue.
#
# So, our goal is to find the largest integer :math:`K \leq K_{max}` that satisfies :math:`K = 4k+2` and adheres to the half-plane condition. This can be carried out using a modulo 4 calculation, which enforces the required condition introduced by :math:`K=4k+2`. Once this is found, the final step is to compute scaling factor :math:`q`, the ratio between the current :math:`K` guess and the previous, which shifts the values relative to the previous step to prevent backsliding. 
#
# The FindNextK function will achieve one of two outcomes. 
#
# 1. Both bounds of the confidence interval fall in the same half-plane, causing the function to return the current :math:`k` guess and a Boolean indicating which half-plane the interval falls in. 
#
# or
#
# 2. The bounds of the confidence interval fall in different half-planes, indicating the current guess is too large. If an adequate guess is not reached while the While loop runs, the previous guess is returned. 
# 

def FindNextK(k_i,theta_min, theta_max, HalfPlane_Bool):
    K_i = 4*k_i+2 #Define coefficient 
    theta_min_i = theta_min*K_i
    theta_max_i = theta_max*K_i
    Kmax = math.floor(math.pi/(theta_max-theta_min)) #Maximum K value
    K = Kmax-(Kmax-2)%4

    while K>=2*K_i:
      q = K/K_i 
      if (q*theta_max_i)%(2*math.pi)<=math.pi and (q*theta_min_i)%(2*math.pi)<=math.pi: #Is this guess in the upper half?
        k_i_it = ((K-2)/4)
        HalfPlane_Bool_it = True
        return k_i_it, HalfPlane_Bool_it

      elif (q*theta_max_i)%(2*math.pi)>=math.pi and (q*theta_min_i)%(2*math.pi)>=math.pi: #Is this guess in the lower half?
        k_i_it = ((K-2)/4)
        HalfPlane_Bool_it = False
        return k_i_it, HalfPlane_Bool_it

      else:
        K-=4 #Decrease guess if the range spans two halves
    return (k_i,HalfPlane_Bool)

##############################################################################
# Implementing the IQAE Algorithm
# -------------------------------
#
# With FindNextK() defined, the IQAE algorithm can now be implemented! The main objective of this function is to apply the :math:`k` value returned by FindNextK() to the previously defined quantum circuit, obtain a measurement, and determine if this measurement is adequately accurate with respect to a defined tolerance or if the confidence interval should be updated and passed back into the classical function for another iteration. The logic is as follows: 
#
# 1. Call circuit() after FindNextK() outputs a guess for :math:`k` and take a probability measurement. 
# 2. Use this value to update the confidence interval, in which both the upper and lower bound on the angles and probabilities are computed from the measured amplitude. 
# 3. Compute the overlap between the previous confidence interval and the new confidence interval, taking this to be your final upper and lower bound definition. 
# 4. Check to see if the difference between the new upper and lower bounds is smaller than :math:`\epsilon`, which represents a chosen accuracy parameter. If not, pass the final upper and lower bounds back into FindNextK() and repeat. If yes, return the probability amplitudes associated with the upper and lower amplitudes. 
#
# There are several well-known statistical methods used to update confidence intervals. A simple, iterative approach is the Chernoff-Hoeffding method, shifts the interval bounds up and down, respectively, by :math:`\epsilon_{a_i}`. From [#Grinko2021]_, the Chernoff-Hoeffding algorithm is as follows
#
# .. math::
#    \epsilon_{a_i}=\sqrt{\frac{1}{2N}\log{\frac{2T}{\alpha}}}
#
# Where :math:`\epsilon_{a_i}` is change between the previous amplitude estimation and current amplitude estimation and :math:`T` defines the maximum number of iterations required to achieve a precision of :math:`\epsilon_{a_i}` and is given by:
#
# .. math::
#    T = \lceil \log_{2}{\frac{\pi}{8\epsilon}} \rceil
#
# Which can be used to estimate the upper and lower bounds of the probability interval estimate.
#
# .. math::
#    p_{max} = \min(1,a_i+\epsilon_{a_i})
#
# .. math::
#    p_{min} = \max(0,a_i-\epsilon_{a_i})
#
# In which :math:`a_i` is the outcome of the quantum circuit measurement for iteration :math:`i`. 
#

def IQAE(eps, alpha, N):
  k_current = 0
  theta_lower = 0
  theta_upper = math.pi/2 #Beginning search in the upper quadrant [0,pi/2]
  HalfPlane_Bool = True
  T = math.ceil(math.log2(math.pi/(8*eps)))

  while (theta_upper-theta_lower)>2*eps:
    k_i, HalfPlane_Bool = FindNextK(k_current, theta_lower, theta_upper, HalfPlane_Bool) #determine current guess of k_i
    K_i = int(4*k_i+2)

    #Call circuit
    a_estimate = (circuit(int(k_i))[1])

    eps_ai = ((1/(2*N))*math.log(2*T/alpha))**0.5

    p_max = (np.clip(a_estimate+eps_ai, 0, 1))
    p_min = (np.clip(a_estimate-eps_ai, 0, 1))

    #Update confidence interval based on the quadrant we are in
    if HalfPlane_Bool: #Upper quadrant
      theta_upper_est = 2*math.asin((p_max)**0.5)
      theta_lower_est = 2*math.asin((p_min)**0.5)
    else: #Lower quadrant - ensure the value stays within [0,2pi]
      theta_upper_est = 2*math.pi - 2*math.asin((p_min)**0.5)
      theta_lower_est = 2*math.pi - 2*math.asin((p_max)**0.5)

    #Compute total rotationk, "unwrapping" from the scaled guess in FindNextK
    new_lower = (2*math.pi*(math.floor(K_i*theta_lower/(2*math.pi))) + theta_lower_est)/K_i
    new_upper = (2*math.pi*(math.floor(K_i*theta_upper/(2*math.pi))) + theta_upper_est)/K_i

    #Intersection calculation
    theta_lower = max(theta_lower, new_lower)
    theta_upper = min(theta_upper, new_upper)

    k_current = k_i

  a_lower = math.sin(theta_lower)**2
  a_upper = math.sin(theta_upper)**2

  return a_lower, a_upper

##############################################################################
# Upon calling IQAE(), the output will consist of the upper and lower bounds between which the true amplitude lies. Since the probability distribution in this example is random, the outcome will change between runs of the full script. To compare the confidence interval obtained by the IQAE algorithm, the :math:`\mathcal{A}` state can be measured in a single shot, though this is not a realistic analogy to a physical system and is used here only for comparison. 
#

dev_exact = qp.device("default.qubit", wires=num_qubits+1)

@qp.qnode(dev_exact)
def circuit_exact():
    A(distribution)
    return qp.probs(wires=[num_qubits])

#Define parameters
eps = 0.0001 #Precision
alpha = 0.01 #Confidence
HalfPlane_Bool = True

a_lower, a_upper = IQAE(eps, alpha, N)
true_a = circuit_exact()[1]

print("Analytic probability:", true_a)
print("Lower prediction bound:", a_lower)
print("Upper prediction bound:", a_upper)
print("Contains true value?", a_lower <= true_a <= a_upper) #Boolean indicating whether the analytical value falls within the final confidence interval
print("-------------------------")
if a_lower <= true_a <= a_upper:
  print("The probability of measuring a multiple of 8 falls between", (100*a_lower),"%", "and", (100*a_upper),"%")
else:
  print("No valid answer found")

##############################################################################
# Conclusion
# ----------
# As quantum hardware continues to develop, it is crucial that methods to make use of current capabilities are developed and implemented. QAE was put forward as a generalization of Grover's algorithm, which hinted at the viability of quantum computers as a notable improvement on existing technologies early on. IQAE pushed forward with this trend, showing not only that quantum hardware has the potential to provide advantages in search algorithms but that these advantages can be accessed in the near-term. Already, demonstrations of IQAE have emerged from applications in finance (specifically in risk assessments and financial modelling), quantum machine learning (QML), and general applications of QML. The simple example shown in this demo provides the framework to extend IQAE to more complex search problems in Pennylane via the redefinition of the success criteria in the :math:`\mathcal{A}` operator.
#
#
# .. _references:
#
# References
# ----------
#
# .. [#Grinko2021] Grinko, D., Gacon, J., Zoufal, C., & Woerner, S. (2021). Iterative quantum amplitude estimation. *npj Quantum Information*, 7(52). https://doi.org/10.1038/s41534-021-00379-1
#
# .. [#Brassard2000] Brassard, G., Høyer, P., Mosca, M., & Tapp, A. (2002). Quantum Amplitude Amplification and Estimation. *Contemporary Mathematics*, 305, 53-74. https://arxiv.org/abs/quant-ph/0005055