r"""
Iterative Quantum Amplitude Estimation in Pennylane
=

Classical search algorithms are expensive. To carry out a search in a string of length N with the goal of identifying a specific element, individual guesses must be made and tested until the target entry is found. The resources required to carry this out scale with the size of the string being queried, reducing the feasibility of using Monte Carlo methods, for example, in complex searches. 'Grover's algorithm <https://pennylane.ai/qml/demos/tutorial_grovers_algorithm/>'_ demonstrated the possbility of circumventing these efficiency challenges by using quantum computing techniques to search a data set in parallel. Grover's algorithm, however, is specifically designed in which a specific state that is known to exist is being sought in a data set. In 2000, the quantum amplitude estimation (QAE) algorithm was put forward as a generalization of Grover's algorithm that seeks to estimate the probability that a certain state exists within a data set, if at all [#Brassard2000]. In this generalization, a superposition state with uneven probabilities is operated on by the Grover operator :math:'n' times, where :math:'n' is the size of the evaluation register. The rotation and amplification induced by the Grover operator should, in theory, make the "good" states (a common term used to refer to the states that meet the success criteria of the search) dominant in amplitude, which can be extracted following processing via 'quantum fourier transform (QFT) <https://pennylane.ai/qml/demos/tutorial_qft/>' to carry out 'quantum phase estimation (QPE) <https://pennylane.ai/qml/demos/tutorial_qp/>'_.

Though QAE is a major improvement on its predecessors, its reliance on QPE limits its practicality. To carry out the required QFT, a large set of qubits must be dedicated to the evaluation register that stores the information introduced by the Grover operator. This causes the required hardware depth to increase very quickly beyond current practicality, which introduces a difficult question, how can we take advantage of the benefits of a quantum implementation without performing the QPE measurements that are required to make sense of the results? In [#Grinko2021], an alternative is proposed: Iterative Quantum Amplitude Estimation (IQAE). In essense, IQAE is a quantum-classical hybrid algorithm that executes the Grover operator :math:'k' times, as defined by a classical processing technique. So, where QAE attempts to find the solution in one shot by interfering all elements of the superposition state after the Grover operator has been applied and prior to measurement, IQAE measures directly after the application of the Grover operator to gather information and narrow the range that the solution may exist in. Since no QPE is taking place, we no longer require the evaluation register and, therefore, have come upon an algorithm that has the potential for near-term hardware implementation!

The goal of this demo is to introduce the IQAE algorithm and implement a simple example in Pennylane.
"""

###############################################################################
#
#IQAE is specifically focused on analyzing data sets composed of an uneven superposition of "good" and "bad" states. In order to make this state searchable, each must be assigned a marker that indicates which of the two categories it falls in. Taking :math:'\|0\rangle' to be a "bad" marker and :math:'\|1\rangle' to be a "good" marker, this state can be defined as follows
# .. math ::
#   |\Psi_{IQAE}\rangle = \sqrt{1-a}|\psi_0\rangle|0\rangle+\sqrt{a}|\psi_1\rangle|1\rangle
#Where :math:'|\psi_0\rangle' is a "bad" state and :math:'|\psi_1\rangle' is a "good" state.

#In this implementation, the goal of the IQAE algorithm will be to identify how many multiples of 8 exist in the given data set. When encoded in binary, multiples of 8 will always have 0 in the last three positions. Thus, this will be our truth criteria. To carry this search out, an operator :math:'\mathcal{A}' that maps a series of input qubits onto the problem. More specifically, :math:'\mathcal{A}' impose a unitary operation on the input states that produces a superposition state that is identical to :math:'|\Psi_{IQAE}\rangle'. In this case, a randomly weighted superposition of all combinations of the input qubits should be generated and the final 3 qubits in each string should be checked for adherence to the success criteria (ie. are they all zero?) via a multi-controlled CNOT gate. If the logic gate is triggered, the marker qubit will be flipped to :math:'|1\rangle', indicating a "good" result. However, the goal will not be to identify all "good" results in one sweep. Instead, several iterations will be carried out in which the Grover operator is applied multiple times with the goal of extracting a probability amplitude with adequate accuracy by refining the interval within which the solution is likely to lie. To do this, each iteration of the IQAE algorithm will yeild the following state:

# .. math::
#    \mathcal{Q}^k\mathcal{A}|0\rangle_n|0\rangle_n = cos((2k+1)\theta_a)|\psi_0\rangle_n|0\rangle+sin((2k+1)\theta_a)|\psi_1\rangle_n|1\rangle

# Where :math:'n' is the number of qubits, :math:'$\mathcal{Q}' is the Grover operator, :math:'\theta_a' is the angle between the current state vector and the "bad" state axis, and :math:'k' is the number of times that the Grover operator is applied to the state in a single IQAE iteration. The specifications of this equation are covered thoroughly in [#Brassard2000], but the important result from this equation is that the probability of measuring a "good" state at the end of an iteration is given by

# .. math ::
#    \mathbb{P}(|1\rangle)=sin^2((2k+1)\theta_a)

# From this, it is clear that the probability is correlated to the angle imposed by the Grover operator, meaning that determining this angle will yeild the desired amplitude result. Thus, if an adequately sized :math:'k' is identified, a high accuracy estimation for the amplitude can be identified by taking :math:'a=sin^2(\theta_a)' to be the amplitude of the "good" state. 

# Defining The Input State and Operators
# --------------------------------------

#First, we can define the circuit specifications, generating a random list of probabilities that will be assigned as weights in the input state.

import pennylane as qp
import numpy as np
import matplotlib.pyplot as plt
import math

#Define system parameters
N = 10000 #Number of shots
num_qubits = 5
n = 2**num_qubits #Number of possible states

#Generate a random list of probabilities to be assigned to the initial state
random_vector = np.sqrt(np.random.rand(n))
distribution = random_vector/np.linalg.norm(random_vector)

#Define which indices should be checked for success criteria
control_wires = [num_qubits-3,num_qubits-2,num_qubits-1,num_qubits]

###############################################################################
# The basic structure of the Grover operator is 
# .. math ::
#    \mathcal{Q}=-\mathcal{A}\mathcal{S}_0\mathcal{A}^{-1}\mathcal{S}_\psi_1
# In which :math:'\mathcal{S}_\psi_1' acts as the oracle and flips the phase of (marks) a "good" state and :math:'\mathcal{S}_0' flips everything except the :math:'|0\rangle' state. This process is outlined in Grover's algorithm. Since this is an uneven superposition, this operator needs to be defined. 
# First, :math:'\mathcal{A}' can be defined according to the following procedure: 
# 1. Generate :math:'n' qubits with amplitudes according to the previously generated random probability distribution using StatePrep.
# 2. Flip the state of the 3 final qubits in the string so that MultiControlledX is triggered by a :math:'|111\rangle' state.
# 3. Implement MultiControlledX such that wire :math:'n+1' takes on the :math:'|1\rangle' state if the success criteria is met.
# 4. Flip the state of the 3 final qubits back to the original. 

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
#Since the "good" state is marked by a :math:'|1\rangle', the oracle can be constructed using a PauliZ gate, which will flip the phase of any state that has this marker. The :math:'\mathcal{S}_0' operator is analogous to a simple FlipSign operation defined to act on the :math:'|0\rangle' state. Thus, the full :math:'\mathcal{Q}' state can be implemented as follows.

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
#These operators can be used to build the final, iterative circuit in which the number of Grover operator applications can vary.

#INSERT CIRCUIT DRAWING HERE

k_i = 0 #Begin with no Grover iterations
#Build the circuit Q^kA|0>n|0>
@qp.qnode(dev, shots=N)
def circuit(k_i):
  A(distribution)
  for i in range(k_i):
    Q()
  return qp.probs(wires=[num_qubits]) #Return the probability of measuring "good" and "bad" state

##############################################################################
#Digesting the FindNextK Function
# ---------------------------------

#As shown, the iteration variable :math:'k' is directly tied to the total angle of the state. Since the :math:'sin^2(x)' function adds complexity to the probability calculations, standard trigonometric identities can be employed to achieve
# .. math::
#    \mathbb{p}(|1\rangle)=\frac{1-cos((4k+2)\theta_a)}{2}=\frac{1-cos(K_i\theta_a)}{2}
#Letting :math:'K_i=4k+2'.

#In [#Grinko2021], the goal of the FindNextK function is to identify the largest possible :math:'k' that adheres what will be refered to as the half-plane condition. The core principle of IQAE is the narrowing of a range of potential amplitudes to, eventually, narrow in on the desired outcome. To do this, each iteration of the algorithm must operated between an upper and lower bound that is referred to as the confidence interval. The bounds of this interval set the upper and lower limits of possible angles that correspond to the range of probabilities and the distance between these angles on a unit circle (and, consequently, the probability results) will be periodic. As such, there is a risk that taking any random guess of the upper and lower bounds will result in uninterpretable results as the location on the probability curve is lost. Think, for example, of measuring a probability of 40%. Without information on the range in which this measurement falls, it is impossible to know whether you are approaching a peak or a plateau. Thus, valid results will always fall in either the upper or lower half-plane of the unit circle.

#HALF PLANE ILLUSTRATION

#The FindNextK function validates this condition. The logic is as follows: for an initial guess :math:'k_i' yeilding confidence interval :math:'[\theta_{min}^i,\theta_{max}]=[theta_{lower}*K_i,theta_{upper}*K_i]', the function will return the current guess of :math:'k' if either both the upper and lower bounds are less than pi (ie. they fall in the upper half of the unit circle) or both the upper and lower bounds are greater than pi (ie. they fall in the lower half of the unit circle).

#To carry out the actual comparison logic, however, some translation is required. First, the maximum possible value of :math:'k' must be defined in relation to the angles. [#Grinko2021] defines this value as:
# .. math::
#    K_{max} = \lfloor \frac{\pi}{\theta_{max}-\theta_{min}} \rfloor
#
#So, our goal is to find the largest integer :math:'K\leqK_{max}' that satisfies :math:'4k+2'. This can be carried out using a modulo 4 calculation, which enforces the required condition. Once this is found, the final step is to compute scaling factor :math:'q', the ratio between the current :math:'K' guess and the previous. This factor shifts the values relative to the previous step to prevent backsliding. 

#The FindNextK function will achieve one of two outcomes. In scenario 1, both bounds of the confidence interval fall in the same half-plane, causing the function to return the current :math:'k' guess and a Boolean indicating which half-plane the interval falls in. In scenario 2, the bounds of the confidence interval fall in difference half-planes, indicating the current guess is too large. If an adequate guess is not reached before the current guess surpasses the maximum, the current guess is returned.

def FindNextK(k_i,theta_min, theta_max, quadrant_bool):
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
#Implementing the IQAE Algorithm 
# ----------------------------------
#With FindNextK defined, the IQAE algorithm can now be implemented! The main objective of this function is to apply the :math:'k' value returned by FindNextK to the previously defined quantum circuit, obtain a measurement, and determine if this measurement is adequate or if the confidence interval should be updated and passed back into the classical function for another iteration. The logic is as follows: call circuit() after FindNextK() outputs a guess for :math:'k'. Use this value to update the confidence interval, in which both the upper and lower bound on the angles and probabilities are computed from the measured amplitude. From this, compute the overlap between the previous confidence interval and the new confidence interval, taking this to be your final upper and lower bound definition. Finally, check to see if the difference between the new upper and lower bounds is smaller than :math:'\epsilon'. If not, pass the final upper and lower bounds back into FindNextK() and repeat. If yes, return the probability amplitudes associated with the upper and lower amplitudes. 

#There are several well-known statistical methods used to update confidence intervals. A simple, iterative approach is the Chernoff-Hoeffding method, shifts the interval bounds up and down, respectively, by :math:'\epsilon_{a_i}'. From [#Grinko2021],
# .. math ::
#    \epsilon_{a_i}=\sqrt{\frac{1}{2N}\log{\frac{2T}{\alpha}}}
#    T = \lceil \log_{2}{\frac{\pi}{2\epsilon}}
#    p_{max} = \max(1,a_i+\epsilon_{a_i})
#    p_{min} = \min(0,a_i-\epsilon_{a_i})

#In which :math:'a_i' is the outcome of the quantum circuit measurement for iteration :math:'i'. 

#Actually implement IQAE!
#Pre-selecting the use of Chernoff-Hoeffding to determine confidence interval
def IQAE(eps, alpha, N):
  k_current = 0
  theta_lower = 0
  theta_upper = math.pi/2 #Beginning search in the upper quadrant - this differs from the paper, is this correct?
  HalfPlane_Bool = True
  T = math.ceil(math.log2(math.pi/(8*eps)))

  while (theta_upper-theta_lower)>2*eps:
    k_i, HalfPlane_Bool = FindNextK(k_current, theta_lower, theta_upper, HalfPlane_Bool) #determine current guess of k_i
    K_i = int(4*k_i+2)

    #Call circuit
    a_estimate = (circuit(int(k_i))[1])

    eps_ai = ((1/(2*N))*math.log(2*T/alpha))**0.5

    # p_max = min(np.minimum(1,a_estimate+eps_ai))
    # p_min = max(np.maximum(0,a_estimate-eps_ai))

    p_max = (np.clip(a_estimate+eps_ai, 0, 1))
    p_min = (np.clip(a_estimate-eps_ai, 0, 1))

    #Update confidence interval based on the quadrant we are in
    if HalfPlane_Bool: #Upper quadrant
      theta_upper_est = 2*math.asin((p_max)**0.5)
      theta_lower_est = 2*math.asin((p_min)**0.5)
    else: #Lower quadrant - ensure the value stays within [0,2pi]
      theta_upper_est = 2*math.pi - 2*math.asin((p_min)**0.5)
      theta_lower_est = 2*math.pi - 2*math.asin((p_max)**0.5)

     #Compute total rotation
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
#Upon calling IQAE(), the output will consist of the upper and lower bounds between which the true amplitude lies. Since the probability distribution in this example is random, the outcome will change between runs of the full script.
a_lower, a_upper = IQAE(eps, alpha, N)

print("Expected probability:", circuit(k_i)[1])
print("Lower prediction bound:", a_lower)
print("Upper prediction bound:", a_upper)


##############################################################################
#References

# .. [#Grinko2021]
#    D. Grinko, J. Gacon, C. Zoufal, and S. Woerner, "Iterative quantum amplitude estimation," Dec. 2019, arXiv:1912.05559
#
# .. [#Brassard2000]
#    G. Brassard, P. Høyer, M. Mosca, and A. Tapp, "Quantum amplitude amplification and estimation," May 2000, arXiv:quant-ph/0005055. 