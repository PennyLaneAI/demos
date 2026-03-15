r"""
Seeing Phase Transitions with Quantum Computers
===============================================




Introduction
------------


A phase transition occurs when a system undergoes an abrupt, qualitative change in one or more of its properties---for 
instance, liquid water freezing to form ice. Phase transitions are of significant importance across many areas
of physics, including condensed matter physics [#Vojta2002]_, cosmology [#Mazumdar2019]_, and high-energy physics [#Mueller2023]_.

They are important for several reasons: they facilitate the discovery of new quantum states of matter, 
help illuminate entanglement and long-range correlations within quantum systems, and help us understand the behaviour of 
many different quantum systems at the same time (due to the property of universality).

This tutorial introduces three quantum phase transitions relevant to condensed matter physics 
and demonstrates how to simulate them on a quantum computer. The transitions covered here include the one-dimensional (1D) 
and two-dimensional (2D) quantum Ising model and a dynamical quantum phase transition (i.e., phase transitions occurring in time evolution).

.. admonition:: Note
   :class: note
   
   *Quantum* phase transitions are fundamentally different from *classical* phase transitions. While classical transitions 
   are driven by thermal fluctuations, quantum phase transitions always occur at zero
   temperature and are induced by quantum fluctuations (i.e., Heisenberg's uncertainty principle). 

.. raw:: html

   <br>

Studying these transitions analytically is challenging; the associated discontinuities can cause mathematical models to break down. 
Although classical numerical simulations have been widely used, the required computational resources for certain cases can be prohibitive. 
But there's another way to study phase transitions: using a quantum computer. Potentially, they can compute aspects of phase
transitions more efficiently than any conventional technique.



To date, quantum computers have been employed to investigate quantum phase transitions in diverse areas, including those related to the early 
universe and high-energy particle colliders [#Mueller2023]_, a  topological transition in an Ising-like model [#Smith2019]_, 
the transverse Ising model [#Haghshenas2024]_, and noisy quantum systems [#Chertkov2022]_.
Furthermore, these systems have been applied to studying scalar quantum field theory [#Thompson2023]_ and the evolution of the universe [#Vodeb2025]_.


Note: This tutorial focuses on the *quantum* Ising model. It complements existing content on this
model: `3-qubit Ising model in PyTorch <https://pennylane.ai/qml/demos/tutorial_isingmodel_PyTorch>`_, `Transverse-field Ising model <https://pennylane.ai/datasets/transverse-field-ising-model>`_, and `Quadratic Unconstrained Binary Optimization (QUBO) <https://pennylane.ai/qml/demos/tutorial_QUBO>`_


What is the Ising model?
------------------------


The simplest Ising model consists of :math:`N` qubits arranged along a line.

"""

##############################################################################
# .. figure:: ../_static/demonstration_assets/quantum_phase_transitions/Fig_1_Ising_chain.png
#    :align: center
#    :width: 50%

######################################################################
# Each qubit interacts with the qubits on either side of it. For example, the second qubit interacts with the first and third qubits.

######################################################################
# The system’s Hamiltonian is
#
# .. math::
#    \begin{equation}
#    H = -J \,\, \Sigma_{i=0}^{N-2} \sigma_{z}^{(i)} \sigma^{(i+1)}_{z},
#    \end{equation}
#
# where :math:`\sigma_{z}^{(i)}` is the Pauli-Z operator for the :math:`i^{th}` qubit and :math:`J` is the interaction strength
# between neighbouring qubits.
#
# The code below creates this Hamiltonian for three qubits:
#
import pennylane as qml
qml.capture.enable()

from pennylane import numpy as np

N = 3
J = 2
wires = range(N)

dev = qml.device("lightning.qubit", wires=N)

coeffs = [-J] * (N - 1)

obs = []
for i in range(N - 1):
    obs.append(qml.Z(i) @ qml.Z(i + 1))
H = qml.Hamiltonian(coeffs, obs)

print(f"H={H}")

######################################################################
# Why is the Ising Model Important?
# ---------------------------------
# At first glance, the Ising model looks like it's simple and unrealistic. However, it correctly
# models many properties of real-world magnets. Also, its simplicity allows us to actually solve it.
# You can think of the Ising model as a sandbox to play in and quickly learn about the essence of
# various complex real-world phenomena.
#
# The Ising model exhibits a wide range of interesting emergent properties, such as phase transitions.
# One calculation that gives us insight into their behaviour is finding the ground state of the Ising
# model and seeing how it changes as the interactions change. Often, we're looking to see if a phase
# transition happens.
#
# Let's look at an example.
#
# Seeing Phase Transitions with Quantum Computers
# -----------------------------------------------
# To do this, we'll use the well-known variational quantum eigensolver (VQE) algorithm to find the
# ground state. You can find an introduction to it `here <https://pennylane.ai/qml/demos/tutorial_vqe>`_.
#
# Let's start by finding the ground state of the Ising model for a fixed value of :math:`J`.
# We'll use the Hardware Efficient Ansatz (HEA) [#Kandala2017]_ to do this. It's a
# general-purpose ansatz that efficiently represents a wide range of quantum states, it consists of:
# 
# 1. Applying three single-qubit rotations to each qubit. Each one is parameterized by a different rotation angle. 
# 
# 2. Applying a CNOT gate to each neighbouring pair of qubits. 
# 
# 3. Applying three single-qubit rotations to each qubit. Again, each one is parameterized by a different rotation angle.



import random

random.seed(a=10)

# params is an array that stores the parameter values of the statevector that we use in VQE.
# Generate some initial random angle values.
params = np.array([2 * np.pi * random.uniform(0, 1)] * (6 * N), requires_grad=True)

# create an ansatz using the hardware efficiency ansatz (HEA)
def create_ansatz(params, N):
    # STEP 1: perform single-qubit rotations on all the qubits
    for i in range(N):
        qml.RZ(phi=params[i], wires=i)
        qml.RX(phi=params[N + i], wires=i)
        qml.RZ(phi=params[2 * N + i], wires=i)
    
    # STEP 2: perform a CNOT gate on each pair of neighbouring qubits
    for i in range(N - 1):
        qml.CNOT(wires=[i, i + 1])

    # STEP 3: perform single-qubit rotations on all the qubits
    for i in range(N):
        qml.RZ(phi=params[3 * N + i], wires=i)
        qml.RX(phi=params[4 * N + i], wires=i)
        qml.RZ(phi=params[5 * N + i], wires=i)

@qml.qnode(dev)
def quantum_circuit(params):
    # Create a quantum state using params
    create_ansatz(params, N)
    return qml.expval(H)

max_iters = 200
tolerance = 1e-04

# create an optimizer
opt = qml.GradientDescentOptimizer(stepsize=0.1)

# energy is a list that stores all the estimates for the ground-state energy
energy = []

# execute the VQE optimization loop
for i in range(max_iters):
    params, prev_energy = opt.step_and_cost(quantum_circuit, params)
    energy.append(prev_energy)

    if i > 1:
        if np.abs(energy[-2] - energy[-1]) < tolerance:
            break

# graph the energy as a function of the number of iterations
import matplotlib.pyplot as plt

plt.plot(list(range(len(energy))), energy)
plt.xlabel("Iteration")
plt.ylabel("Energy")
plt.show()

######################################################################
# The graph above shows that the energy :math:`E` gradually decreases until it reaches :math:`E = - 4`.
# To check that this result makes sense, let's think about the Hamiltonian. Consider the first term, :math:`-2 \sigma_{z}^{(0)} \sigma_{z}^{(1)}`.
# When the first two qubits are in the computational basis state
# :math:`| 0 \rangle` , the product :math:`\langle \sigma_{z}^{(0)} \sigma_{z}^{(1)} \rangle` is :math:`(+1)(+1) = +1`. Multiplying this by :math:`J = -2`
# gives an energy of -2. When the last two qubits are in the state :math:`| 0 \rangle`, the second term :math:`-2 \langle \sigma_{z}^{(1)} \sigma_{z}^{(2)} \rangle` is also :math:`E = -2`. Combining
# these results gives :math:`E = -2 -2 = -4`. When all the qubits are in the other basis state
# :math:`(| 1 \rangle)`, we also get :math:`E = -4`. These two calculations agree with the numerical result from VQE. So far, so good.
# 
# Let's now introduce an extra energy term that's proportional to the sum of all the Pauli X operators:
#
# .. math::
#    - h_{x}\Sigma_{i=0}^{N-1} \sigma_{x}^{(i)}.
#
# If our qubits are actually spin-1/2 particles (e.g., electrons), :math:`h_{x}` is a horizontal
# magnetic field. Often, it's called a *transverse field*.
# We introduce the extra term to increase the system's complexity and to see if it leads to any interesting phase transitions. As you'll see, it does. 

##############################################################################
# .. figure:: ../_static/demonstration_assets/quantum_phase_transitions/Fig_2_transverse_Ising.png
#    :align: center
#    :width: 50%
# 

######################################################################
# The system's Hamiltonian becomes
#
# .. math::
#    H = -J \,\, \Sigma_{i=0}^{N-2} \sigma_{z}^{(i)} \sigma^{(i+1)}_{z} - h_{x}\Sigma_{i=0}^{N-1} \sigma_{x}^{(i)}.
#
# A quantum phase transition happens when we change the ratio :math:`J/h_x`. Physically, this
# corresponds to changing the relative strength of the coupling interaction and the horizontal
# magnetic field. When :math:`J` is much larger than :math:`h_{x}`, the ground state corresponds to
# all the spins (i.e., the qubits) being aligned vertically (parallel to the :math:`z` axis).
# 

##############################################################################
# .. figure:: ../_static/demonstration_assets/quantum_phase_transitions/Fig_3_ground_state_J_large.png
#    :align: center
#    :width: 75%
# 

######################################################################
# But, when :math:`h_{x}` is much greater than :math:`J`, the ground state corresponds to
# all the spins being aligned along the :math:`x` axis parallel to the magnetic field:
# 

##############################################################################
# .. figure:: ../_static/demonstration_assets/quantum_phase_transitions/Fig_4_ground_state_h_large.png
#    :align: center
#    :width: 75%
# 

######################################################################
# When :math:`J/h_{x} = 1`, the ground state suddenly switches from the first state (all
# vertical) to the second one (all horizontal). This is a quantum phase transition. The interplay
# between :math:`J` and :math:`h_{x}` is like a tug of war. The coupling constant :math:`J` tries to
# align all the qubits vertically, in the computational basis. The magnetic field :math:`h_{x}` tries
# to align them horizontally, in the Pauli X basis. Depending on value of :math:`J/h_{x}`, one of the two constants
# will dominate.
# 
# To see the phase transition, let's introduce the total magnetization operator :math:`M` of all
# the qubits:
#
# .. math::
#    M =\frac{1}{N} \Sigma_{i} \sigma_{Z}^{(i)}.
#
# It's just the sum of all the Pauli :math:`Z` operators scaled by the number of qubits. For example,
# for the state :math:`| \psi \rangle = |0 \rangle |0\rangle`,
# :math:`\langle M \rangle = \frac{1}{2} \left( 1 + 1 \right) = 1`. The total magnetization tracks the phase change as
# follows: 
# 
# - When :math:`h_{x} \gg  J`, :math:`\langle M \rangle = 0` as each qubit is in an equal superposition of :math:`|0 \rangle` and :math:`|1 \rangle`. 
# 
# - When :math:`J \gg h_{x}`, :math:`|  \langle M \rangle | = 1` as the qubits are either all in :math:`|0 \rangle` or all in :math:`|1 \rangle`.
#

##############################################################################
# Let's now calculate :math:`\langle M \rangle` for a range of :math:`J/h_{x}` values.
#
N = 5
wires = range(N)

# h_x is the strength of the transverse magnetic field
h_x = 1

# Vary the value of the coupling constant J (change J/h_x)
J_list = [0.0, 0.25, 0.75, 0.9, 1.0, 1.1, 2.0, 5.0, 7.5]

# Store the expectation values of the magnetization operator M for different values of J/h_x
magnetization_list = []

dev_2 = qml.device("lightning.qubit", wires=N)

# This function prepares an estimate of the ground state & calculates its energy.
@qml.qnode(dev_2)
def quantum_circuit_2(params):
    # Generate an estimate of the ground state
    create_ansatz(params, N)
    return qml.expval(H)

# A function that returns the magnetization operator of N qubits.
def magnetization_op(N):
    total_op = qml.PauliZ(0)

    if N > 1:
        for i in range(1, N):
            total_op = total_op + qml.PauliZ(i)

    return total_op / N

#Prepare a parameterized state & return the value of the magnetization operator.
@qml.qnode(dev_2)
def calculate_magnetization(params):
    create_ansatz(params, N)
    return qml.expval(magnetization_op(N))

# Loop through all the different values of J
for i in range(len(J_list)):

    # Build the Hamiltonian

    # Add Pauli Z-Pauli Z interaction terms to the Hamiltonian
    coeffs = [-J_list[i]] * (N - 1)

    obs = []
    for j in range(N - 1):
        obs.append(qml.Z(j) @ qml.Z(j + 1))

    # Add Pauli X terms to the Hamiltonian
    for j in range(N):
        obs.append(qml.X(j))
        coeffs.append(-h_x)

    H = qml.Hamiltonian(coeffs, obs)

    params = np.array([2 * np.pi * random.uniform(0, 1)] * (6 * N), requires_grad=True)

    max_iters = 200
    tolerance = 1e-04

    # create an optimizer
    opt = qml.MomentumOptimizer(stepsize=0.02, momentum=0.9)

    energy = []

    # Run the VQE optimization loop
    for j in range(max_iters):
        params, prev_energy = opt.step_and_cost(quantum_circuit_2, params)
        energy.append(prev_energy)

        if j > 1:
            if np.abs(energy[-2] - energy[-1]) < tolerance:
                break

    magnetization_list.append(calculate_magnetization(params))

##############################################################################
# Now that we've calculated :math:`\langle M \rangle`, let's plot the results.
#

# Plot |<magnetization>| versus J
plt.plot(J_list, np.abs(magnetization_list), marker="x")
plt.xlabel("J")
plt.ylabel(r"$|\langle M \rangle|$")
plt.title(r"$| \langle M \rangle |$ vs. $J$ for $N$=" + str(N))
plt.show()

######################################################################
#Notice how the magnetization increases sharply around :math:`J/h_{x} = 1`. This suggests that a phase transition is happening. 
#(It's also well known that a phase transition does happen at this value [#Subir2011]_.) Why doesn't the graph have a sharp and discontinuous increase at
#exactly :math:`J/h_x=1`? There are two reasons: 
# 
#- Like all other numerical results, this result is just approximate. 
#- The phase transition happens at :math:`J/h_x=1` in the asymptotic limit of large :math:`N`, i.e., as the number of qubits goes to infinity. You can see this by plotting how :math:`M` changes for three different values of :math:`N = 4, 5, 6`.
#

# magnetization values for N = 4
magnetization_4 = [
    0.01705303,
    -0.05617393,
    0.34882499,
    0.38068118,
    0.74856645,
    0.90577316,
    0.9872206,
]
J_list_4 = [0.0, 0.25, 0.75, 0.9, 1.1, 2.0, 5.0]

# magnetization values for N = 6
magnetization_6 = [
    -0.11958867,
    0.00284093,
    0.01237123,
    0.00255386,
    0.81125517,
    0.92437233,
    0.99013448,
]
J_list_6 = J_list_4[:]

# Plot |<M>| for multiple N values versus J
plt.plot(J_list_4, np.abs(magnetization_4), "xk-", label="N=4")
plt.plot(J_list[0:8], np.abs(magnetization_list[0:8]), "xb--", label="N=5")
plt.plot(J_list_6, np.abs(magnetization_6), "sg:", label="N=6")

plt.xlabel("J")
plt.ylabel(r"$|\langle M \rangle|$")
plt.title(r"$| \langle M \rangle |$ vs. $J$ for $N$=4, 5, 6")
plt.legend(loc="lower right")
plt.show()

######################################################################
# Notice how the increase in :math:`|\langle M \rangle |` gets steeper as we increase :math:`N`. You can
# think of this as showing that we're getting closer and closer to the asymptotic behaviour of a truly
# discontinuous phase transition.
#

######################################################################
# Two-dimensional Ising Model 
# ---------------------------
# In the 2D quantum Ising model, the qubits are arranged in a 2D grid.
# 

######################################################################
# .. figure:: ../_static/demonstration_assets/quantum_phase_transitions/Fig_5_2D_Ising_model.png
#    :align: center
#    :width: 25%

##############################################################################
# Compared to the 1D model, the 2D model is richer, harder to solve mathematically, and harder to simulate on
# classical computers. It's also more realistic and is used by physicists to study
# low-dimensional quantum systems. In this section, we'll explore phase transitions
# in the 2D quantum Ising model. The Hamiltonian for the model is
#
# .. math::
#    H = -J \,\, \Sigma_{\langle i,j \rangle} \sigma_{z}^{(i)} \sigma^{(j)}_{z} - h_{x} \Sigma_{ i } \sigma_{x}^{(i)}
#
# The expression :math:`\langle i,j \rangle` includes every pair of neighbouring qubits in the lattice. The
# :math:`\Sigma_{i}` term sums over every qubit in the lattice. The code below creates the Hamiltonian
# using `PennyLane's spin module <https://docs.pennylane.ai/en/stable/code/api/pennylane.spin.transverse_ising.html>`_.
#

N = 2

H = qml.spin.transverse_ising(lattice="square", n_cells=[N, N], h=1.0, boundary_condition=True)

print(f"H={H}")

######################################################################
# Like we did for the 1D model, let's find the ground state using VQE.
#

wires_2D = range(N**2)
dev_2D = qml.device("lightning.qubit", wires=wires_2D)

random.seed(a=10)

# generate random parameter values for the initial statevector
params = np.array([2 * np.pi * random.uniform(0, 1)] * (6 * N), requires_grad=True)

@qml.qnode(dev_2D)
def quantum_circuit_2D(params):
    create_ansatz(params, N)
    return qml.expval(H)

max_iters = 500
tolerance = 3e-04

# create an optimizer
opt = qml.GradientDescentOptimizer(stepsize=0.015)

energy = []

# execute the optimization loop
for i in range(max_iters):
    params, prev_energy = opt.step_and_cost(quantum_circuit_2D, params)
    energy.append(prev_energy)

    if i > 1:
        if np.abs(energy[-2] - energy[-1]) < tolerance:
            break

# print out the results
plt.plot(list(range(len(energy))), energy)
plt.xlabel("Iteration")
plt.ylabel("Energy")
plt.show()

##############################################################################
# The energy in the graph approaches -4.5, which makes sense. The Hamiltonian has four two-qubit interaction terms and the
# smallest that each one can be is -1. So, the ground-state energy must be less than -4. Let's vary the ratio :math:`J/h_{x}` again and calculate the
# average value of the magnetization :math:`\langle M \rangle` each time. Finally, let's plot the results and see if there's a quantum
# phase transition.
#

N = 3
dev_2D_varying_J = qml.device("lightning.qubit", wires=N**2)

# strength of transverse magnetic field
h_x = 1

# Vary J in order to see a phase transition in the magnetization as we change J/h_x
J_list = [0.035, 0.05, 0.1, 0.25, 0.375, 0.5, 0.75, 1.0, 5, 10]

magnetization_list = []

# Prepare a parameterized state & calculate the value of the magnetization operator.
@qml.qnode(dev_2D_varying_J)
def calculate_magnetization_2D(params):
    create_ansatz(params, N)
    return qml.expval(magnetization_op(N))

@qml.qnode(dev_2D_varying_J)
def quantum_circuit_2D_varying_J(params):
    create_ansatz(params, N)
    return qml.expval(H)

# Loop through all values of J
for i in range(len(J_list)):
    H = qml.spin.transverse_ising(
        lattice="square", coupling=J_list[i], n_cells=[N, N], boundary_condition=True
    )
    #Set the initial values of the rotation angle parameters.
    #The values below were chosen through trial and error.
    params = np.zeros(6 * N, requires_grad=True)
    for i in range(N):
        params[i] = 0
        params[N + i] = np.pi / 2
        params[2 * N + i] = np.pi / 2

    max_iters = 500

    # create an optimizer
    opt = qml.MomentumOptimizer(stepsize=0.03, momentum=0.9)

    energy = []

    # execute the optimization loop
    for j in range(max_iters):
        params, prev_energy = opt.step_and_cost(quantum_circuit_2D_varying_J, params)
        energy.append(prev_energy)

        if j > 1:
            if np.abs(energy[-2] - energy[-1]) < tolerance:
                break

    magnetization_list.append(calculate_magnetization_2D(params))

######################################################################
# Let's plot the results.

# Plot |<magnetization>| versus J
plt.plot(J_list, np.abs(magnetization_list), marker="x")
plt.xlabel("J")
plt.ylabel(r"$| \langle M \rangle |$")
plt.title(r"$| \langle M \rangle |$ vs. $J$ for $N$=" + str(N))
plt.show()

######################################################################
# From the graph, it's unclear if there's a phase transition. Looking at it, multiple data points are
# bunched up on the left. To spread them out, let's change the scale by ignoring the last two
# points.
#

plt.plot(J_list[0:8], np.abs(magnetization_list[0:8]), marker="x")
plt.xlabel("J")
plt.ylabel(r"$| \langle M \rangle |$")
plt.title(r"$| \langle M \rangle |$ vs. $J$ for $N$=" + str(N))
plt.show()

######################################################################
# Like in the 1D case, the magnetization displays a rapid increase. This is consistent with a phase change
# but it's not conclusive as the increase is somewhat gradual. This is because :math:`N` is so small.
# Note that the result is consistent with where the phase change is known to occur [#Blote2002]_, [#Hashizume2022]_.
#
# Time Evolution & Dynamical Phase Transitions
# --------------------------------------------
#
# Another important aspect of quantum systems is how they evolve over time.
# Sometimes, this evolution is hard to simulate on classical computers. So, researchers are
# interested in modelling it on quantum computers. Occasionally,
# some property of a quantum system changes abruptly. This is called a *dynamical quantum
# phase transition*: a phase transition that happens during the time evolution
# of a quantum system [#Heyl2013]_.
#
# To evolve the Ising model in time, we'll use the well-known Suzuki-Trotter product approximation. The code below does this.

import math

N = 5
wires = range(N)

# Create H for a 1D Ising model with transverse & longitudinal magnetic fields
# We do this to copy what was done in Reference 13: https://arxiv.org/abs/2008.04894
obs = []
for j in range(N - 1):
    obs.append(qml.Z(j) @ qml.Z(j + 1))
obs.append(qml.Z(N-1) @ qml.Z(0))

# add Pauli X terms to Hamiltonian (transverse field)
for j in range(N):
    obs.append(qml.X(j))

# add Pauli Z terms to Hamiltonian (longitudinal field)
for j in range(N):
    obs.append(qml.Z(j))

dev = qml.device("lightning.qubit", wires=N)

J = -0.1

# strength of transverse field interaction
h_x = 1

# strength of longitudinal field interaction
h_z = -0.15

J_coeffs = [-J] * N

X_coeffs = [h_x] * N

Z_coeffs = [h_z] * N

coeffs = J_coeffs + X_coeffs + Z_coeffs

H = qml.Hamiltonian(coeffs, obs)

# create the circuit that evolves the system in time
@qml.qnode(dev)
def time_evolution_circuit(H, T):
    #Evolve the system via a sequence of short approximate Trotter time steps
    #https://docs.pennylane.ai/en/stable/code/api/pennylane.TrotterProduct.html
    qml.TrotterProduct(H, time=T, n=math.ceil(T / 0.1)+1, order=2)

    # return the final probabilities
    return qml.probs(wires=range(N))


##############################################################################
# To see if a dynamical phase transition happens, let's consider an observable called the *rate function*
# :math:`\gamma`. It depends on the overlap between the quantum state that we start with and the final state at
# some time :math:`t`. More specifically, 
#
# .. math:: \gamma = -\frac{1}{N} \log_{e} (|G|^{2}),
#
# where :math:`G = \langle \psi_{i} | \psi_{f}\rangle`, and :math:`| \psi_{i}\rangle` and :math:`| \psi_{f} \rangle` are the initial and final states,
# respectively. As the system evolves, we'll keep calculating :math:`\gamma`. If it changes discontinuously,
# then a dynamical phase transition has happened.
#
#
# The function below calculates :math:`\gamma` at time :math:`T`.
def rate_function(H, T, N):
    probability_list = time_evolution_circuit(H, T)
    mag_G_squared = probability_list[0]
    return -1 / N * np.log(mag_G_squared)

######################################################################
# Let's now calculate :math:`\gamma` at different times to see how it evolves. Finally, let's graph the value of :math:`\gamma` versus time to see if a dynamical phase transition happens.
#

rate_function_list = []

# time step size for time evolution
deltaT = 0.05

num_time_steps = 50

for i in range(num_time_steps):
    rate_function_list.append(rate_function(H, i * deltaT, N))

plt.plot(np.linspace(0, deltaT * (num_time_steps-1), num_time_steps), rate_function_list)
plt.xlabel("time")
plt.ylabel(r"Rate function, $\gamma$")
plt.title("Rate Function versus time")
plt.legend(["N=" + str(N)])
plt.show()

######################################################################
# Notice the discontinuous change at :math:`t = 1.5`. There, the derivative :math:`\frac{d \gamma}{d t}` is undefined. 
# Phase transitions are characterized by discontinuous changes. So, the discontinuity suggests that a phase change is happening. 
# This conclusion is supported by classical numerical simulations that show a phase
# transition at :math:`t = 1.5` [#Nicola2021]_.
#
# Summary 
# -------
# In this demo, we have shown how you can use quantum computers to simulate quantum phase
# transitions in the 1D and 2D quantum Ising models. We've also shown how to use quantum computers to see dynamical quantum phase transitions.
#
# Acknowledgement 
# ---------------
# Damian Pope would like to thank Professor Matthew Johnson (Perimeter Institute for Theoretical Physics and
# York University) for insightful discussions on quantum phase transitions in cosmology and quantum computing.
# 
# 
# References 
# ------------
#
# .. [#Vojta2002]
#    Thomas Vojta, in K.H. Hoffmann and M. Schreiber (Eds): Computational Statistical Physics, Springer, Berlin 2002 
#
# .. [#Mazumdar2019]
#    Anupam Mazumdar and Graham White. "Cosmic phase transitions: their applications and experimental signatures" Rep. Prog. Phys. 82, 076901, 2019 
#
# .. [#Mueller2023]
#    Niklas Mueller et al. "Quantum Computation of Dynamical Quantum Phase Transitions and Entanglement Tomography in a Lattice Gauge Theory" PRX Quantum 4, 030323, 2023 
#
# .. [#Smith2019]
#    Adam Smith, Bernhard Jobst, Andrew G. Green, and Frank Pollmann. "Crossing a topological phase transition with a quantum computer" `arXiv:1910.05351 [cond-mat.str-el] <https://arxiv.org/abs/1910.05351>`__, 2019 
#
# .. [#Haghshenas2024]
#    Reza Haghshenas et al. "Probing critical states of matter on a digital quantum computer" `arXiv:2305.01650 [quant-ph] <https://arxiv.org/pdf/2305.01650>`__, 2024 
#
# .. [#Chertkov2022]
#    Eli Chertkov, et al., "Characterizing a non-equilibrium phase transition on a quantum computer", `arXiv:2209.12889 [quant-ph] <https://arxiv.org/abs/2209.12889>`__, 2022 
#
# .. [#Thompson2023]
#    Shane Thompson and George Siopsis. "Quantum Computation of Phase Transition in Interacting Scalar Quantum Field Theory" `arXiv:2303.02425 [quant-ph] <https://arxiv.org/abs/2303.02425>`__, 2023 
#
# .. [#Vodeb2025]
#    Jaka Vodeb et al., "Stirring the false vacuum via interacting quantized bubbles on a 5,564-qubit quantum annealer", Nature Physics, 21, 386, 2025 `https://www.nature.com/articles/s41567-024-02765-w <https://www.nature.com/articles/s41567-024-02765-w>`__ 
#
# .. [#Kandala2017]
#    Abhinav Kandala et al., "Hardware-efficient Variational Quantum Eigensolver for Small Molecules and Quantum Magnets", `arXiv:1704.05018 [quant-ph] <https://arxiv.org/abs/1704.05018>`__ 2017 
#
# .. [#Subir2011]
#    Sachdev Subir, "Quantum Phase Transitions", Cambridge University Press, Cambridge 2011 
#
# .. [#Blote2002]
#    Henk W. J. Blöte and Youjin Deng. "Cluster Monte Carlo simulation of the transverse Ising model", Phys. Rev. E 66, 066110, 2002. (See Table II, row labelled 'square lattice'); 
#
# .. [#Hashizume2022]
#    Tomohiro Hashizume, Ian P. McCulloch, and Jad C. Halimeh. "Dynamical phase transitions in the two-dimensional transverse-field Ising model", Phys. Rev. Research 4, 013250, 2022 (See Figure 1 and Section II) 
#
# .. [#Heyl2013]
#    Markus Heyl, Anatoli Polkovnikov, and Stefan Kehrein. "Dynamical Quantum Phase Transitions in the Transverse-Field Ising Model", Phys. Rev. Lett. 110 135704 (2013) 
#
# .. [#Nicola2021]
#    Stefano De Nicola, Alexios A. Michailidis, and Maksym Serbyn. "Entanglement View of Dynamical Quantum Phase Transitions", Phys. Rev. Lett. 126 040602 (2021), Figure 1 (d) 
