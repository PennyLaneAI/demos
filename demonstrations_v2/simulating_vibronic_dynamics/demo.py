r"""
Simulating Vibronic Dynamics
############################

Simulating static systems will not get us very far.

It is difficult to find an advancement in science or technology that did not incorporate simulation into its discovery process. The field of simulation, as a result, is constantly striving to strengthen capabilities in favour of achieving bigger, better results. The tools that we have available to us at present are powerful, but their limitations show as we push forward in our aspirations. 

Classical simulation is well poised to handle systems that, for the most part, stay still. Ground state energies, isolated systems, and other scenarios that deal with a small number of possible states are important but far from everything. Expanding beyond these limited models into dynamic scenarios, though, has proven to be technologically difficult. One such area of interest is `vibronics <https://en.wikipedia.org/wiki/Vibronic_coupling>`_, concerned specifically with electronic and nuclear vibrational interactions. Vibronic simulations push beyond the `Born-Oppenheimer approximation <https://en.wikipedia.org/wiki/Born%E2%80%93Oppenheimer_approximation>`_, opening many useful doors in theoretical chemistry. 

When we go beyond Born-Oppenheimer, it is no longer possible to isolate electronic degrees of freedom (DOFs) from nuclear DOFs, meaning their paths cannot be simulated separately. Long story short, if we want to capture the realistic dynamics of a molecular system, we lose access to approximations that allow for a reduction in the number of simulation variables and, as a result, the size of the system will very quickly exceed standard computational resources. So, what comes to mind when we want to simulate realistic atomic system dynamics that scale exponentially in resource requirements and requires nuanced modelling of phenomena such as entanglement? Quantum computing!

In `"Quantum Algorithm for Vibronic Dynamics" <https://arxiv.org/abs/2411.13669>`_ Motlagh et al. propose a novel quantum algorithm for vibronic simulations. This algorithm leverages several quantum tools such as `phase gradient states <https://pennylane.ai/compilation/phase-gradient>`_, `Trotterization <https://pennylane.ai/challenges/a_simple_trotterization>`_, and `QROM coefficient loading <https://pennylane.ai/demos/tutorial_intro_qrom>`_. This paper lays out an approach that can be generalized to various vibronic Hamiltonians with arbitrary diabatic states and mode interaction specifications, as is shown in `"Quantum Algorithm for Simulating Non-Adiabatic Dynamics at Metallic Surfaces" <https://arxiv.org/pdf/2601.16264>`_. 

Why don't we add this to our quantum toolkits!

The General Vibronics Algorithm
===============================
To carry out a simulation on a specific vibronic Hamiltonian, we need to take the particularities of that system into consideration. Before considering where each Hamiltonian differs, though, we can explore what needs they share. To carry out time-evolution for a vibronic system, the following steps must be fulfilled.

1. Load the initial state of the system into the simulation,
2. Partition the terms of the Hamiltonian into mutually non-commuting fragments for Trotterization,
3. Diagonalize fragments for exponentiation, if necessary,
4. Carry out Trotterization to evolve the Hamiltonian in time,
5. Read out your desired observable.

Sounds like a walk in the park! Or, at least, the start we need to begin building the core of our vibronic simulation. 

Grid Encoding
-------------
To keep our operations efficient, it is important to select a space representation that allows for easy basis transformation. A standard approach is the use of spatial grid discretization to represent the system's operators in real-space. When dealing with a dynamic system, it is expected that we will be mainly concerned with the position operator :math:`Q` and momentum operator :math:`P`, so this representation is convenient. Letting :math:`k` be the number of qubits per mode, the number of grid points required is :math:`K=2^k`. From this, we will take for granted that the eigenvectors of :math:`Q` are given by

.. math::
   Q|x\rangle = \Delta(x-K/2)|x\rangle,

where :math:`\Delta=\sqrt{2\pi/K}` is the grid spacing term and :math:`x` is a position-basis grid point index. It is more convenient to take the signed integer representation

.. math::
   Q|x\rangle = \Delta \cdot x |x\rangle,

letting :math:`x \in \{-\frac{K}{2}, -\frac{K}{2}+1, ..., \frac{K}{2}-1\}`. This discretization method will be kept in mind throughout the implementation as we define our parameters.

The Kinetic Step
----------------
Since position and momentum are non-commuting operators, they must be sliced for Trotterization. To review, Trotterization is a Hamiltonian simulation method used to carry out time evolution in a non-commuting system. It addresses the fact that, if a Hamiltonian is constructed from non-commuting operators, it cannot be exponentiated as a whole and the time evolution operator :math:`e^{iHt}` cannot be conventionally realized. As such, the Hamiltonian can be separated into groups of non-commuting operators called *fragments* to be individually exponentiated and interleaved in partial time steps to simulate simultaneous time evolution.

The kinetic energy Hamiltonian fragment is, comparatively, simple to establish and evolve in time. Cutting to the chase, the kinetic step should:

1. Perform a basis switch on the input state to momentum space,
2. Introduce the kinetic energy coefficients to the state register,
3. Square the state values in the momentum state representation,
4. Apply a rotation to the state register,
5. Uncompute. 

It is specified in "Quantum Algorithm for Vibronic Dynamics" that the basis transformation should take place via the sequence

.. math::
   P = QFT^\dagger X_{k-1} Q X_{k-1} QFT

in which the X-gates are applied as a means of bit-ordering, which is crucial throughout this implementation. These additional gates are sometimes unnecessary depending on the symmetry of the system, such as in symmetric cases which don't necessarily require string reversal upon conversion. For now, though, it will be assumed to be needed.

We also know that the kinetic energy coefficients will always be proportional to :math:`P^2` (where :math:`P` is the momentum operator) and will be independent of electronic state. This makes our lives easy! Rather than loading state-dependent coefficients into the system, we can simply encode calculated coefficients into the register. An intuitive way to understand these coefficients is to take them as describing the motion of the nuclear states. If :math:`C_{kin}=0`, the nucleus is understood to be frozen and, therefore, there is nothing vibronic to simulate.

Executing the kinetic energy step is, at this point, merely a question of determining which tools are ideal to carry out our desired procedure. Once the basis has been switched according to the defined sequence and the coefficient binary representation has been encoded on the appropriate register, an :func:`~qp.OutPoly` operation targeting :math:`f(x)=(x-K/2)^2` (which is equivalent to :math:`x^2` in the signed-integer picture) can be used to carry out the squaring operation on the state register.

To apply the rotation step, a phase gradient state approach will be implemented. Phase gradient rotations work by storing a pre-computed, catalytic state that holds position-dependent rotation angles. Via addition, these angles can be applied to corresponding qubits to carry out cheap rotations. It takes the general form

.. math::
   |R\rangle = \frac{1}{2^{b-1}}\sum e^{-i2\pi y/2^b}|y\rangle,

where :math:`b` is the number of wires in the gradient register.

A major benefit of phase gradient states are that they only need to be prepared once, since the addition step does not change it. To carry out the addition without error, a "slicing" step should be implemented to determine how many wires in the allocated registers are actually required due to the weighted binary representation used in this implementation. This is, essentially, equivalent to carrying out a `logical left shift <https://en.wikipedia.org/wiki/Logical_shift>`_ in classical computing. The rotation procedure is as follows for each index in the simulation grid

1. Determine the current iteration weight by computing the difference between the total grid points and the current index,
2. Compute the number of wires needed to carry out the addition step by finding the difference between the gradient register size and the current weight (since Python indexes at 0, also subtract an extra 1),
3. If the number of wires is 1, perform a :func:`~qp.CNOT` operation between the squared state and the phase gradient register,
4. If the number of wires is not 1, perform a controlled Semi-Out-of-Place addition between the required state wires and required gradient wires, controlled by the squared state register.

The result of this procedure should be a set of electron states that have accumulated a phase as a function of their states and, thus, undergone a time step. This is executed out in the function ``KineticStep()``.

.. figure:: ../demonstrations_v2/simulating_vibronic_dynamics/KineticStepCircuit.png
   :align: center
   :width: 700px
    
   *Kinetic energy step circuit diagram*
"""
import pennylane as qp
import numpy as np
import pennylane.estimator as qre
import math

def KineticStep(time_step, kinetic_coeffs, num_modes, state_wires, gradient_wires, coeff_wires, scratch_wires, cache_wires):
    k = len(state_wires[0])
    K = 2**k
    b = len(gradient_wires)
    AdjointSemiAdder = qp.adjoint(qp.SemiAdder)

    #Set function to be executed by OutPoly()
    def f(x):
        return (x - K//2)**2 #Signed Integer Representation

    #Perform basis transformationD
    for mode in range(num_modes):
        qp.QFT(wires = state_wires[mode])
        qp.X(wires = state_wires[mode][0])

    #Loop full computational procedure over all modes
    for i in range(num_modes):

        #Compute coefficients
        KinCoeffRaw = (kinetic_coeffs[i]*time_step*(2**b)/(2*K))
        C = int(np.floor(KinCoeffRaw+0.5))

        #Flag if the nucleus if frozen
        if C == 0:
            print(f"WARNING: kinetic underflow (raw={KinCoeffRaw:.3f}); nuclei frozen. Raise b or dt.")
        C_binary = format(C, f'0{len(coeff_wires)}b')

        #Encode coefficients
        for j, bit in enumerate(C_binary):
            if bit == '1':
                qp.X(wires=coeff_wires[j])

        #Square state
        qp.OutPoly(f, input_registers = [state_wires[i]], output_wires = cache_wires)

        #Add the squared state to the phase gradient register
        for point in range(2*k):
        #Control on the spatial state register
            ctrl_wire = [cache_wires[point]]

            #Index to the current position in the register
            weight = 2*k - 1 - point
            target_length = len(gradient_wires) - weight

            if target_length <= 0:
                continue

            #Index the coefficient wires to the required size
            x_wire_current = coeff_wires
            if len(x_wire_current)>target_length:
                x_wire_current = coeff_wires[(len(coeff_wires)-target_length):]

            y_wire_current = gradient_wires[:target_length]

            #Apply addition operator based on the size of the numbers being added
            if target_length == 1:
                qp.ctrl(qp.CNOT, control = ctrl_wire)(wires=[x_wire_current[-1], y_wire_current[0]])
            elif target_length >= 2:
                qp.ctrl(qp.SemiAdder, control = ctrl_wire)(x_wires = x_wire_current, y_wires = y_wire_current, work_wires = scratch_wires)

    #Uncompute
        qp.adjoint(qp.OutPoly)(f, input_registers = [state_wires[i]], output_wires = cache_wires)

        for j, bit in enumerate(C_binary):
            if bit == '1':
                qp.X(wires=coeff_wires[j])

    for mode in range(num_modes):
        qp.X(wires = state_wires[mode][0])
        qp.adjoint(qp.QFT)(wires=state_wires[mode])

        
###############################################################################
# With this defined, the potential energy step can now be tackled.
#
# The Potential Step
# ------------------
# The goal of the potential energy step is to construct the full potential energy operator (with coefficients) for each electron state. To do this, we must consider the state-dependent potential coefficients and the vibrational modes of the system. The operations that will need to be carried out by this function are:
#
# 1. Load the electron state-dependent coefficient terms into the coefficient register,
# 2. If there are multiple vibrational mode states involved in this step, multiply them together,
# 3. Multiply the full mode state (either a single mode or the product of multiple modes, depending on step 2) with the coefficient register,
# 4. Add the product of the mode state and coefficient register to the phase gradient register,
# 5. Uncompute.
#
# Okay, we can do that! 
# 
# We mentioned earlier that the kinetic energy coefficients will always be state independent, but the same cannot be said for the potential energy coefficients. So, in this case the coefficients must be determined and stored in a bit-position-dependent fashion prior to Trotterization. The method used to execute this step is highly dependent on the system itself. For now, we will assume that this has been handled externally and simply passed into our potential step function for use.
#
# As stipulated in step 2, the structure of our potential step depends on the dimensions of the mode state being passed to us. For now, we will focus on two scenarios: the linear case (in which a single mode state is being passed to us) and the quadratic case (in which two mode states are being passed to us). If our system is quadratic, we simply need to apply an :func:`~qp.OutPoly` operator that multiplies the two mode registers together, just like we did in the kinetic step. If our system is linear, we don't need to worry about this and can skip right to the addition!
#
# The addition step must be carried out using the same shifting procedure outlined in the kinetic step to ensure proper dimensionality. In the linear case, the mode state register acts as the control, adding the state directly to the phase gradient register. In the quadratic case, the addition into the phase gradient register is, instead, controlled by the product of the two mode states. Feel free to take a second to digest that, or glance down at the diagram that (hopefully) makes that a bit clearer. After this is carried out, we can safely uncompute and move on with our task.
#
# .. figure:: ../demonstrations_v2/simulating_vibronic_dynamics/PotentialEnergyStep.png
#    :align: center
#    :width: 700px
#    
#    *Potential energy step circuit diagram*
#
def PotentialStepLinear(fragment, load_coeffs, mode, time_coeffs, state_wires, electron_wires, gradient_wires, coeff_wires, cache_wires, scratch_wires):

    k = len(state_wires[mode])
    K = 2**k

    AdjointSemiAdder = qp.adjoint(qp.SemiAdder)

    #Load pre-determined, electron state dependent coefficients
    load_coeffs(fragment, time_coeffs, electron_wires, coeff_wires, scratch_wires)

    qp.OutPoly(lambda x: (x-K//2), input_registers = [state_wires[mode]], output_wires = cache_wires)

    for point in range(2*k):
        #Control on the spatial state register
        ctrl_wire = [cache_wires[point]]

        #Index to the current position in the register
        weight = 2*k - 1 - point
        target_length = len(gradient_wires) - weight

        if target_length <= 0:
            continue

        x_wire_current = coeff_wires
        if len(x_wire_current)>target_length:
            x_wire_current = coeff_wires[(len(coeff_wires)-target_length):]

        #Bit-wise shifts
        y_wire_current = gradient_wires[:target_length]

        #If we are dealing with the first point, subtract rather than add
        if point == 0:
            if target_length == 1:
                qp.ctrl(qp.CNOT, control = ctrl_wire)(wires=[x_wire_current[-1], y_wire_current[0]])
            elif target_length >= 2:
                qp.ctrl(AdjointSemiAdder, control = ctrl_wire)(x_wires = x_wire_current, y_wires = y_wire_current, work_wires = scratch_wires)
        #Otherwise, always add
        else:
            if target_length == 1:
                qp.ctrl(qp.CNOT, control = ctrl_wire)(wires=[x_wire_current[-1], y_wire_current[0]])
            elif target_length >= 2:
                qp.ctrl(qp.SemiAdder, control = ctrl_wire)(x_wires = x_wire_current, y_wires = y_wire_current, work_wires = scratch_wires)

    qp.adjoint(qp.OutPoly)(lambda x: (x-K//2), input_registers = [state_wires[mode]], output_wires = cache_wires)

    qp.adjoint(load_coeffs)(fragment, time_coeffs, electron_wires, coeff_wires, scratch_wires)


def PotentialStepQuadratic(fragment, load_coeffs, mode1, mode2, time_coeffs, state_wires, electron_wires, gradient_wires, coeff_wires, cache_wires, scratch_wires):

    k = len(state_wires[mode1])
    K = 2**k

    AdjointSemiAdder = qp.adjoint(qp.SemiAdder)

    load_coeffs(fragment, time_coeffs, electron_wires, coeff_wires, scratch_wires)

    qp.OutPoly(lambda x0,x1: (x0-K//2)*(x1-K//2), input_registers = [state_wires[mode1], state_wires[mode2]], output_wires = cache_wires)

    for point in range(2*k):
        #Control on the spatial state register
        ctrl_wire = [cache_wires[point]]

        #Index to the current position in the register
        weight = 2*k - 1 - point
        target_length = len(gradient_wires) - weight

        if target_length <= 0:
            continue

        x_wire_current = coeff_wires
        if len(x_wire_current)>target_length:
            x_wire_current = coeff_wires[(len(coeff_wires)-target_length):]

        #Bit-wise shifts
        y_wire_current = gradient_wires[:target_length]

        #If we are dealing with the first point, subtract rather than add
        if point == 0:
            if target_length == 1:
                qp.ctrl(qp.CNOT, control = ctrl_wire)(wires=[x_wire_current[-1], y_wire_current[0]])
            elif target_length >= 2:
                qp.ctrl(AdjointSemiAdder, control = ctrl_wire)(x_wires = x_wire_current, y_wires = y_wire_current, work_wires = scratch_wires)
        #Otherwise, always add
        else:
            if target_length == 1:
                qp.ctrl(qp.CNOT, control = ctrl_wire)(wires=[x_wire_current[-1], y_wire_current[0]])
            elif target_length >= 2:
                qp.ctrl(qp.SemiAdder, control = ctrl_wire)(x_wires = x_wire_current, y_wires = y_wire_current, work_wires = scratch_wires)

    #Uncompute
    qp.adjoint(qp.OutPoly)(lambda x0,x1: (x0-K//2)*(x1-K//2), input_registers = [state_wires[mode1], state_wires[mode2]], output_wires = cache_wires)

    qp.adjoint(load_coeffs)(fragment, time_coeffs, electron_wires, coeff_wires, scratch_wires)

###############################################################################
# Now that we have the tools we need to carry out a system-wide time evolution, how exactly can we execute this?
#
# Assembling the Trotter Step
# ---------------------------
# In "Quantum Algorithm for Vibronic Dynamics", it is specified that either a first or second order Trotterization can be carried out. While first order Trotterization is less resource intensive, second order allows for reduced `Trotter error <https://simons.berkeley.edu/sites/default/files/docs/15639/trottererrortheorysimons.pdf>`_ and sets us up for a useful uncompute trick later on. In general, a second order Trotterization is given by
#
# .. math::
#    e^{-iHt} = (e^{-iH_1 \Delta t/2r}e^{-iH_2 \Delta t/r}e^{-iH_1 \Delta t/2r})^r
#
# where :math:`H_1` and :math:`H_2` are non-commuting Hamiltonian fragments, :math:`\Delta t` is a time step, and :math:`r` is the total number of Trotter steps taken. In our case, this implies taking half a time step in kinetic energy, a full time step in potential energy, and another half step forward in energy. 
#
# Before we can assemble our Trotter step, though, we are missing a crucial piece. In order to Trotterize a fragmented Hamiltonian, each fragment must be exponentiated. To do this efficiently and to maintain compatibility with architecture, each fragment must be *diagonal*. In a vibronic system, we are heavily concerned with coupling between atoms and, therefore, must consider off-diagonal terms to fully capture our system. Protocol must be put in place to *diagonalize* off-diagonal fragments. This can be done using cheap `Clifford gates <https://pennylane.ai/demos/tutorial_clifford_circuit_simulations>`_, but the exact method required to diagonalize a given Hamiltonian varies. Thus, the full Trotter step function should
#
# 1. Perform a kinetic half-step on the system,
# 2. Diagonalize each fragment to represent coupling behaviour,
# 3. Perform a full potential step for each fragment,
# 4. Uncompute the diagonalization step,
# 5. Perform another kinetic half-step on the system.
#
# There are some slight system dependent specificities of this step, but the framework is general. We will keep this list in mind as we move forward.
#
# With the foundation laid, let's introduce our star Hamiltonian.
#
# The Köppel-Domcke-Cederbaum Hamiltonian
# =======================================
# The Köppel-Domcke-Cederbaum (KDC) Hamiltonian is a well known, straightforward representation of a coupled nuclear and electronic state. It takes the general form
# 
# .. math::
#    H = \mathcal{I}_{el} \otimes (T_{nuc}+V_0)+\textbf{W}
#
# Where :math:`T_{nuc}=\frac{1}{2}\sum_r \omega_r P_r^2` is the vibrational kinetic operator, :math:`V_0 = \frac{1}{2}\sum_r \omega_r Q_r^2`, and :math:`\textbf{W}` is the "diabatic potential", which is, essentially, a coupling matrix [#Motlagh2025]_. The source paper gives the expansion
#
# .. math::
#    \textbf{W'}_{ij}(\vec{Q})=\lambda^{(i,j)}+\sum_r a_r^{(i,j)}Q_r+\sum_{rr'}b_{rr'}^{(i,j)}Q_rQ_r'
#
# which looks quite intimidating but, in reality, we have already dealt with it! We can interpret this expression by understanding :math:`\lambda^{(i,j)}`, :math:`a_r^{(i,j)}`, and :math:`b_{rr'}^{(i,j)}` as coupling coefficients and :math:`\vec{Q}` as vibrational mode coordinates [#Motlagh2025]_. In this truncation, we deal with only the linear and quadratic coordinate terms, which is exactly what we handled in the potential step functions! This further validates our expectation that the Hamiltonian can be fragmented into terms dependent on :math:`P` (kinetic terms) and terms dependent on :math:`Q` (potential terms).
#
# From this, we can take the kinetic and potential operators as
#
# .. math::
#    T=\mathcal{I}_{el} \otimes \sum_{r=0}^{M-1}\frac{\omega_r}{2}P_r^2
#
# and
#
# .. math::
#    V = (\mathcal{I}_{el}\otimes V_0+\textbf{W'})=\sum_{i,j=0}^{N-1}|j\rangle\langle i| \otimes V_{ji}
#
# respectively, where :math:`V_{ij}` represents the expansion of :math:`\textbf{W'}` in coefficient form, which will be detailed later.
#
# Carrying out a time evolution of the KDC Hamiltonian is useful for spectroscopy and energy transfer dynamics, among other things. For this demonstration, we will aim to simulate electron population evolution for a small, simple system.
#
# Initial State Definition
# ------------------------
# In "Quantum Algorithm for Vibronic Dynamics", the initial state of the KDC system is taken to be a simple vertical excitation represented in product form in relation to electron state :math:`j` as 
#
# .. math::
#    |\psi_0\rangle = |j\rangle_{el} \bigotimes_{r=0}^{M-1}|\chi_0\rangle.
#
# Here,
#
# .. math:: 
#    |\chi_0\rangle = \frac{1}{Z}\sum_{x=0}^{K-1} \text{exp}\left(\frac{-\pi \cdot (x-\frac{K}{2})^2}{K}\right) |x\rangle
#
# is the Hermite-Gauss function representation of the harmonic oscillator ground state, where :math:`Z` is a normalization constant and :math:`x` is, again, a position-basis grid point index. It is stated that one can choose to begin with a superposition state to enable functionalities such as spectroscopy. In this demo, we will be targeting as simple electronic state population time evolution, so an initial superposition is not necessary. This state can be generated using ``KDCStatePrep()``.

#State Preparation - Vertical Excitation of System
def KDCStatePrep(k):
    K = 2**k
    x = np.arange(K)

    amplitudes = np.exp((-np.pi*((x-(K/2))**2))/(K))
    norm_factor = np.linalg.norm(amplitudes)

    chi0 = amplitudes/norm_factor

    return chi0
###############################################################################
# Diagonalization
# ---------------
# As previously stated, it is necessary for here that each fragment of the Hamiltonian is diagonal when exponentiated. Since we are dealing with systems involving coupling, it is inevitable that some fragments will involve non-diagonal configurations.  Motlagh et al. lay out a Clifford gate based scheme for `block-diagonalization <https://pennylane.ai/compilation/diagonal-unitary-decomp/details>`_ that enables uniform exponentiation. To understand the scheme, we must first take each Hamiltonian fragment to be given as
#
# .. math::
#    H_m = \sum_{j=0}^{N-1}|j\rangle \langle m \oplus j|\otimes V_{j,m\oplus j},
#
# recalling :math:`V_{ji}` holds the position operator expansion term, :math:`m` is the fragment index, and :math:`j` is the electronic state index. Since :math:`|j\rangle \langle m \oplus j|\otimes` constructs the matrix geometry of the fragment (being the row and column terms of the Hamiltonian matrix), the difference between :math:`j` and :math:`m\oplus j` (representing the `Hamming weight <https://en.wikipedia.org/wiki/Hamming_weight>`_ in this case) will dictate how the block should be treated. The logic is as follows:
#
# 1. IF :math:`m=0`, we are dealing with a diagonal fragment. Our work here is done!
# 2. IF :math:`j` and :math:`m\oplus j` differ by 1, we can achieve diagonalization by sandwiching the fragments between Hadamard gates.
# 3. ELSE, we must construct a unitary operation using a qubit that satisfies option 2 as a control for a CNOT operation applied to all other qubits in the fragment to bring the Hamming weight down to 1, enabling diagonalization via Hadamard sandwich.
#
# .. figure:: ../demonstrations_v2/simulating_vibronic_dynamics/Diagonalization.png
#    :align: center
#    :width: 700px
#    
#    *Clifford gate diagonalization scheme* [#Motlagh2025]_
#
# This logic can be implemented simply by comparing the two focus indices and applying the required gates.

#Diagonalization Scheme
def KDCDiag(fragment, electron_wires):
    bits = []
    weight = 0

    for j in range(len(electron_wires)):
        if (fragment >> j) & 1:
            weight += 1
            bits.append(j)

    if weight == 1:
        qp.Hadamard(wires = electron_wires[bits[0]])
    elif weight > 1:
        ctrl_wire = electron_wires[bits[0]]
        for bit in bits[1:]:
            qp.CNOT([ctrl_wire, electron_wires[bit]])
        qp.Hadamard(wires=ctrl_wire)
###############################################################################
# We're almost there!
#
# Revisting the Potential Step
# ----------------------------
# Previously, we defined two functions to be used for potential energy evolution in our Trotterization scheme, one that addresses the linear case and one that addresses the quadratic case. The way in which these  are applied will vary by application, but we can take a relatively simple approach for the KDC Hamiltonian. Given an input of mode states, we can evaluate if an entry is a singular integer value, a list containing a single entry, or a list containing two entries, each of which are possible valid inputs. The first two scenarios will be taken as equivalent to a linear case while the third requires the multiplication step of the quadratic function. ``KDCFrag()`` handles this using simple evaluation logic.

#Fragmentation Scheme
def KDCFrag(fragment, load_coeffs, mode_list, coeff_data, state_wires, electron_wires, gradient_wires, coeff_wires, cache_wires, scratch_wires):
    for entry in mode_list:
                if isinstance(entry, int):
                    PotentialStepLinear(fragment, load_coeffs, entry, coeff_data, state_wires, electron_wires, gradient_wires, coeff_wires, cache_wires, scratch_wires)
                if isinstance(entry, tuple) and len(entry) == 1:
                    PotentialStepLinear(fragment, load_coeffs, entry[0], coeff_data, state_wires, electron_wires, gradient_wires, coeff_wires, cache_wires, scratch_wires)
                if isinstance(entry, tuple) and len(entry) == 2:
                    mode1 = entry[0]
                    mode2 = entry[1]
                    PotentialStepQuadratic(fragment, load_coeffs, mode1, mode2, coeff_data, state_wires, electron_wires, gradient_wires, coeff_wires, cache_wires, scratch_wires)
###############################################################################
# Finishing Touches for KDC
# -------------------------
# One conclusive puzzle piece is the method of coefficient loading for the KDC case. Motlagh et al. specify that the coefficients relevant to the KDC case can be easily loaded via a QROM. Using PennyLane's built in :func:`~qp.QROM` function, we can simply take computed coefficients and pass them in to the potential energy step of our choosing.

def LoadCoeffsKDC(fragment, output, electron_wires, coeff_wires, scratch_wires):
    fragment_coeffs = output[fragment]
    
    qp.QROM(fragment_coeffs, control_wires = electron_wires, target_wires = coeff_wires, work_wires = scratch_wires)

###############################################################################
# It was hinted before that taking the second order Trotterization approach allows for a convenient uncompute trick to be implemented. Motlag et al. define the second order Trotter formula to be
#
# .. math::
#    U_2(\theta)=\prod_{m=0}^N e^{i\theta H_m} \prod_{m=N}^0 e^{i\theta H_m},
#
# in which the potential step has been split into two half-steps as well, one evolving forward and one evolving backward. Since a basis change needs to occur between position and momentum, splitting the potential step into these mirrored steps lands a :math:`QFT` next to a :math:`QFT^\dagger`, so we can easily maintain the proper basis without adding additional transformations. Phew!

def TrotterStepKDC(k, dt, frag_list, coupler, PotentialStep, KineticStep, kinetic_args, coupler_args, potential_args):
    K = 2**k

    half_dt = dt/2

    KineticStep(half_dt, *kinetic_args)

    for fragment in frag_list:
        #Diagonalization function
        coupler(fragment, *coupler_args)
        #Pass a function that can handle the potential step in the linear or quadratic case
        PotentialStep(fragment, *potential_args)
        qp.adjoint(coupler)(fragment, *coupler_args)

    #Second-order, reversed potential step
    for fragment in reversed(frag_list):
        #Diagonalization function
        coupler(fragment, *coupler_args)
        PotentialStep(fragment, *potential_args)
        qp.adjoint(coupler)(fragment, *coupler_args)

    KineticStep(half_dt, *kinetic_args)
###############################################################################
# Finally, the registers can be defined according to the requirements of the system.
def WirePrepKDC(num_modes, k, n, delta):

    precision_qubits = int(math.ceil(np.log2(1/delta))) #b

    nuclear_modes = {f"mode_{i}": k for i in range(num_modes)} #Account for linear vs quadratic

    registers = {
        "electrons": n,
        "states": nuclear_modes,
        "gradient": precision_qubits,
        "coefficients": precision_qubits,
        "scratch": precision_qubits + 1,
        "cache": 2*k
    }

    return qp.registers(registers)
###############################################################################
# Time Evolution of Electronic State Population
# ---------------------------------------------
# Finally, we have adequately built up the skeleton of the KDC Hamiltonian simulation! Now, we can combine our previously constructed tools to carry out a Trotterization of this system and observe the population dynamics for a short time scale. 
#
# To keep things simple (and computationally feasible), we will begin by defining a small, single-mode system with 2 electron states and 1 nuclear state. The potential coefficients will be taken to be a simple array of values that will soon be scaled by the required factors. It is worth noting that the flooring step in the kinetic energy coefficient calculation requires a minimum number of precision bits to be present in the system to avoid flooring to 0 and, therefore, supressing all coupling effects. A such, ``delta`` should be small enough to achieve :math:`b\geq 6`.

#Trotterization
qp.decomposition.enable_graph() #enable graph-based decomposition for performance

time_steps = 10
k = 2
n = 1
num_modes = 1
delta = 0.03
mode_list = [0]
omega = [1]
coeff_data = [
    #  |0>    |1>
    [ 1.0,   0.0 ],   # Fragment 0: Diagonal potential energy terms
    [ -1.3,   1.3 ]    # Fragment 1: Off-diagonal electronic coupling terms
]
dt = 0.4
###############################################################################
# Our next administrative task is to call upon ``WirePrepKDC()`` to initialize our registers and assign names that can be referenced in our Trotter function.

#Initialize and label registers
regs = WirePrepKDC(num_modes, k, n, delta)
total_wires = n+(num_modes*k)+(3*int(math.ceil(np.log2(1/delta))))+1+(2*k)

electron_wires = regs["electrons"]
state_wires = [regs[f"mode_{i}"] for i in range(num_modes)]
gradient_wires = regs["gradient"]
coeff_wires = regs["coefficients"]
scratch_wires = regs["scratch"]
cache_wires = regs["cache"]
###############################################################################
# To ensure our coefficients are physically accurate and compatible with the QROM that will be used in the KDC potential step. In the source paper, the full coefficient representation is given as
#
# .. math::
#    \Delta^{\alpha}c^{(j,j)}_\alpha
#
# where :math:`\alpha` is the mode index. Taking the bit-wise representation and considering the time dependence, the full representation of the coefficients that should be passed into the QROM is
#
# .. math::
#    c_{time}=[c_{\alpha} \Delta^\alpha dt \frac{2^b}{2\pi}] \text{mod} 2^b
#
# This form allows for easy computation and conversion to the list-of-bit format required by the QROM. For this demo, :math:`\alpha` will be fixed to 1 since we are only dealing with linear mode behaviour. This can be easily changed depending on the needs and conditions of the system.

#Introduce scaling factor and reformat for compatibility with QROM
width = len(coeff_wires)
max_binary = 2**width
Delta = np.sqrt(2*np.pi/(2**k))
scale = Delta/(2*np.pi)
alpha = 1 #Limiting to linear degree here

time_coeffs = []
for fragment_data in coeff_data:
    fragment_row = []
    for power, val in enumerate(fragment_data):
        v = int(np.round(val * (dt/2) * max_binary * (Delta**alpha) / (2*np.pi))) % max_binary #Include half time step for forward-backward Trotterization
        fragment_row.append(format(v, f"0{width}b"))
    time_coeffs.append(fragment_row)      
################################################################################
# Now, at long last, we can carry out our time evolution. Our implementation function should achieve the following:
#
# 1. Prepare the phase gradient state in the gradient register,
# 2. Prepare the initial state on the electronic state register,
# 3. Carry out the Trotterization protocol for the specified number of time steps,
# 4. Return probability measurements of the electronic state register.
#
# Since we so diligently built up our functionality under the guidance of Motlagh et al.'s innovations, we are well equipped to carry this out smoothly! 

#Define argument lists
kinetic_args = [omega, num_modes, state_wires, gradient_wires, coeff_wires, scratch_wires, cache_wires]
potential_args = [LoadCoeffsKDC, mode_list, time_coeffs, state_wires, electron_wires, gradient_wires, coeff_wires, cache_wires, scratch_wires]
coupler_args = [electron_wires]

dev = qp.device("lightning.qubit", wires=total_wires)
@qp.qnode(dev)
def ElectronPopVibronicsSimulation(steps, gradient_wires, StatePrepFunc, CouplerFunc, PotentialFunc, KineticFunc, kinetic_args, potential_args, coupler_args):
    #Prepare the phase gradient state in the appropriate register
    for wire in gradient_wires:
        qp.X(wires = wire)
    qp.QFT(wires = gradient_wires)

    #Prepare the initial state
    initial_state = StatePrepFunc(k)
    for wire in state_wires:
        qp.StatePrep(state = initial_state, wires = wire)

    #Trotterize
    for t in range(steps):
        TrotterStepKDC(
            k = k,
            dt = dt,
            frag_list = range(2**n),
            coupler = CouplerFunc,
            PotentialStep = PotentialFunc,
            KineticStep = KineticFunc,
            kinetic_args = kinetic_args,
            coupler_args = coupler_args,
            potential_args = potential_args
        )
    return qp.probs(wires=electron_wires)
################################################################################
# Running this simulation for our limited system yields the following result, which shows an incomplete transfer between the ground and first excited states of a two-state system. 
#
# .. figure:: ../demonstrations_v2/simulating_vibronic_dynamics/10StepVibePlot.png
#    :align: center
#    :width: 700px
#  
#    *Electronic state population time evolution after 10 steps with dt=0.4*
# 
# We did it!
#
# Conclusion
# ----------
# Preparing to make the most out of quantum technologies is crucial as we continue to move toward accessible, useful quantum devices. Identifying areas of interest that are known to be important to researchers, companies, and individuals is an important first step. Vibronic simulation has the potential to expand our capacity for material discovery, renewable energy expansion, and drug exploration. Opening this door using the advantages of quantum computation is an important step toward an ever expanding, quantum enabled future.
#
# .. _references:
#
# References
# ----------
# .. [#Motlagh2025] D.\ Motlagh, R. A. Lang, P. Jain, J. A. Campos-Gonzalez-Angulo, W. Maxwell, T. Zeng, A. Aspuru-Guzik, and J. M. Arrazola, "Quantum Algorithm for Vibronic Dynamics: Case Study on Singlet Fission Solar Cell Design," 2025, `arXiv: 2411.13669 <https://arxiv.org/abs/2411.13669>`_.
#
# .. [#Lang2026] R.\ A. Lang, P. Jain, J. M. Arrazola, and D. Motlagh, "Quantum Algorithm for Simulating Non-Adiabatic Dynamics at Metallic Surfaces," 2026, `arXiv: 2601.16264 <https://arxiv.org/abs/2601.16264>`_.