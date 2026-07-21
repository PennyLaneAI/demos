r"""
Simulating Vibronic Dynamics
############################

Simulating static properties will not get us very far.

It is difficult to find an advancement in science or technology that did not
happen through a painstaking process of trial and error. For most of human
history, testing new ideas was only possible in the slow, costly physical world,
making many avenues impossible to explore. With the advent of computers,
this changed. The ability to digitally simulate ideas significantly accelerated
the development cycle of new technologies and cut costs, leading to a substantial increase in the pace of human
innovation. Though we have pushed forward and progressed our technologies
toward the atomic scale where quantum effects become relevant to simulation,
the limitations of our current technologies have been made clear.

When it comes to simulating complex quantum effects, classical simulation 
is well poised to handle systems that, for the most part,
stay still. Ground state energies and other static quantities are good static benchmarks,
but the simulations we need to carry out to continue progressing areas
such as advanced materials discovery are dynamical in nature. This is something 
classical computers just can't seem to crack. 

An important area of interest in dynamic simulation is
`vibronics <https://en.wikipedia.org/wiki/Vibronic_coupling>`_, which is specifically concerned 
with the interaction between electrons and nuclei in
a molecule. Classical simulations tend to take the 
`Born-Oppenheimer approximation
<https://en.wikipedia.org/wiki/Born%E2%80%93Oppenheimer_approximation>`_, which 
assumes electrons and nuclei can be treated as completely separate entities. 
This works very well for ground state energies and simplifies molecular systems
adequately for classical systems to handle. 

When we go beyond Born-Oppenheimer to model dynamic systems, 
it is no longer possible to isolate
electronic degrees of freedom (DOFs) from nuclear DOFs, meaning their paths
cannot be simulated separately. Long story short, if we want to capture the
realistic dynamics of a molecular system, we lose access to approximations that
allow for a reduction in the number of simulation variables and, as a result,
the size of the system will very quickly exceed standard computational
resources. So, what comes to mind when we want to simulate realistic atomic
system dynamics that scale exponentially in resource requirements and require
nuanced modelling of phenomena such as entanglement? Quantum computing!

In `"Quantum Algorithm for Vibronic Dynamics"
<https://arxiv.org/abs/2411.13669>`_ Motlagh et al. propose a novel quantum
algorithm for vibronic simulations. This algorithm leverages several quantum
tools such as `phase gradient states
<https://pennylane.ai/compilation/phase-gradient>`_, `Trotterization
<https://pennylane.ai/challenges/a_simple_trotterization>`_, and `QROM
coefficient loading <https://pennylane.ai/demos/tutorial_intro_qrom>`_ to reach
beyond Born-Oppenheimer and toward more realistic, capable dynamic simulations.

Why don't we add this to our quantum toolkits!

The Köppel-Domcke-Cederbaum Hamiltonian
---------------------------------------
The Köppel-Domcke-Cederbaum (KDC) Hamiltonian is a well known, straightforward
representation of a vibronic system in which the electronic and nuclear
vibrational modes are coupled. It takes the general
form
 
.. math:: 
   H = \mathcal{I}_{el} \otimes (T_{nuc}+V_0)+\textbf{W}

where :math:`T_{nuc}` is the vibrational
kinetic operator, :math:`V_0` is the potential
energy operator, and
:math:`\textbf{W}` is a coupling matrix representing the diabatic
potential [#Motlagh2025]_. The source paper defines this potential
in a truncated quadratic vibronic coupling model as

.. math:: 
   \mathbf{W}'_{ij}(\vec{Q})=\lambda^{(i,j)}+\sum_r a_r^{(i,j)}Q_r+\sum_{r,r'} b_{r,r'}^{(i,j)}Q_rQ_{r^\prime}.

We can interpret this expression by understanding :math:`\lambda^{(i,j)}`,
:math:`a_r^{(i,j)}`, and :math:`b_{rr'}^{(i,j)}` as coupling coefficients and
:math:`\vec{Q}` as mode-dependent position operators [#Motlagh2025]_. In this
truncation, we deal with only the linear and quadratic coordinate terms, which is 
adequate to minimize error and ensure implementability.

We can take the kinetic and potential operators as

.. math:: 
   T=\mathcal{I}_{el} \otimes \sum_{r=0}^{M-1}\frac{\omega_r}{2}P_r^2

and

.. math:: 
   V =\sum_{i,j=0}^{N-1}|j\rangle\langle i| \otimes V_{ji}

respectively, where :math:`P` is the momentum operator,
:math:`V_{ji}` represents the sum of the expansion of
:math:`\textbf{W'}` in coefficient form and the ground-state 
potential :math:`V_0`.

Carrying out a time evolution of the KDC Hamiltonian is useful for
spectroscopy and energy transfer dynamics, among other things. For this
demonstration, we will aim to simulate electron state population evolution for a
small, simple system.


To simulate the time evolution of a specific vibronic Hamiltonian, we need to
take the particularities of that system into consideration. Before considering
where each Hamiltonian differs, though, we can explore what needs they share. To
carry out time-evolution for a vibronic system, the following steps must be
fulfilled.

1. Load the initial state of the system into the simulation,
2. Partition the terms of the Hamiltonian into fragments
   for Trotterization,
3. Diagonalize fragments for exponentiation,
4. Carry out Trotterization to evolve the Hamiltonian in time,
5. Measure the electronic register to extract state populations.

Sounds like a walk in the park! Or, at least, the start we need to begin
building the core of our vibronic simulation. 

Grid Encoding
-------------
To keep our operations efficient, it is important to select a space
representation that allows for easy basis transformation. In classical 
computation, it is difficult to use real space for simulations due to the exponential
memory requirements of storing a grid wave packet. However, using real space provides optimal conditions 
for implementing our Hamiltonian since both potential and kinetic energy operators are diagonal in real space.
In quantum computing, memory is not the main bottleneck and we are free to use real space
as we please.

Letting the total number of required
grid points be :math:`K` (from which we can determine the number of required
qubits in the system :math:`k=\log_2(K)`), we can represent our mode-dependent
:math:`Q` operator as

.. math::
   Q|x\rangle = \Delta(x-K/2)|x\rangle,

where :math:`\Delta=\sqrt{2\pi/K}` is the grid spacing term and :math:`x` is a
position-basis grid point index. It is more convenient to take the signed
integer representation

.. math::
   Q|x\rangle = \Delta \cdot x |x\rangle,

letting :math:`x \in \{-\frac{K}{2}, -\frac{K}{2}+1, \dots, \frac{K}{2}-1\}`. This discretization dictates the 
size of our space and the number of qubits required for our system.

The Kinetic Step
----------------
Since position and momentum are non-commuting operators, they must be fragmented for
:doc:`Trotterization <demos/exploring_trotterization>`. To review, Trotterization is a Hamiltonian
simulation method that addresses the issue of exponentiating non-commuting terms in
a Hamiltonian. In this method, a target Hamiltonian is separated into groups of commuting operators (where these groups
do not commute with each other) called *fragments* to be
individually exponentiated and interleaved in partial time steps to simulate
simultaneous time evolution.

The kinetic energy fragment is, comparatively, simple to establish
and evolve in time. Cutting to the chase, the kinetic step should, for each mode:

1. Perform a change of basis on the input state to momentum space,
2. Square the corresponding mode's register and store the result in an ancillary register,
3. Apply a rotation to the state register via a :doc:`multiplexed phase gradient register <demos/efficient_rotations_with_phase_gradient_states>`,
4. Uncompute the ancillary register. 

Motlagh et al. specify that the basis
transformation should take place via the sequence

.. math::
   P = QFT^\dagger X_{k-1} Q X_{k-1} QFT

in which the X-gates are applied as a means of bit-ordering. These additional gates are sometimes unnecessary
depending on the symmetry of the system, but we will see it through
in our implementation

We also know that the kinetic energy coefficients will always be proportional to
:math:`P^2` and will be independent of the electronic state. This makes our lives easy! Rather than loading
state-dependent coefficients into the system, we can simply encode calculated
coefficients into the initial state. An intuitive way to understand these
coefficients is to take them as describing the motion of the nuclear states. If
the kinetic term coefficient is equal to 0, the nucleus is understood to be frozen and, 
therefore, there is nothing vibronic to simulate.

Once the
basis has been switched, an
:func:`~qp.OutPoly` operation targeting :math:`f(x)=(x-K/2)^2` (which is
equivalent to :math:`x^2` in the signed-integer picture) can be used to carry
out the squaring operation on the state register.

To apply the rotation step, a :doc:`multiplexed phase gradient <demos/efficient_rotations_with_phase_gradient_states>`
approach will be implemented.
Phase gradient rotations work by storing a pre-computed, catalytic state that
holds position-dependent rotation angles. Via quantum addition, these angles can be
applied to corresponding qubits to carry out cheap rotations. It takes the
general form

.. math::
   |R\rangle = \frac{1}{2^{b/2}}\sum e^{i2\pi y/2^b}|y\rangle,

where :math:`b` is the number of wires in the gradient register and determines the
precision of the rotation.

A major benefit of phase gradient states is they only need to be prepared
once, since the addition step does not change it. To carry out the addition
without error, a "slicing" step should be implemented to determine how many
wires in the allocated registers are actually required due to the weighted
binary representation used in this implementation. This is, essentially,
equivalent to carrying out a `logical left shift
<https://en.wikipedia.org/wiki/Logical_shift>`_ in classical computing. 

The result of this procedure is a set of target states that have
accumulated a phase equivalent to the change incurred during a given
kinetic time step.

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

    #Perform basis transformation
    for mode in range(num_modes):
        qp.QFT(wires = state_wires[mode])
        qp.X(wires = state_wires[mode][0])

    #Loop full computational procedure over all modes
    for i in range(num_modes):

        #Compute coefficients
        KinCoeffRaw = (kinetic_coeffs[i]*time_step*(2**b)/(2*K))
        C = int(np.floor(KinCoeffRaw+0.5))

        #Flag if the nucleus is frozen
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
# The goal of the potential energy step is to construct the full potential
# energy operator (with coefficients) for each electron state and coupled
# vibrational states. To do this, we
# must consider the state-dependent potential coefficients and the vibrational
# modes of the system. The operations that will need to be carried out by this
# function are:
#
# 1. Load the electron state-dependent coefficient terms into the coefficient
#    register,
# 2. If there are multiple vibrational mode states involved in this step,
#    multiply them together,
# 3. Multiply the full mode state (either a single mode or the product of
#    multiple modes, depending on step 2) with the coefficient register,
# 4. Add the product of the mode state and coefficient register to the phase
#    gradient register,
# 5. Uncompute.
#
# Okay, we can do that! 
# 
# The state-dependent coefficients must be determined and stored
# in a bit-position-dependent fashion prior to Trotterization. For now, we
# will assume that this has been handled and simply passed into our
# potential step function for use.
#
#
# As shown in our truncated representation of :math:`\textbf{W}`, we are only
# concerned with scenarios with one or two mode states. Thus, in the case our system 
# is quadratic, we simply need to apply an :func:`~qp.OutPoly`
# operator that multiplies the two mode registers together, just like we did in
# the kinetic step. Otherwise, no arithmetic required. The outcome of either of these
# cases is added to the phase gradient register via quantum arithmetic multiplier-adder
# gates, inducing the corresponding rotation.
#
# .. figure::
#    ../demonstrations_v2/simulating_vibronic_dynamics/PotentialEnergyStep.png
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
# Motlagh et al. specify that the coefficients
# can be easily loaded via a QROM. Using PennyLane's built in :func:`~qp.QROM`
# function, we can simply take computed coefficients and pass them in to the
# potential energy step of our choosing.

def LoadCoeffsKDC(fragment, output, electron_wires, coeff_wires, scratch_wires):
    fragment_coeffs = output[fragment]
    
    qp.QROM(fragment_coeffs, control_wires = electron_wires, target_wires = coeff_wires, work_wires = scratch_wires)

###############################################################################
# Given an input of mode states, we can evaluate if an entry is a
# singular integer value, a list containing a single entry, or a list containing
# two entries, each of which are possible valid inputs. The first two scenarios
# will be taken as equivalent to a linear case while the third requires the
# multiplication step of the quadratic function. ``KDCFrag()`` handles this
# using simple evaluation logic.

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
# Now that we have the tools we need to carry out a system-wide time evolution,
# how exactly do we execute?
#
# Assembling the Trotter Step
# ---------------------------
#
# To efficiently exponentiate our fragments and maintain compatibility with
# architecture, each fragment must be *diagonal*. In a vibronic system, off-diagonal terms
# run rampant due to the high degree of coupling. Thus, the
# full Trotter step function should
#
# 1. Perform a kinetic half-step on the system,
# 2. Diagonalize each fragment to represent coupling behaviour,
# 3. Perform a full potential step for each fragment,
# 4. Uncompute the diagonalization step,
# 5. Perform another kinetic half-step on the system.
#
# Motlagh et al. lay out a Clifford gate based
# scheme for `block-diagonalization
# <https://pennylane.ai/compilation/diagonal-unitary-decomp/details>`_ that
# enables uniform exponentiation. To understand the scheme, we must first take
# each Hamiltonian fragment to be given as
#
# .. math:: 
#    H_m = \sum_{j=0}^{N-1}|j\rangle \langle m \oplus j|\otimes
#    V_{j,m\oplus j},
# 
# Since :math:`|j\rangle \langle m \oplus j|\otimes` constructs the matrix geometry of
# the fragment (being the row and column terms of the Hamiltonian matrix), the
# difference between :math:`j` and :math:`m\oplus j` (representing the `Hamming
# weight <https://en.wikipedia.org/wiki/Hamming_weight>`_ in this case) will
# dictate how the block should be treated. The logic is as follows:
#
# 1. IF :math:`m=0`, we are dealing with a diagonal fragment. Our work here is
#    done!
# 2. IF :math:`j` and :math:`m\oplus j` differ by 1, we can achieve
#    diagonalization by sandwiching the fragments between Hadamard gates.
# 3. ELSE, we must construct a unitary operation using a qubit that satisfies
#    option 2 as a control for a CNOT operation applied to all other qubits in
#    the fragment to bring the Hamming weight down to 1, enabling
#    diagonalization via Hadamard sandwich.
#
# .. figure::
#    ../demonstrations_v2/simulating_vibronic_dynamics/Diagonalization.png
#    :align: center :width: 700px
#    
#    *Clifford gate diagonalization scheme* [#Motlagh2025]_
#
# This can be implemented simply by comparing two indices and
# applying the required gates.

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

# In "Quantum Algorithm for Vibronic Dynamics", it is specified that either a
# first or second order Trotterization can be carried out. While first order
# Trotterization is less resource intensive, second order allows for reduced
# `Trotter error
# <https://arxiv.org/html/2606.30738v1>`_
# and sets us up for a useful uncompute trick later on. In general, a second
# order Trotterization is given by
#
# .. math:: e^{-iHt} = (e^{-iH_1 t/2r}e^{-iH_2 t/r}e^{-iH_1 
#    t/2r})^r
#
# where :math:`H_1` and :math:`H_2` are non-commuting Hamiltonian fragments,
# :math:`\Delta t` is a time step, and :math:`r` is the total number of Trotter
# steps taken. In our case, this implies taking half a time step in kinetic
# energy, a full time step in potential energy, and another half step forward in
# kinetic energy. 
#
# Motlagh et al. go a step further and define the second-order Trotter formula
# as
#
# .. math:: U_2(\theta)=\prod_{m=0}^N e^{i\theta H_m} \prod_{m=N}^0 e^{i\theta
#    H_m},
#
# in which the potential step has been split into two half-steps as well, one
# evolving forward and one evolving backward. Since a basis change must
# occur between position and momentum, splitting the potential step into these
# mirrored steps lands a :math:`QFT` next to a :math:`QFT^\dagger`, so we can
# easily maintain the proper basis without adding additional transformations.
# Phew!

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
# We're almost there!
#
# Registers
# ---------

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
# Which can be unpacked and labelled for use.

qp.decomposition.enable_graph() #enable graph-based decomposition for performance

time_steps = 10
k = 2
n = 1
num_modes = 1
delta = 0.03

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
# Initial State Definition
# ------------------------
# In "Quantum Algorithm for Vibronic Dynamics", the initial state of the KDC
# system is taken to be a simple vertical excitation represented in product form
# in relation to electronic state :math:`j` as 
#
# .. math:: |\psi_0\rangle = |j\rangle_{el}
#    \bigotimes_{r=0}^{M-1}|\chi_0\rangle.
#
# Here,
#
# .. math:: |\chi_0\rangle = \frac{1}{Z}\sum_{x=0}^{K-1}
#    \exp\left(\frac{-\pi \cdot (x-\frac{K}{2})^2}{K}\right) |x\rangle
#
# is the Hermite-Gauss function representation of the harmonic oscillator ground
# state, where :math:`Z` is a normalization constant and :math:`x` is, again, a
# position-basis grid point index. It is stated that one can choose to begin
# with a superposition state to enable functionalities such as spectroscopy. In
# this demo, we will be targeting a simple electronic state population time
# evolution, so an initial superposition is not necessary. This state can be
# generated using ``KDCStatePrep()``.

#State Preparation - Vertical Excitation of System
def KDCStatePrep(k):
    K = 2**k
    x = np.arange(K)

    amplitudes = np.exp((-np.pi*((x-(K/2))**2))/(K))
    norm_factor = np.linalg.norm(amplitudes)

    chi0 = amplitudes/norm_factor

    return chi0
###############################################################################
# Time Evolution of Electronic State Population
# ---------------------------------------------
# Finally, we have adequately built up the skeleton of the KDC Hamiltonian
# simulation! Now, we can combine our previously constructed tools to carry out
# a Trotterization of this system and observe the population dynamics for a
# short time scale. 
#
# To keep things simple (and computationally feasible), we will begin by
# defining a small, single-mode system with 2 electron states and 1 vibrational mode. 
# The potential coefficients will be taken to be a simple array of values
# that will soon be scaled by the required factors. It is worth noting that the
# flooring step in the kinetic energy coefficient calculation requires a minimum
# number of precision bits to be present in the system to avoid flooring to 0
# and, therefore, suppressing all coupling effects. As such, ``delta`` should be
# small enough to achieve :math:`b\geq 6`.
#
# In the source paper, the
# full coefficient representation is given as
#
# .. math:: \Delta^{\alpha}c^{(j,j)}_\alpha
#
# where :math:`\alpha` is the expansion degree of the polynomial term. Taking the bit-wise representation and
# considering the time dependence, the full representation of the coefficients
# that should be passed into the QROM is
#
# .. math:: c_{time}=[c_{\alpha} \Delta^\alpha dt \frac{2^b}{2\pi}] \text{mod}
#    2^b
#
# This form allows for easy computation and conversion to the list-of-bit format
# required by the QROM. For this demo, :math:`\alpha` will be fixed to 1 since
# we are only dealing with linear mode behaviour. This can be easily changed
# depending on the needs and conditions of the system.

#Introduce scaling factor and reformat for compatibility with QROM
mode_list = [0]
omega = [1]
coeff_data = [
    #  |0>    |1>
    [ 1.0,   0.0 ],   # Fragment 0: Diagonal potential energy terms
    [ -1.3,   1.3 ]    # Fragment 1: Off-diagonal electronic coupling terms
]
dt = 0.4

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
# Now, at long last, we can carry out our time evolution. 
# Since we so diligently built up our functionality under the guidance of
# Motlagh et al.'s innovations, we are well equipped to carry this out smoothly! 

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
# Running this simulation for our limited system yields the following result,
# which shows an incomplete transfer between the ground and first excited states
# of a two-state system. 
#
# .. figure::
#    ../demonstrations_v2/simulating_vibronic_dynamics/10StepVibePlot.png
#    :align: center 
#    :width: 700px
#  
#    *Electronic state population time evolution after 10 steps with dt=0.4*
# 
# We did it!
#
# Conclusion
# ----------
# Preparing to make the most out of quantum technologies is crucial as we
# continue to move toward accessible, useful quantum devices. Identifying areas
# of interest that are known to be important to researchers, companies, and
# individuals is an important first step. Vibronic simulation has the potential
# to expand our capacity for material discovery, renewable energy expansion, and
# drug exploration. Opening this door using the advantages of quantum
# computation is an important step toward an ever expanding, quantum enabled
# future.
#
# .. _references:
#
# References
# ----------
# .. [#Motlagh2025] D.\ Motlagh, R. A. Lang, P. Jain, J. A.
# Campos-Gonzalez-Angulo, W. Maxwell, T. Zeng, A. Aspuru-Guzik, and J. M.
# Arrazola, "Quantum Algorithm for Vibronic Dynamics: Case Study on Singlet
# Fission Solar Cell Design," 2025, `arXiv: 2411.13669
# <https://arxiv.org/abs/2411.13669>`_.
#
# .. [#Lang2026] R.\ A. Lang, P. Jain, J. M. Arrazola, and D. Motlagh, "Quantum
# Algorithm for Simulating Non-Adiabatic Dynamics at Metallic Surfaces," 2026,
# `arXiv: 2601.16264 <https://arxiv.org/abs/2601.16264>`_.