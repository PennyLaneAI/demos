r"""
Simulating Vibronic Dynamics
############################

Simulating static systems will not get us very far.

The field of simulation has an expansive history of delivering benefit to humanity. The tools that we have had available to us up until now have enabled massive advancements in medicine, energy, smart technologies, fundamental physics ... it would probably be impossible to outline every instance here. The point is, though, that we have been doing pretty well for ourselves but, in doing this, have encountered the limitations of the classical technologies that are available. 

Classical simulations are well poised to handle things that, for the most part, stay still. Ground state energies, isolated systems, and other scenarios that deal with a small number of possible states are, at this point, well studied. The frontiers of the areas mentioned before, however, are expanding beyond these limited models into dynamic scenarios that account for many more interactions and more nuanced behaviours. One such area has been coined `vibronics <https://en.wikipedia.org/wiki/Vibronic_coupling>`_ and is specifically concerned with electronic and nuclear vibrational interactions. Also known as nonadiabtic coupling, vibronics is emerging as an incredibly important tool in theoretical chemistry since it enables movement beyond the Born-Oppenheimer approximation which, put simply, ignores vibronic dynamics completely. 

Whether or not we make the approximation, a molecule composed of :math:`N` atoms will have access to :math:`3N-6` vibrational degrees of freedom (DOFs). When we begin to go beyond Born-Oppenheimer, though, it is no longer possible to isolate electronic DOFs from nuclear DOFs, meaning their paths cannot be simulated seperately. Long story short, if we want to capture the realistic dynamics of entangled electronic and nuclear DOFs, we lose access to approximations that allow for simplification and, as a result, the size of the system will very quickly exceed standard computational resources. So, what comes to mind when we want to simulate realistic atomic system dynamics that scales exponentially in resource requirements and requires nuanced modelling of phenomena such as tunnelling and entanglement? Certainly that this might be an ideal problem for a quantum computer to solve.

In `"Quantum Algorithm for Vibronic Dynamics" <https://arxiv.org/abs/2411.13669>`_ Motlagh et al. propose a novel, quantum-based algorithm for carrying out vibronic simulations. This algorithm leverages several quantum-specific tools (such as `phase gradient states <https://pennylane.ai/compilation/phase-gradient>`_, `Trotterization <https://pennylane.ai/challenges/a_simple_trotterization>`_, and multiplexing via `QROM loading <https://pennylane.ai/demos/tutorial_intro_qrom>`_, which are relevant background topics to this demonstration). This paper lays out an approach that can be generalized to various vibronic Hamiltonians with arbitrary diabatic states and mode interaction specifications, which is particularly notable for its ability to handle beyond two electronic states [#Motlagh2025]_. The generality of this method is emphasized in `"Quantum Algorithm for Simulating Non-Adiabatic Dynamics at Metallic Surfaces" <https://arxiv.org/pdf/2601.16264>`_, which applies the same algorithmic structure to a completely different Hamiltonian. The following demo will explore the implementations carried out in each of these papers by first exploring the general vibronics algorithm, then diving into the nuances of each application to illustrate how the central approach can be generalized. By the end of this, you will be so excited to simulate vibronic dynamics that you won't be able to sit still!

The General Vibronics Algorithm
===============================
The particularities of a specific vibronic system require specific, careful treatment for full capture. As will be shown, to simulate the process of `singlet fission <https://en.wikipedia.org/wiki/Singlet_fission>`_ requires a completely different setup and micro-strategy than, for example, simulating the dynamics of a metallic surface. Before considering where each Hamiltonion differs, though, we can begin by asking "what do they *all* need?". Raising ourselves to the level of abstraction that suppresses these specifics makes clear the general requirements to carry out time-evolution for a dynamic system.

    1. Load the initial state of the system into the simulation,
    2. Partition the terms of the Hamiltonian into mutually non-commuting fragments for Trotterization,
    3. Diagonalize fragments for exponentiation, if necessary,
    4. Carry out a second order Trotterization to evolve the Hamiltonian in time,
    5. Read out your desired observable.

Sounds like a walk in the park! Okay, maybe saying that would be getting ahead of ourselves, but this roadmap should give us the start we need to begin building the framework of our vibronic simulation. 

Grid Encoding
-------------
To keep our operations efficient, it is important to select a space representation that allows for easy basis transformations. A standard approach is the use of spatial grid discretization to represent the system's operators in real-space. When dealing with a dynamic system, it is expected that we will be mainly concerned with the position operator :math:`Q` and momentum operator :math:`P`, so this representation is convenient. Letting :math:`k` be the number of nuclear states in the system, the number of grid points required is :math:`K=2^k`. From this, we will take for granted that the eigenvectors of :math:`Q` are given by

.. math::
   Q|x\rangle = \Delta(x-K/2)|x\rangle,

where :math:`Delta=\sqrt{2\pi/K}` is the grid spacing term and :math:`x` is a specific vibrational mode number. It is more convenient to take the signed integer representation

.. math::
   Q|x\rangle = \Delta \cdot x |x\rangle,

letting :math:`x \in \{-\frac{K}{2}, -\frac{K}{2}+1, ..., \frac{K}{2}-1\}`. This discretization method will be kept in mind throughout the implementation as we define our parameters, but is concretely illustrated in ``GridPrep()``.

"""
def GridPrep(k):
    Delta = np.sqrt(2*np.pi/K)
    x = np.arange(-K//2,K//2)
    Q = np.diag(Delta*x)
    return Q
###############################################################################
# Since position and momentum are non-commuting operators, as stated, they must be seperated into different fragments for Trotterization. To briefly review, Trotterization is a method used in Hamiltonian simulation to carry out time evolution in a complex system. It addresses the fact that if a Hamiltonian is constructed from non-commuting operators, the time evolution operator :math:`e^{iHt}` cannot be realized since the entire Hamiltonian cannot be simultaneously exponentiated. As such, the Hamiltonian can be seperated into groups of non-commuting operators called *fragments* which can be individually exponentiated and interleaved in partial time steps to simulate simultaneous time evolution.
#
# The fact that :math:`P` and :math:`Q` do not share a common basis immediately indicates that each operator will require different treatment to carry out this iterative time evolution step since each will need to be evolved in its own basis. More nuance can be discerned but considering the specific representations of a position and momentum case. The coefficients of the kinetic energy term is always proportional to :math:`T=\frac{P^2}{2}`, implying these coefficients are uniform constants that have no dependence on electronic state. Potential term coefficients, on the other hand, are electron state dependent, meaning each state will have a unique set of coefficients. This implies that the coefficient values cannot be calculated in-situ and must be externally introduced to the exponentiated potential operator step. All this to say, we need different strategies for the time evolution of each operator. 
#
# The Kinetic Step
# ----------------
# The kinetic energy Hamiltonian fragment is, comparatively, simple to establish and evolve in time. Cutting to the chase, the kinetic step should:
#
# 1. Perform a basis switch on the input state to momentum space,
# 2. Square the state values in the momentum state representation to obtain :math:`\hat{P}^2`,
# 3. Add the squared term to the phase gradient register,
# 4. Uncompute. 
#
# Since the coefficients do not depend on electron state, the global coefficient can be computed in the kinetic evolution step. These coefficients should take on the form
#
# .. math::
#    C_{kin} = \frac{2^b \omega_m \Delta t }{2K}
#
# in the computational basis, where :math:`b = \lfloor \log_2(1/\epsilon) \rfloor` is the size of the gradient register and :math:`\omega_m` is the frequency of mode :math:`m`. These states can be written in binary representation and encoded on the state register using ``qp.X()`` gates. An inituive way to understand these coefficients is to take them as describing the motion of the nuclear states. If :math:`C_{kin}=0`, the nuclear state is understood to be frozen and, therefore, there is nothing vibronic to simulate in this step! 
#
# Executing the kinetic energy step is, at this point, merely a question of detemining which tools are ideal to carry out our desired procedure. The first step is low hanging fruit; switching between position and momentum space can be achieved by applying a `quantum fourier transform (QFT) <https://pennylane.ai/demos/tutorial_qft>`_ to each state component. Once that switch takes place, the coefficients can be encoded on the state register as aforementioned and an :func:`~qp.OutPoly` operation targeting :math:`f(x)=(x-K/2)^2` (which is equivalent to :math:`x^2` in the signed-integer picture) can be used to carry out the squaring operation.
#
# Next, the phase gradient addition step can be carried out using a :func:`~qp.SemiAdder` function operating between the register holding the squared state and the phase gradient register. To carry this out properly, a "slicing" step is required to determine how many wires in the allocated registers are actually required to carry out the addition step due to the weighted binary representation used in this implementation. This is, essentially, equivalent to carrying out a `logical left shift <https://en.wikipedia.org/wiki/Logical_shift>`_ in classical computing. The procedure is as follows for each index in the simulation grid:
#
# 1. Determine the current iteration weight by computing the difference between the total grid points and the current index,
# 2. Compute the number of wires needed to carry out the addition step by finding the difference between the gradient register size and the current weight (since Python indexes at 0, also subtract an extra 1),
# 3. If the number of wires is 1, perform a :func:`~qp.CNOT` operation between the squared state and the phase gradient register,
# 4. If the number of wires is not 1, perform a controlled Semi-Out-of-Place addition, between the required state wires and required gradient wires, controlled by squared state wires.
#
# The result of this procedure should be a set of electron states that have accumulated phase gradient invoked rotations as a function of their state and experienced a time step as a result of the uniform kinetic energy operator. This is carried out in the function ``KineticStep()``.

def KineticStep(time_step, omega, num_modes, K, state_wires, gradient_wires, coeff_wires, scratch_wires, cache_wires):
    k_val = len(state_wires[0])
    b = len(gradient_wires)

    #Set function to be executed by OutPoly()
    def f(x):
        return (x - K//2)**2 #Signed Integer Representation

    #Perform basis transformation
    for mode in range(num_modes):
        qp.QFT(wires = state_wires[mode])

    #Loop full computational procedure over all modes
    for i in range(num_modes):

        #Compute coefficients
        KinCoeffRaw = (omega[mode]*time_step*(2**b)/(2*K))
        C = int(np.floor(KinCoeffRaw+0.5))

        #Flag if the nucleus if frozen
        if C == 0:
            print(f"WARNING: kinetic underflow (raw={KinCoeffRaw:.3f}); nuclei frozen. Raise b or dt.")
        C_binary = format(C, f'0{len(coeff_wires)}b')

        #Encode coefficients
        for j, bit in enumerate(reversed(C_binary)):
            if bit == '1':
                qp.X(wires=coeff_wires[j])

        #Square state
        qp.OutPoly(f, input_registers = [state_wires[i]], output_wires = cache_wires)

        #Add the squared state to the phase gradient register
        for point in range(2*k_val):
            #Index to the current position in the register
            weight = 2*k_val-1-point
            target_length = len(gradient_wires) - weight
            
            if target_length <= 0:
                continue
                
            x_wire_current = coeff_wires[:target_length]
            y_wire_current = gradient_wires[:target_length]
        
            ctrl_wire = [cache_wires[point]]
            if target_length == 1:
                qp.ctrl(qp.CNOT, control = ctrl_wire)(wires = [x_wire_current[0], y_wire_current[0]])
            elif target_length >= 2:
                qp.ctrl(qp.SemiAdder, control = ctrl_wire)(x_wires = x_wire_current, y_wires = y_wire_current, work_wires = scratch_wires)

    #Uncompute
        qp.adjoint(qp.OutPoly)(f, input_registers = [state_wires[i]], output_wires = cache_wires)

    for mode in range(num_modes):
        qp.adjoint(qp.QFT)(wires=state_wires[mode])

    for i, bit in enumerate(reversed(C_binary)):
        if bit == '1':
            qp.X(wires=coeff_wires[i])
###############################################################################
# With this defined, the potential energy step can now be tackled.
#
# The Potential Step
# ------------------
# The goal of the potential energy step as a component of the complete Trotter step is to construct the full potential energy operator (with coefficients) for each electron state. The operations that will need to be carried out by this function are:
#
# 1. Upload the electron state-dependent coefficient terms using a QROM controlled by the electronic state register,
# 2. If there are multiple vibrational mode states involved in this step, multiply them together,
# 3. Multiply the full mode state (either a single mode or the product of multiple modes, depending on step 2) with the coefficient register,
# 4. Add the product of the mode state and coefficient register to the phase gradient register,
# 5. Uncompute,
#
# Okay, we can do that! Lucky for us, PennyLane has a build in :func:`~qp.QROM` function that can be used to load the coefficients into the system. As previously mentioned, the potential energy state coefficients are necessarily dependent on electron state. As such, these coefficients are determined prior to Trotterization with methods that vary depending on the target Hamiltonian. For now, we can imagine this as an array of integer coefficient values carrying state information and time dependence that should be translated to a binary string to interface with the QROM. For now, we will assume this has been handled and the prepared coefficients have been properly passed to us for loading. 
#
# As stipulated in step 2, the structure of our potential step depends on the dimensions of the mode state being passed to us. For now, we will focus on two scenarios: the linear case (in which a single mode state is being passed to us) and the quadratic case (in which two mode states are being passed to us). If our system is quadratic, we simply need to apply an :func:`~qp.OutPoly` operator that multiplies the two mode registers together, just like we did in the kinetic step. If our system is linear, we don't need to worry about this and can skip right to the addition!
#
# The addition step must be carried out using the same shifting procedure outlined in the kinetic step to ensure proper dimensionality. In the linear case, the adder should be controlled by the mode state register and target the register containing the product of the mode state and the loaded coefficient and the phase gradient register. In the quadratic case, the adder should be controlled by the product between the two mode states and target the register containing the product of the mode state produce and the loaded coefficient and the phase gradient register. Feel free to take a second to digest that, or glance down at the diagram that (hopefully) makes that a bit clearer. After this is carried out, we can safely uncompute and move on with our day.
#
def PotentialStepLinear(time_step, mode, time_coeffs, state_wires, electron_wires, gradient_wires, coeff_wires, scratch_wires):
    
    k_grid = len(state_wires[mode])

    #Load pre-determined, electron state dependent coefficients
    qp.QROM(time_coeffs, control_wires = electron_wires, target_wires = coeff_wires, work_wires = scratch_wires)

    for point in range(k_grid):
        #Control on the spatial state register
        ctrl_wire = [state_wires[mode][point]]

        #Index to the current position in the register
        weight = k_grid - 1 - point
        target_length = len(gradient_wires) - weight

        #Bit-wise shifts
        x_wire_current = coeff_wires[:target_length]
        y_wire_current = gradient_wires[:target_length]

        if target_length == 1:
            qp.ctrl(qp.CNOT, control = ctrl_wire)(wires = [x_wire_current[0], y_wire_current[0]])
        elif target_length >= 2:
            qp.ctrl(qp.SemiAdder, control = ctrl_wire)(x_wires = x_wire_current, y_wires = y_wire_current, work_wires = scratch_wires)

    qp.adjoint(qp.QROM)(time_coeffs, control_wires = electron_wires, target_wires = coeff_wires, work_wires = scratch_wires)


def PotentialStepQuadratic(time_step, mode1, mode2, time_coeffs, state_wires, electron_wires, gradient_wires, coeff_wires, cache_wires, scratch_wires):
    
    k_grid = len(state_wires[mode1])

    qp.QROM(time_coeffs, control_wires = electron_wires, target_wires = coeff_wires, work_wires = scratch_wires)
    
    qp.OutPoly(lambda x0,x1: (x0-K//2)*(x1-K//2), input_registers = [state_wires[mode1], state_wires[mode2]], output_wires = cache_wires)
        
    for point in range(2*k_grid):
        #Control on the product of the spatial states 
        ctrl_wire = cache_wires[point]

        #Index to the current position in the register
        weight = 2*k_grid - 1 - point
        target_length = len(gradient_wires) - weight
        
        #Bit-wise shifts
        x_wire_current = coeff_wires[:target_length]
        y_wire_current = gradient_wires[:target_length]

        if target_length == 1:
            qp.ctrl(qp.CNOT, control = ctrl_wire)(wires = [x_wire_current[0], y_wire_current[0]])
        elif target_length >= 2:
            qp.ctrl(qp.SemiAdder, control = ctrl_wire)(x_wires = x_wire_current, y_wires = y_wire_current, work_wires = scratch_wires)

    #Uncompute
    qp.adjoint(qp.OutPoly)(lambda x0,x1: (x0-K//2)*(x1-K//2), input_registers = [state_wires[mode1], state_wires[mode2]], output_wires = cache_wires)
    
    qp.adjoint(qp.QROM)(time_coeffs, control_wires = electron_wires, target_wires = coeff_wires, work_wires = scratch_wires)

###############################################################################
# Now that we have the tools we need to carry out a system-wide time evolution, how exactly can we execute this?
#
# Assembling the Trotter Step
# ---------------------------
# In [#Motlagh2025]_, it is specified that a second-order Trotterization approach is taken. In the general sense, a second order trotterization structure is given by
#
# .. math::
#    e^{-iH_1 t}e^{-iH_2 t} = (e^{-iH_1 \Delta t/2r}e^{iH_2 \Delta t}e^{-iH_1 \Delta t/2r})^r
#
# where :math:`H_1` and :math:`H_2` are non-commuting Hamiltonian fragments, :math:`\Delta t` is a time step, and :math:`r` is the total number of Trotter steps taken. Taking the position/momentum operator consideration, this approach can be understood as taking half a time step forward in kinetic energy, a full time step forward in potential energy, and another half step forward in kinetic energy. Taking this approach (or a higher-order approach in general) rather than a full step-full step approach reduces the `Trotter error <https://simons.berkeley.edu/sites/default/files/docs/15639/trottererrortheorysimons.pdf>`_ substantially.
#
# Before we can assemble our Trotter step, though, we are missing a crucial piece. In order to carry out a Trotterization of a fragmented Hamiltonian, each fragment must be exponentiated. In order to do this efficiently and to maintain compatibility with the QROM architecture, this means that each fragment must be *diagonal*. In a vibronic system, we are heavily concerned with coupling between atoms and, therefore, must consider off diagonal terms to fully capture our system. As such, protocol must be put in place to *diagonalize* off-diagonal fragments. This can typically done using cheap `Clifford gates <https://pennylane.ai/demos/tutorial_clifford_circuit_simulations>`_, but the exact method required to diagonalized a given Hamiltonian varies. Space should be build into the general Trotter step function for a diagonalization step to take place. Thus, ``TrotterStep()`` should:
#
# 1. Perform a kinetic half-step,
# 2. Diagonalize each fragment,
# 3. Perform a full potential step for each fragment,
# 4. Uncompute the diagonalization step,
# 5. Perform another kinetic half-step.
#
# Since we have so diligently built up our resources, we can assemble this function simply.

def TrotterStep(k, n, frag_list, num_modes, mode_list, coeff_data, dt, omega, coupler, PotentialStep, state_wires, gradient_wires, coeff_wires, scratch_wires, cache_wires, electron_wires):
    K = 2**k

    half_dt = dt/2
    
    KineticStep(half_dt, omega, num_modes, K, state_wires, gradient_wires, coeff_wires, scratch_wires, cache_wires)

    for fragment in frag_list:

        #Diagonalization function
        coupler(fragment, electron_wires)

        #Pass a function that can handle the potential step in the linear or quadratic case
        PotentialStep(fragment, coeff_data)

        qp.adjoint(coupler)(fragment,electron_wires)

    KineticStep(half_dt, omega, num_modes, K, state_wires, gradient_wires, coeff_wires, scratch_wires, cache_wires)
###############################################################################
# Before we jump into specific examples, one final generalizable step that will keep our lives simple and organized is the wire preparation stage. Using :func:`~qp.registers`, we can build our wires according to the resource requirements of functions we built.
#

def WirePrep(num_modes, k, n, delta):

    precision_qubits = int(math.ceil(np.log2(1/delta)))
    
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
# With the foundation laid, it's time to turn to some specific examples of vibronic simulation.
#
# The Köppel-Domcke-Cederbaum Hamiltonian
# =======================================
# A simple example of a vibronic Hamiltonian is the Köppel-Domcke-Cederbaum (KDC) Hamiltonian. The KDC Hamiltonian represents a simple vibronic coupling between a nuclear state and an electronic state, making it an ideal candidate for this algorithm. It takes the general form
# 
# .. math::
#    H = \mathcal{I}_{el} \otimes (T_{nuc}+V_0)+\textbf{W}
#
# Where :math:`T_{nuc}=\frac{1}{2}\sum_r \omega_r P_r^2` is the vibrational kinetic operator, :math:`V_0 = \frac{1}{2}\sum_r \omega_r Q_r^2`, and :math:`\textbf{W}` is the "diabatic potential", which is, essentially, a coupling matrix. 
#
# Carrying out a time evolution of the KDC Hamiltonian is useful for spectroscopy and energy transfer dynamics, among other things. For this demonstration, we will aim to simulate electron population evolution for a small, simple system. This specific observable will not be relevant until the end of this implementation though, and we can broadly build up a multi-use skeleton for this implementation as well. 
#
# Initial State Definition
# ------------------------
# 
#
# Diagonalization
# ---------------
# 
# Fragmentation
# -------------
#
# Time Evolution of Electronic State Population
# ---------------------------------------------
# .. _references:
#
# References
# ----------
# .. [#Motlagh2025] D.\ Motlagh, R. A. Lang, P. Jain, J. A. Campos-Gonzalez-Angulo, W. Maxwell, T. Zeng, A. Aspuru-Guzik, and J. M. Arrazola, "Quantum Algorithms for Vibronic Dynamics: Case Study on Singlet Fission Solar Cell Design," 2025, `arXiv: 2411.13669 <https://arxiv.org/abs/2411.13669`_.
#
# .. [#Lang2026] R.\ A. Lang, P. Jain, J. M. Arrazola, and D. Motlagh, "Quantum Algorithm for Simulating Non-Adiabatic Dynamics at Metallic Surfaces," 2026, `arXiv: 2601.16264 <https://arxiv.org/abs/2601.16264>`_.