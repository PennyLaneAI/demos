r"""
Simulating Resonant Inelastic X-Ray Scattering
##############################################

Our understanding of the world is only as accurate as our models and,
importantly, our models are only as accurate as our ability to interpret their
results.

In our battery-dependent world, it is very important that we properly understand
how and why our battery technologies age and die. Lithium excess (Li-excess)
batteries are currently being eyed as the next generation of high-capacity
batteries, but they are plagued with short lifespans. In an attempt to figure
out why, resonant inelastic x-ray scattering (RIXS) experiments, an advanced
X-ray spectroscopy technique that monitors momentum changes between input and
output photons that interact with a target molecule, have been carried out.
These tests have indicated that Li-excess cathodes produce an excess of
molecular oxygen becomes trapped inside the battery, leading to decline.

In 2025, Gao et al. published "Clarifying the origin of molecular O2 in cathode
oxides", dropping the bombshell that RIXS spectra show the presense of molecular
oxygen in non-Li-excess batteries as well, meaning it is likely an artifact of
the methodology rather than a result of the battery process. They additionally
point to this as evidence of a much more complex degredation mechanism involving
the bonding of oxygen dimers to transition metals in the battery materials.

Though this wasn't a "back to the drawing board" moment, this shift in
interpretation and understanding shed light on the need for reliable simulations
that can help in the validation and interpretation of experimental results. This
is precisely the case made by Laoiza et al. in "Quantum algorithm for simulating
resonant inelastic X-ray scattering of battery materials". Here, the team puts
forward a quantum algorithm designed to tackle the problem of RIXS simulation
using a novel combination of :doc:`generalized quantum signal processing (GQSP)
<demos/tutorial_estimator_hamiltonian_simulation>`, :doc:`amplitude
amplification <demos/tutorial_intro_amplitude_amplification>`, :doc:`quantum
amplitude estimation (QAE) <demos/iterative_quantum_amplitude_estimation>`, and
:doc:`quantum phase estimation (QPE) <demos/tutorial_qpe>`. 

Today, our goal will be to understand how these quantum building blocks work
together to make way for reliable RIXS simulation and begin to open the door for
more capable advanced materials discovery in the future. Let's get to work!

Getting Started
===============
What is RIXS?
-------------
The goal of RIXS spectroscopy is to monitor how matter interacts with light, as
is the case for spectroscopy in general. In RIXS, a material is illuminated by
X-ray photons with energy that sits very close to a core electron's binding
energy (also known as the "absorption edge") [#Loaiza2026]_. The successful
absorption of this photon with frequency :math:`\omega_I` kicks off a two-step
process in which the absorbed photon promotes a core electron to a valence
orbital, leaving behind a hole in the core orbital that is eventually filled by
a different, lower energy valence electron than the one that was excited from
the core. This is why RIXS is coined a "photon-in, photon-out" process, since
the relaxation of the second valence electron into the core releases a photon of
frequency :math:`\omega_S` that is detected and used to compute the difference
between the input and output photon energies.

.. figure::..demonstrations_v2/pennylane-demo-simulating-resonant-inelastic-xray-scattering-EnergyLevelDiagram.png
   :align: center
   :width: 700px
   :alt: An illustration of the three stages of the RIXS process in the form of an energy level diagram.

So, the three states involved in the RIXS process are:

1. :math:`|E_0\rangle`: The molecule sits in an unexcited state prior to the
      absorption of the incident :math:`\omega_I` photon.
2. :math:`|E_n\rangle`: Following the absorption of the incident
      :math:`\omega_I` photon, a core electron has been excited to a
      higher-energy valence orbital leaving behind a core hole.
3. :math:`|E_f\rangle`: To fill the unfavourable core hole, an electron from a
      lower energy valence orbital has relaxed into the core, leaving behind a
      valence hole and emitting a :math:`\omega_S` photon [#Loaiza2026]_.

The difference between the energies of the photon retrieved from the
:math:`|E_f\rangle` photon and the :math:`|E_0\rangle` is known as the energy
transfer. When plotted for energy versus intensity (as is characteristic of a
RIXS spectrum), the peaks can be interpreted as a specific excitation within the
target molecule and used to understand its structure. Tying back to the battery
example, we could take advantage of RIXS spectra to measure different input
energies and observe both the bandgap of a target material and what byproducts
appear as a result of the charge transfer process, informing decisions on
material viability and lifetime predictions.

REWRAP TK

Why Quantum?
------------
Classical simulation is limited in the amount and type of complexity it can
handle. Basic comparisons of classical and quantum simulation methods make the
case that the size of the system that can be handled in each case varies
greatly, which quantum simulations typically capable to handling many more
states. While this is true, there are additional, potentially more important
advantages that are not merely reliant on an increase in computational power. In
the case of Loaiza et al.'s RIXS algorithm, the following two advantages are
listed for the quantum case:

1. RIXS processes involve several delocalized processes, meaning the positions
   of the electrons involved are probabalistically spread out across the
   molecule. Capturing this requires a large active space that simply cannot be
   handled by the resources of classical computers and by classical bits
   incapable of simulating superposition and entanglement relationships.

2. Strongly correlated systems of interest, such as the transition metal-oxygen
   dimer bonding proposed by Gao et al., experience complex intermediate states
   that dictate the mixing of orbitals in the oxygen-metal bonding process. To
   model this sufficiently, a simple, computationally simple wavefunction is
   insufficient, and a classical computer is once again incapable of carrying
   out the necessary entanglement math [#Loaiza2026]_.

So, even though it *is* a valid argument to say that quantum computers could
more feasibly handle large molecules if necessary, the stronger arguments in
this context is that the use of qubits to carry out RIXS simulation makes it
inherently possible to model the quantum phenomena that it relies on. Good thing
we know our stuff!

The Hamiltonian
---------------
.. admonition:: A note on Fermionic operators
   :class: note

   When describing molecular systems it is, conventional to use :doc:`Fermionic
   operators <demos/tutorial_fermionic_operators>` to describe the behaviour of
   the identical particles that make up the system. In general, the operators of
   concern are:

   1. :math:`a^\dagger`, the **creation operator**. This is used when a particle
      is "created", such as a photon being emitted in a relaxation process.
   2. :math:`a`, the **annihilation operator**. This is used when a particle is
      "destroyed", such as a photon being absorbed in an excitation process.

   Combining these operators for a single particle yeilds **number operators**,
   which "count" the number of a certain particle in a system:

   :math:`\hat{n}_i=a^\dagger_i a_i`

   These operators can be combined to describe various creation/annihilation
   processes and can be converted into a gate-compatible representation using
   techniques such as the :func:`~pennylane.jordan_wigner` transformation.


In "Quantum algorithm for simulating resonant inelastic X-ray scattering of
battery materials", Loaiza et al. specify that a second-quantized Hamiltonian of
the form

.. math::
   \hat{H}=E^{0}+\sum_{p,q=1}^{N_{a}}\sum_{\sigma}h_{pq}\hat{c}_{p\sigma}^{\dagger}\hat{c}_{q\sigma} +\frac{1}{2}\sum_{p,q,r,s=1}^{N_{a}}\sum_{\sigma,\sigma^{\prime}}V_{pqrs}\hat{c}_{p\sigma}^{\dagger}\hat{c}_{q\sigma}\hat{c}_{r\sigma^{\prime}}^{\dagger}\hat{c}_{s\sigma^{\prime}},

where :math: `N_a` is the number of spatial orbitals in the molecule, :math:`p,
q, r,` and :math:`s` are specific orbital indices, :math:`\sigma` and
:math:`\sigma^\prime` are spin states, :math:`h_{pq}` are integrals that compute
the energy of individual active electrons, :math:`V_{pqrs}` are integrals that
compute the energy of correlated electron pairs, and :math:`E^0` is the total
energy of the inner-shell electrons that are apprximated as frozen in the active
space definition. This is a lot and certainly looks complicated, but let's stick
with it!

For our simple, toy implementation, we will not apply this Hamiltonian structure
to the :math:`MnO_7H_6` molecule that the source paper focuses on. Instead, we
will take a simple system consisting of two core orbitals and two valence
orbitals. To do this, we will adapt the given Hamiltonian as:

.. math::
   \hat{H}=\hat{H}_{int}+\hat{H}_{hybrid}+\hat{H}_{spin}=\sum_{\sigma\in {\uparrow,\downarrow}}(\epsilon_{c1}\hat{n}_{c1,\sigma}+\epsilon_{c2}\hat{n}_{c2,\sigma}+\epsilon_{\nu_1}\hat{n}_{\nu_1,\sigma}+\epsilon_{\nu_2}\hat{n}_{\nu_2,\sigma})+h\sum_{\sigma\in {\uparrow,\downarrow}}(\hat{c}_{\nu_1,\sigma}^\dagger\hat{c}_{\nu_2,\sigma}+\hat{c}_{\nu_2,\sigma}^{\dagger}\hat{c}_{\nu_1,\sigma})+V(\hat{n}_{\nu_2,\uparrow}\hat{n}_{\nu_2,\downarrow})

where :math:`c_1` and :math:`c_2` are core orbitals, :math:`\nu_1` and
:math:`\nu_2` are valence orbitals, and :math:`\epsilon_i` are on-site orbital
energies. Since we are dealing with a very small system, we are not taking the
same frozen inner-shell assumption as in the source paper.

To implement this Hamiltonian in PennyLane, we can first call the built in
Fermionic operators :class:`~pennylane.FermiC` (the creation operator) and
:class:`~pennylane.FermiA` (the annihilaton operator).
"""

import pennylane as qp
import numpy as np
from pennylane.fermi import FermiC as create, FermiA as annihilate

#Define Operators

#Core Orbital 1
num_op_c1_up = create(0)*annihilate(0)
num_op_c1_down = create(1)*annihilate(1)
#Core Orbital 2
num_op_c2_up = create(6)*annihilate(6)
num_op_c2_down = create(7)*annihilate(7)
#Valence Orbital 1
num_op_v1_up = create(2)*annihilate(2)
num_op_v1_down = create(3)*annihilate(3)
#Valence Orbital 2
num_op_v2_up = create(4)*annihilate(4)
num_op_v2_down = create(5)*annihilate(5)
###############################################################################
# In order to run this toy model on a classical simulator, the coefficient terms
# will need to be defined as unrealistically low. The following values will be
# taken here, with the scaling factor :math:`s` available for scaling while
# maintaining relative values. TK-ADD NUMBERS FROM PAPER

s = 0.45 #Optional scaling term for runnability

#Orbital energies
eps_c1 = -1.5*s
eps_c2 = -4.5*s
eps_v1 = -1.5*s
eps_v2 = 4.5*s

#One-photon integral
h = 0.5*s

#Two-photon integral
V = 1.0*s
###############################################################################
# With these values defined, the Hamiltonian can be constructed arithmetically.
# Since we will eventually run this Hamiltonian through our quantum circuits,
# the final Hamiltonian will be converted to the Pauli basis using a
# :func:`~pennylane.jordan_wigner` transformation and compressed as much as
# possible using :func:`~pennylane.simplify` for resource optimization.

#Diagonal terms
Hdiag_up = (eps_c1*num_op_c1_up)+(eps_c2*num_op_c2_up)+(eps_v1*num_op_v1_up)+(eps_v2*num_op_v2_up)
Hdiag_down = (eps_c1*num_op_c1_down)+(eps_c2*num_op_c2_down)+(eps_v1*num_op_v1_down)+(eps_v2*num_op_v2_down)

#Hybrid terms
Hhybrid_up = h*((create(2)*annihilate(4))+(create(4)*annihilate(2)))
Hhybrid_down = h*((create(3)*annihilate(5))+(create(5)*annihilate(3)))

#Spin term
Hspin = V*(num_op_v2_up*num_op_v2_down)

H_raw = qp.jordan_wigner((Hdiag_up+Hdiag_down)+(Hhybrid_up+Hhybrid_down)+Hspin).simplify()

#TK TEMPORARY LOCATION - NEEDS DESCRIPTION
coeffs, ops = H_raw.terms()
id_c = sum(c for c, o in zip(coeffs, ops) if len(o.wires) == 0)  
H_traceless = H_raw - id_c * qp.Identity(0)
H_sparse = H_traceless.sparse_matrix(wire_order=range(8)).toarray()
H_evals, H_evecs = np.linalg.eigh(H_sparse)
###############################################################################
# Alright, now we have a goal (simulate a RIXS spectrum), a platform (a quantum
# computer or, here, PennyLane), and a system. Now we just need our strategy!
#
# The Algorithm
# =============
# Algorithm Overview
# ------------------
# To carry out a quantum simulation of a RIXS spectrum, Loaiza et al. summarize
# their algorithm in two steps:
# 
# 1. Prepare the initial RIXS state :math:`|R_{\epsilon_I,
#    \epsilon_S}(\omega_I)\rangle`,
# 2. Carry out a walk-based `quantum phase estimation <demos/tutorial_qpe>` to
#    evolve toward the final state.
#
# ..
#    figure::..demonstrations_v2/pennylane-demo-simulating-resonant-inelastic-xray-scattering-RIXScircuit.png
#    :align: center :width: 700px :alt: An illustrated circuit diagram depicting
#    the general components of Loaiza et al.'s algorithm.
#
# Item 1 on this list does a lot of heavy lifting here. In fact, the process of
# preparing the state *is* the algorithm in many ways. So, we can expand the
# list to capture the complete methodology:
# 
# 1. Prepare the initial RIXS state, :math: `|R_{\epsilon_I,
#    \epsilon_S}(\omega_I)\rangle`
# 
#    a. Select a compatible decomposition of the target Hamiltonian to ensure
#    simulation feasibility, b. Implement a Green's function spectral filter
#    using GQSP :math:`\hat{G}(\omega_I, \Gamma)`, finding the Chebyshev
#    coefficients of the Green's function and translating them to angles for
#    implementation, c. Define the **dipole operator**
#    :math:`\hat{D}_{\epsilon_i}`, which describes the perturbation that occurs
#    as a result of the incident photon excitation, d. Prepare a block encoding
#    :math:`\hat{\mathcal{U}}` of the operator proportional to
#    :math:`\hat{D}_{\epsilon_S}^\dagger \hat{G}(\omega_I, \Gamma)
#    \hat{D}_{\epsilon_I}`, e. Construct a :doc:`Grover operator
#    <demos/tutorial_grovers_algorithm>` using :math:`\hat{\mathcal{U}}` and
#    carry out amplitude estimation to determine the success probability of the
#    block encoding step, f. Carry out :doc:`amplitude amplification
#    <demos/tutorial_intro_amplitude_amplification` on the successful block
#    encoded state to boost the success probability,
# 
# 2. Carry out a walk-based `quantum phase estimation <demos/tutorial_qpe>` to
#    evolve toward the final state.
#
# We have our work cut out for us! Thankfully, most of the tools we need are
# built for us in PennyLane, so let us work through these steps systematically
# to reach our goal. 
#
# BLISS-THC Decomposition TK DOUBLE COMPRESSED, THC HAMILTONIAN RESOURCE OBJECTS
# -----------------------
# In order to carry out subsequent block encoding and minimize resource costs, a
# system's Hamiltonian may need to be decomposed into a :doc:`linear
# combination of unitaries (LCU) <demos/tutorial_lcu_blockencoding>`. Achieving
# this LCU allows for the exploitation of symmetries (in other words, the
# commutation relationships) that exist within the original Hamiltonian for
# optimized implementation. The method selected for decomposition can also aid
# in simplifying the system, particularly through the optimization of the
# system's 1-norm :math:`\lambda`, which comprises the sum of the scalar
# coefficients involved in the expression and will directly contribute to the
# form of the block encoding later.
#
# The toy Hamiltonian that we have defined for this demonstration is small and
# symmetrical enough to forgo the decomposition step. If you are curious,
# though, Laoiza et al. select the block-invariant symmetry-shift technique with
# tensor hypercontraction factorization (BLISS-THC) method for their
# decomposition, which is well known to be suited for compressing molecular
# Hamiltonians [#Loaiza2026]_. This method is carried out as follows:
#
# 1. Define the BLISS Hamiltonian
#    :math:`\hat{H}_B(\alpha,\beta)=\hat{H}-\alpha_1\hat{N}_e-\alpha_2\hat{N}_e^2-\frac{1}{2}\sum_{pq,\sigma}\beta_{pq}(c^\dagger_{p\sigma}c_{q\sigma}(\hat{N_e}-N_e)+\text{h.c})`,
#    where :math:`N_e` is the number of electrons in the ground state,
#    :math:`\hat{N_e}` is the total particle number operator, and
#    :math:`\alpha_1, \alpha_2, \text{and} \beta_{pq}` are optimization
#    parameters.
# 2. Carry out a `THC factorization
#    <https://en.wikipedia.org/wiki/Tensor_contraction>`_ on the BLISS
#    Hamiltonian,
# 3. Minimize the cost function associated with the BLISS-THC representation,
#    focusing on optimizing the BLISS and THC parameters.
#
# A much more detailed guide can be found in the source paper [#Loaiza2026]_.
#
# Operator Preparation
# --------------------
# The overarching goal of step one of two in this algorithm is to create the
# RIXS state
#
# .. math:: |R_{\epsilon_I,
#    \epsilon_S}(\omega_I)\rangle\equiv\frac{\hat{R}_{\epsilon_I,
#    \epsilon_S}(\omega_I)|E_0\rangle}{|R_{\epsilon_I,\epsilon_S}(\omega_I)}.
#
# We will take for granted that this state is equivalent to the block encoded
# operator
#
# .. math:: \hat{\mathcal{U}}_R \equiv \begin{bmatrix}
#    \frac{\Gamma}{\lambda_D^{(\epsilon_S)}} D_{\epsilon_S}^\dagger
#    \hat{G}(\omega_I, \Gamma) \hat{U}_{\epsilon_I} & \cdot \\ \cdot & \cdot
#    \end{bmatrix}
#
# Where :math:`\Gamma` is a scaling factor, :math:`D_{\epsilon_S}^\dagger` is the
# final state dipole operator, :math:`\lambda_D^{(\epsilon_S)}` is the 1-norm of
# the final state dipole operator, :math:`\hat{G}(\omega_I, \Gamma)` is the
# Green's function, and :math:`\hat{U}_{\epsilon_I}` an operator that maps the
# initial dipole perturbed state onto the all-zero state, giving
# :math:`\hat{U}_{\epsilon_I}|0\rangle=|D_{\epsilon_I}\rangle`.
#
# We're almost done with the bulk of the definitions, I promise!
#
# The Dipole Operator 
# ...................
# Laoiza et al. define the dipole
# operator as:
# 
# .. math::
#    \hat{D}^\dagger_{\epsilon_S}=\sum_{pq,\sigma}d_{pq}^{(\epsilon_S)}\hat{c}_{p\sigma}^\dagger\hat{c}_{q\sigma},
#
# where :math:`d_{pq}^{(\epsilon_S)}` are the dipole matrix elements associated
# with the scattering process. 
#
# Since we can interpret this operator as the de-excitation process that occurs
# in the final step, the initial dipole operator can be taken as merely the
# inverse excitation operator since these representations are frequency
# independent. Thus, we can define the base of our dipole operator as only the
# excitation terms (i.e., ignoring the conjugate terms).
#

#TK TEMPORARY LOCATION 
Gamma = 0.99*s
lamb = float(np.sum(np.abs(H_traceless.terms()[0])))
E_0 = H_evals[0]
omega_I = 6.10*s
z = np.linspace(-1, 1, 1000)
K_G = 100
scale = 0.7

#Define dipole matrix elements
d_c1 = 1
d_c2 = 1
d_c3 = 0.3
d_c4 = 0.3

#Spin up terms
D_eps_up   = d_c1*create(2)*annihilate(0) + d_c2*create(4)*annihilate(0) + d_c3*create(2)*annihilate(6) + d_c4*create(4)*annihilate(6)
#Spin down terms
D_eps_down = d_c1*create(3)*annihilate(1) + d_c2*create(5)*annihilate(1) + d_c3*create(3)*annihilate(7) + d_c4*create(5)*annihilate(7)
#Full expression
D_eps = qp.jordan_wigner(D_eps_up+D_eps_down)
###############################################################################
# From here, we can translate the summation expression into a matrix representation, compute the transpose to achieve the de-excitation dipole operator, and normalize.

#Excitation
D_eps_in_mat = qp.matrix(D_eps, wire_order = range(8))
#De-Excitation
D_eps_out_mat = D_eps_in_mat.conj().T

#Normalization
norm_const_in = np.linalg.norm(D_eps_in_mat,2)
norm_const_out = np.linalg.norm(D_eps_out_mat,2)
D_eps_mat_in_norm = D_eps_in_mat/norm_const_in
D_eps_mat_out_norm = D_eps_out_mat/norm_const_out
###############################################################################
#
# Green's Function and GQSP
# .........................
# In spectroscopic
# applications, it is often necessary to select for specific frequencies of
# interest. This can often be carried out using a filter designed to pass
# specific frequency instances to the final measurement. In Laoiza et al., they
# decide to carry this out using a Green's function, which acts as an impulse
# response that amplifies specific intermediate states that match a defined
# resonance frequency :math:`\omega_I`.
#
# The Green's function, scaled by :math:`\Gamma` is given by
# 
# .. math::
#    \Gamma\hat{G}(\omega_I,\Gamma)=\frac{\Gamma}{\omega_I-(\hat{H}-E_0)+i\Gamma}
#
# DISCUSS GREENS FUNCTION DEGREE TK
# 
# To implement this function using GQSP, the phase factor angles must first be
# determined. This is a completely classical process that involves determining
# the `Chebyshev coefficients
# <https://en.wikipedia.org/wiki/Chebyshev_polynomials>`_ and converting them
# into an angle representation for use. ``AngleFinder()`` handles this, taking
# advantage of python and PennyLane tools (such as
# :func:`~pennylane.poly_to_angles`, which handles the conversion as long as the
# found polynomial is represented in the Fourier basis) to get the job done. 

def AngleFinder(Gamma, lamb, E_0, omega_I):
    GreensFunc = lambda x: scale*Gamma/(omega_I-((lamb*x)-E_0)+(1j*Gamma))

    cheb = np.polynomial.chebyshev.Chebyshev.interpolate(GreensFunc, deg = K_G)

    #Convert to Fourier basis 
    d = len(cheb.coef)-1
    GQSPcoefs = np.zeros(2*d+1, dtype = complex)
    GQSPcoefs[d] = cheb.coef[0]

    #shift indices
    for k in range (1, d+1):
        GQSPcoefs[d-k] = cheb.coef[k]/2
        GQSPcoefs[d+k] = cheb.coef[k]/2
    
    GQSPangles = qp.poly_to_angles(poly = GQSPcoefs, routine = "GQSP", angle_solver = "iterative")
    return GQSPangles
###############################################################################
# With the dipole operators defined and the GQSP angles found, we can officially
# carry out our block encoding! To achieve this, we need to: 
#
# 1. Encode the initial dipole excited state onto a register representing the
#    molecular system,
# 2. Carry out the GQSP process, encoding the Green's function onto the system
#    register,
# 3. Load the final dipole operator onto the system register,
# 4. Carry out a controlled X operation that will flag if the block encoding is
#    successful.
#
# .. figure::
#    ../demonstrations_v2/pennylane-demo-simulating-resonant-inelastic-xray-scattering-BlockEncodingCircuit
#    :align: center :width: 700px :alt: An circuit diagram illustration
#    depicting the block encoding operator for the RIXS state.
# 
# While the source paper suggest the use of a :doc:`quantum read only memory
# (QROM) <demos/tutorial_intro_qrom>` to carry out dipole operator loading, we
# will simplify this for our toy model by using :class:`~pennylane.BlockEncode`.

angles = AngleFinder(Gamma, lamb, E_0, omega_I)

def RIXSStateEncodingUnitary(angles):
    #INITIAL STATE |E_0>
    # Prepare initial state
    psi0 = H_evecs[:,0]
    qp.StatePrep(psi0, wires = system_wires)

    #INTERMEDIATE STATE |E_n>
    #Implement excitation dipole operator
    qp.BlockEncode(D_eps_mat_in_norm, wires = list(block_encilla_1) + list(system_wires))

    #Define the GQSP walk operator
    W = qp.Qubitization(H, control = control_wires)
    
    #Implement GQSP and uncompute walk operator
    qp.GQSP(W, angles, control = GQSP_wire)
    for _ in range(K_G):
        qp.adjoint(W)
    
    #FINAL STATE |E_f>
    #Implement de-excitation dipole operator
    qp.BlockEncode(D_eps_mat_out_norm, wires = list(block_encilla_2) + list(system_wires))
    
    #Add success flag
    flag_ctrl = list(GQSP_wire) + list(block_encilla_1) + list(block_encilla_2) + list(control_wires)
    qp.ctrl(qp.X, control = flag_ctrl, control_values = [0]*len(flag_ctrl))(wires=success_wire)
###############################################################################
# With this implemented, we have our RIXS state ready!
# ``RIXSStateEncodingUnitary()`` walks through each of the states present in the
# RIXS process, as indicated in the comments. Since we will be using QPE for our
# readout, though, there are a few additional steps that can be taken to ensure
# the true, optimal outcome can be achieved.
#
# Amplitude Estimation and Amplification
# --------------------------------------
# For successful QPE, we should aim to amplify the success probability of our
# block encoding to as close to 1 as possible. Laoiza et al. achieve this via a
# process of amplitude estimation and amplitude amplification. They note that,
# while you can carry out amplification without prior knowledge of the success
# probability :math:`P_R`, it is "advantagenous to first determine :math:`P_R`
# and then use "textbook" amplitude amplification ... which has better
# prefactors" [#Loaiza2026]_. 
#
# To quickly review, amplitude estimation is a process of determining the
# proportion of a specific "good" state in a data set. In this context, the
# estimation process should return the probability of the block encoding
# architecture returning a successful block encoding, as marked by the success
# flag mentioned earlier. Amplitude amplification, on the other hand, carries
# out a series of strategic reflections that amplify the relative probability of
# measuring the success state.
#
# They define the true success probability as
#
# .. math:: P_R \equiv \left( \frac{\Gamma |R_{\epsilon_I,
#    \epsilon_S}(\omega_I)|}{\lambda_D^{(\epsilon_S)} |D_{\epsilon_I}|}
#    \right)^2.
#
# Which can be used to determine the number of amplitude amplification steps
# :math:`K_A` via
#
# .. math:: \lfloor \frac{\pi}{4\arcsin\sqrt{P_R}} \rfloor
#
# So, if we are able to determine the success probability, we can easily compute
# the amplitude amplification repetition parameter, boost our signal, and move
# forward to our QPE readout with confidence.
#
# ..
#    figure::..demonstrations_v2/pennylane-demo-simulating-resonant-inelastic-xray-scattering-GroverIterateCircuit.png
#    :align: center :width: 700px :alt: An illustrated circuit diagram for
#    constructing the Grover iterate.
#
# To implement the Grover iterate in circuit form, the following operations must
# be executed.
#
# 1. Rotate the entire quantum state about the success state, flagged by
# :math:`|1\rangle`, 2. Uncompute the block encoding hosted in the system and
# ancilla registers (taking :math:`|\cdot\rangle_R` to be a combination of the
# system wires, GQSP wire, ancilla wires, and control wires), 3. Flip all qubits
# in the now empty registers from :math:`|0\rangle` to :math:`|1\rangle`, 4.
# Rotate the entire quantum state about the :math:`|0\rangle` state, 5. Revert
# all :math:`|1\rangle` states back to :math:`|0\rangle` states,
# 6. Recompute the block encoding.
#
# The output of this circuit will act as both the seed for amplitude estimation
# and the state being amplified. 
#
def GroverIterate():
    R_reg = list(system_wires) + list(GQSP_wire) + list(block_encilla_1) + list(block_encilla_2) + list(control_wires)
    
    qp.Z(wires = success_wire)

    qp.adjoint(RIXSStateEncodingUnitary)(angles) #between success and collection register

    qp.X(wires = success_wire)
    
    for wire in R_reg:
        qp.X(wires = wire)
        
    qp.ctrl(qp.Z, control = R_reg)(wires = success_wire)
    
    qp.X(wires = success_wire)
    
    for wire in R_reg:
        qp.X(wires = wire)
        
    RIXSStateEncodingUnitary(angles)
###############################################################################
# Using this, a typical amplitude estimation procedure can be invoked.
dev = qp.device("lightning.qubit")

@qp.qnode(dev)
#Implement QAE
def QAE():
    RIXSStateEncodingUnitary(angles)

    for wire in QAE_wires:
        qp.Hadamard(wires=wire)

    for i, phase_wire in enumerate(QAE_wires):
        exponents = 2**i
        for _ in range(exponents):
            qp.ctrl(GroverIterate, control = phase_wire)()

    qp.adjoint(qp.QFT)(wires = QAE_wires)

    return qp.probs(wires = QAE_wires)
###############################################################################
# Using the output of the amplitude estimation step, we can easily construct a
# high probability RIXS state to guarantee a successful simulation outcome.
#
# ..
#    figure::..demonstrations_v2/pennylane-demo-simulating-resonant-inelastic-xray-scattering-HighProbState.png
#    :align: center :width: 700px :alt: An illustrated circuit diagram of the
#    amplitude amplification step.
#
# 
def HighProbRIXSState(probs):
    wires = int(nQAE)
    PeakProbAngle = (np.argmax(probs)/(2**wires))
    P_R = (np.sin(np.pi*PeakProbAngle))**2
    
    if P_R <= 1e-12:
        K_a = 0                     
    else:
        K_a = int(np.floor(np.pi / (4 * np.arcsin(np.sqrt(P_R)))))

    RIXSStateEncodingUnitary(angles)

    for K in range(K_a):
        GroverIterate()
###############################################################################
#
# Quantum Phase Estimation and Readout
# ------------------------------------
# Another novelty of the Laozia et al. RIXS algorithm is their approach to
# measurement and readout. The second step of the algorithm we laid out
# previously is the application of walk-based QPE. To briefly review, this is a
# strategy in which time evolution is carried out by a "walk operator"
# constructed from a :doc:`qubitized <demos/tutorial_qubitization>`
# representation of the system's Hamiltonian. This walk operator is composed of
# two oracle states, the PREPARE oracle (which loads the Hamiltonian
# coefficients) and the SELECT oracle (which applies non-coefficient, physical
# operators to the system). The source paper defines this operator as
#
# .. math:: \hat{\mathcal{W}}=\hat{\mathcal{R}}\cdot \text{PREP} \cdot
#    \text{SEL} \cdot \text{PREP},
#
# where :math:`\hat{\mathcal{R}}=(\hat{I}-2|0\rangle\langle0|)\otimes\hat{I}`
# [#Loaiza2026]_. This can be taken as an implementable, efficient
# representation of the evolution operator :math:`e^{\pm i \arccos
# \hat{H}/\lambda}`, therefore exponentiating the block encoded state. Carrying
# out controlled applications of the walk operator between a control register
# and a state register results in a phase :math:`\theta_f =
# \arccos(E_f/\lambda)`, where :math:`E_f` is an eigenvalue of the Hamiltonian,
# being kicked back onto the control register for readout. This method is vastly
# more cost effective for quantum chemistry applications than, for example,
# Trotterization, which scales beyond reasonable resources for high-precision
# applications.
# 
# The Kaiser Window
# .................
# Before we go further, let's quickly dig
# into this nebulous "control register". Figure TK depicts the walk operator
# controlled by a register of size :math:`n_omega`, which we will refer to as
# the "phase wires" going forward. Prior to the walk operator, an operator
# :math:`\mathcal{L}_\delta` operates on the register. This operator encodes a
# `Kaiser lineshape <https://en.wikipedia.org/wiki/Kaiser_window>`_ onto the
# wires, replacing the typical Hadamard invoked superposition used to initiate
# similar registers. Laozia et al. state that this is to reduce "errors coming
# from discretization and finite precision" [#Loaiza2026]_, which arise mainly
# from the incapability of our system to replicate an infinite Dirac delta
# function in the QPE step. 
#
# In the following code, one can decide whether they would like to carry out the
# amplitude estimation and amplification steps for higher success yeild, or
# forgo them in favour of resource and runtime minimization.
# 

@qp.qnode(dev)
def QPEReadout(probs, HighProbBool):
    RIXSStateEncodingUnitary(angles)
    
    KaiserWindow = np.kaiser(2**n_omega+1, 2.0)[:-1] #0 corresponds to a rectangular window shape
    KaiserWindowShifted = np.fft.ifftshift(KaiserWindow)
    KaiserWindowNorm = KaiserWindowShifted/np.linalg.norm(KaiserWindowShifted)
    
    qp.StatePrep(KaiserWindowNorm, wires = phase_wires)
    for i, wire in enumerate(phase_wires):
        for _ in range(2**(int(n_omega)-1-i)):
            qp.ctrl(qp.Qubitization, control = wire)(H, control = QPE_wires)
    qp.adjoint(qp.QFT)(wires = phase_wires)

    return qp.probs(wires = list(success_wire)+list(phase_wires))

###############################################################################
# Resource Definition 
# ................... 
# Before we can successfully run our
# RIXS simulation, some bookkeeping is in order. We have built our systems using
# a total of 9 registers, each of which has a different number of wires. 
#
# The GQSP register, success flag register, and two block encoding ancilla
# registers only require one wire each. The number of wires included in the QPE
# register and the qubitization control register can vary depending on the
# desired precision. The system register should be twice the size of the
# molecular system, which, in this case, is 4 (two core orbitals plus two
# valence orbitals).

eps_omega = 1
eps_QAE = 0.3

N_eps_omega = np.ceil((np.pi*lamb)/(np.sqrt(2)*eps_omega))
n_omega = 7

Na = 4 #core plus two valence
Ne = 6
nQAE = np.ceil(np.log2(1/eps_QAE))

registers = {
    "GQSP": 1,
    "success": 1,
    "controllers": 4,
    "block_encilla_1": 1,
    "block_encilla_2": 1,
    "system": int(2*Na),
    "QAE": int(nQAE),
    "phase": int(n_omega),
    "QPE": 4
}

regs = qp.registers(registers)
###############################################################################
# Which can be unpacked and labelled as necessary.

GQSP_wire = regs["GQSP"]
success_wire = regs["success"]
control_wires = regs["controllers"]
block_encilla_1 = regs["block_encilla_1"]
block_encilla_2 = regs["block_encilla_2"]
system_wires = regs["system"]
QAE_wires = regs["QAE"]
phase_wires = regs["phase"]
QPE_wires = regs["QPE"]
###############################################################################
# 
#
# A note on plotting
# ..................
#
# Interpreting the Results
# ========================
#
# .. _references:
#
# References
# ----------
# .. [#Gao2025] X.\ Gao, B. Li, K. Kummer, A. Geondzhian, D. Aksyonov, R. Dedryvère, D. Foix, G. Rousse, M. B. Yahia, M. L. Doublet, et al., "Clarifying the origin of molecular O2 in cathode oxides," Nature Materials, vol. 24, p. 743-752, 2026. doi: `10.1038/s41563-025-02144-7 <https://www.nature.com/articles/s41563-025-02144-7>`_.
#
# .. [#Loaiza2026] I.\ Loaiza, A. Kunitsa, S. Fomichev, D. Motlagh, D. Dhawan, S. Jahangiri, J. H. Fuglsbjerg, A. Izmalov, N. Wiebe, Y. Abu-Lebdeh, J. M. Arrazola, and A. Delgado, "Quantum algorithm for simulating resonant inelastic X-ray scattering of battery materials," 2026. arXiv. doi: `10.48550/arXiv.2602.20270 <https://doi.org/10.48550/arXiv.2602.20270>`_.