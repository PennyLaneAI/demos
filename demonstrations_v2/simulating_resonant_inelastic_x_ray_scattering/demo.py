r"""
Simulating Resonant Inelastic X-Ray Scattering
##############################################

Our understanding of reality is only as accurate as our models. Our models are only as accurate as our ability to interpret their
results.

Lithium excess (Li-excess)
batteries are currently being eyed as the next generation of high-capacity
batteries. As we attempt to determine why they boast such short lifespans, 
**resonant inelastic x-ray scattering (RIXS)** experiments, an advanced
X-ray spectroscopy technique, have suggested Li-excess cathodes may produce
molecular oxygen that becomes trapped inside the battery, leading to decline.

In 2025, Gao et al. published "Clarifying the origin of molecular O2 in cathode
oxides", noting RIXS experiments show the presence of molecular
oxygen in non-Li-excess batteries incapable of producing these molecules as well. 
This implies these molecules are likely an artifact of the RIXS methodology itself rather than the 
battery process. They additionally
point to this as evidence of a much more complex degradation mechanism involving
the bonding of oxygen dimers to transition metals in the battery materials.

This wasn't necessarily a "back to the drawing board" moment, but it did cause a shift in
interpretation and understanding that shed light on the need for reliable simulations
that can help with the validation and interpretation of experimental results. The 
problem? Classical computers simply cannot handle RIXS simulation for significant system sizes. 

This is precisely the case made by Loaiza et al. in "Quantum algorithm for simulating
resonant inelastic X-ray scattering in battery materials". Here, a quantum algorithm
is put forward to tackle the problem of RIXS simulation
using a novel combination of :doc:`generalized quantum signal processing (GQSP)
<demos/tutorial_estimator_hamiltonian_simulation_gqsp>`, :doc:`amplitude
amplification <demos/tutorial_intro_amplitude_amplification>`, :doc:`quantum
amplitude estimation (QAE) <demos/iterative_quantum_amplitude_estimation>`, and
:doc:`quantum phase estimation (QPE) <demos/tutorial_qpe>`. This solution not only
addresses the typical resource limitations of classical computation, but unlocks
access to the quantum processes RIXS relies on.

Today, our goal will be to understand how these quantum building blocks work
together to make way for reliable RIXS simulation and begin to open the door for
more capable advanced materials discovery in the future. After working through 
this demo, you should be better acquainted with RIXS and how it can be simulated
on quantum devices.

Let's get to work!

Getting Started
===============
What is RIXS?
-------------
The goal of RIXS spectroscopy is to monitor how matter interacts with incident light. 
At a high level, RIXS involves a material being illuminated by 
X-ray photons with energy very close to a core electron's binding
energy (also known as the "absorption edge") [#Loaiza2026]_. The successful
absorption of this photon with frequency :math:`\omega_I` kicks off a two-step
process in which the absorbed photon promotes a core electron to a valence
orbital, leaving behind a hole in the core orbital that is eventually filled by
a different, lower-energy valence electron. This is why RIXS is termed a "photon-in, 
photon-out" process, since the relaxation of the second valence electron into the 
core releases a photon of frequency :math:`\omega_S` that is detected 
and used to compute the difference between the input and output photon energies.

Though this simple explanation is sufficient to understand our observables,
it obscures the fact that RIXS is fundamentally a second-order quantum scattering
process. This means the intermediate state (which exists between the ground
state and final excited state) is actually a coherent collection of virtual states.
So, technically no collapse to the intermediate state occurs, but it can be treated
as its own state for practical purposes.

.. figure:: ../demonstrations_v2/simulating_resonant_inelastic_x_ray_scattering/pennylane-demo-simulating-resonant-inelastic-xray-scattering-EnergyLevelDiagram.png
   :align: center
   :width: 700px
   :alt: An illustration of the three stages of the RIXS process in the form of an energy level diagram.

   *Energy level diagrams for the full RIXS process*

To summarize, the three states involved in the RIXS process are:

1. :math:`|E_0\rangle`: The molecule sits in an unexcited state prior to the absorption of the incident :math:`\omega_I` photon.
2. :math:`|E_n\rangle`: Following the absorption of the incident :math:`\omega_I` photon, a core electron has been excited to a higher-energy valence orbital. This results 
in a short-lived superposition of core hole states.
3. :math:`|E_f\rangle`: To fill the unfavourable core hole, an electron from a lower-energy valence orbital has relaxed into the core, leaving behind a valence hole and emitting a :math:`\omega_S` photon [#Loaiza2026]_. The molecule is left in an excited state.

The difference between :math:`E_f` and :math:`E_0` is known as the **energy
loss**. When energy loss versus intensity is plotted (as is characteristic of a
RIXS spectrum), the spectral peaks can be interpreted as a specific excitation within the
target system. The energy values at which these peaks occur indicate what is present in a reaction,
allowing for correlation between observed excitation peaks and known molecular excitation
energies for identification.

Why Quantum?
------------
Classical simulation is limited in the amount of complexity it can
handle. Basic comparisons of classical and quantum simulation methods tend to make the
case that the maximum compatible system size varies greatly between the two, 
with quantum simulations typically capable of handling many more
states. There are, however, additional, potentially more important
advantages that are not merely reliant on an increase in computational power. In
the case of Loaiza et al.'s RIXS algorithm, the following two advantages are
listed for the quantum case:

1. RIXS spectroscopy involves several **delocalized processes**, meaning the positions
   of the electrons involved are probabilistically spread out across the
   molecule. Capturing this requires a large active space that simply cannot be constructed 
   by classical bits incapable of emulating superposition and entanglement relationships.

2. Strongly correlated systems of interest, such as the transition metal-oxygen
   dimer bonding proposed by Gao et al., experience complex intermediate states (like the case we discussed) 
   that dictate the mixing of orbitals in the oxygen-metal bonding process. To
   model this sufficiently, any computationally simple wavefunction is
   insufficient, and a classical computer is once again incapable of carrying
   out the necessary entanglement math [#Loaiza2026]_.

So, even though it *is* a valid argument to say quantum computers can 
handle larger molecules if necessary, the stronger argument in
this context is that the use of qubits to carry out RIXS simulation makes it
possible to model the quantum phenomena that the technique relies on. Good thing
we know our stuff!

The Hamiltonian
---------------
.. admonition:: A note on operators
   :class: note

   When describing electrons in a molecular system represented using second-quantization, 
   it is conventional to use :doc:`Fermionic
   operators <demos/tutorial_fermionic_operators>` to describe the behaviour of
   the indistinguishable particles that make up the system. In general, the operators of
   concern are:

   |

   1. :math:`\hat{c}_i^\dagger`, the **creation operator**. This is used when an electron
      is "created", effectively occupying some orbital :math:`i`.
   2. :math:`\hat{c}_i`, the **annihilation operator**. This is used when an electron is
      "destroyed", effectively vacating some orbital :math:`i`.

   |

   Combining these operators for a given orbital yields **number operators**,
   which "count" the number of electrons occupying a given orbital:

   .. math::
      \hat{n}_i=\hat{c}^\dagger_i \hat{c}_i


In "Quantum algorithm for simulating resonant inelastic X-ray scattering of
battery materials", Loaiza et al. focus on a second-quantized Hamiltonian of
the form

.. math::
   \hat{H}=E^{0}+\sum_{p,q=1}^{N_{a}}\sum_{\sigma\in \{\uparrow, \downarrow\}}h_{pq}\hat{c}_{p\sigma}^{\dagger}\hat{c}_{q\sigma}+\frac{1}{2}\sum_{p,q,r,s=1}^{N_{a}}\sum_{\sigma,\sigma '\in \{\uparrow, \downarrow\}}V_{pqrs}\hat{c}_{p\sigma}^{\dagger}\hat{c}_{q\sigma}\hat{c}_{r\sigma^{\prime}}^{\dagger}\hat{c}_{s\sigma^{\prime}},

where :math:`N_a` is the number of active orbitals used in the simulation, :math:`p,
q, r,` and :math:`s` are specific orbital indices, :math:`\sigma` and
:math:`\sigma^\prime` are spin states, :math:`h_{pq}` are the one-electron
integrals, :math:`V_{pqrs}` are the two-electron integrals, 
and :math:`E^0` is the total energy of the inner-shell electrons, which are approximated as frozen in the active
space definition. 

Let's make things a bit simpler. For our purposes, we will not apply this Hamiltonian structure
to the :math:`MnO_7H_6` molecule that the source paper focuses on. Instead, we
will take a basic system consisting of two core orbitals and two valence
orbitals to be our focus. To do this, we will adapt the given Hamiltonian as

.. math::
   \hat{H}=\sum_{\sigma\in \{\uparrow, \downarrow\}}(\epsilon_{c_1}\hat{n}_{c_1,\sigma}+\epsilon_{c_2}\hat{n}_{c_2,\sigma}+\epsilon_{\nu_1}\hat{n}_{\nu_1,\sigma}+\epsilon_{\nu_2}\hat{n}_{\nu_2,\sigma})+h\sum_{\sigma\in \{\uparrow, \downarrow\}}(\hat{c}_{\nu_1,\sigma}^\dagger\hat{c}_{\nu_2,\sigma}+\hat{c}_{\nu_2,\sigma}^{\dagger}\hat{c}_{\nu_1,\sigma})+V\sum_{\sigma\in \{\uparrow, \downarrow\}}\hat{n}_{\nu_2,\sigma}

where :math:`c_1` and :math:`c_2` are core orbitals, :math:`\nu_1` and
:math:`\nu_2` are valence orbitals, and :math:`\epsilon_i` are on-site orbital
energies. Note that the :math:`\sum_{p,q=1}^{N_{a}}\sum_{\sigma\in \{\uparrow, \downarrow\}}h_{pq}\hat{c}_{p\sigma}^{\dagger}\hat{c}_{q\sigma}`
term has been explicitly broken up into the diagonal (first term) and off-diagonal (second term) components in 
our Hamiltonian.

To implement this Hamiltonian in PennyLane, we can first call the built-in
Fermionic operators :class:`~pennylane.FermiC` (the creation operator) and
:class:`~pennylane.FermiA` (the annihilation operator). These will be used 
to construct the required number operators.
"""

import pennylane as qp
import numpy as np
from pennylane.fermi import FermiC as create, FermiA as annihilate

#Define number operators

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

s = 0.45 #Optional scaling term

#Orbital energies
eps_c1 = -1.5*s
eps_c2 = -4.5*s
eps_v1 = -1.5*s
eps_v2 = 4.5*s

#One-electron integral
h = 0.5*s

#Two-electron integral
V = 1.0*s
###############################################################################
# When building a toy model for demonstration, we choose physically
# meaningless parameters to create a small system that is sufficient to see
# non-trivial effects. In practice, this system would need to include many
# more orbitals and coefficients extracted from specific calculations
# (such as `Hartree-Fock calculations <https://en.wikipedia.org/wiki/Hartree%E2%80%93Fock_method>`_)
# to be useful. 
#
# With our operators defined, we can build our Hamiltonian terms using standard 
# multiplications and additions. In Fermi operator arithmetic,
# operator ordering is crucial due to the anti-commutation relationships
# that govern them. Luckily for us, PennyLane handles the associated algebra
# for us, meaning we can assemble our Hamiltonian using ordinary arithmetic.
#
# Since we will soon run this expression through our quantum circuits,
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
###############################################################################
# Eventually, we will map this Hamiltonian onto our registers, but
# for now we can extract its eigenvalues and eigenvectors for benchmarking. The algorithm itself
# only needs the ground-state energies and the value of :math:`E_0` to function, but we 
# can construct a reference spectrum from the complete ``np.linalg.eigh()`` output later on.

coeffs, ops = H_raw.terms()
id_c = sum(c for c, o in zip(coeffs, ops) if len(o.wires)==0)  
H_traceless = H_raw-id_c*qp.Identity(0)
H_sparse = H_traceless.sparse_matrix(wire_order=range(8)).toarray()
H_evals, H_evecs = np.linalg.eigh(H_sparse)

#Extract the 1-norm and initial energy value from the Hamiltonian
lamb = float(np.sum(np.abs(H_traceless.terms()[0])))
E_0 = H_evals[0] #Extract ground-state eigenvalue
###############################################################################
# Here, the Hamiltonian was converted to a traceless representation to ensure
# the spectrum can be centered around zero and that the 1-norm is reduced. Overall,
# doing this makes our problem cheaper since the associated shift can be effectively
# ignored, so why not!
#
# The Algorithm
# =============
# Algorithm Overview
# ------------------
# To carry out a quantum simulation of a RIXS spectrum, Loaiza et al. summarize
# their algorithm in two steps:
# 
# 1. Prepare the initial RIXS state :math:`|R_{\epsilon_I,\epsilon_S}(\omega_I)\rangle`,
# 2. Carry out a walk-based :doc:`quantum phase estimation <demos/tutorial_qpe>` to read the final state.
#
# .. figure:: ../demonstrations_v2/simulating_resonant_inelastic_x_ray_scattering/pennylane-demo-simulating-resonant-inelastic-xray-scattering-RIXScircuit.png
#    :align: center 
#    :width: 700px 
#    :alt: An illustrated circuit diagram depicting the general components of Loaiza et al.'s algorithm.
#
#    *The entire RIXS circuit involves preparing the RIXS state and executing walk-based QPE*
#
# Item 1 on this list does a lot of heavy lifting here. In fact, the process of
# preparing the state dominantly contributes to the overall cost of the algorithm. So, we can expand the
# list to capture the complete methodology:
# 
# 1. Prepare the initial RIXS state, :math:`|R_{\epsilon_I,\epsilon_S}(\omega_I)\rangle`
# 
#    a. Construct a PREP-SEL-PREP block-encoding of the Hamiltonian, from which we have 
#    direct access to the associated qubitized walk operator, 
#
#    b. Implement a Green's function spectral filter
#    using GQSP with the walk operator :math:`\hat G(\omega_I,\Gamma) \approx \sum_k w_k \hat W^k`, 
#    which amounts to finding the Chebyshev
#    coefficients of the Green's function and translating them to angles for
#    implementation, 
#
#    c. Define incoming and outgoing polarization to obtain the associated the **dipole operators**
#    :math:`\hat{D}_{\epsilon_i}`, which captures, within the dipole approximation, the perturbation that occurs
#    as a result of the incident excitation and outgoing relaxation, 
#
#    d. Prepare a block-encoding
#    :math:`\hat{\mathcal{U}}` of the operator effectively proportional to
#    :math:`\hat{D}_{\epsilon_S}^\dagger \hat{G}(\omega_I, \Gamma)
#    \hat{D}_{\epsilon_I}`, 
#
#    e. Construct a :doc:`Grover operator
#    <demos/tutorial_grovers_algorithm>` using :math:`\hat{\mathcal{U}}` and
#    carry out amplitude estimation to determine the success probability of the
#    block-encoding step, 
#
#    f. Carry out :doc:`amplitude amplification
#    <demos/tutorial_intro_amplitude_amplification>` on the successful block
#    encoded state to boost the success probability,
# 
# 2. Carry out a walk-based :doc:`quantum phase estimation <demos/tutorial_qpe>` to
#    read the final state.
#
# We have our work cut out for us! Thankfully, most of the tools we need are
# built for us in PennyLane, so let us work through these steps systematically
# to reach our goal. 
#
# Resource Definition
# ...................
# First, some bookkeeping is in order. Based on the algorithm
# outline, we will build our functions using 
# a total of 9 registers, each of which requires a different number of wires. 
#
# The GQSP register, success flag register, and block-encoding ancilla
# register only require one wire each. The number of wires included in the QPE walk
# register and the ancilla registers used for qubitization can be defined according to 
# desired precision. The system register should have the same number of qubits
# as the system has active spin-orbitals, which is twice the number of spatial 
# orbitals.
#
# The remaining two registers (the QAE wires and the QPE wires, which facilitate 
# separate circuits) should be
# computed relative to the desired accuracy and resolution of the spectral
# output. The number of wires in the QAE register can be found via the general
# expression:
#
# .. math:: 
#    \lceil \log_2(1/\epsilon) \rceil,
#
# where :math:`\epsilon` is the target precision.
#
# It is defined in the source paper that the QPE register is given by
#
# .. math:: 
#    \lceil \log_2(N(\epsilon_\omega)) \rceil,
#
# where 
# 
# .. math::
#    N(\epsilon_\omega)=\left\lceil \frac{\pi\lambda}{\sqrt{2}\epsilon_\omega}\right\rceil,
#
# So, we can define our thresholds and compute our register sizes, initializing the
# full set of system registers using :func:`~pennylane.registers`.

eps_omega = 0.2
eps_QAE = 0.3
Na = 4 #two core plus two valence

N_eps_omega = np.ceil((np.pi*lamb)/(np.sqrt(2)*eps_omega))
n_omega = np.ceil(np.log2(N_eps_omega))
nQAE = np.ceil(np.log2(1/eps_QAE))

registers = {
    "GQSP": 1,
    "success": 1,
    "GQSP_walk": 4,
    "block_ancilla": 1,
    "system": int(2*Na),
    "QAE": int(nQAE),
    "QPE": int(n_omega),
    "QPE_walk": 4
}

regs = qp.registers(registers)
###############################################################################
# Which can be unpacked and labelled as necessary.

GQSP_wire = regs["GQSP"]
success_wire = regs["success"]
gqsp_walk_wires = regs["GQSP_walk"]
block_ancilla = regs["block_ancilla"]
system_wires = regs["system"]
QAE_wires = regs["QAE"]
QPE_wires = regs["QPE"]
qpe_walk_wires = regs["QPE_walk"]
###############################################################################
# With these registers defined, we can map our Hamiltonian to the system wires
# to ensure nothing gets crossed along the way.

sys_list = list(system_wires)
wire_map = {i: sys_list[i] for i in range(8)}
H = H_traceless.map_wires(wire_map)
###############################################################################
# BLISS-THC Decomposition 
# -----------------------
# In order to carry out subsequent block-encoding and minimize resource costs,
# the target Hamiltonian needs to be decomposed into a :doc:`linear
# combination of unitaries (LCU) <demos/tutorial_lcu_blockencoding>`. This 
# is a necessary step for block-encoding.
#
# Loaiza et al. select the block-invariant symmetry-shift technique with
# tensor hypercontraction factorization (BLISS-THC) method for their
# decomposition, which is known to be well-suited for compressing molecular
# Hamiltonians [#Caesura2025]_. 
# The THC Hamiltonian [#Lee2021]_
# specifically can be implemented natively in PennyLane :doc:`resource estimation <demos/tutorial_re_for_qubitizedQPE>`
# tasks using :class:`~pennylane.estimator.compact_hamiltonian.THCHamiltonian`.
#
# Luckily, we do not have to worry about this as our toy model is simple enough
# to not require compression. Onward!
#
# Operator Preparation
# --------------------
# The overarching goal of the state preparation component of the algorithm is to create the
# RIXS state
#
# .. math:: |R_{\epsilon_I,
#    \epsilon_S}(\omega_I)\rangle\equiv\frac{\hat{R}_{\epsilon_I,
#    \epsilon_S}(\omega_I)|E_0\rangle}{|R_{\epsilon_I,\epsilon_S}(\omega_I)|},
#
# where :math:`|R_{\epsilon_I,\epsilon_S}(\omega_I)|` is the norm of the state. 
# We will take for granted that this state is equivalent to the block-encoded
# operator
#
# .. math:: \hat{\mathcal{U}}_R \equiv \begin{bmatrix}
#    \frac{\Gamma}{\lambda_D^{(\epsilon_S)}} D_{\epsilon_S}^\dagger
#    \hat{G}(\omega_I, \Gamma) \hat{U}_{\epsilon_I} & \cdot \\ \cdot & \cdot
#    \end{bmatrix}
#
# Where :math:`\Gamma` is the inverse of the intermediate state lifetime in units
# energy, :math:`D_{\epsilon_S}^\dagger` is the
# final state dipole operator, :math:`\lambda_D^{(\epsilon_S)}` is the 1-norm of
# the final state dipole operator, :math:`\hat{G}(\omega_I, \Gamma)` is the
# Green's function, and :math:`\hat{U}_{\epsilon_I}` is an operator that maps the
# initial dipole-perturbed state onto the all-zero state, giving
# :math:`\hat{U}_{\epsilon_I}|0\rangle=|D_{\epsilon_I}\rangle` [#Loaiza2026]_. So, our main
# goal for now is to gather the building blocks of the embedded unitary operator
# and construct this block-encoding representation.
#
# The Dipole Operator
# ...................
# For a given polarization :math:`\epsilon`, the (one-electron) dipole operator can be generally defined as
# 
# .. math::
#    \hat{D}_{\epsilon}=\sum_{pq}\sum_{\sigma\in \{\uparrow, \downarrow\}}d_{pq}^{(\epsilon)}\hat{c}_{p\sigma}^\dagger\hat{c}_{q\sigma}+\text{h.c.},
#
# where :math:`d_{pq}^{(\epsilon)}` are the dipole matrix elements. Note that, for simplicity, we do not consider different polarizations in our toy model.
#
# Since the total dipole operator (containing both the excitation and de-excitation terms) is necessarily Hermitian, we can represent it as 
# 
# .. math::
#    \hat{D}=\hat{D}_{exc}+\hat{D}_{exc}^\dagger.
#
# Thus, we can define the excitation operator using our Fermi operators and construct our total operator.
# 

#Define dipole matrix elements
d_c1 = 1
d_c2 = 1
d_c3 = 0.3
d_c4 = 0.3

#Spin up terms
D_up   = d_c1*create(2)*annihilate(0) + d_c2*create(4)*annihilate(0) + d_c3*create(2)*annihilate(6) + d_c4*create(4)*annihilate(6)
#Spin down terms
D_down = d_c1*create(3)*annihilate(1) + d_c2*create(5)*annihilate(1) + d_c3*create(3)*annihilate(7) + d_c4*create(5)*annihilate(7)

#Full expression
D_half = qp.jordan_wigner(D_up+D_down)
###############################################################################
# From here, we can translate the summation expression into a matrix representation, compute the transpose to achieve the de-excitation dipole operator, and normalize.

#Excitation
D_in_mat = qp.matrix(D_half, wire_order = range(8))
#De-Excitation
D_out_mat = D_in_mat.conj().T

D_mat = D_in_mat+D_out_mat
norm_const = np.linalg.norm(D_mat,2)

#Final dipole operator
D = D_mat/norm_const
###############################################################################
# Green's Function and GQSP
# .........................
#
# Even though the RIXS process is formally a second-order process that yields
# a two dimensional spectrum, Loaiza
# et al. chose to instead focus on the quantum simulation of high-resolution
# RIXS spectra for selected incoming photon frequencies, in line with the
# experimental requirements. This is possible because the associated
# two-dimensional spectrum could be directly accessed through the usage of
# generalized QPE [#Loaiza2024]_. The selection of specific frequencies of
# interest is achieved through the implementation of a
# frequency-specific Green's function, which acts as a spectral filter.
#
# The Green's function is given by
# 
# .. math::
#    \Gamma\hat{G}(\omega_I,\Gamma)=\frac{\Gamma}{\omega_I-(\hat{H}-E_0)+i\Gamma}.
# 
# Note that a :math:`\Gamma` factor represents a physical broadening caused by the 
# finite lifetimes of the intermediate states we discussed previously. 
#
# To use GQSP, the phase factor angles, which essentially instruct a system as to
# how a target function can be encoded onto a system,
# must first be determined. This is a completely classical process that involves
# determining the `Chebyshev coefficients
# <https://en.wikipedia.org/wiki/Chebyshev_polynomials>`_ and converting them
# into an angle representation for use. ``AngleFinder()`` handles this, taking
# advantage of python and PennyLane tools (such as
# :func:`~pennylane.poly_to_angles`, which handles the conversion as long as the
# found polynomial is represented in the Fourier basis) to get the job done. 
#
# For a given target accuracy, the Chebyshev expansion of the Green's function
# will have an associated polynomial degree of :math:`K_G`, as fully explained in
# Appendix A of [#Loaiza2026]_. A higher degree will result in a higher-order
# polynomial expansion, yielding higher resolution.

#Define the Gamma parameter and initial photon energy
Gamma = 0.67*s
omega_I = 6.10*s

#Define the Green's function polynomial degree and scaling factor
K_G = 100

#The Green's function must operate between -1 and 1
z = np.linspace(-1, 1, 1000)
def AngleFinder(Gamma, lamb, E_0, omega_I):
    GreensFunc = lambda x: Gamma/(omega_I-((lamb*x)-E_0)+(1j*Gamma))

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

angles = AngleFinder(Gamma, lamb, E_0, omega_I)
###############################################################################
# Block-Encoding
# ..............
# With the dipole operators defined and the GQSP
# angles found, we can carry out our block-encoding. To achieve this, we
# need to: 
#
# 1. Prepare the dipole-perturbed initial state :math:`U_\epsilon =
#    D_\epsilon|E_0\rangle / ||D_\epsilon|E_0\rangle||` 
#    (the state you get from the incoming photon's dipole operator acting on the
#    system's ground state) on the system register,
# 2. Carry out the GQSP process, encoding the Green's function onto the system
#    register and using a qubitized representation of the system Hamiltonian as
#    our walk operator,
# 3. Block-encode the final conjugate dipole operator D\dagger_\epsilon onto the system
#    register (the de-excitation step that fills the core hole),
# 4. Carry out a controlled X operation that will flag if all
#    inner block-encodings were successful.
#
# .. figure::
#    ../demonstrations_v2/simulating_resonant_inelastic_x_ray_scattering/pennylane-demo-simulating-resonant-inelastic-xray-scattering-BlockEncodingCircuit.png
#    :align: center :width: 700px :alt: A circuit diagram illustration depicting
#    the block-encoding operator for the RIXS state.
#
#     *The RIXS state generator*
# 
# While the source paper uses a block-encoding of the dipole operator which
# optimally reduces the 1-norm but requires a more complex implementation, we
# here instead use PennyLane's built-in block-encoding via Pauli operators via
# :class:`~pennylane.BlockEncode`.
#
# Note that :func:`~qp.GQSP` cannot handle negative exponents in the input function.
# Since we centered our function previously, :func:`~qp.GQSP` will shift the register
# to compensate, requiring us to shift the register back using adjoint walk operators.
# Unfortunately, this adds some resource cost to our implementation, but will
# suffice to achieve the desired output. 

def RIXSStateEncodingUnitary(angles):
    #INITIAL STATE |E_0>
    #Prep the initial state
    psi0 = H_evecs[:,0]
    D_psi0_state = D_mat @ psi0 #Apply the unnormalized dipole operator to the ground state
    D_psi0 = D_psi0_state/np.linalg.norm(D_psi0_state) #normalize the combined state
    qp.StatePrep(D_psi0, wires = system_wires)

    #Define the GQSP walk operator
    W = qp.Qubitization(H, control = gqsp_walk_wires)
    
    #Implement GQSP and uncompute walk operator
    qp.GQSP(W, angles, control = GQSP_wire)
    
    #Undo shift induced due to non-negative exponent restriction
    for _ in range(K_G):
        qp.adjoint(W)
    
    #FINAL STATE |E_f>
    #Encode de-excitation dipole operator
    qp.BlockEncode(D, wires = list(block_ancilla) + list(system_wires))
    
    #Add success flag
    flag_ctrl = list(GQSP_wire) + list(block_ancilla) + list(gqsp_walk_wires)
    qp.ctrl(qp.X, control = flag_ctrl, control_values = [0]*len(flag_ctrl))(wires=success_wire)
###############################################################################
# With this implemented, we have our RIXS state ready!
# ``RIXSStateEncodingUnitary()`` walks through each of the states present in the
# RIXS process. Since we will be using QPE for our
# readout, though, there are a few additional steps that can be taken to ensure
# the true, optimal outcome can be achieved.
#
# Amplitude Estimation and Amplification
# --------------------------------------
# So far, we have implemented a block-encoding of the RIXS state, having that
# our target outcome can be thought of as running (walk-based) QPE on the RIXS
# state. Even though the QPE workflow could be run conditioned upon a successful
# application of the RIXS state block-encoding, Loaiza et al. instead chose to
# use amplitude amplification on this block-encoding to guarantee the success
# and reduce the overall runtime. They note that, while you can carry out
# amplification without prior knowledge of the success probability :math:`P_R`
# via fixed-point amplitude amplification,
# it is "advantageous to first determine :math:`P_R` and then use 'textbook'
# amplitude amplification ... which has better prefactors" [#Loaiza2026]_. 
#
# To quickly review, :doc:`amplitude estimation
# <demos/iterative_quantum_amplitude_estimation>` is the process of determining
# the proportion of a specific "good" state in a data set. In this context, the
# estimation process should give the probability of the block-encoding step
# returning a successful block-encoding, as marked by the success flag mentioned
# earlier. :doc:`Amplitude amplification
# <demos/tutorial_intro_amplitude_amplification>`, on the other hand, carries
# out a series of strategic reflections that increase the relative probability
# of measuring the success state.
#
# They define the true success probability as
#
# .. math:: P_R \equiv \left( \frac{\Gamma |R_{\epsilon_I,\epsilon_S}(\omega_I)|}{\lambda_D^{(\epsilon_S)} |D_{\epsilon_I}|}\right)^2.
#
# Which can be used to determine the number of amplitude amplification steps
# :math:`K_A` via
#
# .. math:: K_A = \left\lfloor \frac{\pi}{4\arcsin\sqrt{P_R}} \right\rfloor
#
# So, if we are able to determine the success probability, we can easily compute
# the amplitude amplification repetition parameter, boost our signal, and move
# forward to our QPE step with confidence.
#
# .. figure::
#    ../demonstrations_v2/simulating_resonant_inelastic_x_ray_scattering/pennylane-demo-simulating-resonant-inelastic-xray-scattering-GroverIterateCircuit.png
#    :align: center :width: 700px :alt: An illustrated circuit diagram for
#    constructing the Grover iterate.
#
#    *Amplitude estimation and amplification requires the construction of a
#    Grover iterate* :math:`\hat{Q}_R`. *Note that* :math:`\ket{\cdot}_R` *is
#    a collection of all block-encoding registers in* :math:`U_R`.
#
# A thorough exploration of how this iterate is manipulated for the task at hand can be
# found in :doc:`the PennyLane Grover's Algorithm demo <demos/tutorial_grovers_algorithm>`.
#
#
def GroverIterate():
    R_reg = list(system_wires) + list(GQSP_wire) + list(block_ancilla) + list(gqsp_walk_wires)
    
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
# Using this, a typical amplitude estimation procedure can be carried out.
dev = qp.device("lightning.qubit")

@qp.qnode(dev)
#Implement QAE
def QAE():
    RIXSStateEncodingUnitary(angles)

    for wire in QAE_wires:
        qp.Hadamard(wires=wire)

    for i, qae_wire in enumerate(QAE_wires):
        exponents = 2**i
        for _ in range(exponents):
            qp.ctrl(GroverIterate, control = qae_wire)()

    qp.adjoint(qp.QFT)(wires = QAE_wires)

    return qp.probs(wires = QAE_wires)
###############################################################################
# Having achieved our amplification parameter, we can repeatedly
# execute the ``GroverIterate()`` operator :math:`K_A` times to achieve a 
# high probability RIXS state, ensuring successful QPE.
#
# .. figure:: ../demonstrations_v2/simulating_resonant_inelastic_x_ray_scattering/pennylane-demo-simulating-resonant-inelastic-xray-scattering-HighProbState.png
#    :align: center 
#    :width: 700px 
#    :alt: An illustrated circuit diagram of the amplitude amplification step.
#
# 
def HighProbRIXSState(probs):
    wires = int(nQAE)

    #Extract the highest probability available and compute P_R
    PeakProbAngle = (np.argmax(probs)/(2**wires))
    P_R = (np.sin(np.pi*PeakProbAngle))**2
    
    #Compute K_a
    if P_R <= 1e-12:
        K_a = 0                     
    else:
        K_a = int(np.floor(np.pi / (4 * np.arcsin(np.sqrt(P_R)))))

    RIXSStateEncodingUnitary(angles)

    #Amplify
    for K in range(K_a):
        GroverIterate()
###############################################################################
# Quantum Phase Estimation and Readout
# ------------------------------------
# Following state preparation, the second step of the algorithm is the application of walk-based QPE, which is the final piece
# of the puzzle in Loaiza et al.'s RIXS simulation. This operator
# is defined as
#
# .. math:: \hat{\mathcal{W}}=\hat{\mathcal{R}}\cdot \text{PREP}^\dagger \cdot
#    \text{SEL} \cdot \text{PREP},
#
# where :math:`\hat{\mathcal{R}}=(\hat{I}-2|0\rangle\langle0|)\otimes\hat{I}`
# [#Loaiza2026]_. This can be taken as an implementable, efficient
# representation of the walk operator :math:`e^{\pm i \arccos
# \hat{H}/\lambda}`. Carrying
# out controlled applications of the walk operator between the QPE register
# and the state register results in a phase 
#
# .. math::
#    \theta_f = \pm\arccos(E_f/\lambda), 
#
# where :math:`E_f` is an eigenvalue of the Hamiltonian,
# being kicked back onto the QPE register for readout.
# 
# The Kaiser Window
# .................
# Prior to the walk operator, an operator
# :math:`\mathcal{L}_\delta` operates on the QPE register. This operator encodes a
# `Kaiser lineshape <https://en.wikipedia.org/wiki/Kaiser_window>`_, 
# replacing the typical sinc lineshape produced by the usual Hadamard initialization, which has long tails
# and leads to worse convergence. Loaiza et al. state that this is to reduce "errors coming
# from discretization and finite precision" [#Loaiza2026]_, which arise mainly
# from the incapability of our system to replicate an infinite Dirac delta
# function in the QPE step. 
#
# With that, we're ready to rock! Or, more accurately, we're ready to walk.
# 

@qp.qnode(dev)
def QPEReadout():
    RIXSStateEncodingUnitary(angles)
    
    KaiserWindow = np.kaiser(2**n_omega+1, 2.0)[:-1] #0 corresponds to a rectangular window shape
    KaiserWindowShifted = np.fft.ifftshift(KaiserWindow)
    KaiserWindowNorm = KaiserWindowShifted/np.linalg.norm(KaiserWindowShifted)
    
    qp.StatePrep(KaiserWindowNorm, wires = QPE_wires)
    for i, wire in enumerate(QPE_wires):
        for _ in range(2**(int(n_omega)-1-i)):
            qp.ctrl(qp.Qubitization, control = wire)(H, control = qpe_walk_wires)
    qp.adjoint(qp.QFT)(wires = QPE_wires)

    return qp.probs(wires = list(success_wire)+list(QPE_wires))

###############################################################################
# Note that the amplitude estimation and amplification steps were skipped here
# for computational simplicity. ``HighProbRIXSState()`` can easily replace 
# ``RIXSStateEncodingUnitary()`` at the beginning of the function, where
# the number of calls to amplitude amplification inside ``HighProbRIXSState()``
# would be determined by the amplitude estimation step.
#
# Some Notes on Plotting
# ......................
# When constructing the final RIXS
# spectrum from the algorithm output, it is noted by Loaiza et al. that a
# convolution step is taken to smooth the output, which involves a Dirac delta
# function as a result of the differential representation of the RIXS amplitude,
# given by
#
# .. math:: P_{\epsilon_I, \epsilon_S}(\omega_I,\omega)=\sum_f ||\langle E_f|\hat{R}_{\epsilon_I,\epsilon_S}(\omega_I)|E_0\rangle||^2\delta(\omega-(E_f-E_0)).
#
# To achieve this, the authors apply a `Lorentzian
# <https://en.wikipedia.org/wiki/Lorentzian>`_ with width :math:`\eta=0.2` eV to
# smooth and account for expected broadening in a realistic system.
#
# An additional, relevant trick is the use of **spectral folding**. This compensates for the fact that the
# eigenvalues of the walk operator are :math:`e^{\pm i \arccos(E/\lambda)}`, meaning the 
# phases the QPE step reads out are :math:`\pm \arccos(E/\lambda)`. This means the QPE
# output is mirror symmetric about the middle bin since each energy value appears in both the 
# :math:`+\theta` and :math:`-\theta` phase branches. Folding recombines each mirrored pair
# into a single bin, which prevents double counting and enables accurate value extraction prior to the plotting process.
#
# .. figure:: ../demonstrations_v2/simulating_resonant_inelastic_x_ray_scattering/pennylane-demo-simulating-resonant-inelastic-xray-scattering-PreFoldedBins.png
#    :align: center 
#    :width: 700px 
#    :alt: A plot depicting the output of the QPE run prior to folding.
#
#    *Prior to folding, the QPE output shows a mirrored set of phase
#    values as a result of the mirrored phase branches of the qubitized walk
#    operator*
#
# Finally, the spectral output should be plotted in terms of recovered energy loss (:math:`E_f-E_0`) versus normalized intensity. The recovered energy loss is given by Loaiza et al. as 
#
# .. math::
#    \lambda\cos(\theta_f)-E_0
# 
# The following function should aid in the implementation of these plotting nuances. 

def plot_qpe_spectrum_tools(amplitude, H_traceless, n_omega, eta=0.2, xmax=4.0):

    lamb = float(np.sum(np.abs(H_traceless.terms()[0])))
    Hm = H_traceless.sparse_matrix(wire_order=range(8)).toarray()
    E_0 = float(np.linalg.eigvalsh(Hm)[0])

    N = 2**int(n_omega)
    amp = np.asarray(amplitude).reshape(2, N)
    block = amp[1] / amp[1].sum() #Select the results associated with the success flag

    # Fold phases
    folded = np.zeros(N // 2 + 1)
    folded[0] = block[0]
    folded[N // 2] = block[N // 2]
    for k in range(1, N // 2):
        folded[k] = block[k] + block[N - k]

    fbins = np.arange(N // 2 + 1)
    ftheta  = 2*np.pi*(fbins / N)
    fenergy = lamb*np.cos(ftheta)-E_0  

    #Lorentzian fit
    w = np.linspace(-1.0, xmax, 2000)
    spec = np.zeros_like(w)
    for prob, ef in zip(folded, fenergy):
        spec += prob * (eta/np.pi)/((w-ef)**2 + eta**2)
    if spec.max() > 0:
        spec /= np.trapezoid(spec, w)
###############################################################################
# Interpreting the Results
# ========================
# Since we are dealing with a small toy model, it is easy for us to plot the
# analytical solution of the RIXS spectrum for comparison. This can be achieved
# via diagonalization of the Hamiltonian matrix.
#
# .. figure::
#    ../demonstrations_v2/simulating_resonant_inelastic_x_ray_scattering/pennylane-demo-simulating-resonant-inelastic-xray-scattering-AnalyticalSolution.png
#    :align: center 
#    :width: 500px 
#    :alt: A plot depicting the analytical solution
#    of the target Hamiltonian.
#
#    *Analytical spectrum*
#
# Running the full RIXS simulation with the provided parameters yields the
# following plot.
#
# .. figure::
#    ../demonstrations_v2/simulating_resonant_inelastic_x_ray_scattering/pennylane-demo-simulating-resonant-inelastic-xray-scattering-RIXSspectrum.png
#    :align: center 
#    :width: 500px 
#    :alt: A plot depicting the simulation output
#    of the target Hamiltonian.
#
#    *Simulated spectrum*
#
# We did pretty well! As shown, the peaks occur at approximately the same energy loss value
# with similar intensity in both plots. This was achieved with a :math:`n_\omega=8` value.
# 
# Upon inspection, it is clear that the two plots are not perfectly aligned, though. The result
# is pretty good, but when we are striving for precision for high-importance validations, accuracy
# is everything. Since we
# made no impactful assumptions about our system in our implementation, why is this the 
# case?
#
# The main culprit here is the value of :math:`n_\omega`, which defines the size of the
# QPE register. As the size of this register increases, the angle bins that the QPE
# read out fall into shrink, resulting in higher resolution. Thus, a maximized register size
# leads to the most accurate results. Unfortunately, increasing this register size
# exponentially increases the size of the simulation, meaning computational resources
# are quickly exhausted on classical devices.
#
# .. figure::
#    ../demonstrations_v2/simulating_resonant_inelastic_x_ray_scattering/pennylane-demo-simulating-resonant-inelastic-xray-scattering-PlotRegisterEvolution.gif
#    :align: center 
#    :width: 500px 
#    :alt: Progression of the simulated spectrum with changing QPE register sizes
#
#    *Progression of the simulated spectrum with changing QPE register sizes*
#
# So, striving for high resolution is a key to achieving an exact simulation of 
# a RIXS system. 
#
# These plots depict two peaks, the elastic peak (centered at 0) and the inelastic
# peak (centered at 2.745). The inelastic peak is the value of interest in the battery
# experiments we discussed previously. The frequency at which this peak occurs can
# be compared to known resonance frequencies of various molecules to determine what
# is present in a chemical process. 
#
# Conclusion
# ==========
# Pursuing useful, accessible quantum technologies requires careful
# consideration of which applications and use cases are most important and well
# suited. The problem of RIXS simulation is a clear example of this, in which a
# gap in computational capability is causing confusion in cutting-edge research
# and can be addressed specifically by the capabilities of quantum algorithms.
# Chasing opportunities such as these is a first step in creating a
# quantum-ready future. The algorithm implementation here shows a simple example
# of a powerful algorithm. Play around with it as you become better acquainted
# with the techniques used and explore larger molecules!
# 
# .. _references:
#
# References
# ----------
# .. [#Gao2025] X.\ Gao, B. Li, K. Kummer, A. Geondzhian, D. Aksyonov, R.
# Dedryvère, D. Foix, G. Rousse, M. B. Yahia, M. L. Doublet, et al., "Clarifying
# the origin of molecular O2 in cathode oxides," *Nature Materials*, vol. 24, p.
# 743-752, 2025. doi: `10.1038/s41563-025-02144-7
# <https://www.nature.com/articles/s41563-025-02144-7>`_.
#
# .. [#Loaiza2026] I.\ Loaiza, A. Kunitsa, S. Fomichev, D. Motlagh, D. Dhawan,
# S. Jahangiri, J. H. Fuglsbjerg, A. Izmaylov, N. Wiebe, Y. Abu-Lebdeh, J. M.
# Arrazola, and A. Delgado, "Quantum algorithm for simulating resonant inelastic
# X-ray scattering in battery materials," 2026. arXiv. doi:
# `10.48550/arXiv.2602.20270 <https://doi.org/10.48550/arXiv.2602.20270>`_.
#
# .. [#Caesura2025] A.\ Caesura, C. L. Cortes, W. Pol, S. Sim, M. Steudtner, G.
# R. Anselmetti, M. Degroote, N. Moll, R. Santagati, M. Streif, and C. S.
# Tautermann, "Faster quantum chemistry simulations on a quantum computer with
# improved tensor factorization and active volume compilation," *PRX Quantum*,
# vol. 6, no. 3, 2025. doi: `10.1103/yngp-5fpm
# <https://link.aps.org/doi/10.1103/yngp-5fpm>`_.
# 
# .. [#Lee2021] J.\ Lee, D. W. Berry, C. Gidney, W. J. Huggins, J. R.
# McClean, N. Wiebe, and R. Babbush, "Even more efficient quantum computations
# of chemistry through hypercontraction," *PRX Quantum*, vol. 2, no. 3, 2021.
# doi: `10.1103/PRXQuantum.2.030305
# <https://link.aps.org/doi/10.1103/PRXQuantum.2.030305>`_.
#
# .. [#Loaiza2024] I.\ Loaiza, D. Motlagh, K. Hejazi, M. S. Zini, A. Delgado,
# and J. M. Arrazola, "Nonlinear Spectroscopy via Generalized Quantum Phase
# Estimation", *Quantum*, vol. 9, 2025. doi: `10.22331/q-2025-08-07-1822
# <https://doi.org/10.22331/q-2025-08-07-1822>`_.