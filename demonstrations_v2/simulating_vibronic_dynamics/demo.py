r"""
Simulating Vibronic Dynamics
############################

Simulating static systems will not get us very far.

The field of simulation has an expansive history of delivering benefit to humanity. The tools that we have had available to us up until now have enabled massive advancements in medicine, energy, smart technologies, fundamental physics ... it would probably be impossible to outline every instance here. The point is, though, that we have been doing pretty well for ourselves but, in doing this, have encountered the limitations of the classical technologies that are available. 

Classical simulations are well poised to handle things that, for the most part, stay still. Ground state energies, isolated systems, and other scenarios that deal with a small number of possible states are, at this point, well studied. The frontiers of the areas mentioned before, however, are expanding beyond these limited models into dynamic scenarios that account for many more interactions and more nuanced behaviours. One such area has been coined `vibronics <https://en.wikipedia.org/wiki/Vibronic_coupling>`_ and is specifically concerned with electronic and nuclear vibrational interactions. Also known as nonadiabtic coupling, vibronics is emerging as an incredibly important tool in theoretical chemistry since it enables movement beyond the Born-Oppenheimer approximation which, put simply, ignores vibronic dynamics completely. 

Whether or not we make the approximation, a molecule composed of :math:`N` atoms will have access to :math:`3N-6` vibrational degrees of freedom (DOFs). When we begin to go beyond Born-Oppenheimer, though, it is no longer possible to isolate electronic DOFs from nuclear DOFs, meaning their paths cannot be simulated seperately. Long story short, if we want to capture the realistic dynamics of entangled electronic and nuclear DOFs, we lose access to approximations that allow for simplification and, as a result, the size of the system will very quickly exceed standard computational resources. So, what comes to mind when we want to simulate realistic atomic system dynamics that scales exponentially in resource requirements and requires nuanced modelling of phenomena such as tunnelling and entanglement? Certainly that this might be an ideal problem for a quantum computer to solve.

In `"Quantum Algorithm for Vibronic Dynamics" <https://arxiv.org/abs/2411.13669>`_ Motlagh et al. propose a novel, quantum-based algorithm for carrying out vibronic simulations. This algorithm leverages several quantum-specific tools (such as `phase gradient states <https://pennylane.ai/compilation/phase-gradient>`, `Trotterization <https://pennylane.ai/challenges/a_simple_trotterization>`, and multiplexing via `QROM loading <https://pennylane.ai/demos/tutorial_intro_qrom>`, which are relevant background topics to this demonstration). This paper lays out an approach that can be generalized to various vibronic Hamiltonians with arbitrary diabatic states and mode interaction specifications, which is particularly notable for its ability to handle beyond two electronic states [#Motlagh2025]_. The generality of this method is emphacized in `"Quantum Algorithm for Simulating Non-Adiabatic Dynamics at Metallic Surfaces" <https://arxiv.org/pdf/2601.16264>`_, which applies the same algorithmic structure to a completely different Hamiltonian. The following demo will explore the implementations carried out in each of these papers by first exploring the general vibronics algorithm, then diving into the nuances of each application to illustrate how the central approach can be generalized. By the end of this, you will be so excited to simulate vibronic dynamics that you won't be able to sit still!

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

letting :math:`x \in \{-\frac{K}{2}, -\frac{K}{2}+1, ..., \frac{K}{2}-1}`. This discretization method will be kept in mind throughout the implementation, but is concretely illustrated in ``GridPrep()``.

"""
def GridPrep(k):
    Delta = np.sqrt(2*np.pi/K)
    x = np.arange(-K//2,K//2)
    Q = np.diag(Delta*x)
    return Q
###############################################################################
# Since position and momentum are non-commuting operators, as stated, they must be seperated into different fragments for Trotterization. The fact that :math:`P` and :math:`Q` do not share a common basis immediately indicates that each operator will require different treatment to carry out time evolution. More nuance can be discerned but considering the specific representations of a position and momentum case. The conventional representation of the kinetic energy term is given as :math:`T=\frac{P^2}{2m}`, where :math:`m` is mass. 
#
# The Potential Step
# ------------------
#
#
# The Kinetic Step
# ----------------
#
# Assembling the Trotter Step
# ---------------------------


print("Hello")

###############################################################################
#
# Add comment blocks to separate code blocks
#
#
# .. _references:
#
# References
# ----------
# .. [#Motlagh2025] D.\ Motlagh, R. A. Lang, P. Jain, J. A. Campos-Gonzalez-Angulo, W. Maxwell, T. Zeng, A. Aspuru-Guzik, and J. M. Arrazola, "Quantum Algorithms for Vibronic Dynamics: Case Study on Singlet Fission Solar Cell Design," 2025, `arXiv: 2411.13669 <https://arxiv.org/abs/2411.13669`_.
#
# .. [#Lang2026] R.\ A. Lang, P. Jain, J. M. Arrazola, and D. Motlagh, "Quantum Algorithm for Simulating Non-Adiabatic Dynamics at Metallic Surfaces," 2026, `arXiv: 2601.16264 <https://arxiv.org/abs/2601.16264>`_.