r"""Shadow Hamiltonian simulation
=================================

Shadow Hamiltonian simulation is a new approach (published last week) to quantum simulation on quantum computers [#SommaShadow]_.
Despite its name, it has little to do with :doc:`classical shadows <demos/tutorial_diffable_shadows>`.
In quantum simulation, the goal is typically to simulate the time evolution of expectation values
of :math:`M` observables :math:`O_m,` for :math:`m=1,\ldots ,M.`
The common approach is to evolve the wave function :math:`|\psi\rangle` and then measure the desired observables after the evolution.

In shadow Hamiltonian simulation, we instead directly encode the expectation values in a proxy state — the **shadow state** — 
and evolve that state accordingly. Specifically for time evolution, we can write a shadow Schrödinger equation that governs the
dynamics of the shadow state.

.. figure:: ../_static/demo_thumbnails/opengraph_demo_thumbnails/OGthumbnail_shadow_hamiltonian_simulation.png
    :align: center
    :width: 70%

This is fundamentally different to the common approach. Foremost, the dimensionality of the 
shadow system no longer depends on the number of constituents, :math:`n,` of the system.
In fact, the underlying state can be mixed or even infinite-dimensional.
Instead, the shadow system's size is dependent on the number of observables :math:`M` that we want to measure.
Note that there are conditions of completeness on the observables for the shadow encoding to succeed, called `invariance property` in [#SommaShadow]_.
Further, since the
expectation values are encoded in the amplitudes of states, we cannot directly measure them anymore, but need to resort to some 
form of state tomography.
On the other hand, this gives us entirely new possibilities by letting us sample from the probability distribution
:math:`p_m = |\langle O_m \rangle|^2` and measure the absolute value of all observables simultaneously in the standard Z basis.

In this demo, we are going to introduce the basic concepts of shadow Hamiltonian simulation alongside some easy-to-follow code snippets.
We will also see later how shadow Hamiltonian simulation comes down to :doc:`g-sim <demos/tutorial_liesim>`, 
a Lie-algebraic classical simulation tool, but run on a quantum computer with some simplifications specifically due to considering Hamiltonian simulation.

Shadow Hamiltonian simulation — Definition
------------------------------------------

In common quantum Hamiltonian simulation, we evolve a state vector :math:`|\psi(t)\rangle` according to the Schrödinger equation,

.. math:: \frac{\text{d}}{\text{dt}} |\psi(t)\rangle = -i H |\psi(t)\rangle,

by some Hamiltonian :math:`H,` and then compute expectation values of the evolved state through measurement.
In shadow Hamiltonian simulation, we encode a set of expectation values in the amplitudes of a quantum state,
and evolve those according to some shadow Schrödinger equation.

For that, we first need to define the shadow state,

.. math:: |\rho\rangle = \frac{1}{\sqrt{A}} \begin{pmatrix} \langle O_1 \rangle \\ \vdots \\ \langle O_M \rangle \end{pmatrix},

for a set of operators :math:`S = \{O_m\}` and normalization constant :math:`A = \sum_m |\langle O_m \rangle|^2.`
This means that we can encode these :math:`M` operator expectation values into :math:`n_S` qubits, as long as :math:`2^{n_S} \geq M.`
Note that :math:`\langle O_m \rangle = \text{tr}[O_m \rho],` so we can have mixed or even infinite-dimensional states :math:`\rho.`

The shadow state evolves according to its shadow Schrödinger equation,

.. math:: \frac{\text{d}}{\text{dt}} |\rho\rangle = - i H_S |\rho\rangle.

The Hamiltonian matrix :math:`H_S` is given by the commutation relations between the system Hamiltonian :math:`H` and
the operators in :math:`S = \{O_m\},`

.. math:: [H, O_m] = - \sum_{m'=1}^M \left( H_S \right)_{m m'} O_{m'}.

Let us solve for the matrix elements :math:`(H_S)_{m m'}.`  To do this, recall that a vector :math:`\boldsymbol{v}` can always be decomposed in an orthogonal basis :math:`\boldsymbol{e}_j` via
:math:`\boldsymbol{v} = \sum_j \frac{\langle \boldsymbol{e}_j, \boldsymbol{v}\rangle}{||\boldsymbol{e}_j||^2} \boldsymbol{e}_j.`
Since the operators under consideration are elements of the vector space of Hermitian operators, we can use this to compute :math:`H_S.`

In particular, with the trace inner product, this amounts to

.. math:: [H, O_m] = \sum_{m'=1}^M \frac{\text{tr}\left( O_{m'} [H, O_m] \right)}{|| O_{m'} ||^2} O_{m'},

from which we can read off the matrix elements of :math:`H_S,` i.e.,

.. math:: (H_S)_{m m'} = -\frac{\text{tr}\left( O_{m'} [H, O_m] \right)}{|| O_{m'} ||^2}.

Now, we can see that the operators :math:`O_m` need to be chosen such that all potentially 
new operators :math:`\mathcal{O} = [H, O_m]`, resulting from taking the commutator between :math:`H` and :math:`O_m,` are decomposable
in terms of :math:`O_m` again. In particular, the operators :math:`O_m` need to form a basis for :math:`\{\mathcal{O} | \mathcal{O} = [H, O_m] \}.`
Another way to say this is that :math:`\{O_m\}` need to contain all nested commutators :math:`[[[H, O_m], O_m'], .. ],` which is similar to :func:`~pennylane.lie_closure` but weaker because it revolves around just :math:`H.`
In the paper this is called the **invariance property**.

.. note::

    Take for example :math:`H = X` and :math:`S = \{Y\}`. Then :math:`[H, Y] = iZ,` so there is no linear combination of elements in :math:`S` that can decompose :math:`[H, Y].`
    We need to extend the list such that we have :math:`S = \{Y, Z\}`. Now all results from commutation, :math:`[H, Y] = iZ` and :math:`[H, Z] = -iY,` are supported by :math:`S.` This is similar
    to the Lie closure that we discuss in our :doc:`intro to Lie algebras for quantum practitioners <demos/tutorial_liesim>`, but the requirements are not as strict because
    we only need support with respect to commentators with :math:`H,` and not among all elements in :math:`S.`

How this relates to g-sim
-------------------------

In :doc:`g-sim <demos/tutorial_liesim>`
[#Somma]_ [#Somma2]_ [#Galitski]_ [#Goh]_, we have operators :math:`\{ g_i \}` that are generators or observables for a parametrized quantum circuit,
e.g. :math:`U(\theta) = \prod_\ell \exp(-i \theta_\ell g_\ell)` and :math:`\langle g_i \rangle.`
For that, we are looking at the so-called dynamical Lie algebra (DLA) of the circuit,
:math:`\mathfrak{g} = \langle \{ g_i \} \rangle_\text{Lie} = \{ g_1, .., g_{|\mathfrak{g}|} \},` as well as
the adjoint representation
:math:`(-i \text{ad}_{g_\gamma})_{\alpha \beta} = f^\gamma_{\alpha \beta},` where :math:`f^\gamma_{\alpha \beta}` are the 
:func:`~pennylane.structure_constants` of the DLA.
They are computed via

.. math:: f^\gamma_{\alpha \beta} = \frac{\text{tr}\left(g_\gamma [g_\alpha, g_\beta] \right)}{||g_\gamma||^2}.

The operators in :math:`\frak{g}` can always be orthonormalized via the `Gram–Schmidt process <https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process>`__,
in which case we can drop the denominator. Further, by means of the cyclic property of the trace, we can rewrite this expression to obtain

.. math:: f^\gamma_{\alpha \beta} = \text{tr}\left(g_\beta [g_\gamma, g_\alpha] \right).

From this, we see how :math:`H_S` corresponds to the adjoint representation :math:`i \text{ad}_H` (but we don't require the full Lie algebra here, see below).
For further details on the concept of the adjoint representation, see our
:doc:`demo on g-sim <demos/tutorial_liesim>` that makes extensive use of it.

In g-sim, we also evolve expectation vectors :math:`(\vec{g})_i = \langle g_i \rangle.`
In particular, the circuit of evolving a state according to :math:`U(\theta)` and computing expectation values 
:math:`\langle g_i \rangle` then corresponds to evolving :math:`\vec{g}` by :math:`\prod_\ell \exp(-i \theta_\ell \text{ad}_{g_\ell}).`

Shadow Hamiltonian simulation can thus be viewed as g-sim
with a single, specific gate :math:`U(\theta) = e^{-i \theta H}` and parameter :math:`\theta = t,` and run on a quantum computer.

One striking difference is that, because
we only have one specific "gate", we do not need the full Lie closure of the operators whose expectation values we want to compute.
Instead, here it is sufficient to choose :math:`O_m` such that they build up the full support for all :math:`[H, O_m].`
This could potentially be a significant difference, as the Lie closure in most cases leads to an exponentially large DLA [#Wiersema]_ [#Aguilar]_,
though the scaling of the span of all :math:`[H, O_m]` is unclear at this point.

A simple example
----------------

The abstract concepts of shadow Hamiltonian simulation are best illustrated with a simple and concrete example.
We are interested in simulating the Hamiltonian evolution of 

.. math:: H = X + Y

after a time :math:`t = 1` and computing the expectation values of :math:`S = \{X, Y, Z, I \}.`
In the standard formulation, we simply evolve the initial quantum state :math:`|\psi(0)\rangle = |0\rangle` by :math:`H` in the
following way.

"""

import pennylane as qml
qml.capture.enable()
import numpy as np
from pennylane import X, Y, Z, I

dev = qml.device("default.qubit")

S = [X(0), Y(0), Z(0), I(0)]
H = X(0) + Y(0)

@qml.qnode(dev)
def evolve(H, t):
    qml.evolve(H, t)
    return [qml.expval(Om) for Om in S]

t = 1.
O_t_standard = np.array(evolve(H, t))
O_t_standard

##############################################################################
# We evolved a :math:`2^n = 2` dimensional quantum state and performed :math:`3` independent (non-commuting) measurements.
#
# In shadow Hamiltonian simulation, we encode :math:`4` expectation values in a :math:`2^2 = 4`-dimensional
# quantum state, i.e., :math:`n_S = 2.`
#
# For this specific example, the number of operators is larger than the number of qubits, leading to a shadow system that
# is larger than the original system. This may or may not be a clever choice, but the point here is just to illustrate
# the conceptual difference between both approaches. The authors in [#SommaShadow]_ show various examples where
# the resulting shadow system is significantly smaller than the original system. It should also be noted that having a smaller shadow system may not
# always be its sole purpose, as there are conceptually new avenues one can explore with shadow Hamiltonian simulation, such
# as sampling from the distribution :math:`p_m = |\langle O_m \rangle |^2.`
#
# Let us first construct the initial shadow state :math:`\boldsymbol{O}(t=0)` by computing
# :math:`\langle O_m \rangle_{t=0} = \text{tr}\left(O_m |\psi(0)\rangle \langle \psi(0)| \right)`
# with :math:`|\psi(0)\rangle = |0\rangle.`
# The ``pauli_rep`` attribute of PennyLane operators returns a :class:`~.pennylane.pauli.PauliSentence` instance and lets us efficiently
# compute the trace, where we use the trick that :math:`|0 \rangle \langle 0| = (I + Z)/2.`
#

S_pauli = [op.pauli_rep for op in S]

O_0 = np.zeros(len(S))

for m, Om in enumerate(S_pauli):
    psi0 = (I(0) + Z(0)).pauli_rep

    O_0[m] = (psi0 @ Om).trace()


O_0

##############################################################################
# There are a variety of methods to encode this vector in a qubit basis, but we will just be using
# :class:`~.pennylane.StatePrep` later.
#
# We now go on to construct the shadow Hamiltonian :math:`H_S` by computing the elements
# :math:`(H_S)_{m m'} = \frac{\text{tr}\left( O_{m'} [H, O_m] \right)}{|| O_{m'} ||^2},` and
# we again make use of the :meth:`~.pennylane.pauli.PauliSentence.trace` method.
#

H_pauli = H.pauli_rep

H_S = np.zeros((len(S), len(S)), dtype=complex)

for m, Om in enumerate(S_pauli):
    com = H_pauli.commutator(Om)
    for mt, Omt in enumerate(S_pauli):
        # v = ∑ (v · e_j / ||e_j||^2) * e_j

        value = (Omt @ com).trace()
        value = value / (Omt @ Omt).trace()  
        H_S[m,mt] = value

H_S = -H_S # definition eq. (2) in [1]

##############################################################################
# In order for the shadow evolution to be unitary and implementable on a quantum computer,
# we need :math:`H_S` to be Hermitian.
# 

np.all(H_S == H_S.conj().T)

##############################################################################
# Knowing that, we can write the formal solution to the shadow Schrödinger equation as
# 
# .. math:: \boldsymbol{O}(t) = \exp\left(-i t H_S \right) \boldsymbol{O}(0).
# 

from scipy.linalg import expm

O_t = expm(-1j * t * H_S) @ O_0
O_t

##############################################################################
# Up to this point, this is equivalent to :doc:`g-sim <demos/tutorial_liesim>` if we were doing classical simulation.
# Now, the main novelty for shadow Hamiltonian simulation is to perform this on a quantum computer by encoding the 
# expectation values of :math:`\langle O_m \rangle` in the amplitude of a quantum state, and to translate :math:`H_S`
# accordingly.
#
# This can be done by decomposing the numerical matrix :math:`H_S` into Pauli operators, which can, in turn,
# be implemented on a quantum computer.
#

H_S_qubit = qml.pauli_decompose(H_S)
H_S_qubit

##############################################################################
# Using all these ingredients, we now are able to formulate the shadow Hamiltonian simulation as a quantum algorithm.
# For the amplitude encoding, we need to make sure that the state is normalized. We use that normalization factor to then
# later retrieve the correct result.
#

A = np.linalg.norm(O_0)

@qml.qnode(dev)
def shadow_evolve(H_S_qubit, O_0, t):
    qml.StatePrep(O_0 / A, wires=range(2))
    qml.evolve(H_S_qubit, t)
    return qml.state()

O_t_shadow = shadow_evolve(H_S_qubit, O_0, t) * A

print(O_t_standard)
print(O_t_shadow)

##############################################################################
# We see that the results of both approaches match.
#
# The first result is coming from three independent measurements on a quantum computer after evolution with system Hamiltonian :math:`H.`
# This is conceptually very different from the second result where
# :math:`\boldsymbol{O}` is encoded in the state of the shadow system (note the ``qml.state()`` return), which we evolved according to :math:`H_S.`
#
# In the first case, the measurement is directly obtained, however, 
# in the shadow Hamiltonian simulation, we need to access the amplitudes of the underlying state.
# This can be done naively with state tomography, but in instances where we know 
# that :math:`\langle O_m \rangle \geq 0,` we can just sample bitstrings according to
# :math:`p_m = |\langle O_m\rangle|^2.` The ability to sample from such a distribution 
# :math:`p_m = |\langle O_m\rangle|^2` is a unique and new feature to shadow Hamiltonian simulation.
#
# We should also note that we made use of the abstract quantum sub-routines :func:`~.pennylane.evolve` and :class:`~.pennylane.StatePrep`, which each warrant their
# specific implementation. For example, :class:`~.pennylane.StatePrep` can be realized by :class:`~MottonenStatePreparation` and :func:`~.pennylane.evolve` can be realized
# by :class:`TrotterProduct`, though that is not be the focus of this demo.

##############################################################################
# 
# Conclusion
# ----------
#
# We introduced the basic concepts of shadow Hamiltonian simulation and learned how it fundamentally differs from the common approach to Hamiltonian simulation.
#
# We have seen how classical Hamiltonian simulation is tightly connected to g-sim, but run on a quantum computer.
# A significant difference comes from the fact that the authors in [#SommaShadow]_ specifically look at Hamiltonian simulation, :math:`\exp(-i t H),`
# which allows us to just look at operators :math:`O_m` that support all commutators :math:`[H, O_m],` instead of the full Lie closure.
# There may be some advantage to this feat, because Lie algebras in quantum computing typically scale exponentially [#Wiersema]_ [#Aguilar]_.
# However, the scaling of such sets of operators is unclear at this point and needs further investigation.
#
# Note that even in the case of an exponentially sized set of operators, we have — at least in principle — an exponentially large state vector to store the
# :math:`M \leq 2^{n_S}` values. In the absolute worst case we have :math:`\mathfrak{su}(2^n)` with a dimension of 
# :math:`2^{2n}-1,` so :math:`n_S = 2n` and thus it is just doubling the number of qubits.
#
# The biggest potential to this new persepctive on Hamiltonian simulation most likely lies in finding interesting applications like [#Babbush]_ or [#Barthe]_
# that naturally encode the problem and allow for efficient retrieval of all the relevant information.
#



##############################################################################
# 
# References
# ----------
#
# .. [#SommaShadow]
#
#     Rolando D. Somma, Robbie King, Robin Kothari, Thomas O'Brien, Ryan Babbush
#     "Shadow Hamiltonian Simulation"
#     `arXiv:2407.21775 <https://arxiv.org/abs/2407.21775>`__, 2024.
#
# .. [#Somma]
#
#     Rolando D. Somma
#     "Quantum Computation, Complexity, and Many-Body Physics"
#     `arXiv:quant-ph/0512209 <https://arxiv.org/abs/quant-ph/0512209>`__, 2005.
#
# .. [#Somma2]
#
#     Rolando Somma, Howard Barnum, Gerardo Ortiz, Emanuel Knill
#     "Efficient solvability of Hamiltonians and limits on the power of some quantum computational models"
#     `arXiv:quant-ph/0601030 <https://arxiv.org/abs/quant-ph/0601030>`__, 2006.
#
# .. [#Galitski]
#
#     Victor Galitski
#     "Quantum-to-Classical Correspondence and Hubbard-Stratonovich Dynamical Systems, a Lie-Algebraic Approach"
#     `arXiv:1012.2873 <https://arxiv.org/abs/1012.2873>`__, 2010.
#
# .. [#Goh]
#
#     Matthew L. Goh, Martin Larocca, Lukasz Cincio, M. Cerezo, Frédéric Sauvage
#     "Lie-algebraic classical simulations for variational quantum computing"
#     `arXiv:2308.01432 <https://arxiv.org/abs/2308.01432>`__, 2023.
#
# .. [#Wiersema]
#
#     Roeland Wiersema, Efekan Kökcü, Alexander F. Kemper, Bojko N. Bakalov
#     "Classification of dynamical Lie algebras for translation-invariant 2-local spin systems in one dimension"
#     `arXiv:2309.05690 <https://arxiv.org/abs/2309.05690>`__, 2023.
#
# .. [#Aguilar]
#
#     Gerard Aguilar, Simon Cichy, Jens Eisert, Lennart Bittel
#     "Full classification of Pauli Lie algebras"
#     `arXiv:2408.00081 <https://arxiv.org/abs/2408.00081>`__, 2024.
#
# .. [#Babbush]
#
#     Ryan Babbush, Dominic W. Berry, Robin Kothari, Rolando D. Somma, Nathan Wiebe
#     "Exponential quantum speedup in simulating coupled classical oscillators"
#     `arXiv:2303.13012 <https://arxiv.org/abs/2303.13012>`__, 2023.
#
# .. [#Barthe]
#
#     Alice Barthe, M. Cerezo, Andrew T. Sornborger, Martin Larocca, Diego García-Martín
#     "Gate-based quantum simulation of Gaussian bosonic circuits on exponentially many modes"
#     `arXiv:2407.06290 <https://arxiv.org/abs/2407.06290>`__, 2024.
#
