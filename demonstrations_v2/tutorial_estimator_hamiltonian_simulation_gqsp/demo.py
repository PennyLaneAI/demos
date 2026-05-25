r"""Resource estimation for Hamiltonian simulation with GQSP
============================================================

Simulating the time evolution of a quantum system is arguably the most useful problem with an exponential quantum speedup.
This task is known as **Hamiltonian simulation**, and it is the most common subroutine used in quantum algorithms
for chemistry, materials science, and condensed matter physics. Multiple strategies exist
with unique strengths and weaknesses, but the best asymptotic scaling is achieved by methods based on quantum
signal processing (QSP) [#qsp]_. Quantifying the precise constant-factor cost of these algorithms is challenging,
as we need to compute the cost of block encoding Hamiltonians and determine the correct number of phase factors as
a function of evolution time and target errors. Well, at least it *used* to be challenging.

.. figure:: ../_static/demo_thumbnails/opengraph_demo_thumbnails/pennylane-demo-hamiltonian-simulation-with-gqsp-open-graph.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

In this demo, we use PennyLane's :mod:`~.pennylane.estimator` module to compute the cost of Hamiltonian
simulation with QSP, making it simple to determine how useful it is for your application of interest. We focus
on the modern framework of **generalized quantum signal processing (GQSP)** and study examples of resource estimation
for a simple spin model and for a Heisenberg Hamiltonian for NMR spectral prediction. More information on QSP
can be found in our other demos:

- `Function Fitting using Quantum Signal Processing <function_fitting_qsp>`_
- `Using PennyLane and Qualtran to analyze how QSP can improve measurements of molecular properties <tutorial_qksd_qsp_qualtran>`_
- `Intro to QSVT <tutorial_intro_qsvt>`_

Hamiltonian simulation with GQSP
--------------------------------
QSP is a method for performing polynomial transformations of
block-encoded operators. It consists of a (i) sequence of interleaving *signal* operators that perform the
block-encoding, and (ii) single-qubit *signal-processing* operators that define the polynomial.
GQSP expands on the original approach by considering signal-processing operators that are general
:math:`SU(2)` transformations; this removes restrictions on available polynomial transformations
and facilitates solving for the :math:`SU(2)` *phase factors*, making it the modern method of choice [#gqsp2024]_.

Hamiltonian simulation is the task of implementing the time-evolution operator :math:`e^{-iHt}`.
We do this for an input Hamiltonian :math:`H` and a target evolution time :math:`t`, subject to a target error
:math:`\varepsilon`.
GQSP solves this challenge by expressing the complex exponential :math:`e^{-iHt}` as a polynomial,
truncated to a degree determined by :math:`t` and :math:`\varepsilon`. The corresponding
transformation is then implemented on a block-encoding of :math:`H`.

In practice, it is customary to instead encode the Hamiltonian as a walk operator :math:`\hat{W}`, and
approximate the function :math:`e^{-i\cos (\hat{W})t}` through the rapidly-converging Jacobi-Anger expansion
[#gqsp2024]_. This type of block-encoding is a `qubitization <tutorial_qubitization>`_ of the Hamiltonian.
It can be implemented by a sequence of Prepare and Select operators that are induced by a
`linear combination of unitaries (LCU) decomposition <tutorial_lcu_blockencoding>`_.


Let's explore how to use PennyLane to estimate the cost of Hamiltonian simulation with GQSP.

Spin dynamics
-------------
We focus on the :math:`XX` model Hamiltonian with no external field, defined as

.. math::

    H_{XX} = -  \sum_{i,j=1}^{N} J_{ij} (X_i X_j + Y_iY_j),

where :math:`X,Y` are Pauli matrices acting on a given spin site and :math:`N` is the number of spins.
The coefficients :math:`J_{ij}` can be interpreted as the adjacency matrix of a graph that defines the
coupling between spins. Generating samples from a time-evolved state under this Hamiltonian (as well as
many other spin models) is widely believed to be an intractable classical problem [#spin]_. **But what is
the cost of performing this simulation on a quantum computer?** For example: how many qubits and gates are
needed to perform Hamiltonian simulation with GQSP on a square grid of :math:`100\times 100` spins?

To answer this, we first define the Hamiltonian. For the purpose of resource estimation, the specific coupling coefficients are
unimportant since the algorithm works identically regardless of their concrete values. This allows us to define
**compact Hamiltonians** that are easy to instantiate by specifying only the type and number of Pauli operators.
With periodic boundary conditions, each spin site on the lattice is coupled to four nearest neighbours, so
we have 10,000 qubits with 40,000 :math:`XX` and :math:`YY` couplings respectively. This information is defined as a
dictionary and passed directly to the compact :class:`~.pennylane.estimator.compact_hamiltonian.PauliHamiltonian` class:
"""

import pennylane.estimator as qre
import numpy as np

pauli_dictionary = {"XX": 40000, "YY": 40000}  # 4*100*100

xx_hamiltonian = qre.PauliHamiltonian(
    num_qubits=10000,  # 100*100
    pauli_terms=pauli_dictionary,
)

###################################
# We now construct the walk operator, which consists of a sequence
# of Prepare and Select operators. For Prepare, we need extra qubits to load the coefficients, and will employ a
# standard state preparation algorithm called `QROMStatePreparation <https://docs.pennylane.ai/en/stable/code/api/pennylane.estimator.templates.QROMStatePreparation.html>`_,
# based on `QROM <tutorial_intro_qrom>`_, which is natively supported in PennyLane:


num_terms = xx_hamiltonian.num_terms  # number of terms in the Hamiltonian
num_state_prep_qubits = int(np.ceil(np.log2(num_terms)))

Prep = qre.QROMStatePreparation(
    num_state_qubits=num_state_prep_qubits,
)

print(f"Resources for Prepare")
print(qre.estimate(Prep))


###################################
# For Select, PennyLane provides the
# `SelectPauli <https://docs.pennylane.ai/en/stable/code/api/pennylane.estimator.templates.SelectPauli.html>`_
# resource operator, tailored to Pauli Hamiltonians.

Sel = qre.SelectPauli(xx_hamiltonian)
print(f"Resources for Select")
print(qre.estimate(Sel))

###################################
# We use Prepare and Select to construct the walk operator, which can be built directly using the dedicated
# `Qubitization <https://docs.pennylane.ai/en/stable/code/api/pennylane.estimator.templates.Qubitization.html>`_ operator:

Walk = qre.Qubitization(Prep, Sel)
print(f"Resources for Walk operator")
print(qre.estimate(Walk))

###################################
# Finally, these pieces are brought together to estimate the cost of performing Hamiltonian simulation with GQSP.
# This can be calculated directly with the `GQSPTimeEvolution <https://docs.pennylane.ai/en/stable/code/api/pennylane.estimator.templates.GQSPTimeEvolution.html>`_
# operator. Under the hood, it constructs the `GQSP <https://docs.pennylane.ai/en/stable/code/api/pennylane.estimator.templates.GQSP.html>`_
# sequence and determines the required polynomial degree to simulate the desired dynamics.
#
# As an example, we assume the Hamiltonian is normalized and calculate the degree needed to evolve for
#  :math:`t=100` and a target error of :math:`\epsilon=0.1\%`.

HamSim = qre.GQSPTimeEvolution(Walk, time=100, one_norm=1, poly_approx_precision=0.001)

print(f"Resources for Hamiltonian simulation with GQSP")
print(qre.estimate(HamSim))


###################################
# This is a large system with non-trivial dynamics, yet we can analyze its requirements
# in a straightforward manner using PennyLane.
# We will now study a more practical example applying spin dynamics to NMR spectroscopy.
#
#
# Heisenberg model for NMR spectral prediction
# --------------------------------------------
# We follow work presented in Ref. [#nmr]_ describing quantum simulation of NMR in the zero-to-ultralow field regime.
# The main computational task is to simulate time evolution under a Heisenberg Hamiltonian of the form
#
# .. math::
#   H = \sum_{k\neq l} J_{kl}\vec{\sigma}_k\cdot \vec{\sigma}_l + D_{kl}^{\alpha \beta}\sigma^\alpha_k\sigma^\beta_l
#   - \sum_k\vec{h}_k\cdot \vec{\sigma}_k,
#
# where the sums over :math:`k,l` run over spin sites, and the sum over :math:`\alpha, \beta` run through
# spatial directions :math:`x,y`, and :math:`z`. We're using the notation :math:`\vec{\sigma}\cdot \vec{\sigma} = \sigma_{x}\sigma_{x}+\sigma_{y}\sigma_{y}+\sigma_{z}\sigma_{z}`.
#
# As before, we build a compact Hamiltonian representation by counting the number of Pauli operators of
# each kind, which is straightforward from expanding the Hamiltonian. For :math:`N` spins, sums over :math:`k\neq l`
# run over :math:`N(N-1)/2` terms, and there are 9 possible pairs of Pauli matrices.
# The last sum over :math:`k` contains :math:`N` terms of 3 different Paulis.
# We study a system of :math:`N=32` spins as in the paper, which gives the following description of the Hamiltonian:

pauli_dictionary = {
    "XX": 496,  # 32*31/2 = 496
    "YY": 496,
    "ZZ": 496,
    "XZ": 496 * 2,  # accounting for "ZX"
    "XY": 496 * 2,  # accounting for "YX"
    "YZ": 496 * 2,  # accounting for "ZY"
    "X": 32,
    "Y": 32,
    "Z": 32,
}

nmr_hamiltonian = qre.PauliHamiltonian(
    num_qubits=32,
    pauli_terms=pauli_dictionary,
)
####################################
# We build Prepare and Select operators that are used to define the walk operator.
# For `QROMStatePreparation <https://docs.pennylane.ai/en/stable/code/api/pennylane.estimator.templates.QROMStatePreparation.html>`_,
# we enforce a minimal use of auxiliary qubits via the :code:`select_swap_depths` argument.

num_terms = nmr_hamiltonian.num_terms  # number of terms in the Hamiltonian
num_state_prep_qubits = int(np.ceil(np.log2(num_terms)))

Sel_nmr = qre.SelectPauli(nmr_hamiltonian)
Prep_nmr = qre.QROMStatePreparation(
    num_state_qubits=num_state_prep_qubits,
    positive_and_real=True,
    select_swap_depths=1,  # this minimizes auxiliary qubits
)

Walk_nmr = qre.Qubitization(Prep_nmr, Sel_nmr)

print(f"Resources for NMR Walk operator")
print(qre.estimate(Walk_nmr))

#####################################
# Let's explore how different choices of evolution time and error affect cost. From theoretical arguments,
# one would expect linear growth in cost with time, and logarithmic growth with inverse error.
# We analyze the total number of non-Clifford gates (T+Toffoli) as a function of these parameters:


def nmr_resources(time, one_norm, error):

    gqsp = qre.GQSPTimeEvolution(Walk_nmr, time, one_norm, error)
    resources = qre.estimate(gqsp)
    T_gates = resources.gate_counts["T"]
    Toffoli_gates = resources.gate_counts["Toffoli"]

    return int(T_gates + Toffoli_gates)


#####################################
# We plot the non-Clifford gate cost of the algorithm for different values of total evolution time. This includes
# two cases where the one-norm differs by a factor of 2, to illustrate the linear increase in cost
# as a function of one-norm. This is equivalent to rescaling the units of time by a factor of :math:`\frac{1}{2}`.

import matplotlib.pyplot as plt

one_norm = 1.0
error = 0.001
time_array = np.linspace(1, 1000, 100)  # time from 1 to 1,000
gates = [nmr_resources(t, one_norm, 0.001) for t in time_array]
gates2 = [nmr_resources(t, 2 * one_norm, 0.001) for t in time_array]  # double the one-norm

plt.plot(time_array, gates, label=f"one_norm: {one_norm}")
plt.plot(time_array, gates2, label=f"one_norm: {2*one_norm}")
plt.xlabel("Time")
plt.ylabel("Non-Clifford gates")
plt.grid(True, which="both", linestyle="--")
plt.legend()
plt.show()

#####################################
# Finally, we analyze resources as a function of error to highlight how Hamiltonian simulation with GQSP
# can rapidly converge to very small errors with minimal overhead.
#

error_array = np.logspace(2, 9, 100)
gates = [nmr_resources(10, one_norm, 1 / error) for error in error_array]
plt.plot(error_array, gates)
plt.xlabel("Inverse Error")
plt.xscale("log")
plt.ylabel("Non-Clifford gates")
plt.grid(True, which="both", linestyle="--")
plt.show()


######################################################################
#
# Conclusion
# ----------
# Hamiltonian simulation with GQSP is a well-established quantum algorithm with many useful applications.
# It consists of multiple subroutines that may require time to master, but PennyLane's quantum resource
# :mod:`~.pennylane.estimator` elegantly aggregates them into easy-to-use operations. This library of
# operations allows us to both rapidly and accurately quantify resources for specific use cases, and assess
# their hardware-readiness. Try using :mod:`~.pennylane.estimator` yourself, and see how much it would cost
# for you to study other systems using a quantum computer, such as electronic structure, vibrational, and vibronic Hamiltonians.
#
# References
# ----------
#
# .. [#qsp]
#
#     Guang Hao Low, Isaac L. Chuang,
#     "Hamiltonian simulation by qubitization",
#     `Quantum, 3, 163 <https://quantum-journal.org/papers/q-2019-07-12-163/>`__, 2019.
#
#
# .. [#gqsp2024]
#
#     Danial Motlagh, Nathan Wiebe,
#     "Generalized quantum signal processing",
#     `PRX Quantum 5.2, 020368 <https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.5.020368>`__, 2024.
#
#
# .. [#nmr]
#
#     Justin Elenewski, Christina Camara, Amir Kalev,
#     "Prospects for NMR spectral prediction on fault-tolerant quantum computers",
#     `arXiv:2406.09340 <https://arxiv.org/abs/2406.09340>`__, 2024.
#
# .. [#spin]
#
#     Chae-Yeun Park, Pablo A.M. Casares, Juan Miguel Arrazola, Joonsuk Huh,
#     "The hardness of quantum spin dynamics",
#     `arXiv:2312.07658 <https://arxiv.org/abs/2312.07658>`__, 2023.
#
