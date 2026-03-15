r"""g-sim: Lie-algebraic classical simulations for variational quantum computing
================================================================================

For the most part, we now know the phenomenon of :doc:`barren plateaus <demos/tutorial_local_cost_functions>`
can be reduced to the dimension of the circuit's :doc:`dynamical Lie algebra (DLA) <demos/tutorial_liealgebra>`.
In particular, exponentially sized DLAs lead to exponentially vanishing gradients (barren plateaus).
Conversely, it has been realized that circuits with polynomially sized DLAs can be efficiently simulated using a technique called :math:`\mathfrak{g}`-sim,
leading to discussions on whether all trainable parametrized circuits are also efficiently classically simulable.

So what is all the fuss about? How does :math:`\mathfrak{g}`-sim work? What are its restrictions? How can I run it in PennyLane?
We are going to try to answer all these questions in the demo below.

Introduction
------------

Lie algebras are tightly connected to quantum physics [#Kottmann]_.
While Lie algebra theory is an integral part of high energy and condensed matter physics,
recent developments have shown connections to quantum simulation and quantum computing.
In particular, the infamous :doc:`barren plateau problem <demos/tutorial_barren_plateaus/>` has been fully characterized by the underlying
:doc:`dynamical Lie algebra (DLA) <demos/tutorial_liealgebra/>` [#Fontana]_ [#Ragone]_.
The main result of these works is that the dimension of the circuit's DLA is inversely proportional to the variance of the mean of the gradient
(over a uniform parameter distribution), leading to exponentially vanishing gradients in the uniform average case whenever the 
DLA scales exponentially in system size.

At the same time, there exist Lie algebraic techniques with which one can classically simulate expectation values of circuits with a complexity polynomial
in the dimension of the circuit's DLA [#Somma]_ [#Somma2]_ [#Galitski]_ [#Goh]_.
Hence, circuits with guaranteed non-exponentially vanishing gradients in the uniform average case are classically simulable,
leading to some debate on whether the field of variational quantum computing is doomed or not [#Cerezo]_.
The majority of DLAs are in fact exponentially sized [#Wiersema]_, shifting this debate towards the question of whether or not uniform average case results
are relevant in practice for variational methods [#Mazzola]_, with some arguing for better initialization methods [#Park]_.

In this demo, we want to focus on those cases where efficient classical simulation is possible due to polynomially sized DLAs.
These instances are rather limited as it mainly concerns DLAs of non-interacting systems as well as the transverse-field Ising model and variations thereof (see [#Wiersema]_ for details).

Lie algebra basics
------------------

Before going into the specifics of Lie algebra simulation (:math:`\mathfrak{g}`-sim),
we want to briefly recap the most important concepts of Lie algebra theory that are relevant for us.
More info can be found in our 
:doc:`Intro to (Dynamical) Lie Algebras for quantum practitioners <demos/tutorial_liealgebra/>`.

Given Hermitian operators :math:`G = \{h_i\}` (think Hermitian observables like terms of a Hamiltonian),
the dynamical Lie algebra :math:`\mathfrak{g}`
can be computed via the Lie closure :math:`\langle \cdot \rangle_\text{Lie}` (see :func:`~pennylane.lie_closure`),

.. math:: \mathfrak{g} = \langle \{h_i\} \rangle_\text{Lie} \subseteq \mathfrak{su}(2^n).

That is, by computing all possible nested commutators until no new operators emerge. This leads to a set of operators that is closed under commutation,
hence the name.
In particular, the result of the commutator between any two elements in :math:`\mathfrak{g}`
can be decomposed as a linear combination of other elements in :math:`\mathfrak{g},`

.. math:: [h_\alpha, h_\beta] = \sum_\gamma f^\gamma_{\alpha \beta} h_\gamma.

The coefficients :math:`f^\gamma_{\alpha \beta}` are called the structure constants of the DLA and can be computed via the standard
projection in vector spaces (as is :math:`\mathfrak{g}`),

.. math:: f^\gamma_{\alpha \beta} = \frac{\langle h_\gamma, [h_\alpha, h_\beta]\rangle}{\langle h_\gamma, h_\gamma\rangle}.

The main difference from the usual vector spaces like :math:`\mathbb{R}^N` or :math:`\mathbb{C}^N` is that here we 
use the trace inner product between operators :math:`\langle h_\alpha, h_\beta \rangle = \text{tr}\left[h_\alpha^\dagger h_\beta \right]`

.. note:: 

    Technically, the (dynamical) Lie algebra is formed by skew-Hermitian operators :math:`\{i h_i\}.`
    We avoid this distinction here since for all practical purposes one can also look at Hermitian
    operators and explicitly add imaginary units in the exponents where appropriate.
    For more details, see the note in the "Lie algebras" section of our `Intro to (dynamical) Lie algebras for quantum practitioners <https://pennylane.ai/qml/demos/tutorial_liealgebra/#lie-algebras>`__.

:math:`\mathfrak{g}`-sim theory
-------------------------------

In Lie algebra simulation, :math:`\mathfrak{g}`-sim, we are interested in how expectation values of Lie algebra elements are transformed under unitary evolution.
We start from an initial expectation value vector of the input state :math:`\rho^0` with respect to each DLA element,

.. math:: (\vec{e}^0)_\alpha = \text{tr}\left[h_\alpha \rho^0 \right].

Graphically, we can represent this as a tensor with one leg.

.. figure:: ../_static/demonstration_assets/liesim/e.png
    :align: center
    :width: 33%

When we transform the state :math:`\rho^0` with a unitary evolution :math:`U,` we can use the cyclic property of the trace to shift the
evolution onto the DLA element,

.. math:: (\vec{e}^1)_\alpha = \text{tr}\left[ h_\alpha U \rho^0 U^\dagger \right] = \text{tr}\left[ U^\dagger h_\alpha U \rho^0 \right].

In the context of :math:`\mathfrak{g}`-sim, we assume the unitary operator to be generated by DLA elements :math:`h_\mu \in \mathfrak{g};` in particular, we have

.. math:: U = e^{-i \theta h_\mu}

with some real parameter :math:`\theta \in \mathbb{R}.`

As a consequence of the `Baker–Campbell–Hausdorff formula <https://en.wikipedia.org/wiki/Baker%E2%80%93Campbell%E2%80%93Hausdorff_formula>`__,
we know that any :math:`h_\alpha \in \mathfrak{g}` transformed under such a :math:`U` is again in :math:`\mathfrak{g}`
(because it leads to a sum of nested commutators between DLA elements, and the DLA is closed under commutation).
In fact, it is a well-known result that the resulting operator is given by the exponential of the structure constants

.. math:: e^{i \theta h_\mu} h_\alpha e^{-i \theta h_\mu} = \sum_\beta e^{-i \theta f^\mu_{\alpha \beta}} h_\beta.

This is the identity connecting the adjoint representations of a Lie group, :math:`\text{Ad}_{e^{-ih_\mu}}(x) = e^{ih_\mu} x e^{-ih_\mu},`
and the adjoint representation of the associated Lie algebra, :math:`\left(\text{ad}_{h_\mu}\right)_{\alpha \beta} = f^\mu_{\alpha \beta}.`
It can be summarized as

.. math:: \text{Ad}_{e^{-ih_\mu}} = e^{-i \text{ad}_{h_\mu}}.

To the best of our knowledge there is no universally accepted name for this identity
(see, e.g. `Adjoint representation (Wikipedia) <https://en.wikipedia.org/wiki/Adjoint_representation>`__
or `Lemma 3.14 in Introduction to Lie Groups and Lie Algebras <https://www.math.stonybrook.edu/~kirillov/mat552/liegroups.pdf>`__),
so we shall refer to it as the "adjoint identity" from here on.

With this, we can see how the initial expectation value vector is transformed under unitary evolution,

.. math:: (\vec{e}^1)_\alpha = \sum_\beta e^{-i \theta f^\mu_{\alpha \beta}} \text{tr}\left[h_\beta \rho^0 \right].

This is simply the matrix-vector product between the adjoint representation of the unitary gate and the initial expectation value vector.
For a unitary circuit composed of multiple gates,

.. math:: \mathcal{U} = \prod_j e^{-i \theta_j h_j},

this becomes the product of multiple adjoint representations of said gates,

.. math:: \tilde{U} = \prod_j e^{-i \theta_j \text{ad}_{h_j}}.

So overall, the evolution can be summarized graphically as the following.

.. figure:: ../_static/demonstration_assets/liesim/Ue.png
    :align: center
    :width: 33%

We are typically interested in expectation values of observables composed of DLA elements,
:math:`\langle \hat{O} \rangle = \sum_\alpha w_\alpha h_\alpha.`
Overall, the computation in :math:`\mathfrak{g}`-sim is a vector-matrix-vector product,

.. math:: \langle \hat{O} \rangle = \text{tr}\left[\hat{O} \mathcal{U} \rho^0 \mathcal{U}^\dagger \right] = \sum_{\alpha \beta} w_\alpha \tilde{U}_{\alpha \beta} e_\beta = \vec{w} \cdot \tilde{U} \cdot \vec{e}.

Or, graphically:

.. figure:: ../_static/demonstration_assets/liesim/wUe.png
    :align: center
    :width: 33%

The dimension of :math:`\left(\text{ad}_{h_j}\right)_{\alpha \beta} = f^j_{\alpha \beta}` is
:math:`\text{dim}(\mathfrak{g}) \times \text{dim}(\mathfrak{g}).` So while we evolve a :math:`2^n`-dimensional
complex vector in state vector simulators, we evolve a :math:`\text{dim}(\mathfrak{g})`-dimensional expectation
vector in :math:`\mathfrak{g}`-sim, which is more efficient whenever :math:`\text{dim}(\mathfrak{g}) < 2^n.` In general, it is efficient
whenever :math:`\text{dim}(\mathfrak{g}) = O\left(\text{poly}(n)\right).`

:math:`\mathfrak{g}`-sim in PennyLane
-------------------------------------

Let us put this into practice and write a differentiable :math:`\mathfrak{g}`-simulator in PennyLane.
We start with some boilerplate PennyLane imports.

"""

import pennylane as qml
qml.capture.enable()
from pennylane import X, Z, I
import numpy as np

import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

##############################################################################
#
# System DLA
# ~~~~~~~~~~
#
# As mentioned before, polynomially sized DLAs are rare, with the transverse-field Ising model (TFIM) with nearest neighbors being one of them.
# We take, for simplicity, the one-dimensional variant with open boundary conditions,
#
# .. math:: H_\text{TFIM} = \sum_{j=1}^{n-1} J X_j X_{j+1} + \sum_{i=1}^{n} h Z_j.
#
# We define its generators and compute the :func:`~pennylane.lie_closure`.

n = 10 # number of qubits.
generators = [X(i) @ X(i+1) for i in range(n-1)]
generators += [Z(i) for i in range(n)]

# work with PauliSentence instances for efficiency
generators = [op.pauli_rep for op in generators]

dla = qml.lie_closure(generators, pauli=True)
dim_g = len(dla)

##############################################################################
# We are using the :class:`~pennylane.pauli.PauliSentence` representation of the operators via the ``op.pauli_rep`` attribute for more efficient arithmetic and processing.
#
# Initial expectation value vector
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# With that, we can compute the initial expectation value vector for the :math:`\rho_0 = |0 \rangle \langle 0 |` initial state for every DLA element.
# We are doing a trick of representing the initial state as a Pauli operator, :math:`|0 \rangle \langle 0 |^{\otimes n} = \prod_{i=1}^n (I_i + Z_i)/2.`
# We take advantage of the locality of the DLA elements
# and use the analytic, normalized trace method :meth:`~.pennylane.pauli.PauliSentence.trace`, all to avoid having to go to the full Hilbert space.

# compute initial expectation value vector
e_in = np.zeros(dim_g, dtype=float)

for i, h_i in enumerate(dla):
    # initial state |0x0| = (I + Z)/2, note that trace function
    # below already normalizes by the dimension,
    # so we can ommit the explicit factor /2
    rho_in = qml.prod(*(I(i) + Z(i) for i in h_i.wires))
    rho_in = rho_in.pauli_rep

    e_in[i] = (h_i @ rho_in).trace()

e_in = jnp.array(e_in)
e_in


##############################################################################
# Observable
# ~~~~~~~~~~
#
# We can compute the expectation value of any linear combination of DLA elements. We choose the TFIM Hamiltonian itself,
#
# .. math:: \hat{O} = H_\text{TFIM} = \sum_j J X_j X_{j+1} + h Z_j.
#
# So just the generators with some coefficients. Here we choose :math:`J=h=0.5` for simplicity.
# We generate the :math:`\vec{w}` vector by setting the appropriate coefficients to ``0.5`.`

w = np.zeros(dim_g, dtype=float)
w[:len(generators)] = 0.5
w = jnp.array(w)

##############################################################################
# Forward and backward pass
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Together with the structure constants computed via :func:`~pennylane.structure_constants` we now have all ingredients to define
# the forward pass of the expectation value computation. For demonstration purposes,
# we choose a random subset of ``depth=10`` generators for gates from the DLA.

adjoint_repr = qml.structure_constants(dla)

depth = 10
gate_choice = np.random.choice(dim_g, size=depth)
gates = adjoint_repr[gate_choice]

def forward(theta):
    # simulation
    e_t = e_in
    for i in range(depth):
        e_t = expm(theta[i] * gates[i]) @ e_t

    # final expectation value
    result_g_sim = w @ e_t

    return result_g_sim.real

theta = jax.random.normal(jax.random.PRNGKey(0), shape=(10,))

gsim_forward, gsim_backward = forward(theta), jax.grad(forward)(theta)
gsim_forward, gsim_backward

##############################################################################
# As a sanity check, we compare the computation with the full state vector equivalent circuit.

H = 0.5 * qml.sum(*[op.operation() for op in generators])

@qml.qnode(qml.device("default.qubit"), interface="jax")
def qnode(theta):
    for i, mu in enumerate(gate_choice):
        qml.exp(-1j * theta[i] * dla[mu].operation())
    return qml.expval(H)

statevec_forward, statevec_backward = qnode(theta), jax.grad(qnode)(theta)
statevec_forward, statevec_backward


##############################################################################
# We see that both simulations yield the same results, while full state vector simulation is done with a
# :math:`2^n = 1024` dimensional state vector, and :math:`\mathfrak{g}`-sim with a :math:`\text{dim}(g) = 2n (2n-1)/2 = 190` dimensional
# expectation value vector.

print(
    qml.math.allclose(statevec_forward, gsim_forward), 
    qml.math.allclose(statevec_backward, gsim_backward),
)

##############################################################################
# Beyond 6 qubits, :math:`\mathfrak{g}`-sim is more efficient in simulating circuits generated by the TFIM Hamiltonian.
#

import matplotlib.pyplot as plt
ns = np.arange(2, 17)

plt.plot(ns, 2*ns*(2*ns-1)/2, "x-", label="dim(g)")
plt.plot(ns, 2**ns, ".-", label="2^n")
plt.yscale("log")
plt.legend()
plt.xlabel("n qubits")
plt.show()

##############################################################################
#
# VQE
# ~~~
# 
# Let us do a quick run of the :doc:`variational quantum eigensolver (VQE) <demos/tutorial_vqe/>` on the system at hand.
#
# First, we define our optimization loop in jax. Consider this boilerplate code and see our demo
# on :doc:`how to optimize a QML model using JAX and Optax <demos/tutorial_How_to_optimize_QML_model_using_JAX_and_Optax>`
# for details.

import optax
from datetime import datetime

def run_opt(value_and_grad, theta, n_epochs=100, lr=0.1, b1=0.9, b2=0.999, E_exact=0., verbose=True):

    optimizer = optax.adam(learning_rate=lr, b1=b1, b2=b2)
    opt_state = optimizer.init(theta)

    energy = np.zeros(n_epochs)
    gradients = []
    thetas = []

    @jax.jit
    def step(opt_state, theta):
        val, grad_circuit = value_and_grad(theta)
        updates, opt_state = optimizer.update(grad_circuit, opt_state)
        theta = optax.apply_updates(theta, updates)

        return opt_state, theta, val


    t0 = datetime.now()

    ## Optimization loop
    for n in range(n_epochs):
        opt_state, theta, val = step(opt_state, theta)

        energy[n] = val
        thetas.append(theta)
    t1 = datetime.now()
    if verbose:
        print(f"final loss: {val - E_exact}; min loss: {np.min(energy) - E_exact}; after {t1 - t0}")
    
    return thetas, energy, gradients

##############################################################################
# We can use the Hamiltonian variational ansatz as a natural parametrization of an ansatz circuit to obtain
# the ground-state energy.
#
# In particular, we use the full Hamiltonian generator with a trainable parameter for each term,
#
# .. math:: \prod_{\ell=1}^{10} e^{-i \sum_j \theta^X_j X_j X_{j+1} + \theta^Z_j Z_j},
#
# and repeat that over ``depth=10`` layers.

# Pick the adjoint repr of only the Hamiltonian generators
ham_terms = adjoint_repr[:len(generators)]

def forward(theta):
    # simulation
    e_t = jnp.array(e_in)

    for i in range(depth):
        e_t = expm(jnp.einsum("j,jkl->kl", theta[i], ham_terms)) @ e_t

    # final expectation values
    result_g_sim = w @ e_t

    return result_g_sim.real

##############################################################################
# Now we can run the optimization to find the ground-state energy.

theta = jax.random.normal(jax.random.PRNGKey(0), shape=(depth, len(generators),))

value_and_grad = jax.jit(jax.value_and_grad(forward))

value_and_grad(theta) # jit-compile first

E_exact = H.eigvals().min()

_, energies, _ = run_opt(value_and_grad, theta, E_exact=E_exact, verbose=True)

import matplotlib.pyplot as plt
plt.plot(energies-E_exact)
plt.yscale("log")
plt.ylabel("$E - E_{exact}$")
plt.xlabel("epochs")
plt.show()

##############################################################################
# We see good convergence to the true ground-state energy after ``100`` epochs.

##############################################################################
# 
# Conclusion
# ----------
#
# We learned about the conceptually intriguing connection between unitary evolution and the adjoint representation of the system DLA via the adjoint identity,
# and saw how this connection can be used for classical simulation. In particular, for specific systems like the TFIM we can efficiently simulate circuit 
# expectation values.
#
# In case you are more curious about Lie algebraic simulation techniques, check out the follow-up demo on :doc:`(g+P)-sim <demos/tutorial_liesim_extension>`
# an extension of :math:`\mathfrak{g}`-sim.



##############################################################################
# 
# References
# ----------
#
# .. [#Kottmann]
#
#     Korbinian Kottmann
#     "Introducing (Dynamical) Lie Algebras for quantum practitioners"
#     `PennyLane Demos <https://pennylane.ai/qml/demos/tutorial_liealgebra/>`__, 2024.
#
# .. [#Fontana]
#
#     Enrico Fontana, Dylan Herman, Shouvanik Chakrabarti, Niraj Kumar, Romina Yalovetzky, Jamie Heredge, Shree Hari Sureshbabu, Marco Pistoia
#     "The Adjoint Is All You Need: Characterizing Barren Plateaus in Quantum Ansätze"
#     `arXiv:2309.07902 <https://arxiv.org/abs/2309.07902>`__, 2023.
#
# .. [#Ragone]
#
#     Michael Ragone, Bojko N. Bakalov, Frédéric Sauvage, Alexander F. Kemper, Carlos Ortiz Marrero, Martin Larocca, M. Cerezo
#     "A Unified Theory of Barren Plateaus for Deep Parametrized Quantum Circuits"
#     `arXiv:2309.09342 <https://arxiv.org/abs/2309.09342>`__, 2023.
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
# .. [#Cerezo]
#
#     M. Cerezo, Martin Larocca, Diego García-Martín, N. L. Diaz, Paolo Braccia, Enrico Fontana, Manuel S. Rudolph, Pablo Bermejo, Aroosa Ijaz, Supanut Thanasilp, Eric R. Anschuetz, Zoë Holmes
#     "Does provable absence of barren plateaus imply classical simulability? Or, why we need to rethink variational quantum computing"
#     `arXiv:2312.09121 <https://arxiv.org/abs/2312.09121>`__, 2023.
#
# .. [#Wiersema]
#
#     Roeland Wiersema, Efekan Kökcü, Alexander F. Kemper, Bojko N. Bakalov
#     "Classification of dynamical Lie algebras for translation-invariant 2-local spin systems in one dimension"
#     `arXiv:2309.05690 <https://arxiv.org/abs/2309.05690>`__, 2023.
#
# .. [#Mazzola]
#
#     Guglielmo Mazzola
#     "Quantum computing for chemistry and physics applications from a Monte Carlo perspective"
#     `arXiv:2308.07964 <https://arxiv.org/abs/2308.07964>`__, 2023.
#
# .. [#Park]
#
#     Chae-Yeun Park, Minhyeok Kang, Joonsuk Huh
#     "Hardware-efficient ansatz without barren plateaus in any depth"
#     `arXiv:2403.04844 <https://arxiv.org/abs/2403.04844>`__, 2024.
#
