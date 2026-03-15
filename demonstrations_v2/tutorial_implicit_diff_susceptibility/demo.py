r"""
.. _implicit_diff_susceptibility:

Implicit differentiation of variational quantum algorithms
==========================================================

.. meta::
   :property="og:description": Implicitly differentiating the solution of a VQA in PennyLane.
   :property="og:image": https://pennylane.ai/qml/_static/demonstration_assets/descartes.png

.. related::
    tutorial_backprop Quantum gradients with backpropagation
    tutorial_jax_transformations Using JAX with PennyLane


In 1638, René Descartes, intrigued by (then amateur) Pierre de Fermat's method
of computing tangents, challenged Fermat to find the tangent to
a complicated curve — now called the folium of Descartes:

.. math::

    x^3 + y^3 = 3axy.

.. figure:: ../_static/demonstration_assets/implicit_diff/descartes.png
   :scale: 65%
   :alt: Representation of the folium of Descartes
   :align: center

With its cubic terms, this curve represents an implicit equation which cannot be
written as a simple expression :math:`y = f(x).` Therefore the task of calculating
the tangent function seemed formidable for the method Descartes had then, 
except at the vertex. Fermat successfully provided the tangents at not just the
vertex but at any other point on the curve, baffling Descartes and legitimizing
the intellectual superiority of Fermat. The technique used by Fermat was
*implicit differentiation* [#Paradis2004]_. In the above equation, we can begin
by take derivatives on both sides and re-arrange the terms to obtain :math:`dy/dx.`

.. math::

    \frac{dy}{dx} = -\frac{x^2 - ay}{y^2 - ax}.

Implicit differentiation can be used to compute gradients of such functions that
cannot be written down explicitly using simple elementary operations. It is a
basic technique of calculus that has recently found many applications in machine
learning — from hyperparameter optimization to the training of neural ordinary
differential equations (ODEs), and it has even led to the definition of a whole new class of architectures,
called Deep Equilibrium Models (DEQs) [#implicitlayers]_.

Introduction
------------

The idea of implicit differentiation can be applied in quantum physics to extend
the power of automatic differentiation to situations where we are not able to
explicitly write down the solution to a problem. As a concrete example, consider
a variational quantum algorithm (VQA) that computes the ground-state solution of
a parameterized Hamiltonian :math:`H(a)` using a variational ansatz
:math:`|\psi_{z}\rangle,` where :math:`z` are the variational parameters. This leads to the solution

.. math::

    z^{*}(a) = \arg\,\min_{z} \langle \psi_{z}|H(a)|\psi_z\rangle.

As we change :math:`H(a),` the solution also changes, but we do not obtain an
explicit function for :math:`z^{*}(a).` If we are interested in the properties of the
solution state, we could measure the expectation values of some operator
:math:`A` as

.. math::

    \langle A \rangle (a) = \langle \psi_{z^{*}(a)}| A | \psi_{z^{*}(a)}\rangle.

With a VQA, we can find a solution to the optimization for a fixed :math:`H(a).`
However, just like with the folium of Descartes, we do not have an explicit solution,
so the gradient :math:`\partial_a \langle A \rangle (a)` is not easy to compute.
The solution is only implicitly defined.

Automatic differentiation techniques that construct an explicit computational
graph and differentiate through it by applying the chain rule for gradient
computation cannot be applied here easily. A brute-force application of automatic
differentiation that finds :math:`z^{*}(a)` throughout the full optimization
would require us to keep track of all intermediate variables and steps in the optimization
and differentiate through them. This could quickly become computationally
expensive and memory-intensive for quantum algorithms. Implicit differentiation
provides an alternative way to efficiently compute such a gradient. 

Similarly, there exist various other
interesting quantities that can be written as gradients of a ground-state solution, 
e.g., nuclear forces in quantum chemistry, permanent electric dipolestatic
polarizability, the static hyperpolarizabilities of various orders,
fidelity susceptibilities, and geometric tensors. All such
quantities could possibly be computed using implicit differentiation on quantum
devices. In our recent work we present a unified way to implement such
computations and other applications of implicit differentiation through
variational quantum algorithms [#Ahmed2022]_.

In this demo we show how implicit gradients can be computed using a variational
algorithm written in *PennyLane* and powered by a modular implicit differentiation
implementation provided by the tool *JAXOpt* [#Blondel2021]_. We compute the generalized
susceptibility for a spin system by using a variational ansatz to obtain a
ground-state and implicitly differentiating through it. In order to compare
the implicit solution, we find the exact ground-state through eigendecomposition
and determine gradients using automatic differentiation. Even though
eigendecomposition may become unfeasible for larger systems, for a small number
of spins, it suffices for a comparison with our implicit differentiation approach.

Implicit Differentiation
------------------------

We consider the differentiation of a solution of the root-finding problem, defined by

.. math::

    f(z, a) = 0.

A function :math:`z^{*}(a)` that satisfies :math:`f(z^{*}(a), a) = 0` gives a
solution map for fixed values of :math:`a.` An explicit analytical solution
is, however, difficult to obtain in general. This means that the direct differentiation of
:math:`\partial_a z^{*}(a)` is not always possible. Despite that, some iterative
algorithms may be able to compute the solution by starting from an initial set of values
for :math:`z,` e.g., using a fixed-point solver. The optimality condition 
:math:`f(z^{*}(a), a) = 0` tells the solver when a solution is found.

Implicit differentiation can be used to compute :math:`\partial_a z^{*}(a)`
more efficiently than brute-force automatic differentiation, using only the
solution :math:`z^{*}(a)` and partial derivatives at the solution point. We do
not have to care about how the solution is obtained and, therefore, do not need
to differentiate through the solution-finding algorithm [#Blondel2021]_.

The `Implicit function theorem <https://en.wikipedia.org/wiki/Implicit_function_theorem>`__
is a statement about how the set of zeros of a system of equations is locally
given by the graph of a function under some conditions. It can be extended to
the complex domain and we state the theorem (informally) below [#Chang2003]_.

.. topic:: Implicit function theorem (IFT) (informal)

    If :math:`f(z, a)` is some analytic function where in a local neighbourhood
    around :math:`(z_0, a_0)` we have :math:`f(z_0, a_0) = 0,` there exists an
    analytic solution :math:`z^{*}(a)` that satisfies :math:`f(z^{*}(a), a) = 0.`


.. figure:: ../_static/demonstration_assets/implicit_diff/implicit_diff.png
   :scale: 65%
   :alt: Graph of the implicit function with its solution.
   :align: center

In the figure above we can see solutions to the optimality condition 
:math:`f(z, a) = 0 ` (red stars), which defines a curve :math:`z^{*}(a).` 
According to the IFT, the solution function is analytic, which means it can be
differentiated at the solution points by simply differentiating
the above equation with respect to :math:`a,` as

.. math::
    
    \partial_a f(z_0, a_0) + \partial_{z} f(z_0, a_0) \partial_{a} z^{*}(a) = 0,

which leads to

.. math::

    \partial_{a} z^{*}(a) = - (\partial_{z} f(z_0, a_0) )^{-1}\partial_a f(z_0, a_0).


This shows that implicit differentiation can be used in situations where we can
phrase our optimization problem in terms of an optimality condition 
or a fixed point equation that can be solved. In case of optimization tasks,
such an optimality condition would be that, at the minima, the gradient of the
cost function is zero — i.e., 

.. math::

    f(z, a) = \partial_z g(z, a) = 0.

Then, as long as we have the solution, :math:`z^{*}(a),` and the partial derivatives at the solution (in this case the Hessian of the
cost function :math:`g(z, a)`), :math:`(\partial_a f, \partial_z f),` we can compute implicit gradients. Note that,
for a multivariate function, the inversion
:math:`(\partial_{z} f(z_0, a_0) )^{-1}` needs to be defined and easy to
compute. It is possible to approximate this inversion in a clever way by constructing
a linear problem that can be solved approximately [#Blondel2021]_, [#implicitlayers]_.


Implicit differentiation through a variational quantum algorithm
----------------------------------------------------------------

.. figure:: ../_static/demonstration_assets/implicit_diff/VQA.png
   :scale: 65%
   :alt: Application of implicit differentiation in variational quantum algorithm.
   :align: center

We now discuss how the idea of implicit differentiation can be applied to
variational quantum algorithms. Let us take a parameterized Hamiltonian
:math:`H(a),` where :math:`a` is a parameter that can be continuously varied.
If :math:`|\psi_{z}\rangle` is a variational solution to the ground state of :math:`H(a),`
then we can find a :math:`z^*(a)` that minimizes the ground state energy, i.e.,

.. math::
    
    z^*(a) = \arg\, \min_{z} \langle \psi_{z}| H(a) | \psi_{z}\rangle = \arg \min_{z} E(z, a),


where :math:`E(z, a)` is the energy function. We consider the following
Hamiltonian

.. math::
    
    H(a) = -J \sum_{i} \sigma^{z}_i \sigma^{z}_{i+1} - \gamma \sum_{i} \sigma^{x}_i + \delta \sum_{i} \sigma^z_i - a A,

where :math:`J` is the interaction, :math:`\sigma^{x}` and :math:`\sigma^{z}` 
are the spin-:math:`\frac{1}{2}` operators and :math:`\gamma` is the magnetic
field strength (which is taken to be the same for all spins).
The term :math:`A = \frac{1}{N}\sum_i \sigma^{z}_i`
is the magnetization and a small non-zero magnetization :math:`\delta` is added
for numerical stability. We have assumed a circular chain such that in the
interaction term the last spin (:math:`i = N-1`) interacts with the first
(:math:`i=0`). 

Now we could find the ground state of this Hamiltonian by simply taking the 
eigendecomposition and applying automatic differentiation through the 
eigendecomposition to compute gradients. We will compare this exact computation
for a small system to the gradients given by implicit differentiation through a
variationally obtained solution.

We define the following optimality condition at the solution point:

.. math::
    
    f(z, a) = \partial_z E(z, a) = 0.

In addition, if the conditions of the implicit function theorem
are also satisfied, i.e., :math:`f` is continuously differentiable
with a non-singular Jacobian at the solution, then we can apply the chain rule
and determine the implicit gradients easily.

At this stage, any other complicated function that depends on the variational ground state
can be easily differentiated using automatic differentiation by plugging in the
value of :math:`partial_a z^{*}(a)` where it is required. The 
expectation value of the operator :math:`A` for the ground state
is

.. math::
    
    \langle A\rangle = \langle \psi_{z^*}| A| \psi_{z^*}\rangle.

In the case where :math:`A` is just the energy, i.e., :math:`A = H(a),` the
Hellmann–Feynman theorem allows us to easily compute the gradient. However, for
a general operator we need the gradients :math:`\partial_a z^{*}(a),` which means that implicit differentiation is a very elegant way to go beyond the
Hellmann–Feynman theorem for arbitrary expectation values.

Let us now dive into the code and implementation.

*Talk is cheap. Show me the code.* - Linus Torvalds
"""

##############################################################################
# Implicit differentiation of ground states in PennyLane
# ------------------------------------------------------
#

from functools import reduce

from operator import add

import jax
from jax import jit
import jax.numpy as jnp

import pennylane as qml
qml.capture.enable()
import numpy as np

import jaxopt

import matplotlib.pyplot as plt

jax.config.update("jax_platform_name", "cpu")

# Use double precision numbers
jax.config.update("jax_enable_x64", True)

##############################################################################
# Defining the Hamiltonian and measurement operator
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We define the Hamiltonian by building the non-parametric part separately and
# adding the parametric part to it as a separate term for efficiency. Note that, for
# the example of generalized susceptibility, we are measuring expectation values
# of the operator :math:`A` that also defines the parametric part of the
# Hamiltonian. However, this is not necessary and we could compute gradients for
# any other operator using implicit differentiation, as we have access to the
# gradients :math:`\partial_a z^{*}(a).`

N = 4
J = 1.0
gamma = 1.0


def build_H0(N, J, gamma):
    """Builds the non-parametric part of the Hamiltonian of a spin system.

    Args:
        N (int): Number of qubits/spins.
        J (float): Interaction strength.
        gamma (float): Interaction strength.

    Returns:
        qml.Hamiltonian: The Hamiltonian of the system.
    """
    H = qml.Hamiltonian([], [])

    for i in range(N - 1):
        H += -J * qml.PauliZ(i) @ qml.PauliZ(i + 1)

    H += -J * qml.PauliZ(N - 1) @ qml.PauliZ(0)

    # Transverse
    for i in range(N):
        H += -gamma * qml.PauliX(i)

    # Small magnetization for numerical stability
    for i in range(N):
        H += -1e-1 * qml.PauliZ(i)

    return H


H0 = build_H0(N, J, gamma)
H0_matrix = qml.matrix(H0)
A = reduce(add, ((1 / N) * qml.PauliZ(i) for i in range(N)))
A_matrix = qml.matrix(A)

###############################################################################
# Computing the exact ground state through eigendecomposition
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We now define a function that computes the exact ground state using
# eigendecomposition. Ideally, we would like to take gradients of this function.
# It is possible to simply apply automatic differentiation through this exact
# ground-state computation. JAX has an implementation of differentiation
# through eigendecomposition.
#
# Note that we have some points in this plot that are ``nan``, where the gradient
# computation through the eigendecomposition does not work. We will see later that
# the computation through the VQA is more stable.


@jit
def ground_state_solution_map_exact(a: float) -> jnp.array:
    """The ground state solution map that we want to differentiate
    through, computed from an eigendecomposition.

    Args:
        a (float): The parameter in the Hamiltonian, H(a).

    Returns:
        jnp.array: The ground state solution for the H(a).
    """
    H = H0_matrix + a * A_matrix
    eval, eigenstates = jnp.linalg.eigh(H)
    z_star = eigenstates[:, 0]
    return z_star


a = jnp.array(np.random.uniform(0, 1.0))
z_star_exact = ground_state_solution_map_exact(a)

#################################################################
# Susceptibility computation through the ground state solution map
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Let us now compute the susceptibility function by taking gradients of the
# expectation value of our operator :math:`A` w.r.t `a.` We can use `jax.vmap`
# to vectorize the computation over different values of `a`.


@jit
def expval_A_exact(a):
    """Expectation value of ``A`` as a function of ``a`` where we use the
    ``ground_state_solution_map_exact`` function to find the ground state.

    Args:
        a (float): The parameter defining the Hamiltonian, H(a).

    Returns:
        float: The expectation value of A calculated using the variational state
               that should be the ground state of H(a).
    """
    z_star = ground_state_solution_map_exact(a)
    eval = jnp.conj(z_star.T) @ A_matrix @ z_star
    return eval.real


# the susceptibility is the gradient of the expectation value
_susceptibility_exact = jax.grad(expval_A_exact)
susceptibility_exact = jax.vmap(_susceptibility_exact)

alist = jnp.linspace(0, 3, 1000)
susvals_exact = susceptibility_exact(alist)

plt.plot(alist, susvals_exact)
plt.xlabel("a")
plt.ylabel(r"$\partial_{a}\langle A \rangle$")
plt.show()

###############################################################################
# Computing susceptibility through implicit differentiation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We use PennyLane to find a variational ground state for the Hamiltonian
# :math:`H(a)` and compute implicit gradients through the variational
# optimization procedure. We use the ``jaxopt`` library which contains an
# implementation of gradient descent that automatically comes with implicit
# differentiation capabilities. We are going to use that to obtain
# susceptibility by taking gradients through the ground-state minimization.
#
# Defining the variational state
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# In PennyLane, we can implement a variational state in different ways, by
# defining a quantum circuit. There are also useful template circuits available, such as
# :class:`~pennylane.SimplifiedTwoDesign`, which implements the :doc:`two-design ansatz <demos/tutorial_unitary_designs>`.
# The ansatz consists of layers of Pauli-Y rotations with
# controlled-Z gates. In each layer there are ``N - 1`` parameters for the Pauli-Y gates.
# Therefore, the ansatz is efficient as long as we have enough layers for it
# so that is expressive enough to represent the ground-state.
#
# We set ``n_layers = 5``, but you can redo this example with fewer layers to see
# how a less expressive ansatz leads to error in the susceptibility computation.
#
# .. note::
#
#   The setting ``shots=None`` makes for the computation of gradients using reverse-mode
#   autodifferentiation (backpropagation). It allows us to just-in-time (JIT)
#   compile the functions that compute expectation values and gradients.
#   In a real device we would use a finite number of shots and the gradients would be computed
#   using the parameter-shift rule. However, this may be slower.

variational_ansatz = qml.SimplifiedTwoDesign
n_layers = 5
weights_shape = variational_ansatz.shape(n_layers, N)

dev = qml.device("default.qubit", wires=N)


@jax.jit
@qml.qnode(dev, interface="jax")
def energy(z, a):
    """Computes the energy for a Hamiltonian H(a) using a measurement on the
    variational state U(z)|0> with U(z) being any circuit ansatz.

    Args:
        z (jnp.array): The variational parameters for the ansatz (circuit)
        a (jnp.array): The Hamiltonian parameters.

    Returns:
        float: The expectation value (energy).
    """
    variational_ansatz(*z, wires=range(N))

    return qml.expval(H0 + a * A)


z_init = [jnp.array(2 * np.pi * np.random.random(s)) for s in weights_shape]
a = jnp.array([0.5])
print("Energy", energy(z_init, a))

###############################################################################
# Computing ground states using a variational quantum algorithm (VQA)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We construct a loss function that defines a ground-state minimization
# task. We are looking for variational parameters ``z`` that minimize the energy
# function. Once we find a set of parameters ``z``, we wish to compute the
# gradient of any function of the ground state w.r.t. ``a``.
#
# For the implicit differentiation we will use the tool ``jaxopt``, which implements
# modular implicit differentiation for various cases; e.g., for fixed-point
# functions or optimization. We can directly use ``jaxopt`` to optimize our loss
# function and then compute implicit gradients through it.
# It all works due to :doc:`PennyLane's integration with JAX <demos/tutorial_jax_transformations>`.
#
# The implicit differentiation formulas can even be `implemented manually with JAX <https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html#implicit-function-differentiation-of-iterative-implementations>`__.
# These formulas are implemented in a modular way, using the
# ``jaxopt.GradientDescent`` optimizer with ``implicit_diff=True``.
# We use the seamless integration between PennyLane, JAX
# and JAXOpt to compute the susceptibility.
#
# Since everything is written in JAX, simply calling the
# ``jax.grad`` function works as ``jaxopt`` computes the implicit gradients and
# plugs it any computation used by ``jax.grad``. We can also just-in-time (JIT)
# compile all functions although the compilation may take some time as the
# number of spins or variational ansatz becomes more complicated. Once compiled,
# all computes run very fast for any parameters.


def ground_state_solution_map_variational(a, z_init):
    """The ground state solution map that we want to differentiate
    through.

    Args:
        a (float): The parameter in the Hamiltonian, H(a).
        z_init [jnp.array(jnp.float)]: The initial guess for the variational
                                       parameters.

    Returns:
        z_star (jnp.array [jnp.float]): The parameters that define the
                                        ground-state solution.
    """

    @jax.jit
    def loss(z, a):
        """Loss function for the ground-state minimization with regularization.

        Args:
            z (jnp.array): The variational parameters for the ansatz (circuit)
            a (jnp.array): The Hamiltonian parameters.

        Returns:
            float: The loss value (energy + regularization)
        """
        return energy(z, a) + 0.001 * jnp.sum(jnp.abs(z[0])) + 0.001 * jnp.sum(jnp.abs(z[1]))

    gd = jaxopt.GradientDescent(
        fun=loss,
        stepsize=1e-2,
        acceleration=True,
        maxiter=1000,
        implicit_diff=True,
        tol=1e-15,
    )
    z_star = gd.run(z_init, a=a).params
    return z_star


a = jnp.array(np.random.uniform(0, 1.0))  # A random ``a``
z_star_variational = ground_state_solution_map_variational(a, z_init)

###############################################################################
# Computing gradients through the VQA simply by calling ``jax.grad``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We can compute the susceptibility values by simply using ``jax.grad``. After the
# first call, the function is compiled and subsequent calls become
# much faster.


@jax.jit
@qml.qnode(dev, interface="jax")
def expval_A_variational(z: float) -> float:
    """Expectation value of $A$ as a function of $a$ where we use the
    a variational ground state solution map.

    Args:
        a (float): The parameter in the Hamiltonian, H(a).

    Returns:
        float: The expectation value of M on the ground state of H(a)
    """
    variational_ansatz(*z, wires=range(N))
    return qml.expval(A)


@jax.jit
def groundstate_expval_variational(a, z_init) -> float:
    """Computes ground state and calculates the expectation value of the operator M.

    Args:
        a (float): The parameter in the Hamiltonian, H(a).
        z_init [jnp.array(jnp.float)]: The initial guess for the variational parameters.
        H0 (qml.Hamiltonian): The static part of the Hamiltonian
    """
    z_star = ground_state_solution_map_variational(a, z_init)
    return expval_A_variational(z_star)


susceptibility_variational = jax.jit(jax.grad(groundstate_expval_variational, argnums=0))
z_init = [jnp.array(2 * np.pi * np.random.random(s)) for s in weights_shape]
print("Susceptibility", susceptibility_variational(alist[0], z_init))

susvals_variational = []

for i in range(len(alist)):
    susvals_variational.append(susceptibility_variational(alist[i], z_init))

plt.plot(alist, susvals_variational, label="Implicit diff through VQA")
plt.plot(alist, susvals_exact, "--", c="k", label="Automatic diff through eigendecomposition")
plt.xlabel("a")
plt.ylabel(r"$\partial_{a}\langle A \rangle$")
plt.legend()
plt.show()

##############################################################################
# PennyLane version and details

print(qml.about())

##############################################################################
# Conclusion
# ----------
# We have shown how a combination of JAX, PennyLane and JAXOpt can be used to
# compute implicit gradients through a VQA. The ability to compute such
# gradients opens up new possibilities, e.g., the design of a Hamiltonian such that
# its ground-state has certain properties. It is also possible to perhaps look
# at this inverse-design of the Hamiltonian as a control problem. Implicit
# differentiation in the classical setting allows defining a new type of
# neural network layer — implicit layers such as neural ODEs. In a similar
# way, we hope this demo the inspires creation of new architectures for quantum
# neural networks, perhaps a quantum version of neural ODEs or quantum implicit
# layers.
#
# In future works, it would be important to assess the cost of running implicit
# differentiation through an actual quantum computer and determine the quality
# of such gradients as a function of noise as explored in a related recent work
# [#Matteo2021]_.
#
# References
# ----------
#
# .. [#Paradis2004]
#
#     Jaume Paradís, Josep Pla & Pelegrí Viader
#     "Fermat and the Quadrature of the Folium of Descartes"
#     The American Mathematical Monthly, 111:3, 216-229
#     `10.1080/00029890.2004.11920067 <https://doi.org/10.1080/00029890.2004.11920067>`__, 2004.
#
# .. [#Ahmed2022]
#
#     Shahnawaz Ahmed, Nathan Killoran, Juan Felipe Carrasquilla Álvarez
#     "Implicit differentiation of variational quantum algorithms
#     `arXiv:2211.13765 <https://arxiv.org/abs/2211.13765>`__, 2022.
#
# .. [#Blondel2021]
#
#    Mathieu Blondel, Quentin Berthet, Marco Cuturi, Roy Frostig, Stephan Hoyer, Felipe Llinares-López, Fabian Pedregosa, Jean-Philippe Vert
#    "Efficient and modular implicit differentiation"
#    `arXiv:2105.15183 <https://arxiv.org/abs/2105.15183>`__, 2021.
#
# .. [#implicitlayers]
#
#     Zico Kolter, David Duvenaud, Matt Johnson.
#     "Deep Implicit Layers - Neural ODEs, Deep Equilibirum Models, and Beyond"
#     `http://implicit-layers-tutorial.org <http://implicit-layers-tutorial.org>`__, 2021.
#
# .. [#Matteo2021]
#
#    Olivia Di Matteo, R. M. Woloshyn
#    "Quantum computing fidelity susceptibility using automatic differentiation"
#    `arXiv:2207.06526 <https://arxiv.org/abs/2207.06526>`__, 2022.
#
# .. [#Chang2003]
#
#    Chang, Hung-Chieh, Wei He, and Nagabhushana Prabhu.
#    "The analytic domain in the implicit function theorem."
#    `JIPAM. J. Inequal. Pure Appl. Math 4.1 <http://emis.icm.edu.pl/journals/JIPAM/v4n1/061_02_www.pdf>`__,  (2003).
#
