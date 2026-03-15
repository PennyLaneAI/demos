r"""Fixed depth Hamiltonian simulation via Cartan decomposition
===============================================================

We introduce the powerful Lie-theoretic decomposition technique for Hamiltonians, :math:`H = K h_0 K^\dagger,`
that lets you perform time-evolution by arbitrary times with fixed depth, :math:`e^{-i t H} = K e^{-i t h_0} K^\dagger.`
In particular, we follow the approach in [#Kökcü]_ that directly provides us with a (fixed-depth) circuit
decomposition of the unitaries :math:`K` and :math:`e^{-i t h_0}.`

Sounds too good to be true? There are of course caveats as this fast-forwards the Hamiltonian.
In this algebraic setting, the crucial property is the size of the relevant :doc:`dynamical Lie algebra <demos/tutorial_liealgebra>` (DLA).
Most systems yield an exponentially large DLA, making them hard to manage in practice (i.e., fast-forwarding is not possible).
Yet, this is still an extremely powerful mathematical result integral for quantum compilation,
circuit optimization, and Hamiltonian simulation.

.. figure:: ../_static/demo_thumbnails/opengraph_demo_thumbnails/OGthumbnail_fixed_depth_hamiltonian_simulation_via_cartan_decomposition.png
    :align: center
    :width: 70%
    :target: javascript:void(0)

Introduction
------------

The :doc:`KAK decomposition <demos/tutorial_kak_decomposition>` is an important result from Lie theory that states that any Lie group element :math:`U` can be decomposed
as :math:`U = K_1 A K_2,` where :math:`K_{1, 2}` and :math:`A` are elements of two special sub-groups
:math:`\mathcal{K}` and :math:`\mathcal{A},` respectively. In special cases, the decomposition simplifies to :math:`U = K A K^\dagger.`

You can think of this KAK decomposition as a generalization of
the singular value decomposition to Lie groups. For that, recall that the singular value decomposition states that any
matrix :math:`M \in \mathbb{C}^{m \times n}` can be decomposed as :math:`M = U \Lambda V^\dagger,` where :math:`\Lambda`
are the diagonal singular values and :math:`U \in \mathbb{C}^{m \times \mu}` and :math:`V^\dagger \in \mathbb{C}^{\mu \times n}`
are left- and right-unitary with :math:`\mu = \min(m, n).`

In the case of the KAK decomposition, :math:`\mathcal{A}` is an Abelian subgroup in which all its elements are commuting,
just as is the case for diagonal matrices.

We can use this general result from Lie theory as a powerful circuit decomposition technique.

.. note:: We recommend a basic understanding of Lie algebras, see e.g. our :doc:`introduction to (dynamical) Lie algebras for quantum practitioners <demos/tutorial_liealgebra>`.
    Otherwise, this demo should be self-contained, though for the mathematically inclined, we further recommend our :doc:`demo on the KAK decomposition <demos/tutorial_kak_decomposition>`
    that dives into the mathematical depths of the decomposition and provides more background info.

Goal: Fast-forwarding time evolutions using the KAK decomposition
-----------------------------------------------------------------

Unitary gates in quantum computing are described by the special unitary Lie group :math:`SU(2^n),` so we can use the KAK
decomposition to factorize quantum gates into :math:`U = K_1 A K_2.` While the mathematical statement is rather straightforward,
actually finding this decomposition is not. We are going to follow the recipe prescribed in 
`Fixed Depth Hamiltonian Simulation via Cartan Decomposition <https://arxiv.org/abs/2104.00728>`__ [#Kökcü]_, 
which tackles this decomposition on the level of the associated Lie algebra via Cartan decomposition.

In particular, we are going to consider the problem of time-evolving a Hermitian operator :math:`H` that generates the time-evolution unitary :math:`U = e^{-i t H}.`
We are going to perform a special case of KAK decomposition, a "KhK decomposition" if you will, on the algebraic level in terms of

.. math:: H = K h_0 K^\dagger.

This then induces the KAK decomposition on the group level as

.. math:: e^{-i t H} = K e^{-i t h_0} K^\dagger.

Let us walk through an explicit example, doing theory and code side-by-side.

For that, we are going to use the generators of the `Heisenberg model <https://en.wikipedia.org/wiki/Quantum_Heisenberg_model>`__ Hamiltonian for :math:`n=4` qubits on a one-dimensional chain,

.. math:: \{X_i X_{i+1}, Y_i Y_{i+1}, Z_i Z_{i+1}\}_{i=0}^{2}.

The foundation of a KAK decomposition is a Cartan decomposition of the associated Lie algebra :math:`\mathfrak{g}.`
Now, let us first import some libraries that we are going to use later and construct :math:`\mathfrak{g}.`


"""
from datetime import datetime
import matplotlib.pyplot as plt

import numpy as np
import pennylane as qml
qml.capture.enable()
from pennylane import X, Y, Z
from pennylane.liealg import even_odd_involution, cartan_decomp, horizontal_cartan_subalgebra

import jax
import jax.numpy as jnp
import optax
jax.config.update("jax_enable_x64", True)

n_wires = 4
gens = [X(i) @ X(i+1) for i in range(n_wires-1)]
gens += [Y(i) @ Y(i+1) for i in range(n_wires-1)]
gens += [Z(i) @ Z(i+1) for i in range(n_wires-1)]

H = qml.sum(*gens)

g = qml.lie_closure(gens)
g = [op.pauli_rep for op in g]

##############################################################################
#
# Cartan decomposition
# --------------------
#
# A Cartan decomposition is a bipartition :math:`\mathfrak{g} = \mathfrak{k} \oplus \mathfrak{m}` into a vertical subspace
# :math:`\mathfrak{k}` and an orthogonal horizontal subspace :math:`\mathfrak{m}.` In practice, it can be induced by an
# involution function :math:`\Theta` that fulfills :math:`\Theta(\Theta(g)) = g \ \forall g \in \mathfrak{g}.` Different
# involutions lead to different types of Cartan decompositions, which have been fully classified by Cartan
# (see `the classification of symmetric spaces by Cartan <https://en.wikipedia.org/wiki/Symmetric_space#Classification_result>`__).
#
# .. admonition:: Notation
#     :class: note
#
#     Note that :math:`\mathfrak{k}` is the small letter k in
#     `Fraktur <https://en.wikipedia.org/wiki/Fraktur>`__ and a
#     common — not our — choice for the vertical subspace in a Cartan decomposition.
#
# One common choice of involution is the so-called even-odd involution for Pauli words,
# :math:`P = P_1 \otimes P_2 .. \otimes P_n,` where :math:`P_j \in \{I, X, Y, Z\}.`
# It essentially counts whether the number of non-identity Pauli operators in the Pauli word is even or odd.
# It is readily available in PennyLane as :func:`~.pennylane.liealg.even_odd_involution`, which 
# we already imported above.

even_odd_involution(X(0)), even_odd_involution(X(0) @ Y(3))

##############################################################################
#
# The vertical and horizontal subspaces are the two eigenspaces of the involution, corresponding to the :math:`\pm 1` eigenvalues.
# In particular, we have :math:`\Theta(\mathfrak{k}) = \mathfrak{k}` and :math:`\Theta(\mathfrak{m}) = - \mathfrak{m}.`
# So in order to perform the Cartan decomposition :math:`\mathfrak{g} = \mathfrak{k} \oplus \mathfrak{m},` we simply
# sort the operators by whether or not they yield a plus or minus sign from the involution function.
# This is possible because the operators and involution nicely align with the eigenspace decomposition.

k, m = cartan_decomp(g, even_odd_involution)
len(g), len(k), len(m)

##############################################################################
# We have successfully decomposed the 60-dimensional Lie algebra
# into a 24-dimensional vertical subspace and a 36-dimensional subspace.
#
# Note that not every bipartition of a Lie algebra constitutes a Cartan decomposition.
# For that, the subspaces need to fulfill the following three commutation relations.
#
# .. math::
#     \begin{align}
#     [\mathfrak{k}, \mathfrak{k}] \subseteq \mathfrak{k} & \text{ (subalgebra)}\\
#     [\mathfrak{k}, \mathfrak{m}] \subseteq \mathfrak{m} & \text{ (reductive property)}\\
#     [\mathfrak{m}, \mathfrak{m}] \subseteq \mathfrak{k} & \text{ (symmetric property)}
#     \end{align}
#
# In particular, :math:`\mathfrak{k}` is closed under commutation and is therefore a subalgebra, whereas :math:`\mathfrak{m}` is not.
# This also has the consequence that the associated Lie group :math:`\mathcal{K} := e^{i \mathfrak{k}}` is a subgroup
# of the associated Lie group :math:`\mathcal{G} := e^{i \mathfrak{g}}.`
#
# We mentioned earlier that we are aiming to do a special case of KAK decomposition where the second unitary :math:`K_2 = K_1^\dagger.`
# This is possible whenever the operator that we want to decompose is in the horizontal subspace :math:`\mathfrak{m}`, i.e. we want :math:`\Theta(H) = -H`.
# The chosen :math:`H` and :math:`\Theta` fulfill this property, as can be easily verified.
#

for op in H.operands:
    assert not even_odd_involution(op)

##############################################################################
#
# Cartan subalgebra
# -----------------
#
# With this we have identified the first subgroup, :math:`\mathcal{K},` of the KAK decomposition. The other subgroup
# is induced by the so-called (horizontal) Cartan subalgebra :math:`\mathfrak{h}.` This is a maximal Abelian subalgebra of :math:`\mathfrak{m}` and it is not unique.
# For the case of Pauli words, we can simply pick any element in :math:`\mathfrak{m}` and collect all other operators in :math:`\mathfrak{m}`
# that commute with it.
#
# We then obtain a further split of the vector space :math:`\mathfrak{m} = \tilde{\mathfrak{m}} \oplus \mathfrak{h},`
# where :math:`\tilde{\mathfrak{m}}` is just the remainder of :math:`\mathfrak{m}.` The function
# :func:`~.pennylane.liealg.horizontal_cartan_subalgebra` returns some additional information, which we will
# not use here.

g, k, mtilde, h, _ = horizontal_cartan_subalgebra(k, m, tol=1e-8)
len(g), len(k), len(mtilde), len(h)

##############################################################################
# We now have the Cartan decomposition :math:`\mathfrak{g} = \mathfrak{k} \oplus \tilde{\mathfrak{m}} \oplus \mathfrak{h}`
# and with that all the necessary ingredients for the KAK decomposition.
#
# Variational KhK decomposition
# -----------------------------
#
# The KAK decomposition is not constructive in the sense that it proves that there exists such a decomposition, but there is no general way of obtaining
# it. In particular, there are no linear algebra subroutines implemented in ``numpy`` or ``scipy`` that just compute it for us.
# Here, we follow the construction of [#Kökcü]_ for the special case of :math:`H` being in the horizontal space and the decomposition
# simplifying to :math:`H = K^\dagger h K`.
#
# The authors propose to find a local extremum of the cost function
#
# .. math:: f(\theta) = \langle K(\theta) v K(\theta)^\dagger, H\rangle
#
# where :math:`\langle \cdot, \cdot \rangle` is some inner product (in our case, the trace inner product :math:`\langle A, B \rangle = \text{tr}(A^\dagger B)).`
# This construction uses the operator :math:`v = \sum_j \pi^j h_j \in \mathfrak{h}`
# that is such that :math:`e^{i t v}` is dense in :math:`e^{i \mathcal{h}}.`
# The latter means that for any :math:`e^{i \mathcal{h}}` there is a :math:`t \in \mathbb{R}` such that :math:`e^{i t v}` approximates it.
# Let us construct it. To numerically avoid very large vectors, we take :math:`\pi (\text{mod} 2),` which preserves the dense property of :math:`v`.

gammas = [np.pi**i % 2 for i in range(1, len(h)+1)]

v = qml.dot(gammas, h)
v_m = qml.matrix(v, wire_order=range(n_wires))
v_m = jnp.array(v_m)


##############################################################################
#
# This procedure has the advantage that we can use an already decomposed ansatz
#
# .. math:: K(\theta) = \prod_j e^{-i \theta_j k_j}
#
# for the vertical unitary.
#
# Now we just have to define the cost function and find an extremum.
# In this case, we are going to use gradient descent to minimize the cost function to a minimum.
# We are going to use ``jax`` and ``optax`` and write some boilerplate for the optimization procedure.
#
# .. note::
#     You can check our demos on parameter optimization in JAX with
#     :doc:`Optax <demos/tutorial_How_to_optimize_QML_model_using_JAX_and_Optax>` or
#     :doc:`JAXOpt <demos/tutorial_How_to_optimize_QML_model_using_JAX_and_JAXopt>`.
#

def run_opt(
    loss,
    theta,
    n_epochs=1000,
    lr=0.05,
):
    """Boilerplate JAX optimization"""
    value_and_grad = jax.jit(jax.value_and_grad(loss))
    optimizer = optax.lbfgs(learning_rate=lr, memory_size=100)
    opt_state = optimizer.init(theta)

    energy = []
    gradients = []
    thetas = []

    @jax.jit
    def step(opt_state, theta):
        val, grad_circuit = value_and_grad(theta)
        updates, opt_state = optimizer.update(
            grad_circuit, opt_state, theta, value=val, grad=grad_circuit, value_fn=loss
        )
        theta = optax.apply_updates(theta, updates)

        return opt_state, theta, val

    t0 = datetime.now()
    ## Optimization loop
    for _ in range(n_epochs):
        opt_state, theta, val = step(opt_state, theta)

        energy.append(val)
        thetas.append(theta)
        
    t1 = datetime.now()
    print(f"final loss: {val}; min loss: {np.min(energy)}; after {t1 - t0}")

    return thetas, energy, gradients


##############################################################################
# We can now implement the cost function and find a minimum via gradient descent.

H_m = qml.matrix(H, wire_order=range(n_wires))
H_m = jnp.array(H_m)

def K(theta, k):
    for th, k_j in zip(theta, k):
        qml.exp(-1j * th * k_j.operation())

@jax.jit
def loss(theta):
    K_m = qml.matrix(K, wire_order=range(n_wires))(theta, k)
    A = K_m @ v_m @ K_m.conj().T
    return jnp.trace(A.conj().T @ H_m).real

theta0 = jax.random.normal(jax.random.PRNGKey(0), shape=(len(k),), dtype=float)

thetas, energy, _ = run_opt(loss, theta0, n_epochs=1000, lr=0.05)
plt.plot(energy - np.min(energy))
plt.xlabel("epochs")
plt.ylabel("cost")
plt.yscale("log")
plt.show()


##############################################################################
# This gives us the optimal values of the parameters :math:`\theta_\text{opt}` of :math:`K(\theta_\text{opt}) =: K_c.`

theta_opt = thetas[-1]
Kc_m = qml.matrix(K, wire_order=range(n_wires))(theta_opt, k)

##############################################################################
# The special element :math:`h_0` from the Cartan subalgebra :math:`\mathfrak{h}` is given by
# rotating the Hamiltonian by the critical :math:`K_c,`
#
# .. math:: h_0 = K_c^\dagger H K_c.

h_0_m = Kc_m.conj().T @ H_m @ Kc_m

# decompose h_0_m in terms of the basis of h
basis = [qml.matrix(op, wire_order=range(n_wires)) for op in h]
coeffs = qml.pauli.trace_inner_product(h_0_m, basis)

# ensure that decomposition is correct, i.e. h_0_m is truely an element of just h
h_0_m_recomposed = np.sum([c * op for c, op in zip(coeffs, basis)], axis=0)
print("Decomposition of h_0 is faithful: ", np.allclose(h_0_m_recomposed, h_0_m, atol=1e-10))

# sanity check that the horizontal CSA is Abelian, i.e. all its elements commute
print("All elements in h commute with each other: ", qml.liealg.check_abelian(h))


##############################################################################
#
# The fact that :math:`K_c^\dagger H K_c \in \mathfrak{h}` is crucial for this decomposition to be valid and meaningful.
# Otherwise, :math:`h_0` could be anything and we arrive back at the original problem of decomposing :math:`e^{-i t h_0}`.
# Here we know that :math:`h_0` is composed of elements of an Abelian Lie algebra :math:`\mathfrak{h}`, such that we can
# trivially decompose its unitary as
#
# .. math:: e^{-i t h_0} = e^{-i t \sum_{j=1}^{|\mathfrak{h}|} c_j h_j} = \prod_{j=1}^{|\mathfrak{h}|} e^{-i t c_j h_j}.
#
# Overall, this gives us the KhK decomposition of :math:`H,`
#
# .. math:: H = K_c h_0 K_c^\dagger.
#
# This trivially reproduces the original Hamiltonian.
#

H_re = Kc_m @ h_0_m @ Kc_m.conj().T
np.allclose(H_re, H_m)

##############################################################################
# We can now check if the Hamiltonian evolution is reproduced correctly.
#

t = 1.
U_exact = qml.exp(-1j * t * H)
U_exact_m = qml.matrix(U_exact, wire_order=range(n_wires))
h_0  = qml.dot(coeffs, h)

def U_kak(theta_opt, t):
    qml.adjoint(K)(theta_opt, k)
    qml.exp(-1j * t * h_0)
    K(theta_opt, k)

U_kak_m = qml.matrix(U_kak, wire_order=range(n_wires))(theta_opt, t)

def trace_distance(A, B):
    return 1 - np.abs(np.trace(A.conj().T @ B))/len(A)

trace_distance(U_exact_m, U_kak_m)


##############################################################################
# Indeed we find that the KAK decomposition that we found reproduces the unitary evolution operator.
# Note that this is valid for arbitrary :math:`t,` such that the Hamiltonian simulation operator has a fixed depth.
#
# .. admonition:: Choice of algebra representation
#     :class: note
#
#     In the case of Pauli words, we could make use of the well-known commutation relations of Pauli operators and treat the whole calculation semi-analytically
#     (as is done in [#Kökcü]_).
#     Here, we have chosen the full :math:`2^n` Hilbert space representation of the involved operators as this is actually faster for the chosen size :math:`n=4`.
#
#     More generally, we can use the adjoint identity introdcued in our :doc:`g-sim demo <demos/tutorial_liesim>` and perform all computations in the
#     adjoint representation of the algebra. In that case, the dimension of the algebra is the determining factor. This is useful when a low-dimensional Lie algebra
#     is embedded in a high dimensional Hilbert space. For example, the operators :math:`S^x_\text{total} = \sum_{j=1}^n X_j,`
#     :math:`S^y_\text{total} = \sum_{j=1}^n Y_j` and :math:`S^z_\text{total} = \sum_{j=1}^n Z_j` make up :math:`\mathfrak{su}(2)` of dimension :math:`3`, but are embedded
#     in a :math:`2^n`-dimensional Hilbert space.
#

##############################################################################
# Time evolutions
# ---------------
#
# We compute multiple time evolutions for different times and compare Suzuki–Trotter products with the KAK decomposition circuit.
#

ts = jnp.linspace(1., 5., 10)

Us_exact = jax.vmap(lambda t: qml.matrix(qml.exp(-1j * t * H), wire_order=range(n_wires)))(ts)

def Us_kak(t):
    return Kc_m @ jax.scipy.linalg.expm(-1j * t * h_0_m) @ Kc_m.conj().T

Us_kak = jax.vmap(Us_kak)(ts)
Us_trotter5 = jax.vmap(lambda t: qml.matrix(qml.TrotterProduct(H, time=-t, n=5, order=4), wire_order=range(n_wires)))(ts)
Us_trotter50 = jax.vmap(lambda t: qml.matrix(qml.TrotterProduct(H, time=-t, n=50, order=4), wire_order=range(n_wires)))(ts)

def compute_res(Us):
    # vectorized trace inner product
    res = jnp.abs(jnp.einsum("bij,bji->b", Us_exact.conj(), Us))
    res /= 2**n_wires
    return 1 - res

res_kak = compute_res(Us_kak)
res_trotter5 = compute_res(Us_trotter5)
res_trotter50 = compute_res(Us_trotter50)

plt.plot(ts, res_kak+1e-15, label="KAK") # displace by machine precision to see it still in plot
plt.plot(ts, res_trotter5, "x--", label="5 Trotter steps")
plt.plot(ts, res_trotter50, ".-", label="50 Trotter steps")
plt.ylabel("empirical error")
plt.xlabel("t")
plt.yscale("log")
plt.legend()
plt.show()


##############################################################################
# We see the expected behavior of Suzuki–Trotter product formulas getting worse with an increase in time
# while the KAK error is constant zero.
#
# The KAK decomposition is particularly well-suited for smaller systems as the circuit depth is equal to the
# dimension of the subspaces, in particular :math:`2 |\mathfrak{k}| + |\mathfrak{h}|.` Note, however,
# that these dimensions typically scale exponentially in the system size.
#


##############################################################################
#
# Conclusion
# ----------
#
# The KAK decomposition is a very general mathematical result with far-reaching consequences.
# While there is no canonical way of obtaining an actual decomposition in practice, we followed
# the approach of [#Kökcü]_ which uses a specifically designed loss function and variational
# optimization to find the decomposition.
# This approach has the advantage that the resulting decomposition is itself already decomposed in terms of rotation gates in the original Lie algebra,
# as opposed to other methods such as [#Chu]_ that find :math:`K` as a whole.
# We provided a flexible pipeline that lets users find KAK decompositions in PennyLane for systems with small
# DLA (:doc:`Dynamical Lie Algebras <demos/tutorial_liealgebra>`) and specifically decomposed the Heisenberg model Hamiltonian with :math:`n=4` qubits that has a DLA of dimension :math:`60` (:math:`\left(\mathfrak{s u}(2^{n-2})\right)^{\oplus 4}`).
#
# As most DLAs scale exponentially in the number of qubits, KAK decompositions are limited to small system sizes
# or special cases of systems with small DLAs.
# This is in line with the notion that fast-forwarding is generally not possible and limited to special systems.
# In particular, a KAK decomposition is ultimately always a fast-forwarding of a Hamiltonian.


##############################################################################
#
# References
# ----------
#
# .. [#Kökcü]
#
#     Efekan Kökcü, Thomas Steckmann, Yan Wang, J. K. Freericks, Eugene F. Dumitrescu, Alexander F. Kemper
#     "Fixed Depth Hamiltonian Simulation via Cartan Decomposition"
#     `arXiv:2104.00728 <https://arxiv.org/abs/2104.00728>`__, 2021.
#
# .. [#Chu]
#
#     Moody T. Chu
#     "Lax dynamics for Cartan decomposition with applications to Hamiltonian simulation"
#     `doi:10.1093/imanum/drad018 <https://doi.org/10.1093/imanum/drad018>`__, `preprint PDF <https://mtchu.math.ncsu.edu/Research/Papers/Cartan_02.pdf>`__ 2024.
#
#
