r"""The KAK decomposition
=========================

The KAK decomposition is a beautiful mathematical result from Lie theory, with
particular relevance for quantum computing. It can be seen as a
generalization of the singular value decomposition (SVD), as it decomposes a group
element :math:`U` (think: unitary operator) into :math:`U=K_1AK_2,` where
:math:`K_{1,2}` and :math:`A` belong to special subgroups that we will introduce.
This means the KAK decomposition falls under the large umbrella of matrix factorizations,
and it allows us to break down arbitrary quantum circuits into smaller building blocks,
i.e., we can use it for quantum circuit decompositions.

In this demo, we will dive into Lie algebras and their groups. Then, we will discuss
so-called symmetric spaces, which arise from certain subgroups of those Lie groups.
With these tools in our hands, we will then learn about the KAK decomposition itself,
breaking up the Lie group into the subgroup and the symmetric space.

We will make all steps explicit on a toy example on paper and in code, which splits
single-qubit gates into a well-known sequence of rotations.
Finally, we will get to know a handy decomposition of arbitrary
two-qubit unitaries into rotation gates as another application of the KAK decomposition.

.. figure:: ../_static/demo_thumbnails/opengraph_demo_thumbnails/OGthumbnail_kak_decomposition.png
    :align: center
    :width: 70%
    :target: javascript:void(0)

Along the way, we will put some non-essential mathematical details
as well as a few gotchas regarding the nomenclature into boxes such as this one:

.. admonition:: Prerequisites
    :class: note

    In the following we will assume a basic understanding of vector spaces,
    linear maps, and Lie algebras. To review those topics, we recommend a look
    at your favourite linear algebra material. For the latter, also see our
    :doc:`introduction to (dynamical) Lie algebras <demos/tutorial_liealgebra/>`.

Without further ado, let's get started!

Lie algebras and their groups
-----------------------------

We start with Lie algebras, their Lie groups,
and a particular interaction between the two, the *adjoint action*.

Lie algebras
~~~~~~~~~~~~

As mentioned above, we will assume a basic understanding of the mathematical objects
we will use. To warm up, however, let us briefly talk about Lie algebras (for details
see our :doc:`intro to (dynamical) Lie algebras <demos/tutorial_liealgebra/>`).

A *Lie algebra* :math:`\mathfrak{g}` is a vector space with an additional operation
that takes two vectors to a new vector, the *Lie bracket*. To form an algebra, :math:`\mathfrak{g}` must be
closed under the Lie bracket.
For our purposes, the vectors will always be matrices and the Lie bracket will be the matrix
commutator.

**Example**

Our working example in this demo will be the *special unitary* algebra in two dimensions,
:math:`\mathfrak{su}(2).`
It consists of traceless complex-valued skew-Hermitian :math:`2\times 2` matrices, which we
can conveniently describe using the Pauli matrices:

.. math::

    \mathfrak{su}(2)
    &= \left\{i\left(\begin{array}{cc} a & b-ic \\ b+ic & -a \end{array}\right)
    {\large |} a, b, c \in \mathbb{R}\right\}\\
    &= \left\{i(a Z + b X + c Y)| a, b, c \in \mathbb{R}\right\}.

We will also look at a more involved example at the end of the demo.

.. admonition:: Math detail: our Lie algebras are real
    :class: note

    The algebra :math:`\mathfrak{su}(n)` is a *real* Lie algebra, i.e., it is a vector space over
    real numbers, :math:`\mathbb{R}.` This means that scalar-vector multiplication is
    only valid between vectors (complex-valued matrices) and real scalars.

    There is a simple way to see this. Multiplying a skew-Hermitian matrix
    :math:`x\in\mathfrak{su}(n)` by a complex number :math:`c\in\mathbb{C}` will yield
    :math:`(cx)^\dagger=\overline{c} x^\dagger=-\overline{c} x,` so that
    the result might no longer be skew-Hermitian, i.e. no longer in the algebra! If we keep it to real scalars
    :math:`c\in\mathbb{R}` only, we have :math:`\overline{c}=c,` so that
    :math:`(cx)^\dagger=-cx` and we're fine.

    We will only consider real Lie algebras here.

Let us set up :math:`\mathfrak{su}(2)` in code.
Note that the algebra itself consists of *skew*-Hermitian matrices, but we will work
with the Hermitian counterparts as inputs, i.e., we will skip the factor :math:`i.`
We can check that :math:`\mathfrak{su}(2)` is closed under commutators by
computing all nested commutators, the so-called *Lie closure*, and observing
that the closure is not larger than :math:`\mathfrak{su}(2)` itself.
Of course, we could also check the closure manually for this small example.
"""

from itertools import product, combinations
import pennylane as qml
qml.capture.enable()
from pennylane import X, Y, Z
import numpy as np

su2 = [X(0), Y(0), Z(0)]
print(f"su(2) is {len(su2)}-dimensional")

all_hermitian = all(qml.equal(qml.adjoint(op).simplify(), op) for op in su2)
print(f"The operators are all Hermitian: {all_hermitian}")

su2_lie_closed = qml.lie_closure(su2)
print(f"The Lie closure of su(2) is {len(su2_lie_closed)}-dimensional.")

traces = [op.pauli_rep.trace() for op in su2]
print(f"All operators are traceless: {np.allclose(traces, 0.)}")

######################################################################
# We find that :math:`\mathfrak{su}(2)` is indeed closed, and that it is a 3-dimensional
# space, as expected from the explicit expression above.
# We also picked a correct representation with traceless operators.
#
# .. admonition:: Math detail: (semi)simple Lie algebras
#     :class: note
#
#     Our main result for this demo will be the KAK decomposition, which applies to so-called
#     *semisimple* Lie algebras, which are in turn composed of *simple* Lie algebras as building
#     blocks. Without going into detail, it often is sufficient to think of these building
#     blocks as (1) special orthogonal algebras :math:`\mathfrak{so}(n),` (2) unitary symplectic
#     algebras :math:`\mathfrak{sp}(n),` and (3) special unitary algebras :math:`\mathfrak{su}(n).`
#     In particular, our example here is of the latter type, so it is not only semisimple,
#     but even simple.
#
# Lie group from a Lie algebra
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The topic of Lie groups and Lie algebras is a large field of study and there are many
# things we could talk about in this section. For the sake of brevity, however, we will
# only list a few important properties that are needed further below. For more details
# and proofs, refer to your favourite Lie theory book, which could be [#hall]_ or [#tu]_.
#
# The Lie group :math:`\mathcal{G}` associated to a Lie algebra :math:`\mathfrak{g}` is given
# by the exponential map applied to the algebra:
#
# .. math::
#
#     \mathcal{G}=\exp(\mathfrak{g}).
#
# We will only consider Lie groups :math:`\exp(\mathfrak{g})` arising from a Lie algebra
# :math:`\mathfrak{g}` here.
# As we usually think about the unitary algebras :math:`\mathfrak{u}` and their
# subalgebras, the correspondence is well-known to quantum practitioners: Exponentiate
# a skew-Hermitian matrix to obtain a unitary operation, i.e., a quantum gate.
#
# Interaction between Lie groups and algebras
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We will make use of a particular interaction between the algebra :math:`\mathfrak{g}` and
# its group :math:`\mathcal{G},` called the *adjoint action* of :math:`\mathcal{G}` on
# :math:`\mathfrak{g}.` It is given by
#
# .. math::
#
#     \text{Ad}: \mathcal{G} \times \mathfrak{g} \to \mathfrak{g},
#     \ (\exp(x),y)\mapsto \text{Ad}_{\exp(x)}(y) = \exp(x) y\exp(-x).
#
# Similarly, we can interpret the Lie bracket as a map of :math:`\mathfrak{g}` acting on itself,
# which is called the *adjoint representation* of :math:`\mathfrak{g}` on itself:
#
# .. math::
#
#     \text{ad}: \mathfrak{g} \times \mathfrak{g} \to \mathfrak{g},
#     \ (x, y) \mapsto \text{ad}_x(y) = [x, y].
#
# The adjoint group action and adjoint algebra representation do not only carry a very
# similar name, they are intimately related:
#
# .. math::
#
#     \text{Ad}_{\exp(x)}(y) = \exp(\text{ad}_x) (y),
#
# where we applied the exponential map to :math:`\text{ad}_x,` which maps from :math:`\mathfrak{g}`
# to itself, via its series representation.
# We will refer to this relationship as the *adjoint identity*.
# We talk about :math:`\text{Ad}` and :math:`\text{ad}` in more detail in the box below, and refer to our demo
# :doc:`g-sim: Lie algebraic classical simulations <demos/tutorial_liesim/>` for
# further discussion.
#
# .. admonition:: Derivation: adjoint representations
#     :class: note
#
#     We begin this derivation with the *adjoint action* of :math:`\mathcal{G}` on itself,
#     given by
#
#     .. math::
#
#         \Psi: \mathcal{G} \times \mathcal{G} \to \mathcal{G},
#         \ (\exp(x),\exp(y))\mapsto \Psi_{\exp(x)}(\exp(y)) = \exp(x) \exp(y)\exp(-x).
#
#     The map :math:`\Psi_{\exp(x)}` (with fixed subscript) is a smooth map from the Lie group
#     :math:`\mathcal{G}` to itself, so that we may differentiate it. This leads to the
#     differential :math:`\text{Ad}_{\exp(x)}=d\Psi_{\exp(x)},` which maps the tangent spaces of
#     :math:`\mathcal{G}` to itself. At the identity, where
#     the algebra :math:`\mathfrak{g}` forms the tangent space, we find
#
#     .. math::
#
#         \text{Ad} : \mathcal{G} \times\mathfrak{g} \to \mathfrak{g},
#         \ (\exp(x), y)\mapsto \exp(x) y \exp(-x).
#
#     This is the adjoint action of :math:`\mathcal{G}` on :math:`\mathfrak{g}` as we
#     introduced above.
#
#     Now that we have the adjoint action of :math:`\mathcal{G}` on :math:`\mathfrak{g},`
#     we can differentiate it with respect to the subscript argument:
#
#     .. math::
#
#         \text{ad}_{\circ}(y)&=d\text{Ad}_\circ(y),\\
#         \text{ad}: \mathfrak{g}\times \mathfrak{g}&\to\mathfrak{g},
#         \ (x, y)\mapsto \text{ad}_x(y) = [x, y].
#
#     It is a non-trivial observation that this differential equals the commutator!
#     With :math:`\text{ad}` we arrived at a map that *represents* the action of an algebra element
#     :math:`x` on the vector space that is the algebra itself. That is, we found the
#     *adjoint representation* of :math:`\mathfrak{g}.`
#
#     Finally, note that the adjoint identity can be proven with similar tools as above,
#     i.e., chaining derivatives and exponentiation suitably.
#
# Symmetric spaces
# ----------------
#
# Symmetric spaces are a popular field of study both in physics and mathematics.
# We will not go into depth regarding their interpretation or classification, but refer the
# interested reader to the broad existing literature, including [#arvanitogeorgos]_ and
# [#helgason]_.
# In the following, we mostly care about the algebraic structure of symmetric spaces.
#
# Subalgebras and Cartan decompositions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# A *subalgebra* :math:`\mathfrak{k}` of a Lie algebra :math:`\mathfrak{g}` is a
# vector subspace that is closed under the Lie bracket. Overall, this means that
# :math:`\mathfrak{k}` is closed under addition, scalar multiplication, and the Lie bracket.
# The latter often is simply written as :math:`[\mathfrak{k}, \mathfrak{k}]\subset \mathfrak{k}.`
#
# The algebras we are interested in come with an *inner product* between its elements.
# For our purposes, it is sufficient to assume that it is
#
# .. math::
#
#     \langle x, y\rangle = \text{tr}[x^\dagger y].
#
# Let's implement the inner product and an orthogonality check based on it:
#


def inner_product(op1, op2):
    """Compute the trace inner product between two operators."""
    # Use two wires to reuse it in the second example on two qubits later on
    return qml.math.trace(qml.matrix(qml.adjoint(op1) @ op2, wire_order=[0, 1]))


def is_orthogonal(op, basis):
    """Check whether an operator is orthogonal to a space given by some basis."""
    return np.allclose([inner_product(op, basis_op) for basis_op in basis], 0)


######################################################################
# Given a subalgebra :math:`\mathfrak{k}\subset \mathfrak{g},` the inner product allows
# us to define an orthogonal complement
#
# .. math::
#
#     \mathfrak{p} = \{x\in\mathfrak{g} | \langle x, y\rangle=0 \ \forall \ y\in\mathfrak{k}\}.
#
# In this context, :math:`\mathfrak{k}` is commonly called the *vertical space*,
# :math:`\mathfrak{p}` accordingly is the *horizontal space*.
# The KAK decomposition will apply to scenarios in which these spaces satisfy additional
# commutation relations, which do not hold for all subalgebras:
#
# .. math::
#
#     [\mathfrak{k}, \mathfrak{p}] \subset& \mathfrak{p} \qquad \text{(Reductive property)},\\
#     [\mathfrak{p}, \mathfrak{p}] \subset& \mathfrak{k} \qquad \text{(Symmetric property)}.
#
# The first property tells us that :math:`\mathfrak{p}` is left intact by the adjoint action of
# :math:`\mathfrak{k}.` The second property suggests that :math:`\mathfrak{p}` behaves like the
# *opposite* of a subalgebra, i.e., all commutators lie in its complement, the subalgebra
# :math:`\mathfrak{k}.` Due to the adjoint identity from above, the first property also holds for
# group elements acting on algebra elements; for all :math:`x\in\mathfrak{p}` and
# :math:`K\in\mathcal{K}=\exp(\mathfrak{k}),` we have
#
# .. math::
#
#     K x K^\dagger
#     = \exp(y) x \exp(-y)
#     = \text{Ad}_{\exp(y)}(x)
#     = \exp(\text{ad}_y) (x)
#     = \sum_{n=0}^\infty \frac{1}{n!} \underset{\in\mathfrak{p}}{\underbrace{(\text{ad}_y)^n (x)}}
#     \in \mathfrak{p}.
#
# If the reductive property holds, the quotient space :math:`\mathcal{G}/\mathcal{K}` of the groups
# of :math:`\mathfrak{g}` and :math:`\mathfrak{k}` (see detail box below) is called a
# *reductive homogeneous space*. If both properties hold, :math:`(\mathfrak{k}, \mathfrak{p})` is
# called a *Cartan pair* and we call :math:`\mathfrak{g}=\mathfrak{k} \oplus \mathfrak{p}` a
# *Cartan decomposition*. :math:`(\mathfrak{g}, \mathfrak{k})` is named a *symmetric pair*
# and the quotient :math:`\mathcal{G}/\mathcal{K}` is a *symmetric space*.
# Symmetric spaces are relevant for a wide range of applications in physics
# and have been studied a lot throughout the last hundred years.
#
# .. admonition:: Nomenclature: Cartan decomposition/pair
#     :class: warning
#
#     Depending on context and field, there sometimes is an additional requirement
#     for :math:`\mathfrak{g}=\mathfrak{k}\oplus\mathfrak{p}` to be called a Cartan decomposition.
#     Without going into detail, this requirement is that the so-called *Killing form* must be
#     negative definite on :math:`\mathfrak{k}` and positive definite on :math:`\mathfrak{p}`
#     [#helgason]_.
#
# .. admonition:: Math detail: quotient space
#     :class: note
#
#     The *quotient space* of a Lie group :math:`\mathcal{G}` and a subgroup :math:`\mathcal{K}`
#     is the space of cosets of :math:`\mathcal{K},` i.e.,
#     :math:`\mathcal{G}/\mathcal{K} = \{g\mathcal{K} | g\in G\}.` In this space, two elements are
#     equal if they just differ by multiplying an element from :math:`\mathcal{K}` from the left
#     to one of them. The quotient space is a manifold like the two groups :math:`\mathcal{G}` and
#     :math:`\mathcal{K},` but in general it will *not* be a group itself. For example, a product
#     of two elements is
#     :math:`(g'\mathcal{K})(g\mathcal{K})=g'g(g^{-1} \mathcal{K} g) \mathcal{K},` which only is
#     a coset again if :math:`g^{-1} \mathcal{K} g\subset \mathcal{K}.` Subgroups for which this
#     condition holds for any :math:`g\in \mathcal{G}` are called *normal subgroups*.
#     We are interested in cases where the symmetric property
#     :math:`[\mathfrak{p}, \mathfrak{p}] \subset \mathfrak{k}` holds, which excludes (non-Abelian)
#     normal subgroups, and thus our quotient space :math:`\mathcal{G}/\mathcal{K}` will not be
#     a group.
#
# **Example**
#
# For our example, we consider the subalgebra :math:`\mathfrak{k}=\mathfrak{u}(1)`
# of :math:`\mathfrak{su}(2)` that generates Pauli-:math:`Z` rotations:
#
# .. math::
#
#     \mathfrak{k} = \text{span}_{\mathbb{R}} \{iZ\}.
#
# Let us define it in code, and check whether it gives rise to a Cartan decomposition.
# As we want to look at another example later, we wrap everything in a function.
# A similar function is available in PennyLane as :func:`~.pennylane.liealg.check_cartan_decomp`.
#


def check_cartan_decomp(g, k, space_name):
    """Given an algebra g and an operator subspace k, verify that k is a subalgebra
    and gives rise to a Cartan decomposition. Similar to qml.liealg.check_cartan_decomp"""
    # Check Lie closure of k
    k_lie_closure = qml.lie_closure(k)
    k_is_closed = len(k_lie_closure) == len(k)
    print(f"The Lie closure of k is as big as k itself: {k_is_closed}.")

    # Orthogonal complement of k, assuming that everything is given in the same basis.
    p = [g_op for g_op in g if is_orthogonal(g_op, k)]
    print(
        f"k has dimension {len(k)}, p has dimension {len(p)}, which combine to "
        f"the dimension {len(g)} of g: {len(k)+len(p)==len(g)}"
    )

    # Check reductive property
    k_p_commutators = [qml.commutator(k_op, p_op) for k_op, p_op in product(k, p)]
    k_p_coms_in_p = all([is_orthogonal(com, k) for com in k_p_commutators])

    print(f"All commutators in [k, p] are in p (orthogonal to k): {k_p_coms_in_p}.")
    if k_p_coms_in_p:
        print(f"{space_name} is a reductive homogeneous space.")

    # Check symmetric property
    p_p_commutators = [qml.commutator(*ops) for ops in combinations(p, r=2)]
    p_p_coms_in_k = all([is_orthogonal(com, p) for com in p_p_commutators])

    print(f"All commutators in [p, p] are in k (orthogonal to p): {p_p_coms_in_k}.")
    if p_p_coms_in_k:
        print(f"{space_name} is a symmetric space.")

    return p


u1 = [Z(0)]
space_name = "SU(2)/U(1)"
p = check_cartan_decomp(su2, u1, space_name)

######################################################################
# Cartan subalgebras
# ~~~~~~~~~~~~~~~~~~
#
# The symmetric property of a Cartan decomposition
# :math:`([\mathfrak{p}, \mathfrak{p}]\subset\mathfrak{k})` tells us that :math:`\mathfrak{p}`
# is *very far* from being a subalgebra (commutators never end up in :math:`\mathfrak{p}` again).
# This also gives us information about potential subalgebras *within* :math:`\ \mathfrak{p}.`
# Assume we have a subalgebra :math:`\mathfrak{a}\subset\mathfrak{p}.` Then the commutator
# between any two elements :math:`x, y\in\mathfrak{a}` must satisfy
#
# .. math::
#
#     [x, y] \in \mathfrak{a} \subset \mathfrak{p}
#     &\Rightarrow [x, y]\in\mathfrak{p} \text{(subalgebra property)}, \\
#     [x, y] \in [\mathfrak{a}, \mathfrak{a}] \subset [\mathfrak{p}, \mathfrak{p}]
#     \subset \mathfrak{k} &\Rightarrow [x, y]\in\mathfrak{k}\ \text{(symmetric property)}.
#
# That is, the commutator must lie in *both* orthogonal complements :math:`\mathfrak{k}` and
# :math:`\mathfrak{p},` which only have the zero vector in common. This tells us that *all*
# commutators in :math:`\mathfrak{a}` vanish, making it an *Abelian* subalgebra:
#
# .. math::
#
#     [\mathfrak{a}, \mathfrak{a}] = \{0\}.
#
# Such an Abelian subalgebra is a (horizontal) *Cartan subalgebra (CSA)* if it is *maximal*,
# i.e., if it can not be made any larger (higher-dimensional) without leaving :math:`\mathfrak{p}.`
#
# .. admonition:: Nomenclature: Cartan subalgebra
#     :class: warning
#
#     Depending on context and field, there are inequivalent notions of Cartan subalgebras.
#     In particular, there is a common notion of Cartan subalgebras which are not contained
#     in a horizontal space. Throughout this demo, we always mean a *horizontal*
#     maximal Abelian subalgebra :math:`\mathfrak{a}\subset\mathfrak{p}.` The two notions
#     can be made compatible by being precise about the space of which the subalgebra is a CSA.
#
# How many different CSAs are there? Given a CSA :math:`\mathfrak{a},` we can pick a vertical
# element :math:`y\in\mathfrak{k}` and apply the corresponding group element :math:`K=\exp(y)` to
# all elements of the CSA, using the adjoint action we studied above. This will yield a valid
# CSA again: First, :math:`K\mathfrak{a} K^\dagger` remains in :math:`\mathfrak{p}`
# due to the reductive property, as we discussed when introducing the Cartan decomposition.
# Second, the adjoint action will not change the Abelian property because
#
# .. math::
#
#     [K x_1 K^\dagger, K x_2 K^\dagger] = K [x_1, x_2] K^\dagger = K 0 K^\dagger = 0
#     \quad \forall\ x_{1, 2}\in\mathfrak{a}.
#
# Finally, we are guaranteed that :math:`K\mathfrak{a} K^\dagger` remains maximal:
#
# .. admonition:: Math detail: CSAs remain maximal
#     :class: note
#
#     The reason that :math:`K\mathfrak{a} K^\dagger` is maximal if :math:`\mathfrak{a}` was, is
#     that we assume :math:`\mathfrak{g}` to be a semisimple Lie algebra, for which the
#     adjoint representation is faithful. This in turn implies that linearly
#     independent elements of :math:`\mathfrak{g}` will not be mapped to linearly dependent
#     elements by the adjoint action of :math:`K.`
#
# For most :math:`y\in\mathfrak{k},` applying :math:`K=\exp(y)` in this way will yield a
# *different* CSA, so that we find a whole continuum of them.
# It turns out that they *all* can be found by starting with *any*
# :math:`\mathfrak{a}` and applying all of :math:`\mathcal{K}` to it.
#
# *This is what powers the KAK decomposition.*
#
# **Example**
#
# For our example, we established the decomposition
# :math:`\mathfrak{su}(2)=\mathfrak{u}(1)\oplus \mathfrak{p}` with the two-dimensional horizontal
# space :math:`\mathfrak{p} = \text{span}_{\mathbb{R}}\{iX, iY\}.` Starting with the subspace
# :math:`\mathfrak{a}=\text{span}_{\mathbb{R}} \{iY\},` we see that we immediately reach a maximal Abelian
# subalgebra (a CSA), because :math:`[Y, X]\neq 0.` Applying a rotation
# :math:`\exp(i\eta Z)\in\mathcal{K}` to this CSA gives us a new CSA via
#
# .. math::
#
#     \mathfrak{a}'=\{\exp(i\eta Z) (c iY) \exp(-i\eta Z) | c\in\mathbb{R}\}
#     =\{c\cos(2\eta) iY + c\sin(2\eta) iX | c\in\mathbb{R}\} .
#
# The vertical group element :math:`\exp(i\eta Z)` simply rotates the CSA within
# :math:`\mathfrak{p}.` Let us not forget to define the CSA in code.

# CSA generator: iY
a = p[1]

# Rotate CSA by applying some vertical group element exp(i eta Z)
eta = 0.6
# The factor -2 compensates the convention -1/2 in the RZ gate
a_prime = qml.RZ(-2 * eta, 0) @ a @ qml.RZ(2 * eta, 0)

# Expectation from our theoretical calculation
a_prime_expected = np.cos(2 * eta) * a + np.sin(2 * eta) * p[0]
a_primes_equal = np.allclose(qml.matrix(a_prime_expected), qml.matrix(a_prime))
print(f"The rotated CSAs match between numerics and theory: {a_primes_equal}")

######################################################################
# Cartan involutions
# ~~~~~~~~~~~~~~~~~~
#
# In practice, there often is a more convenient way to obtain a Cartan decomposition
# than by specifying the subalgebra :math:`\mathfrak{k}` or its horizontal counterpart
# :math:`\mathfrak{p}` manually. It goes as follows.
#
# We will look at a map :math:`\theta` from the total Lie algebra :math:`\mathfrak{g}`
# to itself. We demand that :math:`\theta` has the following properties, for
# :math:`x, y\in\mathfrak{g}` and :math:`c\in\mathbb{R}.`
#
# #. It is linear, i.e., :math:`\theta(x + cy)=\theta(x) +c \theta(y),`
# #. It is compatible with the commutator, i.e., :math:`\theta([x, y])=[\theta(x),\theta(y)],` and
# #. It is an *involution*, i.e., :math:`\theta(\theta(x)) = x,`
#    or :math:`\theta^2=\mathbb{I}_{\mathfrak{g}}.`
#
# In short, we demand that :math:`\theta` be an *involutive automorphism* of :math:`\mathfrak{g}.`
#
# As an involution, :math:`\theta` only can have the eigenvalues :math:`\pm 1,` with associated
# eigenspaces :math:`\mathfrak{g}_\pm.` Let's see what happens when we compute commutators between
# elements :math:`x_\pm\in\mathfrak{g}_\pm \Leftrightarrow \theta(x_\pm) = \pm x_\pm:`
#
# .. math::
#
#     &\theta([x_+, x_+]) = [\theta(x_+), \theta(x_+)] = [x_+, x_+]
#     &\ \Rightarrow\ [x_+, x_+]\in\mathfrak{g}_+ ,\\
#     &\theta([x_+, x_-]) = [\theta(x_+), \theta(x_-)] = -[x_+, x_-]
#     &\ \Rightarrow\ [x_+, x_-]\in\mathfrak{g}_- ,\\
#     &\theta([x_-, x_-]) = [\theta(x_-), \theta(x_-)] = (-1)^2 [x_-, x_-]
#     &\ \Rightarrow\ [x_-, x_-]\in\mathfrak{g}_+.
#
# Or, in other words,
# :math:`[\mathfrak{g}_+, \mathfrak{g}_+] \subset \mathfrak{g}_+,`
# :math:`[\mathfrak{g}_+, \mathfrak{g}_-] \subset \mathfrak{g}_-,`
# and :math:`[\mathfrak{g}_-, \mathfrak{g}_-] \subset \mathfrak{g}_+.`
# So an involution is enough to find us a Cartan decomposition, with
# :math:`\mathfrak{k}=\mathfrak{g}_+` and :math:`\mathfrak{p}=\mathfrak{g}_-.`
#
# 🤯
#
# We might want to call such a :math:`\theta` a *Cartan involution*.
#
# .. admonition:: Nomenclature: Cartan involution
#     :class: warning
#
#     Some people do so, some people again require more properties for such an
#     involution to be called Cartan involution.
#     For our purposes, let's go with the more general definition and call all
#     involutions with the properties above Cartan involutions.
#
# Conversely, if we have a Cartan decomposition based on a subalgebra :math:`\mathfrak{k},`
# we can define the map
#
# .. math::
#
#     \theta_{\mathfrak{k}}(x) = \Pi_{\mathfrak{k}}(x)-\Pi_{\mathfrak{p}}(x),
#
# where :math:`\Pi_{\mathfrak{k},\mathfrak{p}}` are the projectors onto the two vector
# subspaces. Clearly, :math:`\theta_{\mathfrak{k}}` is linear because projectors are.
# It is also compatible with the commutator due to the commutation relations
# between :math:`\mathfrak{k}` and :math:`\mathfrak{p}` (see box below).
# Finally, :math:`\theta_{\mathfrak{k}}` is an involution because
#
# .. math::
#
#     \theta_{\mathfrak{k}}^2=(\Pi_{\mathfrak{k}}-\Pi_{\mathfrak{p}})^2
#     = \Pi_{\mathfrak{k}}^2-\Pi_{\mathfrak{k}}\Pi_{\mathfrak{p}}
#     -\Pi_{\mathfrak{p}}\Pi_{\mathfrak{k}}+\Pi_{\mathfrak{p}}^2
#     =\Pi_{\mathfrak{k}}+\Pi_{\mathfrak{p}}
#     = \mathbb{I}_{\mathfrak{g}},
#
# where we used the projectors' property :math:`\Pi_{\mathfrak{k}}^2=\Pi_{\mathfrak{k}}` and
# :math:`\Pi_{\mathfrak{p}}^2=\Pi_{\mathfrak{p}},` as well as the fact that
# :math:`\Pi_{\mathfrak{k}}\Pi_{\mathfrak{p}}=\Pi_{\mathfrak{p}}\Pi_{\mathfrak{k}}=0` because
# the spaces :math:`\mathfrak{k}` and :math:`\mathfrak{p}` are orthogonal to each other.
#
# .. admonition:: Math detail: :math:`\theta_{\mathfrak{k}}` is a homomorphism
#     :class: note
#
#     To see that :math:`\theta_{\mathfrak{k}}` is compatible with the commutator, i.e.,
#     an algebra homomorphism, see how a commutator :math:`[x, y]` splits:
#
#     .. math::
#
#         [x, y]
#         &= [\Pi_{\mathfrak{k}}(x) + \Pi_{\mathfrak{p}}(x), \Pi_{\mathfrak{k}}(y) + \Pi_{\mathfrak{p}}(y)] \\
#         &= \underset{\in \mathfrak{k}}{\underbrace{[\Pi_{\mathfrak{k}}(x), \Pi_{\mathfrak{k}}(y)]}}
#         +\underset{\in \mathfrak{p}}{\underbrace{[\Pi_{\mathfrak{k}}(x), \Pi_{\mathfrak{p}}(y)]}}
#         +\underset{\in \mathfrak{p}}{\underbrace{[\Pi_{\mathfrak{p}}(x), \Pi_{\mathfrak{k}}(y)]}}
#         +\underset{\in \mathfrak{k}}{\underbrace{[\Pi_{\mathfrak{p}}(x), \Pi_{\mathfrak{p}}(y)]}},
#
#     where we used :math:`\mathbb{I}_{\mathfrak{g}} = \Pi_{\mathfrak{k}} + \Pi_{\mathfrak{p}}` and the
#     commutation relations between :math:`\mathfrak{k}` and :math:`\mathfrak{p}.`
#     We can use this split to compute
#
#     .. math::
#
#         \theta_{\mathfrak{k}} ([x, y])
#         &=\Pi_{\mathfrak{k}}([x, y]) - \Pi_{\mathfrak{p}}([x, y])\\
#         &=[\Pi_{\mathfrak{k}}(x), \Pi_{\mathfrak{k}}(y)] + [\Pi_{\mathfrak{p}}(x), \Pi_{\mathfrak{p}}(y)]
#         - [\Pi_{\mathfrak{k}}(x), \Pi_{\mathfrak{p}}(y)] - [\Pi_{\mathfrak{p}}(x), \Pi_{\mathfrak{k}}(y)]\\
#         &=[\Pi_{\mathfrak{k}}(x) -\Pi_{\mathfrak{p}}(x), \Pi_{\mathfrak{k}}(y)-\Pi_{\mathfrak{p}}(y)]\\
#         &=[\theta_{\mathfrak{k}} (x),\theta_{\mathfrak{k}} (y)].
#
#     Thus, :math:`\theta_{\mathfrak{k}}` indeed is compatible with the commutator.
#
# This shows us that we can easily switch between a Cartan involution and a Cartan
# decomposition, in either direction!
#
# **Example**
#
# In our example, an involution that reproduces our choice
# :math:`\mathfrak{k}=\text{span}_{\mathbb{R}} \{iZ\}` is :math:`\theta_Z(x) = Z x Z`
# (convince yourself that it is an involution that respects commutators, or verify that
# it matches :math:`\theta_{\mathfrak{k}}` from above).


def theta_Z(x):
    return qml.simplify(Z(0) @ x @ Z(0))


theta_of_u1 = [theta_Z(x) for x in u1]
u1_is_su2_plus = all(qml.equal(x, theta_of_x) for x, theta_of_x in zip(u1, theta_of_u1))
print(f"U(1) is the +1 eigenspace: {u1_is_su2_plus}")

theta_of_p = [theta_Z(x) for x in p]
p_is_su2_minus = all(qml.equal(-x, theta_of_x) for x, theta_of_x in zip(p, theta_of_p))
print(f"p is the -1 eigenspace: {p_is_su2_minus}")

######################################################################
# We can easily get a new subalgebra by modifying the involution, say, to
# :math:`\theta_Y(x) = Y x Y,` expecting that
# :math:`\mathfrak{k}_Y=\text{span}_{\mathbb{R}} \{iY\}` becomes the new subalgebra.


def theta_Y(x):
    return qml.simplify(Y(0) @ x @ Y(0))


eigvals = []
for x in su2:
    if qml.equal(theta_Y(x), x):
        eigvals.append(1)
    elif qml.equal(theta_Y(x), -x):
        eigvals.append(-1)
    else:
        raise ValueError("Operator not purely in either eigenspace.")

print(f"Under theta_Y, the operators\n{su2}\nhave the eigenvalues\n{eigvals}")

######################################################################
# This worked! A new involution gave us a new subalgebra and Cartan decomposition.
#
# .. admonition:: Math detail: classification of Cartan decompositions
#     :class: note
#
#     You might already see that the two different decompositions created by :math:`\theta_Z`
#     and :math:`\theta_Y` are very similar. There is a whole field of study that
#     characterizes---and even fully classifies---the possible Cartan decompositions
#     of semisimple Lie algebras. This classification
#     plays a big role when talking about decompositions without getting stuck on details
#     like the choice of basis or the representation of the algebra as matrices.
#     For example, there are only three types of Cartan decompositions of the special unitary
#     algebra :math:`\mathfrak{su}(n),` called AI, AII, and AIII. The subalgebras
#     :math:`\mathfrak{k}` for these decompositions are the special orthogonal algebra
#     :math:`\mathfrak{so}(n)` (AI), the unitary symplectic algebra :math:`\mathfrak{sp}(n)` (AII),
#     and a sum of (special) unitary algebras
#     :math:`\mathfrak{su}(p)\oplus\mathfrak{su}(q)\oplus\mathfrak{u}(1)` (AIII, :math:`p+q=n`).
#     For a quick overview, see for example the `Wikipedia entry on symmetric spaces
#     <https://en.wikipedia.org/wiki/Symmetric_space#Classification_of_Riemannian_symmetric_spaces>`__.
#     Their involutions are usually represented by complex conjugation (AI), by the adjoint
#     action with a Pauli operator (AIII, for qubits, :math:`p=q=2^{N-1}`), or by both
#     (AII). It is instructive to try and see why those three are *not* equivalent
#     under a unitary basis change!
#
# The KAK decomposition
# ---------------------
#
# Now that we covered all prerequisites, we are ready for our main result. It consists of two
# steps that are good to know individually, so we will look at both of them in sequence.
# We will not conduct formal proofs but leave those to the literature references.
# In the following, let :math:`\mathfrak{g}` be a compact real semisimple Lie algebra and
# :math:`\mathfrak{k}` a subalgebra such that :math:`\mathfrak{g}=\mathfrak{k}\oplus \mathfrak{p}`
# is a Cartan decomposition.
#
# The first step is a decomposition of the Lie group :math:`\mathcal{G}=\exp(\mathfrak{g})`
# into the Lie subgroup
# :math:`\mathcal{K}=\exp(\mathfrak{k})` and the exponential of the horizontal space,
# :math:`\mathcal{P}=\exp(\mathfrak{p}),` *which is not a group* (see box on quotient spaces).
# The decomposition is a simple product within :math:`\mathcal{G}:`
#
# .. math::
#
#     \mathcal{G} &= \mathcal{K}\mathcal{P}, \text{ or }\\
#     \forall\ G\in\mathcal{G}\ \ \exists K\in\mathcal{K}, x\in\mathfrak{p}: \ G &= K \exp(x).
#
# This *KP* decomposition can be seen as the *group version* of
# :math:`\mathfrak{g} = \mathfrak{k} \oplus\mathfrak{p}` and is known as a *global* Cartan
# decomposition of :math:`\mathcal{G}.`
#
# The second step is the further decomposition of the space :math:`\mathcal{P}=\exp(\mathfrak{p}).`
# For this we first need to fix a Cartan subalgebra (CSA) :math:`\mathfrak{a}\subset\mathfrak{p}.`
# The CSA might be given through some application or from context, but there is no
# canonical choice.
# Given a horizontal vector :math:`x\in\mathfrak{p},` we can always construct a second CSA
# :math:`\mathfrak{a}_x\subset\mathfrak{p}` that contains :math:`x.` As any two CSAs can be mapped
# to each other by some subalgebra element :math:`y\in\mathfrak{k}` using the adjoint action :math:`\text{Ad},`
# we know that a :math:`y` exists such that
#
# .. math::
#
#     \exp(y)\mathfrak{a}_x\exp(-y)=\mathfrak{a}
#     \quad\Rightarrow\quad x\in \exp(-y) \mathfrak{a}\exp(y).
#
# Generalizing this statement across all horizontal elements :math:`x\in\mathfrak{p},` we find
#
# .. math::
#
#     \mathfrak{p} \subset \{\exp(-y) \mathfrak{a} \exp(y) | y\in\mathfrak{k}\}.
#
# As we discussed, the converse inclusion also must hold for a reductive space, so that we
# may even replace :math:`\subset` by an equality.
# Now we can use :math:`\exp(\text{Ad}_{K} x)=\text{Ad}_{K}\exp(x)` to move
# this statement to the group level,
#
# .. math::
#
#     \mathcal{P}
#     =\exp(\mathfrak{p})
#     = \{\exp(\exp(-y) \mathfrak{a} \exp(y)) | y\in\mathfrak{k}\}
#     = \{\exp(K^{-1} \mathfrak{a} K) | K\in\mathcal{K}\}
#     = \{K^{-1} \mathcal{A} K | K\in\mathcal{K}\},
#
# where we abbreviated :math:`\mathcal{A} = \exp(\mathfrak{a}).`
#
# Chaining the two steps together and combining the left factor :math:`K^{-1}` with the group
# :math:`\mathcal{K}` in the *KP* decomposition, we obtain the *KAK decomposition*
#
# .. math::
#
#     \mathcal{G}
#     =\mathcal{K}\mathcal{P}
#     = \mathcal{K}\{K^{-1} \mathcal{A} K | K\in\mathcal{K}\},
#     = \{K_1 \mathcal{A} K_2 | K_{1,2}\in\mathcal{K}\},
#     = \mathcal{K} \mathcal{A} \mathcal{K} \qquad\textbf{(KAK Decomposition).}
#
# It teaches us that any group element can be decomposed into two factors from the Lie subgroup and
# the exponential of a CSA element, i.e., of commuting elements from the horizontal subspace
# :math:`\mathfrak{p}.` This may already hint at the usefulness of the KAK decomposition for matrix
# factorizations in general, and for quantum circuit decompositions in particular.
# Given a group operation :math:`G=\exp(x)` with :math:`x\in\mathfrak{g},` there are two
# subalgebra elements :math:`y_{1,2}\in\mathfrak{k}` (or subgroup elements
# :math:`K_{1,2}=\exp(y_{1,2})\in \mathcal{K}`) and a Cartan subgalgebra element
# :math:`a\in\mathfrak{a}` so that
#
# .. math::
#
#     G\in\mathcal{G} \quad\Rightarrow\quad G=K_1 \exp(a) K_2.
#
# If :math:`x` happens to be from the horizontal subspace :math:`\mathfrak{p},` so that
# :math:`G\in \mathcal{P}\subset\mathcal{G},` we know that the two subgroup elements :math:`K_1`
# and :math:`K_2` will in fact be related, namely
#
# .. math::
#
#     G\in\mathcal{P} \quad\Rightarrow\quad G=K\exp(a)K^\dagger.
#
# **Example**
#
# Applying what we just learned to our example on :math:`\mathfrak{su}(2),` we can state that
# any single-qubit gate can be implemented by running a gate from
# :math:`\mathcal{K}=\{\exp(i\eta Z) | \eta\in\mathbb{R}\},` a CSA gate
# :math:`\mathcal{A}=\{\exp(i\varphi Y) | \eta\in\mathbb{R}\},` and another gate from
# :math:`\mathcal{K}.` We rediscovered a standard decomposition of an arbitrary
# :math:`SU(2)` gate! PennyLane produces it with :func:`~.pennylane.ops.one_qubit_decomposition`:

x = 0.2j * su2[0] - 0.1j * su2[1] - 0.2j * su2[2]
G = qml.math.linalg.expm(qml.matrix(x))
print("Decomposition:")
print(*qml.ops.one_qubit_decomposition(G, 0, rotations="ZYZ"), sep="\n")

######################################################################
# If we pick a *horizontal gate*, i.e., a gate :math:`G\in\mathcal{P}`, we obtain the same
# rotation angle for the initial and final :math:`R_Z` rotations, up to the expected sign, and
# a shift by some multiple of :math:`2\pi.`

horizontal_x = -0.1j * p[0] - 4.1j * p[1]
print(horizontal_x)
P = qml.math.linalg.expm(qml.matrix(horizontal_x))
decomp = qml.ops.one_qubit_decomposition(P, 0, rotations="ZYZ")
print("Decomposition:", *decomp, sep="\n")
angle_match = np.isclose((decomp[0].data[0] + decomp[-1].data[0]) % (2 * np.pi), 0.0)
print(f"First and last rotation angle match up to sign and shift by 2kπ: {angle_match}")

######################################################################
# Other choices for involutions or---equivalently---subalgebras :math:`\mathfrak{k}` will
# lead to other decompositions of ``Rot``. For example, using :math:`\theta_Y` from above
# together with the CSA :math:`\mathfrak{a}_Y=\text{span}_{\mathbb{R}} \{iX\},` we find the
# decomposition
#
# .. math::
#
#     \text{Rot}(\phi, \theta, \omega) = R_Y(\eta_1) R_X(\vartheta) R_Y(\eta_2).
#
# And that's it for our main discussion. We conclude this demo by applying the
# KAK decomposition to the group of arbitrary two-qubit gates.
#
# Application: Two-qubit gate decomposition
# -----------------------------------------
#
# Two-qubit operations are described by the special unitary group :math:`SU(4)` and
# here we will use a decomposition of its algebra :math:`\mathfrak{su}(4)` to decompose
# such gates.
# Specifically, we use the subalgebra that generates single-qubit operations independently
# on either qubit, :math:`\mathfrak{su}(2)\oplus\mathfrak{su}(2).` Let's set it up with our
# tool from earlier:

# Define su(4). Skip first entry of Pauli group, which is the identity
su4 = list(qml.pauli.pauli_group(2))[1:]
print(f"su(4) is {len(su4)}-dimensional")

# Define subalgebra su(2) ⊕ su(2)
su2_su2 = [X(0), Y(0), Z(0), X(1), Y(1), Z(1)]
space_name = "SU(4)/(SU(2)xSU(2))"
p = check_cartan_decomp(su4, su2_su2, space_name)

######################################################################
# .. admonition:: Math detail: involution for two-qubit decomposition
#     :class: note
#
#     The accompanying involution sorts operators by the number of qubits on which they are
#     supported (:math:`\mathfrak{k}` is supported on one, :math:`\mathfrak{p}` on two).
#     This can be realized with the operation
#
#     .. math::
#
#         \theta(x) = -Y_0Y_1 x^T Y_0Y_1.
#
#     Intuitively, the conjugation by :math:`Y_0Y_1` adds a minus
#     sign for each :math:`X` and :math:`Z` factor in :math:`x,` and the transposition
#     adds a minus sign for each :math:`Y.` Taken together, each Pauli operator contributes
#     a minus sign. Finally, as we want the single-qubit operators to receive no sign in total
#     (:math:`\mathfrak{k}` is the :math:`+1` eigenspace), we add a minus sign overall.
#
# Now we can pick a Cartan subalgebra within :math:`\mathfrak{p},` the vector space
# of all two-qubit Paulis. A common choice is
#
# .. math::
#
#     \mathfrak{a} = \text{span}_{\mathbb{R}}\{iX_0X_1, iY_0Y_1, iZ_0Z_1\}.
#
# These three operators commute, making :math:`\mathfrak{a}` Abelian.
# They also form a *maximal* Abelian algebra within :math:`\mathfrak{p},` which is less obvious.
#
# The KAK decomposition now tells us that any two-qubit gate :math:`U,` being part of
# :math:`SU(4),` can be implemented by a sequence
#
# .. math::
#
#     U &= \exp(y_1) \exp(a)\exp(y_2)\\
#     &= \exp(i[\varphi^x_0 X_0 + \varphi^y_0 Y_0 + \varphi^z_0 Z_0])
#     \exp(i[\varphi^x_1 X_1 + \varphi^y_1 Y_1 + \varphi^z_1 Z_1])\\
#     &\times \exp(i [\eta^x X_0X_1 + \eta^y Y_0Y_1 + \eta^z Z_0Z_1])\\
#     &\times \exp(i[\vartheta^x_0 X_0 + \vartheta^y_0 Y_0 + \vartheta^z_0 Z_0])
#     \exp(i[\vartheta^x_1 X_1 + \vartheta^y_1 Y_1 + \vartheta^z_1 Z_1]).
#
# Here we decomposed the exponentials of the vertical elements :math:`y_{1,2}` further by
# splitting them into exponentials acting on the first and second qubit, respectively.
#
# The three parameters :math:`\eta^{x, y, z}` sometimes are called the *Cartan coordinates*
# of :math:`U,` and they can be used, e.g., to assess the smallest-possible duration to
# implement the gate in hardware.
#
# With this result, we can implement a template that can create any two-qubit gate.
# We'll use :class:`~.pennylane.Rot` for the single-qubit exponentials (which changes
# the meaning of the angles, but maintains the coverage) and are allowed to
# split the Cartan subalgebra term :math:`\exp(a)` into three exponentials, as its
# terms commute.
#


def su4_gate(params):
    phi0, phi1, eta, theta0, theta1 = np.split(params, range(3, 15, 3))
    qml.Rot(*phi0, wires=0)
    qml.Rot(*phi1, wires=1)
    qml.IsingXX(eta[0], wires=[0, 1])
    qml.IsingYY(eta[1], wires=[0, 1])
    qml.IsingZZ(eta[2], wires=[0, 1])
    qml.Rot(*theta0, wires=0)
    qml.Rot(*theta1, wires=1)


params = np.random.random(15)
fig, ax = qml.draw_mpl(su4_gate, wire_order=[0, 1])(params)

######################################################################
# And that's a wrap on our application of the KAK decomposition for two-qubit gates!
#
# You may have noticed that this mathematical result only states the existence of a
# decomposition, but does not provide a constructive way of finding
# :math:`y_{1,2}` and :math:`a` for a given gate :math:`U.` For this,
# some additional work is required, as explained in [#kokcu_fdhs]_, for example.
#
# Conclusion
# ----------
#
# In this demo we learned about the KAK decomposition and how it breaks up a
# Lie group using the Cartan decomposition of its Lie algebra.
# This allows us to break down arbitrary quantum gates from that group,
# as we implemented in code for the groups of single-qubit and two-qubit gates,
# :math:`SU(2)` and :math:`SU(4).`
#
# If you are interested in other applications of Lie theory in the field of
# quantum computing, you are in luck! It has been a handy tool throughout the last
# decades, e.g., for the simulation of quantum circuits [#somma]_ [#goh]_ and their
# compression [#kokcu_comp]_ [#gu]_, in quantum optimal control [#dirr]_, and for trainability
# analyses [#fontana]_ [#ragone]_. For Lie algebraic classical simulation of quantum circuits,
# also take a look at the :doc:`g-sim <demos/tutorial_liesim/>` and
# :doc:`(g+P)-sim <demos/tutorial_liesim_extension/>` demos.
#
# References
# ----------
#
# .. [#hall]
#
#     Brian C. Hall
#     "Lie Groups, Lie Algebras, and Representations. An Elementary Introduction"
#     `Graduate Texts in Mathematics, Springer <https://link.springer.com/book/10.1007/978-3-319-13467-3>`__, 2015.
#
# .. [#tu]
#
#     Loring W. Tu
#     "An Introduction to Manifolds"
#     `Universitext, Springer <https://link.springer.com/book/10.1007/978-1-4419-7400-6>`__, 2011.
#
# .. [#arvanitogeorgos]
#
#     Andreas Arvanitogeorgos
#     "An Introduction to Lie Groups and the Geometry of Homogeneous Spaces"
#     `Student Mathematical Library 22 <https://bookstore.ams.org/stml-22>`__, 2003
#
# .. [#helgason]
#
#     Sigurdur Helgason
#     "Differential geometry, Lie groups, and symmetric spaces"
#     `Graduate Studies in Mathematics 34 <https://bookstore.ams.org/gsm-34/>`__, 2001
#
# .. [#goh]
#
#     Matthew L. Goh, Martin Larocca, Lukasz Cincio, M. Cerezo, Frédéric Sauvage
#     "Lie-algebraic classical simulations for variational quantum computing"
#     `arXiv:2308.01432 <https://arxiv.org/abs/2308.01432>`__, 2023.
#
# .. [#somma]
#
#     Rolando D. Somma
#     "Quantum Computation, Complexity, and Many-Body Physics"
#     `arXiv:quant-ph/0512209 <https://arxiv.org/abs/quant-ph/0512209>`__, 2005.
#
# .. [#kokcu_fdhs]
#
#     Efekan Kökcü, Thomas Steckmann, Yan Wang, J. K. Freericks, Eugene F. Dumitrescu, Alexander F. Kemper
#     "Fixed Depth Hamiltonian Simulation via Cartan Decomposition"
#     `arXiv:2104.00728 <https://arxiv.org/abs/2104.00728>`__, 2021.
#     `PRL (closed access) <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.129.070501>`__, 2022.
#
# .. [#kokcu_comp]
#
#     Efekan Kökcü, Daan Camps, Lindsay Bassman, James K. Freericks, Wibe A. de Jong, Roel Van Beeumen, Alexander F. Kemper
#     "Algebraic Compression of Quantum Circuits for Hamiltonian Evolution"
#     `arXiv:2108.03282 <https://arxiv.org/abs/2108.03282>`__, 2021.
#     `PRA (closed access) <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.105.032420>`__, 2022.
#
# .. [#gu]
#
#     Shouzhen Gu, Rolando D. Somma, Burak Şahinoğlu
#     "Fast-forwarding quantum evolution"
#     `Quantum 5 <https://quantum-journal.org/papers/q-2021-11-15-577/>`__, 2021.
#
# .. [#dirr]
#
#     G. Dirr, U. Helmke
#     "Lie Theory for Quantum Control"
#     `GAMM-Mitteilungen 31 <https://onlinelibrary.wiley.com/doi/abs/10.1002/gamm.200890003>`__, 2008.
#
# .. [#fontana]
#
#     Enrico Fontana, Dylan Herman, Shouvanik Chakrabarti, Niraj Kumar, Romina Yalovetzky, Jamie Heredge, Shree Hari Sureshbabu, Marco Pistoia
#     "The Adjoint Is All You Need: Characterizing Barren Plateaus in Quantum Ansätze"
#     `Nat. Commun. 15 <https://www.nature.com/articles/s41467-024-49910-w>`__, 2024.
#
# .. [#ragone]
#
#     Michael Ragone, Bojko N. Bakalov, Frédéric Sauvage, Alexander F. Kemper, Carlos Ortiz Marrero, Martin Larocca, M. Cerezo
#     "A Unified Theory of Barren Plateaus for Deep Parametrized Quantum Circuits"
#     `Nat. Commun. 15 <https://www.nature.com/articles/s41467-024-49909-3>`__, 2024.
#
