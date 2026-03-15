r""".. role:: html(raw)
   :format: html

Unitary designs
===============

.. meta::
    :property="og:description": Learn about designs and their uses in quantum computing.

    :property="og:image": https://pennylane.ai/qml/_static/demonstration_assets/fano.png

.. related::

    tutorial_haar_measure Understanding the Haar measure

*Author: Olivia Di Matteo — Posted: 07 September 2021. Last updated: 07 September 2021.*


.. note::

   This demo is intended to be a sequel to the
   :doc:`demo about the Haar measure <demos/tutorial_haar_measure>`.
   If you are not familiar with the Haar measure, we recommend going through
   that demo first before exploring this one.

Take a close look at the following mathematical object:

.. figure:: /_static/demonstration_assets/unitary_designs/fano_no_labels.svg
   :align: center
   :width: 30%

|

There are many things we can say about it: it consists of seven points and seven
lines (the circle counts as a line); each line contains three points, and each
point is contained in three lines. Furthermore, any pair of points occur
together in exactly one line. This object, called the `Fano plane
<https://en.wikipedia.org/wiki/Fano_plane>`__, is an instance of a mathematical
structure called a `projective plane
<https://en.wikipedia.org/wiki/Projective_plane>`__, which is just one example
of a `combinatorial design
<https://en.wikipedia.org/wiki/Combinatorial_design>`__. Designs are sets of
objects, and groups of those objects, that satisfy certain balance properties
and symmetries. They have been studied for hundreds of years in a huge variety
of contexts [#Handbook]_, from `error correcting codes
<http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.50.5465>`__, to `card
games <https://homepages.warwick.ac.uk/staff/D.Maclagan/papers/set.pdf>`__, and
even `agriculture
<http://www-groups.mcs.st-and.ac.uk/~rab/histLShand.pdf>`__. So, what about
quantum computing?

Designs are actually quite prevalent in quantum computing. You've almost
certainly come across one before, though you may not have realized it. At the
end of the Haar measure demo, we asked a very important question: "do we always
*need* to sample from the full Haar measure?". The answer to this is "no", and
the reasoning lies in the study of *unitary designs*.

In this demo, you'll learn the definition of :math:`t`-designs, what it means to
generalize them to unitary :math:`t`-designs, and you'll see some canonical
examples of designs in quantum computing. You'll also learn about their
connection with the Haar measure, what it means to *twirl* a quantum channel,
and explore how to leverage 2-designs in PennyLane to compute the average
fidelity of a noisy channel. You will experience directly a situation where we can
use a :math:`t`-design as a shortcut over the full Haar measure to greatly improve
the efficiency of a task 🎉.


From spheres to unitary :math:`t`-designs
-----------------------------------------

Spherical designs
^^^^^^^^^^^^^^^^^

Before diving into unitary designs, let's look at the sphere for some
intuition.  Suppose we have a polynomial in :math:`d` variables, and we would
like to compute its average over the surface of a real, :math:`d`-dimensional
unit sphere, :math:`S(R^d).` We can do so by integrating that function over the
sphere (using the proper measure), but that would be a lot of parameters to
keep track of. 

One could alternatively approximate the average by sampling thousands of points
uniformly at random on the sphere, evaluating the function at those points, and
computing their average value. That will always work, and it will get us close,
but it will not be exact.

In fact, both of those approaches may be overkill in some special cases—if the
terms in the polynomial have the same degree of at most :math:`t,` you can
compute the average **exactly** over the sphere using only a small set of points
rather than integrating over the entire sphere.  That set of points is called a
spherical :math:`t`-design. More formally [#Handbook]_, [#Delsarte]_:

.. admonition:: Definition
    :class: defn

    Let :math:`p_t: \mathcal{S}(R^d)\rightarrow R` be a polynomial in :math:`d`
    variables, with all terms homogeneous in degree at most :math:`t.` A
    set :math:`X = \{x: x \in \mathcal{S}(R^d)\}` is a spherical :math:`t`-design if

    .. math::

        \frac{1}{|X|} \sum_{x \in X} p_t(x) = \int_{\mathcal{S}(R^d)} p_t (u) d\mu(u)

    holds for all possible :math:`p_t,` where :math:`d\mu` is the uniform,
    normalized spherical measure. A spherical :math:`t`-design is also a
    :math:`k`-design for all :math:`k < t.`



Now this is a pretty abstract picture, so let's consider the 3-dimensional
sphere. This definition tells us that if we want to take the average of a
polynomial over a sphere where all terms have the same degree of at most 2, we
can do so using a small, representative set of points called a 2-design,
rather than the whole sphere. Similarly, if all terms of the polynomial have the
same degree of at most 3, we could use a 3-design. 

But what are these representative sets of points?  Since we are using these
points as a stand-in for averaging over the whole sphere, we'd want the points
in the set to be distributed in a way that provides sufficient "coverage". In
the 3-dimensional case, the vertices of some familiar solids form :math:`t`-designs
[#Handbook]_, [#sph4design]_:

.. figure:: /_static/demonstration_assets/unitary_designs/shapes.svg
   :align: center
   :width: 80%

|

We see from these illustrations that spherical designs are sets of
*evenly-spaced points*. As :math:`t` increases, the configurations become
increasingly sphere-like. Looking at this in a different way, the more complex a
function becomes as its degree increases, the closer the :math:`t`-design must
be to a sphere; we need to evaluate the function at more points in order to gain
sufficient information when a function is varying more quickly due to a higher
degree. In 3 dimensions, we can compute the average of a polynomial with degree
2 by evaluating it only at the points of a tetrahedron, despite the fact that it
doesn't look spherical at all. More complex functions require more points
and thus more intricate configurations for the design.  Spherical designs exist
for all :math:`t` and dimension :math:`d` [#Handbook]_. They are not always
unique, and may have varying numbers of points.

To show that this really works, let's look at an explicit example. Consider the
following polynomial in 3 variables:

.. math::

   f(x, y, z) = x^4 - 4 x^3 y + y^2 z^2

We can compute the average value of :math:`f` by integrating over a unit sphere:
the result is :math:`4/15 \approx 0.26667.` However, this integral is
non-trivial to evaluate by hand; the most straightforward way is to convert to
polar coordinates, and even then, it involves integrating functions with 4th and
5th powers of trigonometric functions.

Instead, this is a case where we can leverage the fact that all terms in the
polynomial have degree 4, and compute the average exactly using only a subset of
points that form a 4-design. We choose a dodecahedron for convenience; while
this is actually a 5 design, it also forms a 4-design, and is a more familiar
shape than the 4-design depicted above.

First, we define the set of points that comprise a dodecahedron:

"""

import numpy as np

# The golden ratio
g = (1 + np.sqrt(5)) / 2

# A dodecahedron has 20 points
dodecahedron = np.array([
    # 8 of them form a cube within the sphere
    [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
    [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],

    # 4 of them form a rectangle within the y-z plane
    [0, g, 1/g], [0, g, -1/g], [0, -g, 1/g], [0, -g, -1/g],

    # 4 of them form a rectangle within the x-z plane
    [1/g, 0, g], [1/g, 0, -g], [-1/g, 0, g], [-1/g, 0, -g],

    # 4 of them form a rectangle within the x-y plane
    [g, 1/g, 0],[g, -1/g, 0], [-g, 1/g, 0], [-g, -1/g, 0],
])

# Normalize the points so they all fit in the unit sphere
dodecahedron = np.array(
   [point / np.linalg.norm(point) for point in dodecahedron]
)

######################################################################
# Now we define our function and compute the average over the dodecahedron:

def f(x, y, z):
    return (x ** 4) - 4 * (x ** 3) * y +  y ** 2 * z ** 2

dodeca_average = np.mean([f(*point) for point in dodecahedron])
print(dodeca_average)

######################################################################
# This is exactly the value we expect. What happens if we try to do this using
# only a 3-design, the cube?

# The first 8 points of the dodecahedron are a cube
cube = dodecahedron[:8]

cube_average = np.mean([f(*point) for point in cube])
print(cube_average)


######################################################################
# This clearly differs from the true value. We need a design with :math:`t=4`
# or better in order to compute this average, and when such a design is
# available, we may save significant computational time.
# 
# Unitary designs
# ^^^^^^^^^^^^^^^
# 
# We've learned now that spherical designs are sets of evenly-spaced points, and
# saw how they can be used as a shortcut to evaluate the average of a
# polynomial up to a given degree :math:`t.` However, there was nothing quantum
# about this; there weren't even any complex numbers involved. A *unitary
# design* extends this concept from evenly-distributed points to
# evenly-distributed unitaries.  More formally, instead of averaging polynomials
# over spheres, we consider polynomials that are functions of the entries of
# unitary matrices [#Dankert]_, [#Gross]_.
# 
# .. admonition:: Definition
#     :class: defn
# 
#     Let :math:`P_{t,t}(U)` be a polynomial with homogeneous degree at most :math:`t` in
#     :math:`d` variables in the entries of a unitary matrix :math:`U,` and degree
#     :math:`t` in the complex conjugates of those entries. A unitary
#     :math:`t`-design is a set of :math:`K` unitaries :math:`\{U_k\}` such that
# 
#     .. math::
# 
#         \frac{1}{K} \sum_{k=1}^{K} P_{t,t}(U_k) = \int_{\mathcal{U}(d)}
#         P_{t,t} (U) d\mu(U)
# 
#     holds for all possible :math:`P_{t,t},` and where :math:`d\mu` is the
#     uniform *Haar measure*.
# 
# We stress again that this expression is **exact**. The unitaries in a unitary
# design are a representative set of points that are "evenly spaced" across the
# unitary group. With just a subset of the full group, we can evaluate complex
# expressions that would be otherwise intractable.
# 
# A surprising result about unitary designs is that they exist for all possible
# combinations of :math:`t` and :math:`d` [#Roy]_. There are some known lower
# bounds for the number of unitaries required; for example, a 2-design in
# dimension :math:`d` has at least :math:`d^4 - 2d^2 + 2` elements [#Gross]_,
# [#Roy]_.  However, actually finding the sets (and in particular, finding ones
# with minimal size), is a challenging problem [#Bannai]_, though very recently
# some constructions have been put forward [#Nakata]_.
#
#
# .. admonition:: Fun fact
#
#     Applying the elements of a unitary design to a fixed pure state produces a
#     set of vectors that form a *complex projective design* [#DankertThesis]_.
#     These are much like spherical designs, but they live in a complex vector
#     space. If you've ever studied the characterization of quantum systems, you
#     may have come across some special sets of measurements called mutually
#     unbiased bases (MUBs), or symmetric, informationally complete positive
#     operator valued measurements (SIC-POVMs). Both of these sets of vectors
#     are complex projective 2-designs [#Klappenecker]_.
#
#     .. figure:: /_static/demonstration_assets/unitary_designs/sic-povm.svg
#        :align: center
#        :width: 80%
# 
#        The vectors of the simplest SIC-POVM in dimension 2, plotted on a Bloch sphere.
#
# Unitary :math:`t`-designs in action
# -----------------------------------
#
# Unitary designs come into play in applications that require randomization, or
# sampling of random unitaries—essentially, they can be used as a stand-in for
# the Haar measure. The way in which the unitaries are used in the application may
# place restrictions on the value of :math:`t` that is required; arguably the most
# common is the unitary 2-design.
#
# While in general unitary designs are hard to construct, there are well known
# results for unitary 1-, 2-, and 3-designs based on familiar objects in quantum
# computing. Before we see what those are, let's explore an important use case.
#
# Average fidelity
# ^^^^^^^^^^^^^^^^
# A key application of unitary 2-designs is benchmarking quantum
# operations. Suppose we have a noisy quantum channel :math:`\Lambda` that should
# perform something close to the unitary operation :math:`V.`  What can we say
# about the performance of this channel?
#
# One metric of interest is the *fidelity*. Consider the state :math:`|0\rangle.`
# In an ideal case, we apply :math:`V` and obtain :math:`V|0\rangle.`  But applying the
# channel :math:`\Lambda` gives us something a little different. Since it's noisy,
# we must consider the state as a density matrix. The action of :math:`\Lambda` on
# our starting state is :math:`\Lambda(|0\rangle \langle 0|).`  If :math:`\Lambda`
# was perfect, then :math:`\Lambda(|0\rangle \langle 0|) = V|0\rangle \langle
# 0|V^\dagger`, and the fidelity is
#
# .. math::
# 
#     F(\Lambda, V) = \langle 0 | V^\dagger \cdot \Lambda(|0\rangle \langle 0|) \cdot V|0\rangle = 1.
#
# In reality, :math:`\Lambda` is not going to implement :math:`V` perfectly, and
# :math:`F < 1.` More importantly though, all we've computed so far is the fidelity when
# the initial state is :math:`|0\rangle.` What if the initial state is something
# different? What is the fidelity *on average*?
# 
# To compute an average fidelity, we must do so with respect to the full set
# of Haar-random states. We usually generate random states by applying a
# Haar-random unitary :math:`U` to :math:`|0\rangle.` Thus to compute the average
# over all such :math:`U` we must evaluate
# 
# .. math::
# 
#     \bar{F}(\Lambda, V) = \int_{\mathcal{U}} d\mu(U) \langle 0 | U^\dagger V^\dagger \Lambda(U |0\rangle \langle 0| U^\dagger) V U |0\rangle.
# 
# This is known as *twirling* the channel :math:`\Lambda.` Computing the average
# fidelity in this way would be a nightmare—we'd have to compute the fidelity
# with respect to an infinite number of states!
# 
# However, consider the expression in the integral above. We have an inner product
# involving two instances of :math:`U,` and two instances of
# :math:`U^\dagger.` This means that the expression is a polynomial of degree 2 in
# both the elements of :math:`U` and its complex conjugates—this matches exactly
# the definition of a unitary 2-design. This means that if we can find a set of
# :math:`K` unitaries that form a 2-design, we can compute the average fidelity
# using only a finite set of initial states:
#
# .. math::
# 
#     \frac{1}{K} \sum_{j=1}^K \langle 0 | U_j^\dagger V^\dagger \Lambda(U_j |0\rangle \langle 0|
#     U_j^\dagger) V^\dagger U_j |0\rangle = \int_{\mathcal{U}} d\mu(U) \langle 0
#     | U^\dagger V^\dagger \Lambda(U |0\rangle \langle 0| U^\dagger) V U |0\rangle
#
# This is great, but a question remains: what is the representative set of unitaries?
#
# The Clifford group
# ^^^^^^^^^^^^^^^^^^
#
# A beautiful result in quantum computing is that some special groups you may
# already be familiar with are unitary designs:
#
# - the Pauli group forms a unitary 1-design, and 
# - the Clifford group forms a unitary 3-design. 
#
# By the definition of designs, this means the Clifford group is also a 1-
# and 2-design.
#
# The :math:`n`-qubit Pauli group, :math:`\mathcal{P}(n),` is the set of all tensor
# products of Pauli operations :math:`X`, :math:`Y`, :math:`Z,` and :math:`I.` The
# :math:`n`-qubit Clifford group, :math:`\mathcal{C}(n),` is the *normalizer* of the
# Pauli group. In simpler terms, the Clifford group is the set of operations that
# send Paulis to Paulis (up to a phase) under conjugation i.e.,
#
# .. math::
#
#    C P C^\dagger = \pm P^\prime, \quad \forall P, P^\prime \in \mathcal{P}(n), \quad C \in \mathcal{C}(n).
#
# The Clifford group has some profoundly interesting properties and countless uses
# across quantum computing, from circuit compilation to error correcting
# codes. For a single qubit, the group is built from just two operations. One is
# the Hadamard:
#
# .. math::
#
#    H X H^\dagger = Z, \quad H Y H^\dagger = -Y, \quad H Z H^\dagger = X.
#
# This clearly maps Paulis to Paulis (up to a phase). The other is the phase gate :math:`S:`
#
# .. math::
#
#    S X S^\dagger = Y, \quad S Y S^\dagger = -X, \quad S Z S^\dagger = Z.
#
# If both :math:`H` and :math:`S` map Paulis to Paulis, then products of them do
# as well. In group theory terms, the single-qubit Clifford group is
# generated by :math:`H` and :math:`S.`  For example, consider the action of
# :math:`HS:`
#
# .. math::
#
#    (HS) X (HS)^\dagger = -Y, \quad (HS) Y (HS)^\dagger = -Z, \quad (HS) Z (HS)^\dagger = X.
#
# Since :math:`Y = iXZ,` it is enough to specify Clifford operations by how they
# act on :math:`X` and :math:`Z.`  For a particular Clifford, there are 6 possible
# ways it can transform :math:`X`, namely :math:`\pm X, \pm Y,` or :math:`\pm Z.`  Once
# that is determined, there are four remaining options for the transformation of
# :math:`Z,` leading to 24 elements total.
#
# It takes some work, but you can take combinations of :math:`H` and :math:`S`
# and evaluate their action on :math:`X` and :math:`Z` (or look at their matrix
# representations) until you find all 24 unique elements. The results of
# this endeavour are expressed below as strings:

single_qubit_cliffords = [
 '',
 'H', 'S',
 'HS', 'SH', 'SS',
 'HSH', 'HSS', 'SHS', 'SSH', 'SSS',
 'HSHS', 'HSSH', 'HSSS', 'SHSS', 'SSHS',
 'HSHSS', 'HSSHS', 'SHSSH', 'SHSSS', 'SSHSS',
 'HSHSSH', 'HSHSSS', 'HSSHSS'
]

######################################################################
# To see for yourself how this set of unitaries is evenly distributed, try
# applying each of the Cliffords to the initial state :math:`|0\rangle,` and
# plot the resulting states on the Bloch sphere. You'll find they are
# symmetric and evenly spaced; in fact, they are all eigenstates of :math:`X,`
# :math:`Y,` and :math:`Z.` Furthermore, under the full group action, the result
# is balanced in the sense that each eigenstate is obtained the same number of
# times.
#
# The multi-qubit Clifford group can also be
# specified by only a small set of generators (in fact, only one more
# than is needed for the single-qubit case). Together, :math:`H`, :math:`S,` and
# CNOT (on every possible qubit or pair of qubits) generate the :math:`n`-qubit
# group. Be careful though—the size of the group increases exponentially. The
# 2-qubit group alone has 11520 elements! The size can be worked out in a manner
# analogous to that we used above in the single qubit case: by looking at the
# combinatorics of the possible ways the gates can map Paulis with only
# :math:`X` and :math:`Z` to other Paulis.
#
# An experiment
# ^^^^^^^^^^^^^
# The whole idea of unitary designs may sound too good to be true. Can we
# *really* compute the exact average fidelity using just 24 operations? In this
# section, we put them to the test: we'll compute the average fidelity of an
# operation first with experiments using a large but finite amount of Haar-random
# unitaries, and then again with only the Clifford group.

import pennylane as qml
qml.capture.enable()

# Scipy allows us to sample Haar-random unitaries directly
from scipy.stats import unitary_group

# set the random seed
np.random.seed(42)

# Use the mixed state simulator
dev = qml.device("default.mixed", wires=1)

######################################################################
# Let's set up a noisy quantum channel. To keep things simple, assume it
# consists of applying :class:`~.pennylane.SX`, the square-root of
# :math:`X` gate, followed by a few different types of noise. First, write a
# quantum function for our ideal experiment:

def ideal_experiment():
    qml.SX(wires=0)
    return qml.state()

######################################################################
# Next, we apply some noise. We do so by making use of a relatively new feature
# in PennyLane called `quantum function transforms <https://pennylane.readthedocs.io/en/latest/code/qml_transforms.html>`__. Such transforms work by
# modifying the underlying, low-level quantum tapes which queue the quantum
# operations. Suppose the noisy channel is composed of the following:

def noisy_operations(damp_factor, depo_factor, flip_prob):
    qml.AmplitudeDamping(damp_factor, wires=0)
    qml.DepolarizingChannel(depo_factor, wires=0)
    qml.BitFlip(flip_prob, wires=0)


######################################################################
# Let's create a transform that applies this noise to any quantum function
# *after* the original operations, but before the measurements.  We use the
# convenient :func:`~.pennylane.transform` decorator:

@qml.transform
def apply_noise(tape, damp_factor, depo_factor, flip_prob):
    # Capture the operations from the noisy sequence
    noisy_tape = qml.tape.make_qscript(noisy_operations)(damp_factor, depo_factor, flip_prob)

    # Apply the original operations, then the noisy ones
    noisy_ops = tape.operations + noisy_tape.operations

    def null_postprocessing_fn(results):
        return results[0]

    # Apply the original measurements
    return (type(tape)(noisy_ops, tape.measurements, shots=tape.shots),), null_postprocessing_fn

######################################################################
# We can now apply this transform to create a noisy version of our ideal
# quantum function:

# The strengths of various types of noise
damp_factor = 0.02
depo_factor = 0.02
flip_prob = 0.01

noisy_experiment = apply_noise(ideal_experiment, damp_factor, depo_factor, flip_prob)

######################################################################
# The last part of the experiment involves applying a random unitary matrix
# before all the operations, and its inverse right before the measurements.  We
# can write another transform here to streamline this process:

@qml.transform
def conjugate_with_unitary(tape, matrix):
    new_ops = [
        qml.QubitUnitary(matrix, wires=0),
        *tape.operations,
        qml.QubitUnitary(matrix.conj().T, wires=0),
    ]

    def null_postprocessing_fn(results):
        return results[0]

    return (type(tape)(new_ops, tape.measurements, shots=tape.shots),), null_postprocessing_fn

######################################################################
# Finally, in order to perform a comparison, we need a function to compute the
# `fidelity <https://en.wikipedia.org/wiki/Fidelity_of_quantum_states>`__
# compared to the ideal operation.

from scipy.linalg import sqrtm

def fidelity(rho, sigma):
    # Inputs rho and sigma are density matrices
    sqrt_sigma = sqrtm(sigma)
    fid = np.trace(sqrtm(sqrt_sigma @ rho @ sqrt_sigma))
    return fid.real

######################################################################
# Let's now compute the average fidelity, averaging over 50000 Haar-random unitaries:

n_samples = 50000

fidelities = []

for _ in range(n_samples):
    # Select a Haar-random unitary
    U = unitary_group.rvs(2)

    # Apply transform to construct the ideal and noisy quantum functions
    conjugated_ideal_experiment = conjugate_with_unitary(ideal_experiment, U)
    conjugated_noisy_experiment = conjugate_with_unitary(noisy_experiment, U)

    # Use the functions to create QNodes
    ideal_qnode = qml.QNode(conjugated_ideal_experiment, dev)
    noisy_qnode = qml.QNode(conjugated_noisy_experiment, dev)

    # Execute the QNodes
    ideal_state = ideal_qnode()
    noisy_state = noisy_qnode()

    # Compute the fidelity
    fidelities.append(fidelity(ideal_state, noisy_state))

fid_mean = np.mean(fidelities)
print(f"Mean fidelity = {fid_mean}")

######################################################################
# Now let's repeat the procedure using only Clifford group elements. First, we
# write a quantum function that performs a Clifford operation (or its inverse)
# based on its string representation.

def apply_single_clifford(clifford_string, inverse=False):
    for gate in clifford_string:
        if gate == 'H':
            qml.Hadamard(wires=0)
        else:
            sign = -1 if inverse else 1
            qml.PhaseShift(sign * np.pi/2, wires=0)

######################################################################
# Next, we write a transform that applies a Clifford in the context of the full
# experiment, i.e., apply the Clifford, then the operations, followed by the
# inverse of the Clifford.

@qml.transform
def conjugate_with_clifford(tape, clifford_string):
    make_single_clifford = qml.tape.make_qscript(apply_single_clifford)
    non_inv_tape = make_single_clifford(clifford_string, inverse=False)
    inv_tape = make_single_clifford(clifford_string, inverse=True)

    new_ops = non_inv_tape.operations + tape.operations + inv_tape.operations

    def null_postprocessing_fn(results):
        return results[0]

    return (type(tape)(new_ops, tape.measurements, shots=tape.shots),), null_postprocessing_fn

######################################################################
# You may have noticed this transform has exactly the same form as
# ``conjugate_with_unitary`` from above. Only the input type has changed, since
# the application of Cliffords here is specified by their string representation.
#
# It's now time to run the experiments:

fidelities = []

for C in single_qubit_cliffords:
    conjugated_ideal_experiment = conjugate_with_clifford(ideal_experiment, C)
    conjugated_noisy_experiment = conjugate_with_clifford(noisy_experiment, C)

    ideal_qnode = qml.QNode(conjugated_ideal_experiment, dev)
    noisy_qnode = qml.QNode(conjugated_noisy_experiment, dev)

    ideal_state = ideal_qnode()
    noisy_state = noisy_qnode()

    fidelities.append(fidelity(ideal_state, noisy_state))

######################################################################
# Let's see how our results compare to the earlier simulation:

clifford_fid_mean = np.mean(fidelities)

print(f"Haar-random mean fidelity = {fid_mean}")
print(f"Clifford mean fidelity    = {clifford_fid_mean}")

######################################################################
# Incredible 🤯 🤯 🤯 We were able to compute the average fidelity using only
# 24 experiments. Furthermore, the mean fidelity obtained from the Clifford
# experiments is **exact**; even with 50000 Haar-random experiments, we see
# deviations starting a few decimal places in. Consider the resources that would
# be saved if you were actually implementing this in a lab! It's not hard to see
# why the Clifford group plays such an important role in characterization
# procedures.
#
# Conclusion
# ----------
#
# In this demo, we've barely scratched the surface of designs and their
# applications in quantum computing. While benchmarking is a key application
# area, there are many others.  The Pauli group as a unitary 1-design has
# applications in the construction of private quantum channels [#PQC]_.
# *Approximate* unitary :math:`t`-designs (where the equality in the definition
# is replaced by approximately equal up to some finite precision) are also of
# interest, as there ways to construct them that are more efficient than those
# of exact designs [#Dankert]_. In particular, it has been shown that
# approximate complex projective 4-designs have applications to the state
# discrimination problem [#Ambainis]_.
#
# Furthermore, unitary designs are not the only designs that you'll encounter
# in quantum computing. The familiar Hadamard gate is just a 2-dimensional
# example of a broader family of *Hadamard designs*, on which there has been
# extensive research [#Seberry]_. Some sets of `mutually orthogonal Latin
# squares <https://en.wikipedia.org/wiki/Mutually_orthogonal_Latin_squares>`__
# have a direct correspondence with mutually unbiased bases, which are optimal
# quantum measurements [#Gaeta]_, as well as complex projective designs; and Latin
# squares themselves have direct correspondence with affine and projective
# planes, bringing us full circle back to the Fano plane from which we began.
#
#
# .. figure:: /_static/demonstration_assets/unitary_designs/affine-latin.svg
#    :align: center
#    :width: 80%
#
#    An affine plane, Hadamard matrix, and a depiction of mutually orthogonal Latin squares.
#
# References
# ----------
#
# .. [#Handbook]
#
#     C. J. Colbourn and J. H. Dinitz (2006) *Handbook of Combinatorial Designs,
#     Second Edition*.  Chapman & Hall/CRC.
#
# .. [#Delsarte]
#
#    P. Delsarte, J.M. Goethals, J.J. Seidel (1977) *Spherical Codes and Designs*. Geometriae
#    Dedicata 6 363-388.
#
# .. [#sph4design]
#
#    R. H. Hardin and N. J. A. Sloane (1992) *New spherical 4-designs*. Discrete
#    Mathematics, 106-107 (255-264). `(PDF)
#    <https://www.sciencedirect.com/science/article/pii/0012365X9290552Q>`__.
#
# .. [#Ambainis]
#
#    A. Ambainis and J. Emerson (2007) *Quantum t-designs: t-wise independence
#    in the quantum world.* Twenty-Second Annual IEEE Conference on
#    Computational Complexity 129-140.
#    `(arXiv) <https://arxiv.org/abs/quant-ph/0701126>`__.
#
# .. [#Klappenecker]
#
#    A. Klappenecker and M. Roetteler (2005) *Mutually unbiased bases, spherical
#    designs, and frames.* Proceedings of SPIE Vol. 5914.
#
# .. [#Dankert]
#
#    C. Dankert, R. Cleve, J. Emerson, and E. Levine (2009) *Exact and
#    Approximate Unitary 2-Designs: Constructions and Applications.* Phys. Rev. A 80, 012304.
#    `(arXiv) <https://arxiv.org/abs/quant-ph/0606161>`__.
#
# .. [#DankertThesis]
#
#    C. Dankert (2005) *Efficient Simulation of Random Quantum States and
#    Operators.* MSc Thesis, University of Waterloo. `(arXiv)
#    <https://arxiv.org/abs/quant-ph/0512217>`__.
#
# .. [#Gross]
#
#    D. Gross, K. Audenaert, and J. Eisert (2007) *Evenly distributed unitaries:
#    on the structure of unitary designs*. J. Math. Phys. 48, 052104.
#    `(arXiv) <https://arxiv.org/abs/quant-ph/0611002>`__.
#
# .. [#Roy]
#
#    A. Roy and A. J. Scott (2009) *Unitary designs and codes*. Des. Codes Cryptogr. 53 13-31.
#    `(arXiv) <https://arxiv.org/abs/0809.3813>`__.
#
# .. [#Bannai]
#
#    E. Bannai, M. Nakahara, D. Zhao, and Y. Zhu (2019) *On the explicit constructions of
#    certain unitary t-designs.* J. Phys. A: Math. Theor. 52 495301.
#    `(arXiv) <https://arxiv.org/abs/1906.04583>`__.
#
# .. [#Nakata]
#
#    Y. Nakata et al. (2021) *Quantum circuits for exact unitary t-designs and
#    applications to higher-order randomized benchmarking.* `(arXiv)
#    <https://arxiv.org/abs/2102.12617>`__.
#
# .. [#PQC]
#
#    A. Ambainis, M. Mosca, A. Tapp, and R. de Wolf (2000) *Private Quantum Channels*. Proc.
#    41st FOCS, 547-553. `(PDF) <https://homepages.cwi.nl/~rdewolf/publ/qc/AMTW00.pdf>`__.
#
# .. [#Seberry]
#
#    J. Seberry and M. Yamada (1992) *Hadamard matrices, sequences, and block
#    designs.* Contemporary Design Theory – A Collection of Surveys
#    (D. J. Stinson and J. Dinitz, Eds.), John Wiley and Sons, 431-560.
#    `(PDF) <http://mathscinet.ru/files/YamadaSeberry1992.pdf>`__.
#
# .. [#Gaeta]
#
#    M. Gaeta, O. Di Matteo, A. B. Klimov, and H. de Guise (2014) *Discrete phase-space
#    approach to mutually orthogonal Latin squares*. J. Phys. A: Math. Theor. 47 (43) 435303.
#    `(arXiv) <https://arxiv.org/abs/1408.6742>`__.
#
