r"""
It's all about groups: From Fast Fourier Transforms to QFTs
=========================================================================

Quantum Fourier Transforms (QFTs) are unitary operations that turn a quantum state :math:`\sum_x f(x) |x \rangle` of amplitudes :math:`f(x)` into
another quantum state whose amplitudes are the Fourier coefficients :math:`\hat{f}(x)` of :math:`f(x)`. They appear
literally everywhere in quantum computing: even if you're not interested in `Shor's algorithm
<https://pennylane.ai/codebook/shors-algorithm/shors-algorithm>`__, `hidden subgroup problems <https://pennylane.ai/qml/demos/tutorial_period_finding>`__,
`quantum phase estimation <https://pennylane.ai/qml/demos/tutorial_qpe>`__
or `quantum arithmetics <https://pennylane.ai/qml/demos/tutorial_qft_arithmetics>`__,
you probably have worked with a circuit that applies Hadamard gates to each qubit.
Well, a layer of Hadamards is a Quantum Fourier Transform as well! The reason why you might not have appreciated this fact is that Hadamards do not form the
`famous QFT <https://pennylane.ai/qml/demos/tutorial_qft>`__ we know from Nielsen \& Chuang's standard textbook. They move into a
Fourier basis nevertheless---only of a different *group*.

.. figure:: ../_static/demonstration_assets/qft_groups/pennylane-demo-fft-to-qft-hadamard-image.png
    :align: center
    :width: 100%
    :target: javascript:void(0);

    Figure 1. Applying Hadamards to each qubit is a Quantum Fourier Transform, but with respect to the "boolean" group :math:`Z_2^n`.

Sometimes, knowing about the Fourier-theoretic interpretation of a quantum algorithm helps to understand what is
going on under the hood. But group theory comes with a lot of jargon that can be overwhelming at first. This demo
illuminates the fascinating link between (Quantum) Fourier Transforms and groups, for those who have
not taken a course in group theory yet.

We will see that a group can be used to *define* what a Fourier transform (and hence a Quantum Fourier Transform) is, a fact
that explains a lot of seemingly arbitrary assumptions in the standard Fourier transform.
Groups are also implicitly used to design one of the world's most important scientific subroutines, the *Fast Fourier Transform* (FFT),
which is an algorithmic implementation of a Fourier transform that is polynomially faster than the naive one. (This may not
sound like much to a quantum computing researcher, but the difference between quadratic and near-linear runtime is a game changer
when it comes to data analysis.)  Lastly, it turns out [#Moore]_ that the group-based recipe of a Fast Fourier Transform
can be implemented in "quantum parallel", which offers a way to understand why QFTs are exponentially faster!

In short, groups are the fundamental structure behind quantum and classical Fourier transforms, and exploiting this
structure is one of the reasons we believe that quantum computers could change how humans process information!

But let us start with the basics...
"""

#####################################################################
# Fourier transforms through a group-theoretic lens
# -------------------------------------------------
#
# The common discrete Fourier transform
# ++++++++++++++++++++++++++++++++++++++++
#
# Let's focus on the well-known `discrete Fourier transform <https://en.wikipedia.org/wiki/Discrete_Fourier_transform>`__
# for now. As a reminder, this mathematical operation
# transforms a sequence :math:`f_1,...,f_N` of complex numbers into another sequence of complex numbers.
# Sometimes the complex values are written as a
# function :math:`f(x)` evaluated or "sampled" at regular intervals, for example using
# integer values :math:`x \in \{0,...,N-1\}`. The Fourier coefficients are then given as
#
# .. math:: \hat{f}(k) = \frac{1}{\sqrt{N}}\sum_{x} f(x) e^{2 \pi i  \frac{k x}{N}},
#
# where the frequencies :math:`k` are the same integers as the :math:`x` values. Note that the normalisation factor is a
# matter of convention. The expressions :math:`e^{2 \pi i  \frac{k x}{N}}`
# correspond to Fourier basis functions with integer-valued
# frequencies :math:`k`, and a Fourier coefficient :math:`\hat{f}(x_i)`
# can be seen as the projection of :math:`f(x)` onto the :math:`i`'th basis function. The function beyond
# the interval :math:`0,..,N-1` is thought to be "periodically continued", which means that :math:`f(x_i) = f(x_i + N)`.
#
# Let's look at an example of such a Fourier transform:
#

import matplotlib.pyplot as plt
import numpy as np

N = 16

def f(x):
    """Some function over the integers 0,...,N-1."""
    x = x % N
    return 0.5*(x-4)**3

def f_hat(k):
    """Fourier coefficients of f."""
    projection = [f(x) * np.exp(2 * np.pi * 1j * k * x / N)/np.sqrt(N) for x in range(N)]
    return  np.sum(projection)

integers = np.array(range(N))
f_vec = np.array([f(x) for x in integers])
f_hat_vec = np.array([f_hat(k) for k in integers])

# plot this
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4))
ax1.bar(integers, np.real(f_vec), color='dimgray')  # casting to real is needed in case we perform an inverse FT
ax1.set_title(f"function f")
ax2.bar(integers + 0.05, np.imag(f_hat_vec), color='lightpink', label="imaginary part")
ax2.bar(integers, np.real(f_hat_vec), color='dimgray', label="real part")
ax2.set_title("Fourier coefficients")
plt.legend()
plt.tight_layout()
plt.show()

#####################################################################
# But why do---at least if we don't want to incur additional headaches---the :math:`x` values have to be equidistant?
# Why this notion of "periodic continuation"? Why are the basis functions of exponential form? Why
# are the :math:`k` also integers? And what makes the Fourier basis special?
# It turns out that all of these questions have an elegant answer if we
# interpret the function values as a group!
#
# .. admonition:: Group
#     :class: note
#
#     A group is a set of elements that has:
#       1. an operation that maps two elements :math:`a` and :math:`b` of the set into a third element of the set, for example :math:`c = a + b`,
#       2. an "identity element" :math:`e` such that :math:`e + a = a` for any element :math:`a`, and
#       3. an inverse :math:`-a` for every element :math:`a`, such that :math:`a + (-a) = e`.
#
# More precisely, we can consider the :math:`x` as elements from the set of integers :math:`\{0,...,N-1\}`,
# together with a prescription of how to combine two integers to a third from this set. Choosing "addition modulo N" for
# this operation (which means that :math:`(N-1)+1 = 0`), we get the *cyclic group* :math:`Z_N`.
#
# This choice explains the equidistant property: integers are by nature equally spaced in :math:`\mathbb{R}`. It also explains the
# periodic continuation, as :math:`x = x + N` implies :math:`f(x) = f(x +N)`. The Fourier basis functions are
# central group-theoretic objects called the *characters* of the group. Furthermore, the integer-valued frequencies :math:`k` turn out to be elements
# from the so-called "dual group" :math:`\hat{G}`, which in this case looks exactly like the original one.
# We can therefore think of the :math:`k`
# also as integers in :math:`\{0,...,N-1\}` and treat Fourier space as a "mirror image" of the original space---a convenient
# fact that we tend to take for granted, but which does not hold for every group.
#
# .. note::  From a group-theoretic perspective, the Fourier basis has a special property that can be used to *define* what
#            a Fourier transform is: In this basis, the so-called *regular representation* of the group is (block-)diagonal.
#            The regular representation can be thought of as a collection of matrices, one for each group element.
#
#            The blocks in those matrices, when written in the Fourier basis,
#            reveal "invariant subspaces" which capture symmetries modeled by the group. In our case this
#            means that if we take a function
#            :math:`g(x) = a e^{2 \pi i  \frac{k x}{N}}` that lives in the subspace spanned by
#            a Fourier basis function, and shift it by any :math:`h \in Z_N`,
#            then :math:`g(x+h) = a e^{2 \pi i  \frac{k x+h}{N}} = \underbrace{a e^{2 \pi i  \frac{k h}{N}}}_{b} e^{2 \pi i  \frac{k x}{N}} = b e^{2 \pi i  \frac{k x}{N}}`
#            stays in the same subspace.
#
# Changing the group
# ++++++++++++++++++
#
# What happens if we exchange the cyclic group :math:`Z_N` by another one? First of all, if we change to the infinite group :math:`\mathbb{R}`,
# which are just the real numbers under addition, we get the `continuous Fourier transform <https://en.wikipedia.org/wiki/Fourier_transform>`__,
# whose characters or basis functions look similar to the discrete ones.
# We could also change to a direct product of groups, such as :math:`Z_N \times Z_N \times ...` or :math:`\mathbb{R} \times \mathbb{R} \times ...` and get the
# multi-dimensional Fourier transform whose basis functions are products of characters, one for each dimension.
# Choosing :math:`N=2` we get the group :math:`Z_2^n` mentioned above, where we consider :math:`N` "copies" of the binary set :math:`\{0,1\}`.
# This captures boolean logic ubiquitous in quantum algorithms. The characters for :math:`Z_2^n` are of the form
#
# .. math:: e^{i k_0 x_0} \dots e^{i k_{N-1} x_{N-1}}, \;\; k_0,...,k_{N-1} \in \{0,1\}, \; x_0,...,x_{N-1} \in \{0,1\},
#
# where the product simply multiplies the two bits :math:`x_i k_i`. Each character evaluates to either -1 or 1. A
# character measures the *parity* of the subset of bits in
# the binary vector :math:`x` that are 1 in the binary vector :math:`k`. A Fourier coefficient can therefore be
# interpreted as an average subset parity with respect to a function!
#
# The Fourier transform with these characters is also known as the
# "Walsh transform", and reads
#
# .. math::
#           \hat{f}(k) = \sum_{x_0=0}^1 \dots \sum_{x_{N-1}=0}^1 f(x_0 \dots x_{N-1}) e^{i k_0 x_0} \dots e^{i k_{N-1} x_{N-1}}.
#
# The Fourier transform over a single component group :math:`Z_2` can be written as a Hadamard matrix, and the full :math:`N`-dimensional
# version is therefore a tensor product of N Hadamards. This is exactly the reason why a quantum
# circuit that contains a layer of Hadamards moves into a Fourier basis as claimed above.
#
# Let's code up an example of a Fourier transform of a function ``g(x)`` over the group :math:`Z^4_2`. Such a function
# takes bit strings, which means that the Fourier frequencies are bitstrings as well.
#

n = 4
bitstrings = [[int(c) for c in format(i, f'0{n}b')] for i in range(2**n)]

def g(x):
    """Majority function for length-n bitstrings."""
    count_ones = sum(x[i] for i in range(n))
    return count_ones/n

def character(x, k):
    """Fourier basis function for Z_2^n"""
    return np.prod([np.exp(1j * k[i] * x[i]) for i in range(n)])

def g_hat(k):
    """Fourier coefficients of f."""
    projection = [g(x) * character(x, k) for x in bitstrings]
    return  np.sum(projection)


g_vec = np.array([g(x) for x in bitstrings])
g_hat_vec = np.array([g_hat(k) for k in bitstrings])

# plot this
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4))
x_labels = ["".join([str(x_) for x_ in x]) for x in bitstrings]
ax1.bar(range(len(bitstrings)), np.real(g_vec), color='dimgray')  # casting to real is needed in case we perform an inverse FT
ax1.set_xticks(range(len(bitstrings)))
ax1.set_xticklabels(x_labels, rotation=45, ha='right')
ax1.set_title(f"function g")
ax2.bar(np.array(range(len(bitstrings))) + 0.05, np.imag(g_hat_vec), color='lightpink', label="imaginary part")
ax2.bar(range(len(bitstrings)), np.real(g_hat_vec), color='dimgray', label="real part")
ax2.set_xticks(range(len(bitstrings)))
ax2.set_xticklabels(x_labels, rotation=45, ha='right')
ax2.set_title("Fourier coefficients")
plt.legend()
plt.tight_layout()
plt.show()

#####################################################################
# But we can also move to any other compact group.
# This even includes *non-Abelian* groups, which are those where the composition of group elements does not commute, or
# :math:`g_1 g_2 \neq g_2 g_1`. Here, Fourier coefficients become matrix-valued objects, as the characters are replaced
# by matrix-valued functions called *irreducible representations*. (If this interests you, Parsi Diaconis' book [#Diaconis]_ on
# data analysis with group Fourier transforms is a great resource...)
#

#####################################################################
# Transforming amplitudes: QFTs and groups
# ----------------------------------------
#
# As previously mentioned, a Quantum Fourier Transform is just a Fourier transform of the amplitudes of a quantum state.
# You might have already come across the `standard QFT <https://pennylane.ai/qml/demos/tutorial_qft>`_ that implicitly
# uses the group :math:`Z_N`.
# Each computational basis state :math:`|x \rangle` is associated with an integer from the group, and the corresponding amplitude encodes
# the original function value :math:`f(x)`. The QFT then maps to a new quantum state where the computational basis states
# are interpreted as :math:`k` values from the dual group, and the amplitudes are the Fourier coefficients:
#
# .. math:: \sum_{x=0}^{N-1} f(x) |x \rangle \rightarrow \sum_{k=0}^{N-1} \hat{f}(k) |k \rangle.
#
# This works because the Fourier transform, if normalised appropriately, is a unitary transform!
#
# Let us verify this by comparing the DFT of a function, as coded up above, with the result of a quantum circuit
# that implements ``qml.QFT``. However, now we need a function that, written as a vector, can be interpreted
# as a normalised quantum state. We therefore apply a slight modification to our function ``f`` from before.
#

norm_f = np.linalg.norm(np.array([f(x_) for x_ in range(N)]))

def h(x):
    """Normalised version of f above, such that \sum_x |h(x)|^2 = 1."""
    x = x % N
    return f(x) / norm_f

# compute the Fourier coefficients
def h_hat(k):
    """Fourier coefficients of h."""
    projection = [h(x) * np.exp(2 * np.pi * 1j * k * x / N)/np.sqrt(N) for x in range(N)]
    return  np.sum(projection)

#####################################################################
# We can easily code up a quantum circuit that prepares a state :math:`\sum_x f(x) |x\rangle`, applies the QFT and extracts
# the output state vector from the simulator.
#

import pennylane as qml
qml.capture.enable()

dev = qml.device("default.qubit", wires=4, shots=None)

@qml.qnode(dev)
def qft(state):
    """Prepare a state \sum_x f(x) |x> and apply the discrete Fourier transform."""
    qml.StatePrep(state, wires=range(4))
    qml.QFT(wires=range(4))
    return qml.state()


#####################################################################
# Now we can check that the QFT produces a state that is equivalent to the "classical" discrete Fourier transform.
#

h_vec = np.array([h(x) for x in range(N)])
h_hat_vec = np.array([h_hat(x) for x in range(N)])
h_hat_state = qft(h_vec)

print("QFT and DFT coincide:", np.allclose(h_hat_state, h_hat_vec))

#####################################################################
# But of course, ``qml.QFT`` only implements the Fourier transform with respect to the group :math:`Z_N`, which
# interprets computational basis states as integers.
# As mentioned before, we can alternatively interpret bit strings as a collection of binary variables, which changes the group
# to :math:`Z_2^n`. Both map into a Fourier basis, but with respect a different
# underlying structure! And this alternative QFT is just a bunch of Hadamards.
#


@qml.qnode(dev)
def qft2(state):
    """Prepare a state \sum_x f(x) |x> and apply the Fourier transform over Z_2^4."""
    qml.StatePrep(state, wires=range(4))
    for i in range(4):
        qml.Hadamard(wires=i)
    return qml.state()

h_hat_state2 = qft2(h_vec)
print("QFTs over different groups coincide:", np.allclose(h_hat_state, h_hat_state2))

#####################################################################
# The FFT: divide-and-conquer the group structure
# -----------------------------------------------
#
# Ok, Fourier transforms are all about groups, and so are quantum Fourier transforms.
# But how is this used to come up with the Fast Fourier Transform (FFT)? (Remember, the FFT was the
# workhorse implementation of the discrete Fourier transforms that takes "only" time :math:`\mathcal{O}(|G| \log |G|)`
# instead of :math:`\mathcal{O}(|G|^2)` or worse.)
# And how do FFTs give rise to quantum Fourier transforms with *poly-logarithmic* runtimes?
#
# As it turns out, the famous Cooley-Tukey implementation of a Fast Fourier transform can be interpreted as
# performing Fourier transforms on *subgroups* of the original group and stitching them together.
#
# .. admonition:: Subgroup
#    :class: note
#
#    A subgroup is a subset of the original set that, under the group operation, fulfills all group axioms.
#
# These subgroups can sometimes be decomposed into even smaller subgroups, leading to a recursive "divide-and-conquer"
# algorithm. This technique is always possible for Abelian groups, but also for some non-Abelian groups such as the symmetric
# group of permutations. Even more
# important for us, this recursive strategy can be parallelised on a quantum computer, and it is know that every
# FFT gives rise to an efficient QFT [#Moore]_---but more on that later.
#
# We will illustrate the basic idea of the FFT using the cyclic group :math:`Z_{6}`. It will get a little dense, but
# the math is not complicated: all we do is change the variables and apply cosmetic rearrangements to the DFT equation.
#

#######################################################################
# The idea of Cooley-Tukey
# +++++++++++++++++++++++++
#
# Consider the Fourier transform of a function on :math:`Z_{6}`---which for all matters and purposes one can think of
# as a function on the integers from 0 to 5:
#
# .. math::
#           \hat{f}(k) = \sum_{x=0}^5 f(x) e^{\frac{2 \pi i}{6} x k }
#
# The trick of the Cooley-Tukey algorithm is a change of variables such as this:
#
# .. math::
#         \begin{align}
#          x &\rightarrow 3 x_1 + x_2, \quad x_1 = 0,1, \;\; x_2 = 0,1,2 \\
#          k &\rightarrow 2k_2 + k_1, \quad k_1 = 0,1,2 \;\; k_2 = 0,1
#         \end{align}
#
# Implicitly, we are representing the set of integers :math:`\{0,1,2,3,4,5\}` first as:
#
# .. math:: \{3\cdot 0+0, 3\cdot 0+1, 3\cdot 0+2, 3\cdot 1+0, 3\cdot 1+1, 3\cdot 1+2\},
#
# and then as:
#
# .. math:: \{2\cdot 0+0, 2\cdot 0+1, 2\cdot 0+2, 2\cdot 1+0, 2\cdot 1+1, 2\cdot 1+2\}.
#
# While doubling the amount of variables, the
# new variables run over fewer integers. This
# will allow us to "chop" the full Fourier transform into Fourier transforms over :math:`x_1, k_1` only, and is the
# secret of the runtime reduction.
#
# Rewriting the Fourier transform in the new variables reads
#
# .. math::
#           \hat{f}(k_2, k_1) = \frac{1}{\sqrt{3}}  \sum_{x_2=0}^2 e^{\frac{2 \pi i}{6} (2k_2+k_1) x_2} \frac{1}{\sqrt{2}} \sum_{x_1=0}^1 e^{\frac{2 \pi i}{2} x_1 k_1 } f(x_1, x_2).
#
# Besides some reordering, there is one slightly non-trivial identity we used above:
#
# .. math::
#           e^{\frac{2 \pi i}{6} x_1 3(2k_2+k_1) } = e^{\frac{2 \pi i}{2} x_1k_1 } \underbrace{e^{2 \pi i x_1 k_2 }}_{1}  = e^{\frac{2 \pi i}{2} x_1 k_1 }.
#
# Essentially, the change in variables turns the Fourier basis function of period 6 into a Fourier basis function of period 2,
# which makes the dependency on :math:`k_2` disappear. This effectively turns the Fourier transform into a sum of "smaller" Fourier transforms
# over :math:`3x_1`, namely  :math:`\{0, 3\}`, :math:`\{1, 4\}`, and :math:`\{2, 5\}`. These are combined with
# an appropriate "twiddle factor" :math:`e^{\frac{2 \pi i}{6} (2k_2+k_1) x_2}` that weighs the building blocks.
#
# We explicitly implement this with the function used at the very beginning:
#

N = 6

def f2(x1, x2):
    """Function f from before, but with input split
       into two variables, and running from 0,...5."""
    x = 3*x1 + x2
    x = x % N
    return 0.5*(x-4)**3

def f_subgroup(x2, k1):
    """Fourier transform over a smaller group or coset."""
    projection = [f2(x1, x2) * np.exp(2*np.pi*1j * k1*x1/2)/np.sqrt(2) for x1 in range(2)]
    return  np.sum(projection)

def f_hat_fft(k1, k2):
    """Putting the smaller transforms together."""
    res = 0
    for x2 in range(3):
        twiddle_factor = np.exp(2 * np.pi * 1j * (2*k2+k1) * x2 / 6) / np.sqrt(3)
        res += twiddle_factor * f_subgroup(x2, k1)
    return res

f_hat_vec_fft = np.array([f_hat(2*k2+k1) for k1 in range(2) for k2 in range(3)])

#######################################################################
# Let's compare the result of this implementation to how we implemented the DFT before.
#

def f(x):
    """Function f from before, but restricted to the integers 0,...,5."""
    x = x % N
    return 0.5*(x-4)**3

def f_hat(k):
    """Fourier coefficients of f."""
    projection = [f(x) * np.exp(2 * np.pi * 1j * k * x / N)/np.sqrt(N) for x in range(N)]
    return  np.sum(projection)

f_hat_vec = np.array([f_hat_fft(k1, k2) for k1 in range(2) for k2 in range(3)])
print("FFT and previous DFT implementation coincide:", np.allclose(f_hat_vec, f_hat_vec_fft))

#######################################################################
# Quantifying the runtime gains
# +++++++++++++++++++++++++++++
#
# The FFT algorithm devises to compute :math:`\hat{f}(k) = \hat{f}(2k_2+k_1)` in two steps:
#
# 1. First, compute the terms in the inner sum, :math:`\tilde{f}(x_2, k_1) = \sum_{x_1=0}^1 e^{\frac{2 \pi i}{2} x_1 k_1 } f(3 x_1 + x_2)`
#    as done in ``f_subgroup``.
#    There are 9 such terms (as we have 3 options to pick :math:`k_1` and 3 options to pick :math:`x_2`). Each term requires
#    summing over 2 expressions (as :math:`x_1` runs over 2 values). Overall, there are 18 terms involved.
#
# 2. Second, combine the 9 terms using the outer sum :math:`\hat{f}(k_2, k_1) = \sum_{x_2=0}^2 e^{\frac{2 \pi i}{6} (2k_2+k_1) x_2}  \tilde{f}(k_1, x_2)`.
#    This computes :math:`2 \cdot 3=6` Fourier coefficients, and each has to sum over 3 terms (as :math:`x_2` runs over 3 values).
#    Overall, there are 12 terms involved.
#
# Together, the FFT algorithm includes 12 + 18 = 30 terms. The naive sum, instead, would compute the 6 Fourier coefficients
# with a sum over 6 terms each, using :math:`6 \cdot 6=36` terms altogether.
#
# Of course, in this "mini" example the savings are not very
# impressive. However, the overall scaling reduces from :math:`O(N^2)` to :math:`O(N \mathrm{log}N)`, which is a quadratic
# speedup, and made it possible for the FFT to become one of the most important algorithms in science and engineering.
#

#######################################################################
# A group-theoretic interpretation
# ++++++++++++++++++++++++++++++++
#
# As hinted at before, the change of variables that was the core idea behind the Cooley-Tukey implementation of the Fast Fourier Transform is
# in fact a decomposition of the original group, here :math:`Z_{6}`, into a subgroup with the
# elements :math:`\{0,3\}`, and its *cosets* or "shifted copies", :math:`\{1,4\}` and :math:`\{2,5\}` (see [#Maslen]_, [#Rockmore]_).
# The subgroup is "isomorphic to" :math:`Z_{2}`, which loosely speaking means that it "behaves like" :math:`Z_{2}`.
#
# The new variable :math:`x_2` selects *between* cosets, while :math:`x_1` picks *within* a coset.
# For example, for :math:`5=3 \cdot 1 + 2`, we have that :math:`x_2=2` selects the third coset of the subgroup,
# while :math:`x_1=1` selects the second element within the coset. It is a fundamental result in group theory that
# "shifted" copies of a subgroup tile the entire group, which is why the new coordinates can always be used to express any
# element :math:`x \in G`.
#
# .. figure:: ../_static/demonstration_assets/qft_groups/pennylane-demo-fft-to-qft-divide-image.png
#     :align: center
#     :width: 50%
#     :target: javascript:void(0);
#
#     Figure 2. The FFT algorithm divides the group of integers :math:`\{0,...,5\}` into the subgroup :math:`\{0,3\}` isomorphic
#     to :math:`Z_2`, and its copies or "cosets" :math:`\{2,4\}` and :math:`\{3,5\}`.
#
# The splitting of the variable :math:`k` is related to the concept of "restricting a character" to the subgroup.
# Essentially, it allows us to turn the characters, or Fourier basis functions :math:`e^{\frac{2 \pi i}{6} x k }`
# related to the original group, into characters of the subgroup, :math:`e^{\frac{2 \pi i}{2} x_1 k_1 }` by changing the period.
# Surprisingly, such a trick generalises to much more complicated groups, including those where there is no "cyclic" notion of
# ordered integers.
#

#######################################################################
# From FFTs to QFTs
# -----------------
#
# Well then, Fourier transforms are all about groups, and sometimes allow the construction of classical algorithms that
# scale almost-linearly with the group size.
# But we also claimed that
# any FFT can be turned into a QFT which scales logarithmically in the group size.
# In other words, there are efficient quantum algorithms that transform the amplitudes
# of a quantum state according to any FFT.
#
# The generic blueprint is explained in a paper from 2003 [#Moore]_ (when QFTs were all the rage in quantum computing).
# Essentially, and following the example from before, we start with a quantum state of the form
#
# .. math:: \sum_{x_1=0}^1 \sum_{x_2=0}^2 f(x_1, x_2) | x_1 x_2 \rangle | 0 0 \rangle
#
# where :math:`| x_1 x_2 \rangle` encodes the two variables :math:`3x_1 + x_2 = x` in binary representation into
# two different computational basis registers of sufficient size. From here, we first prepare a state that encodes
# the intermediate functions :math:`\tilde{f}(k_1, x_2)`,
#
# .. math:: \sum_{k_1=0}^2 \sum_{x_2=0}^2 \tilde{f}(k_1, x_2) | 0 x_2 \rangle | k_1 0 \rangle
#
# and then move from there to the final state
#
# .. math:: \sum_{k_1=0}^2 \sum_{k_2=0}^1 \hat{f}(k_1, k_2) | 0 0 \rangle | k_1 k_2 \rangle.
#
# (Of course, this is just a didactic sketch, for example, we can be much more frugal with the number of qubits by
# sharing the registers in a clever way.)
# The crucial point is that the "smaller Fourier transforms", :math:`\tilde{f}(k_1, x_2)`, can be computed
# and combined in quantum parallel!
#
# You can also look closely at the standard textbook circuit for the QFT to recognise
# the divide-and-conquer strategy
# of the FFT: the algorithm acts recursively on subgroups and their cosets.
#
# .. figure:: ../_static/demonstration_assets/qft_groups/qft3.jpeg
#     :align: center
#     :width: 75%
#     :target: javascript:void(0);
#
#     Figure 3. Circuit of the standard QFT for 3 qubits.
#
#
# To see this, consider that the most significant bit in a binary representations of the cyclic group,
# say :math:`Z_{2^3}` of elements
#
# .. math::  \{0,...,7\}, \text{ (or } \{000,...,111\}\text{ in binary notation)}
#
# "cuts" the integers into a
# subgroup :math:`Z_{2^2}` with elements
#
# .. math:: \{0,1,2,3\}, \text{ (or } \{000, 001, 010 ,011\}\text{)},
#
# and its coset
#
# .. math:: \{4,5,6,7\}, \text{ (or } \{100, 101, 110,111\}\text{)}.
#
# Hence, applying gates to qubits 1, 2, 3, and then to
# 2, 3 and finally to qubit 3 implements an operation on smaller and smaller subgroups, working the divide-and-conquer
# strategy backwards!
#
# Furthermore, you should find it suspicious that the controlled operations introduce a phase that looks somewhat like a "twiddle factor" used
# to combine the smaller Fourier transforms.
# For example, the controlled operations in the blue box apply a phase
# :math:`e^{\frac{2 \pi i 2^0}{2^3} (2^1 q_2 + 2^0 q_3)}` to the first qubit, where :math:`q_2, q_3\in \{0,1\}` are
# the respective states of the second and third qubit queried by the control.
#
# And lastly, the Hadamard gate, which sums amplitudes,
# combines the "smaller" Fourier transforms performed on the :math:`q_1 = 0` and :math:`q_1=1`
# branches together (with a phase that is taken into account by the previously tuned twiddle factor).
#
# In short, and while the details are more complex than we can cover here, our favourite quantum subroutine, the
# QFT, is nothing but a Fast Fourier Transform in parallel---enabled by groups!
#

#######################################################################
# Conclusion
# ++++++++++++++++++++++++++++++++++++++++++++++++++++
# Congratulations if you made it to the end, this was a lot of material to cover!
# Let's summarise everything we learned as three important insights:
#
# 1. The *Fourier transform* can be defined with respect to functions over groups. The usual discrete Fourier transform, for
#    example, uses the cyclic group of integers.
# 2. The *Fast Fourier Transform* exploits that for some groups, a Fourier transform can be decomposed into "smaller" Fourier transforms
#    that consider subgroups (and their copies).
#    This provides a speedup to nearly-linear runtime in the size of the group.
# 3. The *Quantum Fourier Transform*, which transforms amplitudes of a quantum state,
#    calculates these smaller blocks in "quantum parallel", thereby realising a runtime
#    that is logarithmic in the size of the group.
#
# Group structure and Fourier transforms are important building blocks of quantum algorithms, and a fascinating
# reason for why quantum computers might offer unique ways of information processing. Of course, the caveat is
# that quantum implementations of Fourier transforms cannot compute Fourier coefficients directly. After a QFT, the
# Fourier coefficients are the amplitudes of a quantum state.
# All we can do is sample from the final state, or manipulate it in interesting ways---
# creating a wonderful playground for quantum algorithm design.
#

#######################################################################
# References
# ------------
#
# .. [#Moore]
#    Moore, C., Rockmore, D. and Russell, A., 2006.
#    Generic quantum Fourier transforms. ACM Transactions on Algorithms (TALG), 2(4), pp.707-723.
# 
# .. [#Diaconis]
#    Diaconis, P., 1988. Group representations in probability and statistics.
#    Lecture notes-monograph series, 11, pp.i-192.
#
# .. [#Maslen]
#    Maslen, D.K. and Rockmore, D.N., 2001. The Cooley-Tukey FFT and group theory.
#    Notices of the AMS, 48(10), pp.1151-1160. `<https://library2.msri.org/books/Book46/files/11maslen.pdf>`__
#
# .. [#Rockmore]
#    Rockmore, D.N., 2002, November. Recent progress and applications in group FFTs.
#    In Conference Record of the Thirty-Sixth Asilomar Conference on Signals, Systems and Computers, 2002.
#    (Vol. 1, pp. 773-777). IEEE. `<https://www.cs.dartmouth.edu/~rockmore/nato-1.pdf>`__.
#
