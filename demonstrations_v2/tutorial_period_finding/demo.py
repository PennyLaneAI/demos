r"""
Period finding: A problem at the heart of quantum computing
=============================================================

You might have heard that Shor's algorithm is an instance of "period finding". You might also have heard that, more generally, this is an example of an *Abelian hidden subgroup problem* solved by a quantum computer's stunning ability to implement the Fourier transform efficiently for an intractable number of function values. Hidden subgroup problems and quantum Fourier transforms were all the rage in the quantum computing literature in the 2000s.

While trends may have moved on, the idea of extracting group structure from the Fourier spectrum is still at the very core of what quantum computers could be useful for. Scott Aaronson, for example, in his 2022 commentary "How Much Structure Is Needed for Huge Quantum Speedup?" presents the following hierarchy:

.. figure:: ../_static/demonstration_assets/period_finding/aaronson_fig6.png
    :align: center
    :width: 75%
    :target: javascript:void(0);

    Figure 1. Aaronson's 'hierarchy of structure' for quantum algorithms (Fig. 6 in [#Aaronson22]_)

However, group theory is a huge hurdle for even some of the more seasoned quantum enthusiasts. This demo wants to give a glimpse of what this "Abelian structure" is all about. Luckily, the fruit-fly example of a hidden subgroup problem is just the task of finding the period of an integer valued function—something one can appreciate without much group jargon. A fantastic resource to read up on the basics is the review of hidden subgroup problems by Childs & van Dam (2010) [#Childs2010]_.
"""

#####################################################################
# The period finding problem
# -------------------
#
# Consider a function :math:`f(x)` that maps from integers, say the numbers between 0 and 11, to
# some other domain, such as the numbers between 0 and 3. The function is guaranteed to have a
# periodic structure, which means that it repeats after a certain number of integers. We need a
# further technical requirement, which is that the function does not have the same value within a
# period. Here is an example of such a function with a period of 4:
#
# .. figure:: ../_static/demonstration_assets/period_finding/periodic_function.png
#    :align: center
#    :width: 90%
#    :target: javascript:void(0);
#    :figwidth: 80%
#    
#    Figure 2. Example of a discrete periodic function f(x) over the integers x = 0,...,11. 
#    The function only takes the same value when moving exactly 4 integers on the x-axis.
#
# Importantly, we assume that we have *black-box access* to that function. For a python coder, this 
# abstract technical term is actually quite intuitive: imagine some part of your code returns a python function ``f`` 
# that you can call by using integers :math:`x \in \{0,..,11\}` as arguments, like ``f(2)``. However, 
# you have no knowledge of the definition ``def f(x)``. Your only way to
# learn about the function is to evaluate it at every point. We're ultimately interested in cases
# where there are too many x-values to evaluate each function value and recover the period in a
# brute force manner.
#
# What does period finding have to do with groups? Well, in the language of group theory, the
# integers from 0 to 11 (together with an appropriate operation, like addition modulo 12) form a
# so-called *cyclic group*, which is an example of *Abelian* groups that Aaronson referred to above. 
#
# .. admonition:: Abelian group
#     :class: note
#
#     A group is a set of elements that has:
#       1. an operation that maps two elements a and b of the set into a third element of the set, for example c = a + b,
#       2. an "identity element" e such that e + a = a for any element a, and
#       3. an inverse -a for every element a, such that a + (-a) = e.
#
#     A group is called "Abelian" if a + b = b + a for all a and b, otherwise it is called non-Abelian. 
#     In general, Abelian groups are simpler to work with than non-Abelian groups.
#
#     With this definition we can easily see that the group of integers {0,1,...,11} with addition modulo 12 forms an Abelian group: 
#     there is an addition operation such that a + b = b + a, the element 0 is the identity element since 0 + a = a for 
#     all a, and the inverse of element a modulo 12 is 12 - a.
#
# The values {0,4,8} form a *subgroup* that is "generated" by the period 4. Finding the period
# means to find the subgroup. The function is said to "hide" the subgroup, since it has the same
# value on all its elements, effectively labeling them. It is also constant on the *cosets* of
# the subgroups, which are shifted sets of the form {0+s, 4+s, 8+s} (where s is an
# element of the group, or a number between 0 and 11). 
#
# .. figure:: ../_static/demonstration_assets/period_finding/periodic_function_groups.png
#    :align: center
#    :width: 70%
#    :target: javascript:void(0);
#    
#    Figure 3. Periodic function from above, from the perspective of group theory.
#
# This jargon hopefully does not scare you, but should illustrate that
# period finding—and the algorithm we will have a look at here—can be generalised to other groups and
# subgroups. It can even be efficiently run for some instances of non-Abelian groups.
#
# The quantum algorithm to find the period of the function above is really simple. Encode the
# function into a quantum state of the form :math:`\sum_x |x \rangle |f(x) \rangle`, apply the quantum
# Fourier transform on the first register, and measure. We then need to do a bit of post-processing:
# The state we measure, written as an integer, is a multiple of the number of periods
# that "fit" into the x-domain. The number of periods is then the biggest number that divides all the measurement outcomes, 
# and the period is the number of integers divided by the number of periods. 
# In the example above, the quantum algorithm would return random bitstrings that can be
# interpreted as integers {0, 3, 6, 9}. By sampling only two distinct values, say 6 and
# 9, we can determine the largest common divisor as 3, which is the number of periods fitting
# into 12. From there we can recover the number of periods 12/3 = 4.
#
# We'll now implement this recipe, but use a more convenient cyclic group of 16 elements
# {0,..,15}, which can be exactly represented by 4 qubits.


#####################################################################
# Implementation of period finding in PennyLane
# ----------------

import pennylane as qml
qml.capture.enable()
import numpy as np
import matplotlib.pyplot as plt

#####################################################################
# First we need to define the periodic function :math:`f(x)`. As mentioned, this function is considered
# unknown in principle, but we can call it on a classical computer, and—theoretically, by turning the
# classical logical circuit into a reversible quantum circuit—we assume we can also call it on a quantum computer. The call
# on a quantum computer can be implemented in parallel for a superposition of inputs, which is
# part of the trick of this algorithm.


def f(x):
    """
    Function whose period we want to find.

    Args:
      x (int): integer in {0,..,15}

    Returns:
      integer in {0,..,3}
    """
    return x % 8


# let's plot this!
x = range(16)
y = [f(x_) for x_ in x]
fig = plt.figure()
ax = fig.add_axes((0.1, 0.2, 0.8, 0.7))
ax.scatter(x, y)
ax.set_title("Periodic function")
ax.set_ylabel("f(x)")
ax.set_xlabel("x")
fig.text(0.5, 0.05, "Figure 4. Periodic function f(x)", horizontalalignment='center')
plt.show()

#####################################################################
# 
# We will represent the :math:`x` and :math:`f(x)` values of this function as computational basis states
# :math:`| x \rangle | f(x) \rangle`. The amplitudes belonging to that state can be interpreted as a
# "weight" for a specific function value. 
# To give an example, the point :math:`f(2) = 2` can be expressed by preparing a state that has a
# nonzero uniform amplitude at :math:`| 0010 \rangle | 11 \rangle`, but zero amplitudes at the states
# :math:`| 0010 \rangle | 00 \rangle`, :math:`| 0010 \rangle | 01 \rangle`, :math:`| 0010 \rangle | 10 \rangle`.
#
# Since we move between integers and computational basis states, we need two utility conversion functions.


def to_int(binary):
    return int(binary, 2)


def to_binary(integer, n):
    return format(integer, "b").zfill(n)


#####################################################################
# Now we need an oracle that implements the function ``f`` in "quantum parallel". Applied to a
# superposition of inputs, :math:`\sum_x |x \rangle |0 \rangle`, this unitary prepares the state
# :math:`\sum_x |x \rangle |f(x) \rangle`. There are many such unitaries and here we'll somewhat hack it 
# together by defining a matrix that does the job. Of course, in a real application, this 
# would be a quantum circuit defined by a sequence of gates.


def Oracle(f):
    """
    Defines the unitary that implements a function f:{0,..,15} -> {0,..,7}.

    Args:
      f (func): function to implement

    Returns:
      ndarray representing the unitary
    """

    U = np.zeros((2**7, 2**7))

    for x in range(2**4):
        for f_x in range(2**3):
            # we know that the initial state has only support on basis states
            # of the form |x>|0>, and therefore write all necessary information
            # into the entries of the unitary acting on those states
            if f_x == 0:
                i = to_int(to_binary(x, 4) + "0" * 3)
                j = to_int(to_binary(x, 4) + to_binary(f(x), 3))
                U[i, j] = 1
                U[j, i] = 1
            else:
                # in all other cases we use trivial entries
                i = to_int(to_binary(x, 4) + to_binary(f_x, 3))
                U[i, i] = 1

    # check that this is a unitary
    assert np.allclose(U @ np.linalg.inv(U), np.eye(2**7))

    return qml.QubitUnitary(U, wires=range(7))


#####################################################################
# Now we're ready to go. Let's implement the famous period finding algorithm. It consists of a
# quantum routine and a classical post-processing step. As mentioned, the quantum part is simple: prepare the
# desired initial state :math:`\sum_x |x \rangle |f(x) \rangle`, apply the quantum Fourier transform
# onto the :math:`|x\rangle`-register, and measure in the computational basis. Effectively, this
# measures in the "Fourier basis", which is where all the magic happens.
#
# We only have to get two unique samples, from which we can compute the period of the function. For this 
# reason we define a device with 2 shots. We also add some snapshots to the circuit that we will look at later.


dev = qml.device("default.qubit", wires=7)


@qml.set_shots(2)
@qml.qnode(dev)
def circuit():
    """Circuit to implement the period finding algorithm."""

    for i in range(4):
        qml.Hadamard(wires=i)

    qml.Snapshot("initial_state")

    Oracle(f)

    qml.Snapshot("loaded_function")

    qml.QFT(wires=range(4))

    qml.Snapshot("fourier_spectrum")

    return qml.sample(wires=range(4))


# take two samples from the circuit
samples = circuit()

#####################################################################
# The classical post-processing is relatively simple for period finding (whereas 
# for general hidden subgroup problems it requires some less trivial algebra). 
# If you are curious about the details, you'll find them in
# Childs & van Dam (2013) [#Childs2010]_, Section IVa.

from fractions import Fraction
from math import lcm

# convert the bistrings to integers
sample1_int = to_int("".join(str(s) for s in samples[0]))
sample2_int = to_int("".join(str(s) for s in samples[1]))

# get the denominator of the fraction representation
denominator1 = Fraction(sample1_int / (2**4)).denominator
denominator2 = Fraction(sample2_int / (2**4)).denominator

# get the least common multiple
result = lcm(denominator1, denominator2)

print(f"Hidden period: {result}")

#####################################################################
# Yay, we've found the hidden period! Now, of course this is only impressive if we increase the 
# number of qubits. We can find the hidden period of a function defined on :math:`2^n` values in time
# :math:`O(n \mathrm{log}(n))`.


#####################################################################
# A peep at the states involved
# -----------------------------
#
# Somehow the quantum Fourier transform exposed the information we needed. Let's have a closer
# look at the states that were prepared by making use of the snapshots we recorded during the
# circuit simulation.

dev = qml.device("default.qubit", wires=7)
qnode = qml.set_shots(qml.QNode(circuit, dev), shots = 1)
intermediate_states = qml.snapshots(circuit)()

#####################################################################
# We can plot them as discrete functions, where the size of a point indicates the absolute value
# of the amplitude.

import matplotlib.patches as mpatches

x = [i for i in range(2**4) for _ in range(2**3)]
y = list(range(2**3)) * (2**4)

states_to_plot = ["initial_state", "loaded_function", "fourier_spectrum"]

fig, axes = plt.subplots(3, 1, sharex=True, figsize=(7, 6))

for state, ax in zip(states_to_plot, axes):

    real = np.real(intermediate_states[state])
    imag = np.imag(intermediate_states[state])

    # scale the points by the absolute value of the amplitude
    sizes_real = []
    for x_, y_ in zip(x, y):
        idx = to_int(to_binary(x_, 4) + to_binary(y_, 3))
        sizes_real.append(200 * np.abs(real[idx]))

    sizes_imag = []
    for x_, y_ in zip(x, y):
        idx = to_int(to_binary(x_, 4) + to_binary(y_, 3))
        sizes_imag.append(200 * np.abs(imag[idx]))

    ax.scatter(np.array(x), y, s=sizes_real, c="green", alpha=0.5)
    ax.scatter(np.array(x), y, s=sizes_imag, c="pink", alpha=0.5)
    ax.set_title(state)
    ax.set_ylabel("f(x)")

plt.xlabel("x")
green = mpatches.Patch(color="green", label="abs(real part)")
pink = mpatches.Patch(color="pink", label="abs(imaginary part)")
plt.legend(handles=[green, pink])
fig.suptitle("Figure 5. Representing the Fourier spectrum", horizontalalignment='center')
fig.tight_layout()
plt.show()

#####################################################################
# 
# First of all, we see that the oracle really prepared a quantum state that we can interpret as 
# our periodic function ``f``.
#
# Furthermore, the state representing the Fourier spectrum looks actually quite interesting. But
# the important feature of this state is also clearly visible: Amplitudes concentrate in :math:`x`
# values that are multiples of 2. And 2 was exactly the amount of periods of 8 that fit into 16.

#####################################################################
# What is the "Abelian structure" exploited by the quantum Fourier transform?
# ----------------------------------------------------------------------------
#
# Why does the Fourier transform prepare a state with such a convenient structure? Let us have a
# look how it acts on a superposition of inputs :math:`x` for which :math:`f(x)` has the same value, such as
# :math:`\frac{1}{\sqrt{2}} (|2\rangle + |10\rangle)` (where we translated bitstrings to integers for
# better readability). The quantum Fourier transform prepares a new state whose amplitudes are
# proportional to
#
# .. math::
#
#     \sum_k \left(e^{\frac{2 \pi i 2 k}{16}} + e^{\frac{2 \pi i 10 k}{16}} \right)  |k \rangle.
#
# In the exponent you find the values 2 and 10, as well as the size of the group, 16.
# Somewhat magically, for some :math:`|k \rangle`, all exponential functions in the sum evaluate to :math:`1`, 
# while for all others, the functions cancel each other out and evaluate exactly to zero.

for k in range(16):
    res = 0.5*(np.exp(2 * np.pi * 1j * 2 * k / 16) + np.exp(2 * np.pi * 1j * 10 * k / 16))
    print(f"k={k} --- {np.round(res, 13)}")

#####################################################################
# This pattern is the same for whichever set of :math:`x` with the same value :math:`f(x)` 
# (in other words, which "coset") we picked. It is also
# true for whichever period we encoded, function we chose, and even which (Abelian) group we
# started with.
#
# The "magic" interference, of course, would not surprise a group theorist; it is inherent to the
# structure of the group. The functions :math:`e^{\frac{2 \pi i x k}{16}}`, which we are so used to
# see in quantum theory, are nothing but the *characters* of a group: functions that take group
# elements :math:`x` to complex numbers, and -- a little like eigenvalues in linear algebra -- capture
# the essentials of the group. The destructive interference we see in the quantum Fourier
# transform is nothing other than the orthogonality theorem for these characters.
#
# It is strangely beautiful how quantum interference leverages the structure of Abelian groups in
# the quantum Fourier transform. So far this has only found application for rather abstract
# problems. Even though cryptography, impacted by Shor's algorithm, is what one might consider a
# "real-world" application, it is an outlier, since problems in cryptography are artificially
# *constructed* from abstract mathematical theory. A fascinating question is whether "real"
# real-world applications, like material science or data analysis, could benefit from this
# remarkable confluence of group structure and quantum theory.
#
#
# References
# ------------
#
# .. [#Aaronson22] 
#    
#     Scott Aaronson, "How Much Structure Is Needed for Huge Quantum Speedups?", 
#     `arXiv:2209.06930 <https://arxiv.org/pdf/2209.06930>`__, 2022
# 
# .. [#Childs2010] 
#    
#     Andrew Childs, Vim van Dam, "Quantum algorithms for algebraic problems", 
#     `Reviews of Modern Physics 82.1: 1-52. <https://journals.aps.org/rmp/abstract/10.1103/RevModPhys.82.1>`__, 2010
#
