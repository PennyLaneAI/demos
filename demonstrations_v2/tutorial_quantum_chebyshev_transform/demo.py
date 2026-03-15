r"""Quantum Chebyshev Transform
=============================================================

Looking for ways to leverage the speed of the `quantum Fourier transform <https://pennylane.ai/qml/demos/tutorial_qft/>`__ is a common way to design quantum algorithms with exponential speed ups over classical algorithms. 
Working in the Fourier basis can be a more natural choice than the standard basis for some computations. 
Swapping bases is feasible due to the efficiency of the quantum Fourier transform.
In the paper `"Quantum Chebyshev transform: mapping, embedding, learning and sampling distributions" <https://arxiv.org/abs/2306.17026>`__ [#williams2023]_, the authors describe a different basis, the *Chebyshev basis*, and its associated transformation, the *quantum Chebyshev transform*. 
They demonstrate the use of the Chebyshev basis space in generative modelling of probability distributions. 
Further work also proposes a protocol for learning and sampling multivariate probability distributions that arise in high-energy physics [#delejarza2025]_. 
Crucial to their implementation of the learning models is the quantum Chebyshev transform which utilizes the quantum Fourier transform to allow for fast transformations between the standard and the Chebyshev basis.

In this demo we will show how the quantum Chebyshev transform can be implemented in PennyLane. 
To start, we'll describe what Chebyshev polynomials are, and what the classical discrete Chebyshev transform is. After that, we'll look at the quantum Chebyshev basis and its transform.

.. figure:: ../_static/demo_thumbnails/regular_demo_thumbnails/thumbnail_quantum_chebyshev_transform.png
   :align: center
   :width: 60%
   :target: javascript:void(0)
   :alt: Quantum Chebyshev Transform mapping between the computational basis and the non-uniform Chebyshev basis.

   Figure 1: Quantum Chebyshev transform -- a map between the computational basis and the non-uniform Chebyshev basis.

What are Chebyshev polynomials?
---------------------------------------
The :math:`n` -th order Chebyshev polynomial of the first kind is defined as 

.. math::
  T_n(x) \equiv \cos(n \arccos(x))\,.

`Chebyshev polynomials <https://en.wikipedia.org/wiki/Chebyshev_polynomials>`__ of the first kind are a set of orthogonal polynomials. They are *complete*  on the interval :math:`[-1,1]`, meaning that any polynomial :math:`f(x)` of degree :math:`N` can be written exactly as a series in :math:`T_n(x)` up to order :math:`N` as :math:`f(x) = \sum_{j=0}^N a_j T_j(x)` on that interval.

.. note::
    There are more types of Chebyshev polynomials, but in this demo, we will only discuss those of the first kind. 

We can write out the first few orders explicitly.

.. math::
  T_0(x) &= 1 \\
  T_1(x) &= x \\
  T_2(x) &= 2x^2-1 \\
  T_3(x) &= 4x^3 - 3x \\
  T_4(x) &= 8x^4 - 8x^2 + 1 \\
  &\ \,\vdots \\
  T_{n+1}(x) &= 2xT_n(x) - T_{n-1}(x)\,.

The recursion relation in the last line can be used to compute the next orders. 
Observe that odd and even order :math:`T_n` are odd and even functions, respectively. 
The roots of the :math:`T_n(x)` occur at the values 

.. math::
  x_k^\mathrm{Ch} = \cos\left(\frac{2k+1}{2n}\pi\right)\,, \quad k=0, ..., n-1\,. 

These are known as the `Chebyshev nodes <https://en.wikipedia.org/wiki/Chebyshev_nodes>`__.

.. figure:: ../_static/demonstration_assets/quantum_chebyshev_transform/chebyshev_polynomials.png
    :align: center
    :width: 100%
    :target: javascript:void(0)
    :alt: Plot of first six Chebyshev polynomials, with nodes densely packed near the boundary.

    Figure 2. The first six Chebyshev polynomials, along with their corresponding nodes.

The nodes are plotted above along with the corresponding polynomials. 
Note that the polynomials are normalized such that $T_n(1)=1$, and they satisfy the following discrete orthogonality condition on the nodes of :math:`T_N(x)` for :math:`k, \ell<N`

.. math::
  \sum^{N-1}_{j=0}T_k(x_j^\mathrm{Ch})T_\ell(x_j^\mathrm{Ch}) =  
    \begin{cases}
      0 & k \neq \ell\,,\\
      N & k = \ell = 0\,,\\
      N/2 & k = \ell \neq 0\,.
    \end{cases}

The Chebyshev polynomials have a lot of *nice* properties that make them a good choice for expanding a function into a series on a real interval. They are often utilized in numerical integration and interpolation methods, where the `discrete Chebyshev transformation <https://en.wikipedia.org/wiki/Discrete_Chebyshev_transform>`__  is needed to map between the function evaluated on a grid of points and the coefficients of the Chebyshev expansion of that function. Let's take a look at the discrete transform.

Discrete Chebyshev transform
---------------------------------------
A common definition of the discrete Chebyshev transform uses the grid of Chebyshev nodes, also known as the roots grid, as the sampling points. The transform computes the coefficients for the :math:`(N-1)` -degree expansion of a function :math:`f(x)` sampled at the :math:`N` nodes of :math:`T_N(x)`. That is, the :math:`j` -th coefficient, for :math:`0<j<N`, is

.. math::
    a_j = \frac{2}{N}\sum_{k=0}^{N-1} f(x_k^\mathrm{Ch}) T_j(x_k^\mathrm{Ch})\,,

and for :math:`j=0`, there is a slightly different normalization factor

.. math::
    a_0 = \frac{1}{N}\sum_{k=0}^{N-1} f(x_k^\mathrm{Ch})\,.

The inverse of the transform is then just the series expansion evaluated on the grid

.. math::
    f(x_k^\mathrm{Ch}) = \sum_{j=0}^{N-1} a_j T_j(x_k^\mathrm{Ch})\,.

Since a function expanded in Chebyshev polynomials in this way will be sampled on the non-uniformly spaced Chebyshev nodes, it will have more resolution at the boundary than in the middle. In fact, the grid of Chebyshev nodes minimizes the problem of `Runge's phenomenon <https://en.wikipedia.org/wiki/Runge%27s_phenomenon>`__. This can be beneficial if you are, for example, solving a differential equation and expect more interesting features at the boundary. Furthermore, the Chebyshev expansion gives the best polynomial approximation to a continuous function under the `maximum norm <https://en.wikipedia.org/wiki/Uniform_norm>`__.

In general, computing the expansion of a function in a complete set on a classical computer for a discrete number of sampling points would take :math:`\mathcal{O}(N^2)` operations 🐌. 
However, because of the way the Chebyshev polynomials are defined in terms of cosine, the discrete Chebyshev transformation is related to the `discrete cosine transform <https://en.wikipedia.org/wiki/Discrete_cosine_transform>`__. This allows the discrete Chebyshev transform to be implemented in a way that leverages the efficiency of the `fast-Fourier-transform <https://en.wikipedia.org/wiki/Fast_Fourier_transform>`__-style algorithms for expansion, which take :math:`\mathcal{O}(N \log N)` operations 🚀. 

We can see the relation to the cosine transform by plugging in the definition of the Chebyshev polynomials and the nodes into the inverse transform and simplifying. Starting with the polynomials

.. math::
    f(x_k^\mathrm{Ch}) = \sum_{j=0}^{N-1} a_j \cos\left(j \arccos(x_k^\mathrm{Ch})\right)\,,

then, using the definition of the nodes we obtain 

.. math:: 
    f(x_k^\mathrm{Ch}) = \sum_{j=0}^{N-1} a_j \cos\left(\frac{j\pi}{N}(N + k + 1/2)\right)\,.

Finally, we can use the cyclical property of cosine to convert a :math:`j \pi` term in the argument to a :math:`(-1)^{j}` factor in the coefficient, resulting in

.. math:: 
    f(x_k^\mathrm{Ch}) = \sum_{j=0}^{N-1} a_j (-1)^{j}\cos\left(\frac{j\pi}{N}(k + 1/2)\right)\,,

which looks just like a discrete cosine transform.

The quantum analogue of the discrete Chebyshev transform, the quantum Chebyshev transform, inherits the relation to the Fourier transform, allowing the transform to be designed efficiently by utilizing the `quantum Fourier transform <https://pennylane.ai/qml/demos/tutorial_qft/>`__. Next we will discuss the quantum Chebyshev basis, where the Chebyshev polynomials appear in the state amplitudes.


Quantum Chebyshev basis
---------------------------------------
We can define the :math:`j` -th Chebyshev basis state using :math:`N` qubits as

.. math::
  |\tau(x_j^\mathrm{Ch})\rangle = \frac1{2^{N/2}}T_0(x_j^\mathrm{Ch})|0\rangle + \frac1{2^{(N-1)/2}}\sum_{k=1}^{2^N-1}T_k(x_j^\mathrm{Ch})|k\rangle\,,

where :math:`|k\rangle` are the computational basis states and :math:`x_j^\mathrm{Ch}` is the :math:`j` -th node of the Chebyshev polynomial of order :math:`2^N-1`. 
Notice how the amplitudes of the basis state components are the Chebyshev polynomials evaluated at the :math:`j` -th Chebyshev node, and that the normalization of the :math:`|0\rangle` component is different from the rest, like in the classical transform.
Due to the orthogonality of the Chebyshev polynomials and the normalization factors used, this construction guarantees the states are orthonormal, that is

.. math::
  \langle\tau(x_j^\mathrm{Ch})|\tau(x_{j'}^\mathrm{Ch})\rangle = 
    \begin{cases}
      0 & j \neq j'\,,\\
      1 & j = j'\,.
    \end{cases}

The quantum Chebyshev transform circuit described in Ref. [#williams2023]_ maps computational basis states :math:`\{|x_j\rangle\}_{j=0}^{2^N-1}` to Chebyshev basis states :math:`\{|\tau(x_j^\mathrm{Ch})\rangle\}_{j=0}^{2^N-1}`. Our goal is to design a circuit in PennyLane that applies the operation :math:`\mathcal{U}_\mathrm{QChT} = \sum_{j=0}^{2^N-1} |\tau(x_j^\mathrm{Ch})\rangle\langle x_j|`.


Designing the transform circuit
---------------------------------------
Let's start from the end and look at the circuit diagram generated from the code we want to write. 
An auxiliary qubit is required, which will be the :math:`0` indexed qubit, and the rest compose the state :math:`|x\rangle` which starts in the computational basis, shown below as :math:`|\psi\rangle`. We demonstrate for :math:`N=4` non-auxiliary qubits. 

.. figure:: ../_static/demonstration_assets/quantum_chebyshev_transform/qcht_circuit_diagram.png
    :align: center
    :width: 100%
    :target: javascript:void(0)
    :alt: Quantum Chebyshev transform circuit diagram drawn using PennyLane

    Figure 3. Quantum Chebyshev transform circuit. 

The intuition for the structure of the above circuit comes from the link between the discrete Chebyshev transform and the discrete cosine transform. 
Notice the use of the `quantum Fourier transform (QFT) <https://pennylane.ai/qml/demos/tutorial_qft/>`__ applied on all qubits. 
The quantum Chebyshev transform is an extended QFT circuit with some added interference and mixing of the elements. 
Note the auxiliary qubit starts and ends in the state :math:`|0\rangle`, and the amplitudes of the transformed state are all real valued.
Let's break down the circuit above into pieces that we will use inside our circuit function. 

First, a Hadamard gate is applied to the auxiliary qubit, and then a CNOT ladder is applied, controlled on the auxiliary qubit. 
To start, we will define a function for the CNOT ladder.
"""

import pennylane as qml
qml.capture.enable()

# Number of qubits (non-auxiliary qubit).
N = 4


def CNOT_ladder():
    for wire in range(1, N + 1):
        qml.CNOT([0, wire])


#############################################
# After the initial CNOT ladder comes an :math:`N+1` QFT circuit, which can be implemented using ``qml.QFT``.
#
# Next are phase rotations and shifts.
# For the auxiliary qubit, there is a :math:`Z` rotation by and angle of :math:`-\pi(2^N - 1)/2^{N+1}` followed by a phase shift of :math:`-\pi/2^{(N+1)}` .
# The other qubits are rotated in :math:`Z` by :math:`\pi/2^{(j+1)}`, where :math:`j` is the index of the qubit as labelled in the circuit diagram.

import numpy as np

pi = np.pi


def rotate_phases():
    """Rotates and shifts the phase of the auxiliary qubit and rotates the jth qubit by
    pi/2^(j+1) in Z."""
    qml.RZ(-pi * (2**N - 1) / 2 ** (N + 1), wires=0)
    qml.PhaseShift(-pi / 2 ** (N + 1), wires=0)
    for wire in range(1, N + 1):
        qml.RZ(pi / 2 ** (wire + 1), wires=wire)


#############################################
# Now a permutation of the qubits is used to reorder them.
# This is built using a multicontrolled NOT gate applied to each qubit from the initial state, which is controlled on the auxiliary qubit and all qubits with larger index than the target.
# The multicontrolled NOT gate can be implemented using a multicontrolled Pauli X gate.
# Let's see what that looks like.


def permute_elements():
    """Reorders amplitudes of the conditioned states."""
    for wire in reversed(range(1, N + 1)):
        control_wires = [0] + list(range(wire + 1, N + 1))
        qml.MultiControlledX(wires=(*control_wires, wire))


#############################################
# In the above code, we use ``reversed`` to loop over the qubits in reverse order, to apply the controlled gate to the last qubit first.
# After the permutation is another CNOT ladder, which we already have a function for.
#
# The last part is a phase adjustment of the auxiliary qubit: a rotation in :math:`Y` by :math:`\pi/2`, a phase shift of :math:`-\pi/2` and a multicontrolled :math:`X` rotation by :math:`\pi/2`.
# All of the other qubits control the :math:`X` rotation, but the control is sandwiched by Pauli :math:`X` operators.
# We can implement the multicontrolled :math:`X` rotation by using the function ``qml.ctrl`` on ``qml.RX``, specifying the target wire in ``qml.RX`` and the control wires as the second argument of ``qml.ctrl``.


def adjust_phases():
    """Adjusts the phase of the auxiliary qubit."""
    qml.RY(-pi / 2, wires=0)
    qml.PhaseShift(-pi / 2, wires=0)

    # First Pauli X gates.
    for wire in range(1, N + 1):
        qml.PauliX(wires=wire)

    # Controlled RX gate.
    qml.ctrl(qml.RX(pi / 2, wires=0), range(1, N + 1))

    # Second Pauli X gates.
    for wire in range(1, N + 1):
        qml.PauliX(wires=wire)


#############################################
# All together, we can construct the circuit.
# We have added :class:`qml.BasisState` to initialize the input in any computational basis state with the optional argument ``state``.


def QChT():
    """Performs the quantum Chebyshev transform."""
    qml.Hadamard(wires=0)
    CNOT_ladder()
    qml.QFT(wires=range(N + 1))
    rotate_phases()
    permute_elements()
    CNOT_ladder()
    adjust_phases()


dev = qml.device("default.qubit")


@qml.qnode(dev)
def circuit(state=0):
    qml.BasisState(state=state, wires=range(1, N + 1))
    QChT()
    return qml.state()


#############################################
# Finally, we can reproduce the circuit diagram shown at the beginning of this section using ``qml.draw_mpl``.


def circuit_to_draw():
    qml.BasisState(state=0, wires=range(1, N + 1))
    QChT()


fig, ax = qml.draw_mpl(circuit_to_draw, decimals=2, style="pennylane")()
fig.show()

#############################################
# Note we defined a new function for the circuit to simplify the drawing, removing the returned ``qml.state``.
#
# Testing the quantum Chebyshev transform
# ----------------
# With our quantum Chebyshev transform circuit, let's first check if the auxiliary qubit ends in the state :math:`|0\rangle` and the output state amplitudes are real valued. 
# To do this, we'll input the computational basis state :math:`|7\rangle`, which will transform into :math:`|\tau(x_7^\mathrm{Ch})\rangle`.
# We expect the full output state to be :math:`|0\rangle|\tau(x_7^\mathrm{Ch})\rangle`, which means the second half of the amplitude vector should be zero (since there should be no state components with the auxiliary qubit in :math:`|1\rangle`).

j = 7  # Initial state in computational basis.

total_state = circuit(state=j)  # State with auxiliary qubit.

# Round very small values to zero.
total_state = np.where(np.abs(total_state) < 1e-12, 0, total_state)
print(np.round(total_state, 3))

#############################################
# Indeed, we see the second half of the amplitude vector is zero. 
# Furthermore, the first :math:`2^N` entries are real valued, but let's check if the amplitudes of the state components in the computational basis agree with our definition. 
# Remember we expect 
# 
# .. math::
#   |\tau(x_j^\mathrm{Ch})\rangle = \frac1{2^{N/2}}T_0(x_j^\mathrm{Ch})|0\rangle + \frac1{2^{(N-1)/2}}\sum_{k=1}^{2^N-1}T_k(x_j^\mathrm{Ch})|k\rangle\,.
# We can just check the circuit output by computing each state component directly.


def T_n(x, n):
    """Chebyshev polynomial of order n."""
    return np.cos(n * np.arccos(x))


def ch_node(j):
    """The jth node of Chebyshev polynomial 2^N - 1."""
    return np.cos(pi * (2 * j + 1) / 2 ** (N + 1))


def tau_amplitudes(x, k):
    """Computes the state amplitudes of tau in the computational basis."""
    prefactor = np.where(k == 0, 1 / 2 ** (N / 2), 1 / 2 ** ((N - 1) / 2))
    return prefactor * T_n(x, k)


# Reduce state size, effectively removing the auxiliary qubit.
state = np.real(total_state[: 2**N])  # Discard the small imaginary components.

k = range(2**N)  # Computational basis indices.
expected_state = tau_amplitudes(ch_node(j), np.array(k))

assert np.allclose(expected_state, state, atol=1e-9)  # Compare circuit output to calculated values.

#############################################
# The output state from the circuit is exactly what we want.
#
# To give some intuition for how the quantum Chebyshev transform is related to the quantum Fourier transform,
# we can plot the overlap of :math:`|\tau(x_7^\mathrm{Ch})\rangle` with the computational basis states :math:`|k\rangle`. 
# This should look like a cosine plot, since Chebyshev polynomials are just the cosine of an argument which is proportional to :math:`k`. 

import matplotlib.pyplot as plt

plt.style.use("pennylane.drawer.plot")

ks = np.linspace(0, max(k), 100)  # Continuous list of points.

fig = plt.figure(figsize=(6.4, 3.2))
ax = fig.add_axes((0.15, 0.23, 0.80, 0.72))  # Make room for caption.
ax.plot(k, state, "o", label="circuit")
ax.plot(ks, [tau_amplitudes(ch_node(j), kk) for kk in ks], label="interpolation")
ax.set(xlabel=r"$|k\rangle$", ylabel=r"Overlap $\langle k|\tau(x_7^\mathrm{Ch})\rangle$")
ax.legend()
fig.text(0.5, 0.05,
    r"Figure 4. Overlaps of $|\tau(x_7^\mathrm{Ch})\rangle$ with computational basis states.",
    horizontalalignment="center",
    size="small", weight="normal",
)
plt.show()

#############################################
# Notice that, unlike the quantum Fourier transform, these amplitudes are real instead of complex values. Also note that the overlap at :math:`|0\rangle` is discontinuous because the :math:`|0\rangle` amplitude of the Chebyshev state was adjusted in the definition to guarantee orthonormality of the Chebyshev basis.
#
# Next, let's test that orthonormality by computing the overlap at the nodes with all other Chebyshev basis states :math:`|\tau(x_j^\mathrm{Ch})\rangle`.

# Compute the overlap with the other basis states using np.vdot().
js = list(range(int(len(state))))
overlaps = [np.vdot(state, circuit(state=i)[: 2**N]) for i in js]

#############################################
# We can compare these circuit-calculated overlaps to the definition, plotting the squared overlaps at all values of :math:`x`.
# This continuous overlap function can be derived analytically as [#williams2023]_
#
# .. math::
#  |\langle\tau(x_j^\mathrm{Ch})|\tau(x)\rangle|^2 = \frac{\left(T_{2^N+1}(x_j^\mathrm{Ch})T_{2^N}(x)-T_{2^N}(x_j^\mathrm{Ch})T_{2^N+1}(x)\right)^2}{2^{2N}(x_j^\mathrm{Ch}-x)^2}\,,
#
# where :math:`\tau(x)` is a generalization of one of Chebyshev basis states defined earlier and :math:`x` can be any value in :math:`[-1,1]` rather than just one of the nodes.


def overlap_sq(x, xp):
    """Computes the squared overlap between Chebyshev states."""
    numerator = T_n(xp, 2**N + 1) * T_n(x, 2**N) - T_n(xp, 2**N) * T_n(x, 2**N + 1)
    return numerator**2 / (2 ** (2 * N)) / (xp - x) ** 2

fig = plt.figure(figsize=(6.4, 3.2))
ax = fig.add_axes((0.15, 0.25, 0.8, 0.6))  # Make room for the caption.
ax.set(xlabel=r"x", ylabel="Squared Overlap")
ax_top = ax.twiny()

# Plot squared overlaps computed in the circuit.
nodes = [ch_node(i) for i in js]
ax.plot(nodes, np.abs(overlaps) ** 2, marker="o", label="circuit")

# Plot expected squared overlaps.
xs = np.linspace(-1, 1, 1000)
ax.plot(xs, [overlap_sq(x, nodes[j]) for x in xs], label="expectation")

# Set the second axis to show computational basis state indices.
ax_top.set_xlim(ax.get_xlim())
tick_labels = [str(i) for i in js]
tick_labels[0::2] = [''] * len(tick_labels[0::2])  # Omit even-numbered labels.
ax_top.set(xlabel=r"Chebyshev node $j$", 
           xticks=nodes, xticklabels=tick_labels)
ax_top.grid(False)

ax.legend()
fig.text(0.5, 0.05,
    "Figure 5. Squared overlap of Chebyshev states " 
    + r"$|\langle \tau(x_k^\mathrm{Ch})|\tau(x_7^\mathrm{Ch})\rangle|^2$",
    horizontalalignment="center",
    size="small", weight="normal",
)
plt.show()


#############################################
# We can see that the squared overlap between the basis states and the :math:`j=7` state :math:`|\tau(x_7^\mathrm{Ch})\rangle` is 0, unless :math:`x=x_7^\mathrm{Ch}\approx 0.1`, then the overlap is 1.
#
# Conclusion
# ----------
# In this tutorial, we've gone through how to implement the quantum Chebyshev transform from the paper by Williams *et al.*, and tested the circuit output by looking at the state amplitudes and the orthonormality. 
# The properties of Chebyshev polynomials and the speed at which the quantum Chebyshev transform can be implemented make the Chebyshev basis an interesting choice of state space for quantum algorithms, such as generative modelling of probability distributions.
# To build a generative model in the Chebyshev basis, one could implement the quantum Chebyshev feature map from the same paper [#williams2023]_, which prepares a state in the Chevyshev space via a parameter :math:`x``.
#
#
# References
# ----------
#
# .. [#williams2023]
#
#   Chelsea A. Williams, Annie E. Paine, Hsin-Yu Wu, Vincent E. Elfving and Oleksandr Kyriienk. "Quantum Chebyshev transform: mapping, embedding, learning and sampling distributions." `arxiv:2306.17026 <https://arxiv.org/abs/2306.17026>`__ (2023).
#
# .. [#delejarza2025]
#
#   Jorge J. Martínez de Lejarza, Hsin-Yu Wu, Oleksandr Kyriienko, Germán Rodrigo, Michele Grossi. "Quantum Chebyshev probabilistic models for fragmentation functions." `arxiv:2503.16073 <https://arxiv.org/abs/2503.16073>`__ (2025).
#
