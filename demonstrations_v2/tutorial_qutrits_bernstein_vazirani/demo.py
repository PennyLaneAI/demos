r"""

Qutrits and quantum algorithms
==============================

.. meta::
    :property="og:description": Learn how to interpret the Bernstein-Vazirani algorithm with qutrits
    :property="og:image": https://pennylane.ai/qml/_static/demonstration_assets/thumbnail_tutorial_qutrits_bernstein_vazirani.png

.. related::


A qutrit is a basic quantum unit that can exist in a superposition of three possible quantum states, represented as :math:`|0\rangle`, :math:`|1\rangle`, and :math:`|2\rangle,` which functions as a generalization of the qubit.
There are many problems to which we can apply these units, among which we can highlight an improved decomposition of the Toffoli gate.
Using only qubits, it would take at least 6 CNOTs to decompose the gate, whereas with qutrits it would be enough to use 3 [#toffoli_qutrits]_.
This is one of the reasons why it is important to develop the intuition behind this basic unit of information, to see where qutrits can provide an advantage. The goal of this demo is to start working with qutrits from an algorithmic point of view. To do so, we will start with the Bernstein–Vazirani algorithm, which we will initially explore using qubits, and later using qutrits.



Bernstein–Vazirani algorithm
------------------------------

The Bernstein–Vazirani algorithm is a quantum algorithm developed by Ethan Bernstein and Umesh Vazirani [#bv]_.
It was one of the first examples that demonstrated an exponential advantage of a quantum computer over a traditional one. So, in this first section we will understand the problem that they tackled.


Suppose there is some hidden bit string "a" that we are trying to learn, and that we have access to a function :math:`f(\vec{x})` that implements the following scalar product:

.. math::
 f(\vec{x}) := \vec{a}\cdot\vec{x} \pmod 2,

where :math:`\vec{a}=(a_0,a_1,...,a_{n-1})` and :math:`\vec{x}=(x_0,x_1,...,x_{n-1})` are bit strings of length :math:`n` with :math:`a_i, x_i \in \{0,1\}`. Our challenge will be to discover the hidden value of :math:`\vec{a}` by using the function :math:`f.` We don't know anything about :math:`\vec{a}` so the only thing we can do is to evaluate :math:`f` at different points :math:`\vec{x}` with the idea of gaining hidden information.


To give an example, let's imagine that we take :math:`\vec{x}=(1,0,1)` and get the value :math:`f(\vec{x}) = 0`. Although it may not seem obvious, knowing the structure that :math:`f` has, this gives us some information about :math:`\vec{a}.` In this case, :math:`a_0` and :math:`a_2` have the same value. This is because taking that value of :math:`\vec{x}`, the function will be equivalent to :math:`a_0 + a_2 \pmod 2,` which will only take the value 0 if they are equal. 
I invite you to take your time to think of a possible strategy (at the classical level) in order to determine :math:`\vec{a}` with the minimum number of evaluations of the function :math:`f.`

The optimal solution requires only :math:`n` calls to the function! Let's see how we can do this.
Knowing the form of :math:`\vec{a}` and :math:`\vec{x},` we can rewrite :math:`f` as:

.. math::
  f(\vec{x})=\sum_{i=0}^{n-1}a_ix_i \pmod 2.

The strategy will be to deduce one element of :math:`\vec{a}` with each call to the function. Imagine that we want to determine the value :math:`a_i.` We can simply choose :math:`\vec{x}` as a vector of all zeros except a one in the i-th position, since in this case:



.. math::
    f(\vec{x})= 0\cdot a_0 + 0\cdot a_1 + ... + 1\cdot a_i + ... + 0\cdot a_{n-1} \pmod 2 \quad= a_i.

It is trivial to see, therefore, that :math:`n` evaluations of :math:`f` are needed. The question is: can we do it more efficiently with a quantum computer? The answer is yes, and in fact, we only need to make one call to the function! 

The first step is to see how we can represent this statement in a circuit. In this case, we will assume that we have an oracle :math:`U_f` that encodes the function, as we can see in the picture below.


.. figure:: ../_static/demonstration_assets/qutrits_bernstein_vazirani/oracle_qutrit.jpg
   :scale: 35%
   :alt: Oracle definition.
   :align: center

   Oracle representation of the function.


In general, :math:`U_f` sends the state :math:`|\vec{x} \rangle |y\rangle` to the state :math:`| \vec{x} \rangle |y + \vec{a} \cdot \vec{x} \pmod{2} \rangle.`

Suppose, for example, that :math:`\vec{a}=[0,1,0]`. Then :math:`U_f|111\rangle |0\rangle = |111\rangle|1\rangle`, since we are evaluating :math:`f` at the point :math:`\vec{x} = [1,1,1]`. The scalar product between the two values is :math:`1,` so the last qubit of the output will take the value :math:`1.`

The Bernstein–Vazirani algorithm makes use of this oracle according to the following circuit:

.. figure:: ../_static/demonstration_assets/qutrits_bernstein_vazirani/bernstein_vazirani_algorithm.jpg
   :scale: 35%
   :alt: Bernstein-Vazirani's algorithm
   :align: center

   Bernstein–Vazirani algorithm.


What we can see is that, by simply using Hadamard gates before and after the oracle, after a single run, the output of the circuit is exactly the hidden value of :math:`\vec{a}.` Let's do a little math to verify that this is so.

First, the input to our circuit is :math:`|0001\rangle.` The second step is to apply Hadamard gates to this state, and for this we must use the following property:

.. math::
    H^{\otimes n}|\vec{x}\rangle = \frac{1}{\sqrt{2^n}}\sum_{\vec{z} \in \{0,1\}^n}(-1)^{\vec{x}\cdot\vec{z}}|\vec{z}\rangle.

Taking as input the value :math:`|0001\rangle,` we obtain the state

.. math::
    |\phi_1\rangle=H^{\otimes 4}|0001\rangle = H^{\otimes 3}|000\rangle\otimes H|1\rangle = \frac{1}{\sqrt{2^3}}\left(\sum_{z \in \{0,1\}^3}|\vec{z}\rangle\right)\left(\frac{|0\rangle-|1\rangle}{\sqrt{2}}\right).

As you can see, we have separated the first three qubits from the fourth for clarity.
If we now apply our operator :math:`U_f,`

.. math::
  |\phi_2\rangle= U_f |\phi_1\rangle = \frac{1}{\sqrt{2^3}}\left(\sum_{\vec{z} \in \{0,1\}^3}|\vec{z}\rangle\frac{|\vec{a}\cdot\vec{z} \pmod 2\rangle-|1 + \vec{a}\cdot\vec{z} \pmod 2\rangle}{\sqrt{2}}\right).

Depending on the value of :math:`f(\vec{x}),` the final part of the expression can take two values and it can be checked that

.. math::
  |\phi_2\rangle = \frac{1}{\sqrt{2^3}}\left(\sum_{\vec{z} \in \{0,1\}^3}|\vec{z}\rangle(-1)^{\vec{a}\cdot\vec{z}}\frac{|0\rangle-|1\rangle}{\sqrt{2}}\right).

This is because, if :math:`\vec{a}\cdot\vec{z}` takes the value :math:`0`, we will have the :math:`\frac{|0\rangle - |1\rangle}{\sqrt{2}}`, and if it takes the value :math:`1,` the result will be :math:`\frac{|1\rangle - |0\rangle}{\sqrt{2}} = - \frac{|0\rangle - |1\rangle}{\sqrt{2}}.` Therefore, by calculating :math:`(-1)^{\vec{a}\cdot\vec{z}}` we cover both cases.
After this, we can include the :math:`(-1)^{\vec{a}\cdot\vec{z}}` factor in the :math:`|\vec{z}\rangle` term and disregard the last qubit, since we are not going to use it again:

.. math::
    |\phi_2\rangle =\frac{1}{\sqrt{2^3}}\sum_{\vec{z} \in \{0,1\}^3}(-1)^{\vec{a}\cdot\vec{z}}|\vec{z}\rangle.

Note that you cannot always disregard a qubit. In cases where there is entanglement with other qubits that is not possible, but in this case the state is separable.

Finally, we will reapply the property of the first step to calculate the result after using the Hadamard:

.. math::
    |\phi_3\rangle = H^{\otimes 3}|\phi_2\rangle = \frac{1}{2^3}\sum_{\vec{z} \in \{0,1\}^3}(-1)^{\vec{a}\cdot\vec{z}}\left(\sum_{\vec{y} \in \{0,1\}^3}(-1)^{\vec{z}\cdot\vec{y}}|\vec{y}\rangle\right).

Rearranging this expression, we obtain:

.. math::
    |\phi_3\rangle  = \frac{1}{2^3}\sum_{\vec{y} \in \{0,1\}^3}\left(\sum_{\vec{z} \in \{0,1\}^3}(-1)^{\vec{a}\cdot\vec{z}+\vec{y}\cdot\vec{z}}\right)|\vec{y}\rangle.

Perfect! The only thing left to check is that, in fact, the previous state is exactly :math:`|\vec{a}\rangle`. It may seem complicated, but I invite you to demonstrate it by showing that :math:`\langle \vec{a}|\phi_3\rangle = 1.` Let's go to the code and check that it works.

Algorithm coding with qubits
------------------------------

We will first code the classical solution. We will do this inside a quantum circuit to understand how to use the oracle, but we are really just programming the qubits as bits.

"""


import pennylane as qml
qml.capture.enable()

dev = qml.device("default.qubit", wires = 4)

def Uf():
    # The oracle in charge of encoding a hidden "a" value.
    qml.CNOT(wires=[1, 3])
    qml.CNOT(wires=[2 ,3])


@qml.set_shots(1)
@qml.qnode(dev)
def circuit0():
    """Circuit used to derive a0"""


    # Initialize x = [1,0,0]
    qml.PauliX(wires = 0)

    # Apply our oracle

    Uf()

    # We measure the last qubit
    return qml.sample(wires = 3)

@qml.set_shots(1)
@qml.qnode(dev)
def circuit1():
    # Circuit used to derive a1

    # Initialize x = [0,1,0]
    qml.PauliX(wires = 1)

    # We apply our oracle
    Uf()

    # We measure the last qubit
    return qml.sample(wires = 3)

@qml.set_shots(1)
@qml.qnode(dev)
def circuit2():
    # Circuit used to derive a2
    # Initialize x = [0,0,1]
    qml.PauliX(wires = 2)

    # We apply our oracle
    Uf()

    # We measure the last qubit
    return qml.sample(wires = 3)

# We run for x = [1,0,0]
a0 = circuit0()

# We run for x = [0,1,0]
a1 = circuit1()

# We run for x = [0,0,1]
a2 = circuit2()

print(f"The value of 'a' is [{a0},{a1},{a2}]")

##############################################################################
#
# In this case, with 3 queries (:math:`n=3`), we have discovered the value of :math:`\vec{a}.` Let's run the Bernstein–Vazirani subroutine (using qubits as qubits this time) to check that one call is enough:


@qml.set_shots(1)
@qml.qnode(dev)
def circuit():

    # We initialize to |0001>
    qml.PauliX(wires = 3)

    # We run the Hadamards
    for i in range(4):
        qml.Hadamard(wires = i)

    # We apply our function
    Uf()

    # We run the Hadamards
    for i in range(3):
        qml.Hadamard(wires = i)

    # We measure the first 3 qubits
    return qml.sample(wires = range(3))

a = circuit()

print(f"The value of a is {a}")


##############################################################################
# Great! Everything works as expected, and we have successfully executed the Bernstein–Vazirani algorithm.
# It is important to note that, because of how we defined our device, we are only using a single shot to find this value!
#
# Generalization to qutrits
# ------------------------------
#
# To make things more interesting, let's imagine a new scenario. We are given a function of the form :math:`f(\vec{x}) := \vec{a}\cdot\vec{x} \pmod 3` where, :math:`\vec{a}=(a_0,a_1,...,a_{n-1})` and :math:`\vec{x}=(x_0,x_1,...,x_{n-1})` are strings of length :math:`n` with :math:`a_i, x_i \in \{0,1,2\}`. How can we minimize the number of calls to the function to discover :math:`\vec{a}?` In this case, the classical procedure to detect the value of :math:`\vec{a}` is the same as in the case of qubits: we will evaluate the output of the inputs :math:`[1,0,0],` :math:`[0,1,0]` and :math:`[0,0,1].`
#
# But how can we work with these kinds of functions in a simple way? To do this we must use a qutrit and its operators.
# By using this new unit of information and unlocking the third orthogonal state, we will have states represented with a vector of dimension :math:`3^n` and the operators will be :math:`3^n \times 3^n` matrices where :math:`n` is the number of qutrits.
# Specifically, we will use the :class:`~.pennylane.TShift` gate, which is equivalent to the :class:`~.pennylane.PauliX` gate for qutrits. It has the following property:
#
# .. math::
#   \text{TShift}|0\rangle = |1\rangle
#
# .. math::
#   \text{TShift}|1\rangle = |2\rangle
#
# .. math::
#   \text{TShift}|2\rangle = |0\rangle
#
# This means we can use this gate to initialize each of the states.
# Another gate that we will use for the oracle definition is the :class:`~.pennylane.TAdd` gate, which is the generalization of the :class:`~.pennylane.Toffoli` gate for qutrits.
# These generalizations simply adjust the addition operation to be performed in modulo 3 instead of modulo 2.
# So, with these ingredients, we are ready to go to the code.

dev = qml.device("default.qutrit", wires=4)

def Uf():
    # The oracle in charge of encoding a hidden "a" value.
    qml.TAdd(wires = [1,3])
    qml.TAdd(wires = [1,3])
    qml.TAdd(wires = [2,3])

@qml.set_shots(1)
@qml.qnode(dev)
def circuit0():

    # Initialize x = [1,0,0]
    qml.TShift(wires = 0)

    # We apply our oracle
    Uf()

    # We measure the last qutrit
    return qml.sample(wires = 3)

@qml.set_shots(1)
@qml.qnode(dev)
def circuit1():

    # Initialize x = [0,1,0]
    qml.TShift(wires = 1)

    # We apply our oracle
    Uf()

    # We measure the last qutrit
    return qml.sample(wires = 3)

@qml.set_shots(1)
@qml.qnode(dev)
def circuit2():

    # Initialize x = [0,0,1]
    qml.TShift(wires = 2)

    # We apply our oracle
    Uf()

    # We measure the last qutrit
    return qml.sample(wires = 3)

# Run to obtain the three trits of a
a0 = circuit0()
a1 = circuit1()
a2 = circuit2()


print(f"The value of a is [{a0},{a1},{a2}]")

##############################################################################
#
# The question is, can we perform the same procedure as we have done before to find :math:`\vec{a}` using a single shot? The Hadamard gate also generalizes to qutrits (also denoted as :class:`~.pennylane.THadamard`), so we could try to simply substitute it and see what happens!
#
#
# The definition of the Hadamard gate in this space is:
#
# .. math::
#   \text{THadamard}=\frac{-i}{\sqrt{3}}\begin{pmatrix}
#   1 & 1 & 1\\
#   1 & w & w^2\\
#   1 & w^2 & w
#   \end{pmatrix},
#
# where :math:`w = e^{\frac{2 \pi i}{3}}.`
# Let's go to the code and see how to run this in PennyLane.


@qml.set_shots(1)
@qml.qnode(dev)
def circuit():

    # We initialize to |0001>
    qml.TShift(wires = 3)

    # We run the THadamard
    for i in range(4):
        qml.THadamard(wires = i)

# We run the oracle
    Uf()

# We run the THadamard again
    for i in range(3):
        qml.THadamard(wires = i)

    # We measure the first 3 qutrits
    return qml.sample(wires = range(3))

a = circuit()

print(f"The value of a is {a}")

##############################################################################
#
# Awesome! The Bernstein–Vazirani algorithm generalizes perfectly to qutrits! Let's do the mathematical calculations again to check that it does indeed make sense.
#
# As before, the input of our circuit is :math:`|0001\rangle.`
# We will then use the Hadamard definition applied to qutrits:
#
# .. math::
#          H^{\otimes n}|\vec{x}\rangle = \frac{1}{\sqrt{3^n}}\sum_{\vec{z} \in \{0,1,2\}^n}w^{\vec{x}\cdot\vec{z}}|\vec{z}\rangle.
#
# In this case, we are disregarding the global phase of :math:`-i` for simplicity.
# Applying this to the state :math:`|0001\rangle,` we obtain
#
# .. math::
#       |\phi_1\rangle=H^{\otimes 4}|0001\rangle = H^{\otimes 3}|000\rangle\otimes H|1\rangle = \frac{1}{\sqrt{3^3}}\left(\sum_{z \in \{0,1,2\}^3}|\vec{z}\rangle\frac{|0\rangle+w|1\rangle+w^2|2\rangle}{\sqrt{3}}\right).
#
# After that, we apply the operator :math:`U_f` to obtain
#
# .. math::
#       |\phi_2\rangle= U_f |\phi_1\rangle = \frac{1}{\sqrt{3^3}}\left(\sum_{\vec{z} \in \{0,1,2\}^3}|\vec{z}\rangle\frac{|0 + \vec{a}\cdot\vec{z} \pmod 3 \rangle+w|1+ \vec{a}\cdot\vec{z} \pmod 3 \rangle+w^2|2+ \vec{a}\cdot\vec{z} \pmod 3 \rangle}{\sqrt{3}}\right).
#
# Depending on the value of :math:`f(\vec{x}),` as before, we obtain three possible states:
#
# - If :math:`\vec{a}\cdot\vec{z} = 0,` we have :math:`\frac{1}{\sqrt{3}}\left(|0\rangle+w|1\rangle+w^2|2\rangle\right).`
# - If :math:`\vec{a}\cdot\vec{z} = 1,` we have :math:`\frac{w^2}{\sqrt{3}}\left(|0\rangle+|1\rangle+w|2\rangle\right).`
# - If :math:`\vec{a}\cdot\vec{z} = 2,` we have :math:`\frac{w}{\sqrt{3}}\left(|0\rangle+w^2|1\rangle+|2\rangle\right).`
#
# Based on this, we can group the three states as :math:`\frac{1}{\sqrt{3}}w^{-\vec{a}\cdot\vec{z}}\left(|0\rangle+w|1\rangle+w^2|2\rangle\right).`
#
# After this, we can enter the coefficient in the :math:`|\vec{z}\rangle` term and, as before, disregard the last qutrit, since we are not going to use it again:
#
# .. math::
#         |\phi_2\rangle =\frac{1}{\sqrt{3^3}}\sum_{\vec{z} \in \{0,1,2\}^3}w^{-\vec{a}\cdot\vec{z}}|\vec{z}\rangle.
#
# Finally, we reapply the THadamard:
#
# .. math::
#         |\phi_3\rangle := H^{\otimes 3}|\phi_2\rangle = \frac{1}{3^3}\sum_{\vec{z} \in \{0,1,2\}^3}w^{-\vec{a}\cdot\vec{z}}\left(\sum_{\vec{y} \in \{0,1,2\}^3}w^{\vec{z}\cdot\vec{y}}|\vec{y}\rangle\right).
#
# Rearranging this expression, we obtain:
#
# .. math::
#          |\phi_3\rangle  = \frac{1}{3^3}\sum_{\vec{y} \in \{0,1,2\}^3}\left(\sum_{\vec{z} \in \{0,1,2\}^3}w^{-\vec{a}\cdot\vec{z}+\vec{y}\cdot\vec{z}}\right)|\vec{y}\rangle.
#
#
# In the same way as before, it can be easily checked that :math:`\langle \vec{a}|\phi_3\rangle = 1` and therefore, when measuring, one shot will be enough to obtain the value of :math:`\vec{a}!`
#
# Conclusion
# ----------
#
# In this demo, we have practised the use of basic qutrit gates such as TShift or THadamard by applying the Bernstein–Vazirani algorithm. In this case, the generalization has been straightforward and we have found that it makes mathematical sense, but we cannot always substitute qubit gates for qutrit gates as we have seen in the demo. To give an easy example of this, we know the property that :math:`X = HZH`, but it turns out that this property does not generalize! The general property is actually :math:`X = H^{\dagger}ZH.` In the case of qubits it holds that :math:`H = H^{\dagger},` but in other dimensions it does not. I invite you to continue practising with other types of algorithms. For instance, will the `Deutsch–Jozsa algorithm <https://en.wikipedia.org/wiki/Deutsch–Jozsa_algorithm>`__ generalize well? Take a pen and paper and check it out!
#
# References
# ----------
#
# .. [#bv]
#
#     Ethan Bernstein, Umesh Vazirani, "Quantum Complexity Theory". `SIAM Journal on Computing Vol. 26 (1997).
#     <https://epubs.siam.org/doi/10.1137/S0097539796300921>`__
#
# .. [#toffoli_qutrits]
#     Alexey Galda, Michael Cubeddu, Naoki Kanazawa, Prineha Narang, Nathan Earnest-Noble, `"Implementing a Ternary Decomposition of the Toffoli Gate on Fixed-FrequencyTransmon Qutrits".
#     <https://arxiv.org/pdf/2109.00558.pdf>`__
#