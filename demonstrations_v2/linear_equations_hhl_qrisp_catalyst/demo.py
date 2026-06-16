r"""Solving systems of linear equations via HHL using Qrisp and Catalyst
===========================================================

The Harrow-Hassidim-Lloyd (HHL) quantum algorithm offers an exponential speed-up over classical
methods for solving linear system problems :math:`Ax=b` for certain sparse matrices :math:`A`. In
this demo, you will learn how to implement this algorithm in the high-level language
`Qrisp <https://www.qrisp.eu>`__.

The program features hybrid quantum-classical workflows and is
compiled using `Catalyst <https://docs.pennylane.ai/projects/catalyst/en/stable/index.html>`__, a
quantum just-in-time (QJIT) compiler framework. The goal of this demo is to showcase how Qrisp and
Catalyst complement each other for implementing advanced quantum algorithms and compiling them for
practically relevant problem sizes.

.. figure:: ../_static/demo_thumbnails/opengraph_demo_thumbnails/OGthumbnail_linear_equations_hhl_qrisp_catalyst.png
    :align: center
    :width: 70%
    :target: javascript:void(0)

In order to make this demo self-contained, it begins with a brief
introduction to Qrisp, QuantumVariables and QuantumTypes, and the implementation of Quantum Phase
Estimation (QPE). It then shifts into a higher gear with the HHL implementation in Qrisp and showcasing 
of the Catalyst features.

Intro to Qrisp
--------------

Since this demo is featured here, you may not have heard about Qrisp
yet. If you feel addressed with this assumption, please don’t fret—we explain all the concepts
necessary for you to follow along even if not already fluent in Qrisp. So, join us on this Magical
Mystery Tour in the world of Qrisp, Catalyst, and HHL.

What is Qrisp? Qrisp [#qrisp_2024]_ is a high-level open-source programming framework. It offers an alternative
approach to circuit construction and algorithm development with its defining
feature—QuantumVariables. This approach allows the user to step away from focusing on qubits and
gates for circuit construction when developing algorithms, and instead allows the user to code in
terms of variables and functions similarly to how one would program classically.

Qrisp also supports a modular architecture, allowing you to use, replace, and optimize code
components easily. We'll demonstrate this later in this demo.

You can install Qrisp to experiment with this implementation yourself, at your own pace, by calling
``pip install qrisp``. Please note that `qrisp` is only compatible with PennyLane versions
<= 0.41.1 and Catalyst versions <= 0.11.0.

QuantumVariable
~~~~~~~~~~~~~~~

The QuantumVariable is the primary building block in Qrisp. It abstracts away the details of qubit
management, offering a range of features like infix arithmetic syntax and strong typing through
class inheritance. This simplification allows developers to focus on the algorithm logic rather than
the underlying circuit construction details.

Creating your first QuantumVariable is easy:
"""

import jax
import qrisp

qv = qrisp.QuantumVariable(5)

######################################################################
# Here, the number 5 indicates the amount of qubits represented by the QuantumVariable. One can then
# manipulate the QuantumVariable by applying quantum gates:


# Apply gates to the QuantumVariable.
qrisp.h(qv[0])
qrisp.z(qv)  # Z gate applied to all qubits
qrisp.cx(qv[0], qv[3])

# Print the quantum circuit.
print(qv.qs)

######################################################################
# .. rst-class:: sphx-glr-script-out
#
#   .. code-block:: none
#
#      QuantumCircuit:
#      ---------------
#              ┌───┐┌───┐
#      qv_0.0: ┤ H ├┤ Z ├──■──
#              ├───┤└───┘  │
#      qv_0.1: ┤ Z ├───────┼──
#              ├───┤       │
#      qv_0.2: ┤ Z ├───────┼──
#              ├───┤     ┌─┴─┐
#      qv_0.3: ┤ Z ├─────┤ X ├
#              ├───┤     └───┘
#      qv_0.4: ┤ Z ├──────────
#              └───┘
#      Live QuantumVariables:
#      ----------------------
#      QuantumVariable qv_0

######################################################################
# So far, this doesn’t seem that different from what you are used to, but it provides the
# nurturing ground for other neat features like QuantumTypes.
#
# QuantumFloat
# ~~~~~~~~~~~~
#
# Qrisp offers the functionality to represent floating point numbers, both signed and unsigned, up to
# an arbitrary precision.


a = qrisp.QuantumFloat(msize=3, exponent=-2, signed=False)
qrisp.h(a)
print(a)

######################################################################
# .. rst-class:: sphx-glr-script-out
#
#   .. code-block:: none
#
#     {0.0: 0.125, 0.25: 0.125, 0.5: 0.125, 0.75: 0.125, 1.0: 0.125, 1.25: 0.125, 1.5: 0.125, 1.75: 0.125}

######################################################################
# Here, ``msize=3`` indicates the amount of mantissa qubits, which represent the significant digits of
# a number in a floating-point representation, and ``exponent=-2`` indicates, you guessed correctly,
# the exponent.
#
# .. tip::
#     For unsigned QuantumFloats, the decoder function is given by :math:`f_k(i) = i2^k`, where :math:`i`
#     is represented by the mantissa qubits, and
#     :math:`k` is the exponent.
#
# The output is a dictionary of measurement results, where the keys represent the values of the
# QuantumFloat ``a``, and the corresponding values are the measurement probabilities. By applying the
# Hadamard to ``a``, we create a superposition of all the values our QuantumFloat can hold.
#
# Recalling the demo on :doc:`How to use quantum arithmetic operators <demos/tutorial_how_to_use_quantum_arithmetic_operators>`, here
# is how you can do simple arithmetic operations (and gates) out of the gate:


a = qrisp.QuantumFloat(3)
b = qrisp.QuantumFloat(3)

a[:] = 5
b[:] = 2

c = a + b
d = a - c
e = d * b
f = a / b

print(c)
print(d)
print(e)
print(f)

######################################################################
# .. rst-class:: sphx-glr-script-out
#
#   .. code-block:: none
#
#     {7: 1.0}
#     {-2: 1.0}
#     {-4: 1.0}
#     {2.5: 1.0}

######################################################################
# Another QuantumType we will use in this demo are the QuantumBools, representing boolean truth
# values. They can be either used for comparison, or as a return type of comparison operators. We can
# also perform operations on them:


qb = qrisp.QuantumBool()
qrisp.h(qb)
print(qb)

######################################################################
# .. rst-class:: sphx-glr-script-out
#
#   .. code-block:: none
#
#     {False: 0.5, True: 0.5}

######################################################################
# With a second QuantumBool, we can demonstrate and evaluate some logical functions:


qb_1 = qrisp.QuantumBool()
print(qb_1)
print(qb | qb_1)
print(qb & qb_1)

######################################################################
# .. rst-class:: sphx-glr-script-out
#
#   .. code-block:: none
#
#     {False: 1.0}
#     {False: 0.5, True: 0.5}
#     {False: 1.0}

######################################################################
# Comparisons, however, are not limited to only QuantumBools, but also for other types, like the
# previously mentioned QuantumFloats:


a = qrisp.QuantumFloat(4)
qrisp.h(a[3])
qb_3 = a >= 4
print(a)
print(qb_3)
print(qb_3.qs.statevector())

######################################################################
# .. rst-class:: sphx-glr-script-out
#
#   .. code-block:: none
#
#     {0: 0.5, 8: 0.5}
#     {False: 0.5, True: 0.5}
#     sqrt(2)*(|0>*|False> + |8>*|True>)/2

######################################################################
# We can also compare one QuantumFloat to another:


b = qrisp.QuantumFloat(3)
b[:] = 4
comparison = a < b
print(comparison.qs.statevector())

######################################################################
# .. rst-class:: sphx-glr-script-out
#
#   .. code-block:: none
#
#     sqrt(2)*(|0>*|False>*|4>*|True> + |8>*|True>*|4>*|False>)/2

######################################################################
# Jasp
# ~~~~
#
# Jasp is a submodule of Qrisp that allows you to scale up your Qrisp code to practically relevant
# problem sizes. Apart from significantly speeding up compilation, Jasp provides the functionality to
# integrate classical real-time computations.
#
# .. admonition:: What is a real-time computation?
#     :class: note
#
#     A real-time computation is a classical computation that happens during the quantum
#     computation, while the quantum computer stays in superposition. This computation has to happen much
#     faster than the coherence time, so performing that computation by waiting for the Python interpreter
#     is impossible. Real-time computations are essential for many techniques in error correction, such as
#     syndrome decoding or :doc:`magic state distillation <demos/tutorial_magic_state_distillation>`. On the algorithmic level, real-time computations have also
#     become more popular since they are so much cheaper than the quantum equivalent. Examples are
#     Gidney’s adder or repeat until success protocols like HHL.
#
# Jasp is a module that provides `JAX <https://jax.readthedocs.io/en/latest/quickstart.html>`__
# primitives for Qrisp syntax and therefore makes Qrisp JAX-traceable. How does this work in practice?
# The central class here is Jaspr, which is a subtype of Jaxpr. Jaspr objects are intermediate
# representations of Qrisp programs and can be compiled to
# `QIR <https://learn.microsoft.com/en-us/azure/quantum/concepts-qir>`__ using the Catalyst framework.
#
# The Qrisp syntax is intended to work seamlessly in the regular Python mode, primarily for learning,
# testing, and experimenting with small scale quantum programs, as well as in Jasp mode for large scale
# quantum programs with real-time hybrid computations running on fault-tolerant quantum hardware.
#
# QPE in Qrisp
# ~~~~~~~~~~~~
#
# For deepening your understanding of QPE, we would like to refer you to the :doc:`Intro to Quantum Phase Estimation demo <demos/tutorial_qpe>`, and instead focus on how QPE is implemented
# in Qrisp, and later showcase how to use it in the HHL implementation.
#
# .. admonition:: Quantum Phase Estimation (QPE)
#     :class: note
#
#     Quantum Phase Estimation is an important subroutine in many quantum algorithms, like
#     the HHL as you will get to learn. A short summary of what problem QPE solves can be stated as:
#
#     Given a unitary :math:`U` and quantum state :math:`|\psi\rangle` which is an eigenvector of :math:`U`:
#
#     .. math::  U |\psi\rangle = e^{i 2 \pi \phi}|\psi\rangle
#
#     applying quantum phase estimation to :math:`U` and :math:`|\psi\rangle` returns a quantum register
#     containing an estimate for the value of :math:`\phi`,
#     i.e. :math:`\text{QPE}_{U} |\psi\rangle |0\rangle = |\psi\rangle |\phi\rangle`.
#
# This can be implemented within a few lines of code in Qrisp:


def QPE(psi, U, precision=None, res=None):

    if res is None:
        res = qrisp.QuantumFloat(precision, -precision)

    qrisp.h(res)

    # Performs a loop with a dynamic bound in Jasp mode.
    for i in qrisp.jrange(res.size):
        with qrisp.control(res[i]):
            for j in qrisp.jrange(2**i):
                U(psi)

    return qrisp.QFT(res, inv=True)


######################################################################
# The first step here is to create the QuantumFloat ``res`` which will contain the result. The first
# argument specifies the amount of mantissa qubits the QuantumFloat should contain and the second
# argument specifies the exponent. Having :math:`n` mantissa qubits and exponent of :math:`-n`
# means that this QuantumFloat can represent values between 0 and 1 with a granularity of
# :math:`2^{-n}`. Subsequently, we apply an Hadamard gate to all qubits of ``res`` and continue by
# performing controlled evaluations of :math:`U`. This is achieved by using the
# ``with control(res[i]):`` statement. This statement enters a ControlEnvironment such that every
# quantum operation inside the indented code block will be controlled on the :math:`i`-th qubit of
# ``res``. The ``for i in jrange(res.size)``\ statement performs a loop with a dynamic bound in Jasp
# mode. The intricacies of what exactly this means are beyond the scope of this demo. You can just
# treat it like a normal ``range``. We conclude the algorithm by performing an inverse quantum fourier
# transformation of ``res``.
#
# Note that compared to other similar implementations, :math:`U` is given as a Python function
# (instead of a circuit object), allowing for slim and elegant implementations.
#
# Let’s take a look at a simple example.


import numpy as np


def U(psi):
    phi_1 = 0.5
    phi_2 = 0.125

    qrisp.p(phi_1 * 2 * np.pi, psi[0])  # p applies a phase gate
    qrisp.p(phi_2 * 2 * np.pi, psi[1])


psi = qrisp.QuantumFloat(2)
qrisp.h(psi)

res = QPE(psi, U, precision=3)

######################################################################
# In this code snippet, we define a function ``U`` which applies a phase gate onto the first two
# qubits of its input. We then create the QuantumVariable ``psi`` and bring it into uniform
# superposition by applying Hadamard gates onto each qubit. Subsequently, we evaluate QPE on ``U`` and
# ``psi`` with the precision 3.
#
# The quantum state is now:
#
# .. math::  \frac{1}{2} \text{QPE}_{U}(|0\rangle + |1\rangle + |2\rangle + |3\rangle)|0\rangle = \frac{1}{2} (|0\rangle|0\rangle + |1\rangle|\phi_1\rangle + |2\rangle|\phi_2\rangle +|3\rangle|\phi_1 + \phi_2\rangle)
#
# We verify by measuring ``psi`` together with ``res``:


print(qrisp.multi_measurement([psi, res]))

######################################################################
# .. rst-class:: sphx-glr-script-out
#
#   .. code-block:: none
#
#     {(0, 0.0): 0.25, (1, 0.5): 0.25, (2, 0.125): 0.25, (3, 0.625): 0.25}

######################################################################
# Here, the measurement results are given by a dictionary, where the keys are are tuples representing
# the values of ``psi`` and ``res``, and the corresponding values are the measurement probabilities.
#
# This example can also seamlessly be executed in Jasp mode: In this case, the ``terminal_sampling``
# decorator performs a hybrid simulation and afterwards samples from the resulting quantum state.


@qrisp.terminal_sampling
def main():
    qf = qrisp.QuantumFloat(2)
    qf[:] = 3

    res = QPE(qf, U, precision=3)

    return res

main()

######################################################################
# .. rst-class:: sphx-glr-script-out
#
#   .. code-block:: none
#
#     {0.625: 1.0}

######################################################################
# Executing the ``main`` function with the ``terminal_sampling`` decorator returns the measurement
# results for the variable ``res``. As expected, setting ``qf`` to :math:`3`, we obtain the correct
# result :math:`\phi_1 + \phi_2 = 0.625`.
#
#
# The HHL algorithm
# -----------------
#
# Given an :math:`N`-by-:math:`N` Hermitian matrix :math:`A` and an :math:`N`-dimensional vector
# :math:`b`, the Quantum Linear Systems Problem (QSLP) consists of preparing a quantum state
# :math:`|x\rangle` with amplitudes proportional to the solution :math:`x` of the linear system of
# equations :math:`Ax=b`. Thereby, it can exhibit an exponential speedup over classical methods for
# certain sparse matrices :math:`A`. The HHL quantum algorithm [#HHL_2009]_ [#HHL_step-by-step_2023]_ and, more generally, quantum linear
# systems algorithms, hold significant promise for accelerating computations in fields that rely
# heavily on solving linear systems of equations, such as `solving differential
# equations <https://arxiv.org/abs/2202.01054v4>`__, or accelerating machine learning.
#
# In its eigenbasis, the matrix :math:`A` can be written as
#
# .. math::  A = \sum_i \lambda_i|u_i\rangle\langle u_i|
#
# where :math:`|u_i\rangle` is an eigenvector of :math:`A` corresponding to the eigenvalue
# :math:`\lambda_i`.
#
# We define the quantum states :math:`|b\rangle` and :math:`|x\rangle` as
#
# .. math::  |b\rangle = \dfrac{\sum_i b_i|i\rangle}{\|\sum_i b_i|i\rangle\|} = \sum_i \beta_i|u_i\rangle \quad\text{and}\quad |x\rangle = \dfrac{\sum_i x_i|i\rangle}{\|\sum_i x_i|i\rangle\|} = \sum_i \gamma_i|u_i\rangle
#
# where :math:`|b\rangle` and :math:`|x\rangle` are expressed in the eigenbasis of :math:`A`.
#
# Solving the linear system amounts to
#
# .. math:: \begin{align}|x\rangle&=A^{-1}|b\rangle\\&=\bigg(\sum_{i=0}^{N-1}\lambda_i^{-1}|u_i\rangle\langle u_i|\bigg)\sum_j\beta_j|u_j\rangle\\&=\sum_{i=0}^{N-1}\lambda_i^{-1}\beta_i|u_i\rangle\end{align}
#
# You might wonder why we can’t just apply :math:`A^{-1}` directly to :math:`|b\rangle`? This is
# because, in general, the matrix :math:`A` is not unitary. However, we will circumnavigate this by
# exploiting that the Hamiltonian evolution :math:`U=e^{itA}` is unitary for a Hermitian matrix
# :math:`A`. And this brings us to the HHL algorithm.
#
# In theory, the HHL algorithm can be described as follows:
#
# - Step 1: We start by preparing the state
#
#   .. math::  |\Psi_1\rangle = |b\rangle = \sum_i \beta_i|u_i\rangle
#
# - Step 2: Applying **Quantum Phase Estimation** with respect to the Hamiltonian evolution
#   :math:`U=e^{itA}` yields the state
#
#   .. math::  |\Psi_2\rangle = \sum_i \beta_i|u_i\rangle|\lambda_it/2\pi\rangle = \sum_i \beta_i|u_i\rangle|\widetilde{\lambda}_i\rangle
#
#   To simplify notation, we write :math:`\widetilde{\lambda}_i=\lambda_it/2\pi`.
#
# - Step 3: Performing the inversion of the eigenvalues
#   :math:`\widetilde{\lambda}_i\rightarrow\widetilde{\lambda}_i^{-1}` yields the state
#
#   .. math::  |\Psi_3\rangle = \sum_i \beta_i|u_i\rangle|\widetilde{\lambda}_i\rangle|\widetilde{\lambda}_i^{-1}\rangle
#
# - Step 4: The amplitudes are multiplied by the inverse eigenvalues
#   :math:`\widetilde{\lambda}_i^{-1}` to obtain the state
#
#   .. math::  |\Psi_4\rangle = \sum_i \lambda_i^{-1}\beta_i|u_i\rangle|\widetilde{\lambda}_i\rangle|\widetilde{\lambda}_i^{-1}\rangle
#
#   This is achieved by means of a repeat-until-success procedure that applies **Steps 1-3** as a
#   subroutine. Stay tuned for more details below!
#
# - Step 5: As a final step, we uncompute the variables :math:`|\widetilde{\lambda}^{-1}\rangle` and
#   :math:`|\widetilde{\lambda}\rangle`, and obtain the state
#
#   .. math::  |\Psi_5\rangle = \sum_i \lambda_i^{-1}\beta_i|u_i\rangle = |x\rangle
#
# This concludes the HHL algorithm. The variable initialized in state :math:`|b\rangle` is now found in
# state :math:`|x\rangle`. As shown in the original HHL paper, the
# runtime of this algorithm is :math:`\mathcal{O}(\log(N)s^2\kappa^2/\epsilon)` where :math:`s` and
# :math:`\kappa` are the sparsity and condition number of the matrix :math:`A`, respectively, and
# :math:`\epsilon` is the precision of the solution. The logarithmic dependence on the dimension
# :math:`N` is the source of an exponential advantage over classical methods.
#
#
#
# The HHL algorithm in Qrisp
# --------------------------
# Let’s put theory into practice and dive into an implementation of the HHL algorithm in Qrisp.
#
# As a first step, we define a function that performs the inversion :math:`\lambda\mapsto\lambda^{-1}`.
# In this example, we restrict ourselves to an implementation that works for values
# :math:`\lambda=2^{-k}` for :math:`k\in\mathbb N`. (As shown above, a general inversion is available
# in Qrisp and will soon be updated to be compatible with QJIT compilation!)


def fake_inversion(qf, res=None):

    if res is None:
        res = qrisp.QuantumFloat(qf.size + 1)

    for i in qrisp.jrange(qf.size):
        qrisp.cx(qf[i], res[qf.size - i])

    return res


######################################################################
# .. tip::
#     Essentially, the controlled-NOT operations in the loop reverse the positions of the bits in input
#     variable and place them in the result variable in the opposite order. For example, for
#     :math:`\lambda=2^{-3}`, which is :math:`0.001` in binary, the function would produce
#     :math:`\lambda^{-1}=2^3`, which in binary is 1000.
#
# Let’s see if it works as intended.


qf = qrisp.QuantumFloat(3, -3)
qrisp.x(qf[2])
qrisp.dicke_state(qf, 1)
res = fake_inversion(qf)
print(qrisp.multi_measurement([qf, res]))

######################################################################
# .. rst-class:: sphx-glr-script-out
#
#   .. code-block:: none
#
#      {(0.125, 8): 0.3333333333333333, (0.25, 4): 0.3333333333333333, (0.5, 2): 0.3333333333333333}

######################################################################
# Next, we define the function ``HHL_encoding`` that performs **Steps 1-4** and prepares the state
# :math:`|\Psi_4\rangle`. But, how do get the values :math:`\widetilde{\lambda}^{-1}_i` into the
# amplitudes of the states, i.e. how do we go from :math:`|\Psi_3\rangle` to :math:`|\Psi_4\rangle`?
#
# Recently, efficient methods for black-box quantum state preparation that avoid arithmetic were
# proposed, see Bausch [#Bausch_2022]_ and Sanders et al. [#Sanders_2019]_. In this demo, we use a routine proposed in the latter
# reference which is based on a comparison between integers. This is implemented via the
# aforementioned comparisons for QuantumFloats.
#
# To simplify the notation, we write :math:`y^{(i)}=\widetilde{\lambda}^{-1}_i`. Recall that the
# values :math:`y^{(i)}` represent unsigned integers between :math:`0` and :math:`2^n-1$`.
#
# Starting from the state
#
# .. math::  |\Psi_3\rangle = \sum_i \beta_i|u_i\rangle|\widetilde{\lambda}_i\rangle|y^{(i)}\rangle_{\text{res}}
#
# we prepare a uniform superposition of :math:`2^n` states in a ``case_indicator`` QuantumFloat.
#
# .. math::  |\Psi_3'\rangle = \sum_i \beta_i|u_i\rangle|\widetilde{\lambda}_i\rangle|y^{(i)}\rangle_{\text{res}}\otimes\frac{1}{\sqrt{2^n}}\sum_{x=0}^{2^n-1}|x\rangle_{\text{case}}
#
# Next, we calculate the comparison :math:`a\geq b` between the ``res`` and the ``case_indicator`` into a QuantumBool ``qbl``.
#
# .. math::  |\Psi_3''\rangle = \sum_i \beta_i|u_i\rangle|\widetilde{\lambda}_i\rangle|y^{(i)}\rangle_{\text{res}}\otimes\frac{1}{\sqrt{2^n}}\left(\sum_{x=0}^{y^{(i)}-1}|x\rangle_{\text{case}}|0\rangle_{\text{qbl}} + \sum_{x=y^{(i)}}^{2^n-1}|x\rangle_{\text{case}}|1\rangle_{\text{qbl}}\right)
#
# Finally, the ``case_indicator`` is unprepared with :math:`n` Hadamards and we obtain the state
#
# .. math::  |\Psi_3'''\rangle = \sum_i \dfrac{y^{(i)}}{2^n}\beta_i|u_i\rangle|\widetilde{\lambda}_i\rangle|y^{(i)}\rangle_{\text{res}}|0\rangle_{\text{case}}|0\rangle_{\text{qbl}} + |\Phi\rangle
#
# where :math:`|\Phi\rangle` is an orthogonal state with the last variables not in :math:`|0\rangle_{\text{case}}|0\rangle_{\text{qbl}}`. Hence, upon measuring the ``case_indicator`` in state :math:`|0\rangle` and the target ``qbl`` in state :math:`|0\rangle`, the desired state is prepared.
#
# **Steps 1-4** are performed as a repeat-until-success (RUS) routine. This decorator converts the function to be executed within a RUS procedure. The function must return a Boolean value as first return value and is repeatedly executed until the first return value is True.


@qrisp.RUS(static_argnums=[0, 1])
def HHL_encoding(b, hamiltonian_evolution, n, precision):

    # Prepare the state |b>. Step 1
    qf = qrisp.QuantumFloat(n)
    # Reverse the endianness for compatibility with Hamiltonian simulation.
    qrisp.prepare(qf, b, reversed=True)

    qpe_res = QPE(qf, hamiltonian_evolution, precision=precision)  # Step 2
    inv_res = fake_inversion(qpe_res)  # Step 3

    case_indicator = qrisp.QuantumFloat(inv_res.size)

    with qrisp.conjugate(qrisp.h)(case_indicator):
        qbl = (case_indicator >= inv_res)

    cancellation_bool = (qrisp.measure(case_indicator) == 0) & (qrisp.measure(qbl) == 0)

    # The first return value is a boolean value. Additional return values are QuantumVariables.
    return cancellation_bool, qf, qpe_res, inv_res


######################################################################
# The probability of success could be further increased by oblivious :doc:`amplitude amplification <demos/tutorial_intro_amplitude_amplification>` to achieve
# optimal asymptotic scaling.
#
# Finally, we put all things together into the ``HHL`` function.
#
# This function takes the following arguments:
#
# - ``b``: The vector :math:`b`.
# - ``hamiltonian_evolution``: A function performing hamiltonian_evolution :math:`e^{itA}`.
# - ``n``: The number of qubits encoding the state :math:`|b\rangle` (:math:`N=2^n`).
# - ``precision``: The precision of the quantum phase estimation.
#
# The HHL function uses the previously defined subroutine to prepare the state :math:`|\Psi_4\rangle`
# and subsequently uncomputes the :math:`|\widetilde{\lambda}\rangle` and :math:`|\lambda\rangle` quantum
# variables leaving the first variable, that was initialized in state :math:`|b\rangle`, in the target
# state :math:`|x\rangle`.


def HHL(b, hamiltonian_evolution, n, precision):

    qf, qpe_res, inv_res = HHL_encoding(b, hamiltonian_evolution, n, precision)

    # Uncompute qpe_res and inv_res
    with qrisp.invert():
        QPE(qf, hamiltonian_evolution, res=qpe_res)
        fake_inversion(qpe_res, res=inv_res)

    # Reverse the endianness for compatibility with Hamiltonian simulation.
    for i in qrisp.jrange(qf.size // 2):
        qrisp.swap(qf[i], qf[n - i - 1])

    return qf


######################################################################
# Application: Solving systems of linear equations
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Let’s try a first simple example. First, the matrix :math:`A` is represented as a Pauli operator
# :math:`H` and the Hamiltonian evolution unitary :math:`U=e^{itH}` is obtained by Trotterization with
# one step (as the Pauli terms commute in this case). We choose :math:`t=\pi` to ensure that
# :math:`\widetilde{\lambda}_i=\lambda_i t/2\pi` are of the form :math:`2^{-k}` for a positive integer
# :math:`k`. This is enabled by the Qrisp’s ``QubitOperator`` class providing the tools to describe,
# optimize and efficiently simulate quantum Hamiltonians.


from qrisp.operators import QubitOperator
import numpy as np

A = np.array([[3 / 8, 1 / 8], [1 / 8, 3 / 8]])
b = np.array([1, 1])

H = QubitOperator.from_matrix(A).to_pauli()


# By default e^{-itH} is performed. Therefore, we set t=-pi.
def U(qf):
    H.trotterization()(qf, t=-np.pi, steps=1)


######################################################################
# .. _terminal_sampling:
#
# The ``terminal_sampling`` decorator performs a hybrid simulation and afterwards samples from the
# resulting quantum state. We convert the resulting measurement probabilities to amplitudes by applying
# the square root. Note that minus signs of amplitudes cannot be recovered from measurement
# probabilities.


@qrisp.terminal_sampling
def main():
    x = HHL(tuple(b), U, 1, 3)
    return x


res_dict = main()

for k, v in res_dict.items():
    res_dict[k] = v**0.5

print(res_dict)

######################################################################
# .. rst-class:: sphx-glr-script-out
#
#   .. code-block:: none
#
#     {0.0: 0.7071067811865476, 1.0: 0.7071067811865476}

######################################################################
# Finally, let’s compare to the classical result.


x = (np.linalg.inv(A) @ b) / np.linalg.norm(np.linalg.inv(A) @ b)
print(x)

######################################################################
# .. rst-class:: sphx-glr-script-out
#
#   .. code-block:: none
#
#     [0.70710678 0.70710678]

######################################################################
# Et voila! Now, let’s tackle some more complicated examples! Next, we try some randomly generated
# matrices whose eigenvalues are inverse powers of 2, i.e. of the form :math:`2^{-k}` for :math:`k<K`.
# To facilitate fast simulations, we restrict ourselves to :math:`K=4` (required ``precision`` of
# QPE) as the runtime of the HHL algorithm scales linearly in the inverse precision
# :math:`\epsilon=2^{-K}` (and therefore exponentially in :math:`K`).


def hermitian_matrix_with_power_of_2_eigenvalues(n):
    # Generate eigenvalues as inverse powers of 2.
    eigenvalues = 1 / np.exp2(np.random.randint(1, 4, size=n))

    # Generate a random unitary matrix.
    Q, _ = np.linalg.qr(np.random.randn(n, n))

    # Construct the Hermitian matrix.
    A = Q @ np.diag(eigenvalues) @ Q.conj().T

    return A


# Example
n = 3
A = hermitian_matrix_with_power_of_2_eigenvalues(2**n)

H = QubitOperator.from_matrix(A).to_pauli()


def U(qf):
    H.trotterization()(qf, t=-np.pi, steps=5)


b = np.random.randint(0, 2, size=2**n)

print("Hermitian matrix A:")
print(A)

print("Eigenvalues:")
print(np.linalg.eigvals(A))

print("b:")
print(b)


######################################################################
# .. rst-class:: sphx-glr-script-out
#
#   .. code-block:: none
#
#     Hermitian matrix A:
#     [[ 0.19252221  0.03518761 -0.08725666  0.02288045  0.04538538  0.03247052
#       -0.06519315 -0.06229545]
#      [ 0.03518761  0.28758984  0.03595881  0.06702119  0.03489919  0.07145221
#        0.07085617 -0.10555209]
#      [-0.08725666  0.03595881  0.28859598  0.00165777 -0.05503796 -0.03801707
#        0.13179087  0.03443008]
#      [ 0.02288045  0.06702119  0.00165777  0.26810863 -0.12651043  0.00385685
#       -0.01492278 -0.10505962]
#      [ 0.04538538  0.03489919 -0.05503796 -0.12651043  0.34469893  0.07605711
#        0.01253655  0.02660667]
#      [ 0.03247052  0.07145221 -0.03801707  0.00385685  0.07605711  0.31567327
#        0.080171   -0.01770181]
#      [-0.06519315  0.07085617  0.13179087 -0.01492278  0.01253655  0.080171
#        0.30119828  0.03446737]
#      [-0.06229545 -0.10555209  0.03443008 -0.10505962  0.02660667 -0.01770181
#        0.03446737  0.25161286]]
#     Eigenvalues:
#     [0.25  0.5   0.5   0.125 0.125 0.5   0.125 0.125]
#     b:
#     [1 0 0 0 0 0 0 1]


@qrisp.terminal_sampling
def main():
    x = HHL(tuple(b), U, n, 4)
    return x


res_dict = main()

for k, v in res_dict.items():
    res_dict[k] = v**0.5

np.array([res_dict[key] for key in sorted(res_dict)])

######################################################################
# .. rst-class:: sphx-glr-script-out
#
#   .. code-block:: none
#
#     array([0.73688311, 0.11916253, 0.09056467, 0.11650071, 0.09336856,
#            0.04641304, 0.04607175, 0.63877596])

######################################################################
# Let’s compare to the classical solution:


x = (np.linalg.inv(A) @ b) / np.linalg.norm(np.linalg.inv(A) @ b)
print(x)

######################################################################
# .. rst-class:: sphx-glr-script-out
#
#   .. code-block:: none
#
#     [ 0.73761732  0.11838953  0.08853343  0.11632373 -0.09299411 -0.04581715
#       0.04179596  0.63878104]

######################################################################
# First of all, kudos for making it to this point of the demo. Prior to proceeding to the final
# part of the demo showcasing the awesomeness of QJIT compilation with Catalyst, let’s rewind for
# a second, take a deep breath, and go through the steps and concepts you learned so far.
#
# .. admonition:: Intermezzo: Quick Recap
#     :class: note
#
#     Starting with getting familiar with Qrisp’s approach to programming through the use of
#     QuantumVariables, we foreshadowed the two QuantumTypes used in the implementation of HHL:
#     QuantumFloats, and QuantumBools. Following a quick mention of Jasp and the ability of doing
#     real-time computations using this module, the 'Qrisp-y' QPE implementation was demonstrated with a
#     simple example.
#
#     Equipped with a theoretical introduction to HHL and outlining the steps required to perform this
#     algorithm, you got to see how to first encode the first 4 steps and make use of the repeat until
#     success feature of Jasp.
#
#     Then, putting everything together, we combined the previously defined building blocks (read: Python
#     functions)—the HHL_encoding and QPE—into a simple function. With a brief feature appearance of
#     Hamiltonian simulation you then successfully managed to solve two systems of linear equations.
#
# Before moving on to the part of the demo showcasing Catalyst's capabilities, let’s
# appreciate one last time how elegantly we can call the HHL algorithm:
# ``x = HHL(b, hamiltonian_evolution, n, precision)``.
#

######################################################################
# QJIT compilation with Catalyst
# ------------------------------
# The solution that we saw so far only ran within the Python-based Qrisp internal simulator, which
# doesn’t have to care about silly things such as coherence time. Unfortunately, this is not (yet) the
# case for real quantum hardware so it is of paramount importance that the classical real-time
# computations (such as the while loop within the repeat-until-success steps) are executed as fast as
# possible. For this reason the QIR specification was created.
#
# QIR essentially embeds quantum aspects into `LLVM <https://llvm.org/docs/>`__, which is the
# foundation of a lot of modern compiler infrastructure. This implies QIR based software stacks are
# able to integrate a large part of established classical software and also express real-time control
# structures. Read more about QIR at the `QIR Alliance website <https://www.qir-alliance.org/>`__ and
# the `Advancing hybrid quantum–classical computation with real-time execution paper
# <https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2022.940293/full>`__.
#
# Jasp has been built with a direct Catalyst integration implying Jasp programs can be converted to
# QIR via the Catalyst pipeline. This conversion is straightforward: You simply capture the Qrisp
# computation using the ``make_jaspr`` function and then call ``to_qir``.


def main():
    x = HHL(tuple(b), U, n, 4)
    # Note that we have to return a classical value
    # (in this case the measurement result of the
    # quantum variable returned by the HHL algorithm)
    # Within the above examples, we used the terminal_sampling
    # decorator, which is a convenience feature and allows
    # a much faster sampling procedure.
    # The terminal_sampling decorator expects a function returning
    # quantum variables, while most other evaluation modes require
    # classical return values.
    return qrisp.measure(x)


jaspr = qrisp.make_jaspr(main)()
qir_str = jaspr.to_qir()
# Print only the first few lines - the whole string is very long.
print(qir_str[:2000])

######################################################################
# .. rst-class:: sphx-glr-script-out
#
#   .. code-block:: none
#
#     ; ModuleID = 'LLVMDialectModule'
#     source_filename = "LLVMDialectModule"
#     target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
#     target triple = "x86_64-unknown-linux-gnu"
#
#     @"{'shots': 0, 'mcmc': False, 'num_burnin': 0, 'kernel_name': None}" = internal constant [66 x i8] c"{'shots': 0, 'mcmc': False, 'num_burnin': 0, 'kernel_name': None}\00"
#     @LightningSimulator = internal constant [19 x i8] c"LightningSimulator\00"
#     @"/home/positr0nium/miniforge3/envs/qrisp/lib/python3.10/site-packages/pennylane_lightning/liblightning_qubit_catalyst.so" = internal constant [120 x i8] c"/home/positr0nium/miniforge3/envs/qrisp/lib/python3.10/site-packages/pennylane_lightning/liblightning_qubit_catalyst.so\00"
#     @__constant_xi64_3 = private constant i64 -4
#     @__constant_1024xi64 = private constant [1024 x i64] zeroinitializer
#     @__constant_xi1 = private constant i1 false
#     @__constant_xi64_2 = private constant i64 4
#     @__constant_xi64_1 = private constant i64 3
#     @__constant_xi64_0 = private constant i64 30
#     @__constant_30xi64 = private constant [30 x i64] [i64 30, i64 29, i64 28, i64 27, i64 26, i64 25, i64 24, i64 23, i64 22, i64 21, i64 20, i64 19, i64 18, i64 17, i64 16, i64 15, i64 14, i64 13, i64 12, i64 11, i64 10, i64 9, i64 8, i64 7, i64 6, i64 5, i64 4, i64 3, i64 2, i64 1]
#     @__constant_xf64_34 = private constant double 0xBF919CF5D85DB47A
#     @__constant_xf64_33 = private constant double 0xBF89C4C2643A2DEA
#     @__constant_xf64_32 = private constant double 0xBF8696E980F4B09E
#     @__constant_xf64_31 = private constant double 0x3F69A6304F46D56B
#     @__constant_xf64_30 = private constant double 0xBF97F300BE6AA82E
#     @__constant_xf64_29 = private constant double 0x3F75120CFFA6C0C9
#     @__constant_xf64_28 = private constant double 0x3FA0EEDA934AC632
#     @__constant_xf64_27 = private constant double 0x3F9B793155FE33E5
#     @__constant_xf64_26 = private constant double 0x3EFEF5A8835D9B8E
#     @__constant_xf64_25 = private constant double 0xBF826D7D6D20BB88
#     @__constant_xf64_24 = private const

######################################################################
# The Catalyst runtime
# --------------------
#
# Jasp is also capable of targeting the Catalyst execution runtime (i.e., the Lightning simulator).
# There are, however, still some simulator features to be implemented on Jasp side, which prevents
# efficient sampling. We restrict the demonstration to the smaller examples from above to limit the
# overall execution time required. (Warning: The execution may take more than 15 minutes.)


A = np.array([[3 / 8, 1 / 8], [1 / 8, 3 / 8]])

b = np.array([1, 1])

H = QubitOperator.from_matrix(A).to_pauli()


# By default e^{-itH} is performed. Therefore, we set t=-pi.
def U(qf):
    H.trotterization()(qf, t=-np.pi, steps=1)


@qrisp.qjit
def main():
    x = HHL(tuple(b), U, 1, 3)

    return qrisp.measure(x)


samples = []
for i in range(5):
    samples.append(float(main()))

print(samples)

######################################################################
# .. rst-class:: sphx-glr-script-out
#
#   .. code-block:: none
#
#     [1.0, 1.0, 0.0, 1.0, 0.0]

######################################################################
# Scrolling back to the :ref:`terminal_sampling` cell, we see that the
# expected distribution is 50/50 between one and zero, which roughly agrees to the result of the
# previous cell and concludes this demo.
#
# Conclusion
# ----------
#
# In this demo, we have shown how to implement the HHL algorithm featuring classical real-time
# computations in the high-level language Qrisp. This algorithm is important in a variety of use cases
# such as solving differential equations, accelerating machine learning, and more generally, any task
# that involves solving linear systems of equations. Along the way, we have dipped into advanced
# concepts such as Linear Combination of Unitaries and Hamiltonian simulation. Moreover, we have
# demonstrated how Qrisp and Catalyst complement each other for translating a high-level
# implementation into low-level QIR.
#
#
#
# References
# ----------
#
# .. [#qrisp_2024]
#
#     R. Seidel, S. Bock, R. Zander, M. Petrič, N. Steinmann, N. Tcholtchev, M. Hauswirth,
#     "Qrisp: A Framework for Compilable High-Level Programming of Gate-Based Quantum Computers",
#     `arXiv:2406.14792 <https://arxiv.org/abs/2406.14792>`__, 2024.
#
# .. [#HHL_2009]
#
#     A. W. Harrow, A. Hassidim, S. Lloyd,
#     "Quantum Algorithm for Linear Systems of Equations",
#     `Physical Review Letters 103(15), 150503 <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.103.150502>`__, 2009.
#
# .. [#HHL_step-by-step_2023]
#
#     A. Zaman, H. J. Morrell, H. Y. Wong,
#     "A Step-by-Step HHL Algorithm Walkthrough to Enhance Understanding of Critical Quantum Computing Concepts",
#     `IEEE Access 11 <https://ieeexplore.ieee.org/document/10189828>`__, 2023.
#
# .. [#Bausch_2022]
#
#     J. Bausch,
#     "Fast Black-Box Quantum State Preparation",
#     `Quantum 6, 773 <https://quantum-journal.org/papers/q-2022-08-04-773/#>`__, 2022.
#
# .. [#Sanders_2019]
#
#     Y. R. Sanders, G. H. Low, A. Scherer, D. W. Berry,
#     "Black-Box Quantum State Preparation without Arithmetic",
#     `Physical review letters 122(2), 020502 <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.122.020502>`__, 2019.
#
