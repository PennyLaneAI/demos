r"""

.. _adjoint_differentiation:

Adjoint Differentiation
=======================

.. meta::
    :property="og:description": Learn how to use the adjoint method to compute gradients of quantum circuits."
    :property="og:image": https://pennylane.ai/qml/_static/demonstration_assets/icon.png

.. related::

   tutorial_backprop  Quantum gradients with backpropagation
   tutorial_quantum_natural_gradient Quantum natural gradient
   tutorial_general_parshift Generalized parameter-shift rules
   tutorial_stochastic_parameter_shift The Stochastic Parameter-Shift Rule


"""

##############################################################################
#
# `Classical automatic differentiation <https://en.wikipedia.org/wiki/Automatic_differentiation#The_chain_rule,_forward_and_reverse_accumulation>`__
# has two methods of calculation: forward and reverse.
# The optimal choice of method depends on the structure of the problem; is the function
# many-to-one or one-to-many? We use the properties of the problem to optimize how we
# calculate derivatives.
#
# Most methods for calculating the derivatives of quantum circuits are either direct applications
# of classical gradient methods to quantum simulations, or quantum hardware methods like parameter-shift
# where we can only extract restricted pieces of information.
#
# Adjoint differentiation straddles these two strategies, taking benefits from each.
# On simulators, we can examine and modify the state vector at any point. At the same time, we know our
# quantum circuit holds specific properties not present in an arbitrary classical computation.
#
# Quantum circuits only involve:
#
# 1) initialization,
#
# .. math:: |0\rangle,
#
# 2) application of unitary operators,
#
# .. math:: |\Psi\rangle = U_{n} U_{n-1} \dots U_0 |0\rangle,
#
# 3) measurement, such as estimating an expectation value of a Hermitian operator,
#
# .. math:: \langle M \rangle = \langle \Psi | M | \Psi \rangle.
#
# Since all our operators are unitary, we can easily "undo" or "erase" them by applying their adjoint:
#
# .. math:: U^{\dagger} U | \phi \rangle = |\phi\rangle.
#
# The **adjoint differentiation method** takes advantage of the ability to erase, creating a time- and
# memory-efficient method for computing quantum gradients on state vector simulators. Tyson Jones and Julien Gacon describe this
# algorithm in their paper
# `"Efficient calculation of gradients in classical simulations of variational quantum algorithms" <https://arxiv.org/abs/2009.02823>`__ .
#
# In this demo, you will learn how adjoint differentiation works and how to request it
# for your PennyLane QNode. We will also look at the performance benefits.
#
# Time for some code
# ------------------
#
# So how does it work? Instead of jumping straight to the algorithm, let's explore the above equations
# and their implementation in a bit more detail.
#
# To start, we import PennyLane and Jax's numpy:

import pennylane as qml
qml.capture.enable()
import pennylane.numpy as pnp


##############################################################################
# We also need a circuit to simulate:
#

dev = qml.device("default.qubit", wires=2)

x = pnp.array([0.1, 0.2, 0.3], requires_grad=True)

@qml.qnode(dev, diff_method="adjoint")
def circuit(a):
    qml.RX(a[0], wires=0)
    qml.CNOT(wires=(0, 1))
    qml.RY(a[1], wires=1)
    qml.RZ(a[2], wires=1)
    return qml.expval(qml.PauliX(wires=1))

##############################################################################
# The fast c++ simulator device ``"lightning.qubit"`` also supports adjoint differentiation,
# but here we want to quickly prototype a minimal version to illustrate how the algorithm works.
# We recommend performing
# adjoint differentiation on ``"lightning.qubit"`` for substantial performance increases.
#
# We will use the ``circuit`` QNode just for comparison purposes.  Throughout this
# demo, we will instead use a list of its operations ``ops`` and a single observable ``M``.

n_gates = 4
n_params = 3

ops = [
    qml.RX(x[0], wires=0),
    qml.CNOT(wires=(0, 1)),
    qml.RY(x[1], wires=1),
    qml.RZ(x[2], wires=1),
]
M = qml.PauliX(wires=1)

##############################################################################
# We will be using internal functions to manipulate the nuts and bolts of a statevector
# simulation.
#
# Internally, the statevector simulation uses a 2x2x2x... array to represent the state, whereas
# the result of a measurement ``qml.state()`` flattens this internal representation. Each dimension
# in the statevector corresponds to a different qubit.
#
# The internal functions ``create_initial_state`` and ``apply_operation``
# make additional assumptions about their inputs, and will fail or give incorrect results
# if those assumptions are not met. To work with these simulation tools, all operations should provide
# a matrix, and all wires must corresponding to dimensions of the statevector. This means all wires must already
# be integers starting from ``0``, and not exceed the number of dimensions in the state vector.
#

from pennylane.devices.qubit import create_initial_state, apply_operation

state = create_initial_state((0, 1))

for op in ops:
    state = apply_operation(op, state)

print(state)

##############################################################################
# We can think of the expectation :math:`\langle M \rangle` as an inner product between a bra and a ket:
#
# .. math:: \langle M \rangle = \langle b | k \rangle = \langle \Psi | M | \Psi \rangle,
#
# where
#
# .. math:: \langle b | = \langle \Psi| M = \langle 0 | U_0^{\dagger} \dots U_n^{\dagger} M,
#
# .. math:: | k \rangle =  |\Psi \rangle = U_n U_{n-1} \dots U_0 |0\rangle.
#
# We could have attached :math:`M,` a Hermitian observable (:math:`M^{\dagger}=M`), to either the
# bra or the ket, but attaching it to the bra side will be useful later.
#
# Using the ``state`` calculated above, we can create these :math:`|b\rangle` and :math:`|k\rangle`
# vectors.

bra = apply_operation(M, state)
ket = state

##############################################################################
# Now we use ``np.vdot`` to take their inner product.  ``np.vdot`` sums over all dimensions
# and takes the complex conjugate of the first input.

M_expval = pnp.vdot(bra, ket)
print("vdot  : ", M_expval)
print("QNode : ", circuit(x))

##############################################################################
# We got the same result via both methods! This validates our use of ``vdot`` and
# device methods.
#
# But the dividing line between what makes the "bra" and "ket" vector is actually
# fairly arbitrary.  We can divide the two vectors at any point from one :math:`\langle 0 |`
# to the other :math:`|0\rangle.` For example, we could have used
#
# .. math:: \langle b_n | = \langle 0 | U_1^{\dagger} \dots  U_n^{\dagger} M U_n,
#
# .. math:: |k_n \rangle = U_{n-1} \dots U_1 |0\rangle,
#
# and gotten the exact same results.  Here, the subscript :math:`n` is used to indicate that :math:`U_n`
# was moved to the bra side of the expression.  Let's calculate that instead:

bra_n = create_initial_state((0, 1))

for op in ops:
    bra_n = apply_operation(op, bra_n)
bra_n = apply_operation(M, bra_n)
bra_n = apply_operation(qml.adjoint(ops[-1]), bra_n)

ket_n = create_initial_state((0, 1))

for op in ops[:-1]:  # don't apply last operation
    ket_n = apply_operation(op, ket_n)

M_expval_n = pnp.vdot(bra_n, ket_n)
print(M_expval_n)

##############################################################################
# Same answer!
#
# We can calculate this in a more efficient way if we already have the
# initial ``state`` :math:`| \Psi \rangle.` To shift the splitting point, we don't
# have to recalculate everything from scratch. We just remove the operation from
# the ket and add it to the bra:
#
# .. math:: \langle b_n | = \langle b | U_n,
#
# .. math:: |k_n\rangle = U_n^{\dagger} |k\rangle .
#
# For the ket vector, you can think of :math:`U_n^{\dagger}` as "eating" its
# corresponding unitary from the vector, erasing it from the list of operations.
#
# Of course, we actually work with the conjugate transpose of :math:`\langle b_n |,`
#
# .. math:: |b_n\rangle = U_n^{\dagger} | b \rangle.
#
# Once we write it in this form, we see that the adjoint of the operation :math:`U_n^{\dagger}`
# operates on both :math:`|k_n\rangle` and :math:`|b_n\rangle` to move the splitting point right.
#
# Let's call this the "version 2" method.

bra_n_v2 = apply_operation(M, state)
ket_n_v2 = state

adj_op = qml.adjoint(ops[-1])

bra_n_v2 = apply_operation(adj_op, bra_n_v2)
ket_n_v2 = apply_operation(adj_op, ket_n_v2)

M_expval_n_v2 = pnp.vdot(bra_n_v2, ket_n_v2)
print(M_expval_n_v2)

##############################################################################
# Much simpler!
#
# We can easily iterate over all the operations to show that the same result occurs
# no matter where you split the operations:
#
# .. math:: \langle b_i | = \langle b_{i+1}| U_{i},
#
# .. math:: |k_{i+1} \rangle = U_{i} |k_{i}\rangle.
#
# Rewritten, we have our iteration formulas
#
# .. math:: | b_i \rangle = U_i^{\dagger} |b_{i+1}\rangle,
#
# .. math:: | k_i \rangle  = U_i^{\dagger} |k_{i+1}\rangle.
#
# For each iteration, we move an operation from the ket side to the bra side.
# We start near the center at :math:`U_n` and reverse through the operations list until we reach :math:`U_0.`

bra_loop = apply_operation(M, state)
ket_loop = state

for op in reversed(ops):
    adj_op = qml.adjoint(op)
    bra_loop = apply_operation(adj_op, bra_loop)
    ket_loop = apply_operation(adj_op, ket_loop)
    print(pnp.vdot(bra_loop, ket_loop))

##############################################################################
# Finally to Derivatives!
# -----------------------
#
# We showed how to calculate the same thing a bunch of different ways. Why is this useful?
# Wherever we cut, we can stick additional things in the middle. What are we sticking in the middle?
# The derivative of a gate.
#
# For simplicity's sake, assume each unitary operation :math:`U_i` is a function of a single
# parameter :math:`\theta_i.`  For non-parametrized gates like CNOT, we say its derivative is zero.
# We can also generalize the algorithm to multi-parameter gates, but we leave those out for now.
#
# Remember that each parameter occurs twice in :math:`\langle M \rangle:` once in the bra and once in
# the ket. Therefore, we use the product rule to take the derivative with respect to both locations:
#
# .. math::
#       \frac{\partial \langle M \rangle}{\partial \theta_i} =
#       \langle 0 | U_1^{\dagger} \dots \frac{\text{d} U_i^{\dagger}}{\text{d} \theta_i} \dots M \dots U_i \dots U_1 | 0\rangle.
#
#
# .. math::
#     + \langle 0 | U_1^{\dagger} \dots U_i^{\dagger} \dots M \dots \frac{\text{d} U_i}{\text{d} \theta_i}  \dots U_1 |0\rangle
#
# We can now notice that those two components are complex conjugates of each other, so we can
# further simplify.  Note that each term is not an expectation value of a Hermitian observable,
# and therefore not guaranteed to be real.
# When we add them together, the imaginary part cancels out, and we obtain twice the
# value of the real part:
#
# .. math::
#       = 2 \cdot \text{Re}\left( \langle 0 | U_1^{\dagger} \dots U_i^{\dagger} \dots M \dots \frac{\text{d} U_i}{\text{d} \theta_i}  \dots U_1 |0\rangle \right).
#
# We can take that formula and break it into its "bra" and "ket" halves for a derivative at the :math:`i` th position:
#
# .. math::
#    \frac{\partial \langle M \rangle }{\partial \theta_i } =
#    2 \text{Re} \left( \langle b_i | \frac{\text{d} U_i }{\text{d} \theta_i} | k_i \rangle \right)
#
# where
#
# .. math:: \langle b_i | = \langle 0 | U_1^{\dagger} \dots U_n^{\dagger} M U_n \dots U_{i+1},
#
#
# .. math :: |k_i \rangle = U_{i-1} \dots U_1 |0\rangle.
#
# Notice that :math:`U_i` does not appear in either the bra or the ket in the above equations.
# These formulas differ from the ones we used when just calculating the expectation value.
# For the actual derivative calculation, we use a temporary version of the bra,
#
# .. math:: | \tilde{k}_i \rangle = \frac{\text{d} U_i}{\text{d} \theta_i} | k_i \rangle
#
# and use these to get the derivative
#
# .. math::
#       \frac{\partial \langle M \rangle}{\partial \theta_i} = 2 \text{Re}\left( \langle b_i | \tilde{k}_i \rangle \right).
#
# Both the bra and the ket can be calculated recursively:
#
# .. math:: | b_{i} \rangle = U^{\dagger}_{i+1} |b_{i+1}\rangle,
#
# .. math:: | k_{i} \rangle = U^{\dagger}_{i} |k_{i+1}\rangle.
#
# We can iterate through the operations starting at :math:`n` and ending at :math:`1.`
#
# We do have to calculate initial state first, the "forward" pass:
#
# .. math:: |\Psi\rangle = U_{n} U_{n-1} \dots U_0 |0\rangle.
#
# Once we have that, we only have about the same amount of work to calculate all the derivatives,
# instead of quadratically more work.
#
# Derivative of an Operator
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#
# One final thing before we get back to coding: how do we get the derivative of an operator?
#
# Most parametrized gates can be represented in terms of Pauli Rotations, which can be written as
#
# .. math:: U = e^{i c \hat{G} \theta}
#
# for a Pauli matrix :math:`\hat{G}`, a constant :math:`c,` and the parameter :math:`\theta.`
# Thus we can easily calculate their derivatives:
#
# .. math:: \frac{\text{d} U}{\text{d} \theta} = i c \hat{G} e^{i c \hat{G} \theta} = i c \hat{G} U .
#
# Luckily, PennyLane already has a built-in function for calculating this.

grad_op0 = qml.operation.operation_derivative(ops[0])
print(grad_op0)

##############################################################################
# Now for calculating the full derivative using the adjoint method!
#
# We loop over the reversed operations, just as before.  But if the operation has a parameter,
# we calculate its derivative and append it to a list before moving on. Since the ``operation_derivative``
# function spits back out a matrix instead of an operation,
# we have to use :class:`~pennylane.QubitUnitary` instead to create :math:`|\tilde{k}_i\rangle.`

bra = apply_operation(M, state)
ket = state

grads = []

for op in reversed(ops):
    adj_op = qml.adjoint(op)
    ket = apply_operation(adj_op, ket)

    # Calculating the derivative
    if op.num_params != 0:
        dU = qml.operation.operation_derivative(op)
        ket_temp = apply_operation(qml.QubitUnitary(dU, op.wires), ket)

        dM = 2 * pnp.real(pnp.vdot(bra, ket_temp))
        grads.append(dM)

    bra = apply_operation(adj_op, bra)


# Finally reverse the order of the gradients
# since we calculated them in reverse
grads = grads[::-1]

print("our calculation: ", [float(grad) for grad in grads])

grad_compare = qml.grad(circuit)(x)
print("comparison: ", grad_compare)

##############################################################################
# It matches!!!
#
# If you want to use adjoint differentiation without having to code up your own
# method that can support arbitrary circuits, you can use ``diff_method="adjoint"`` in PennyLane with
# ``"default.qubit"`` or PennyLane's fast C++ simulator ``"lightning.qubit"``.


dev_lightning = qml.device("lightning.qubit", wires=2)

@qml.qnode(dev_lightning, diff_method="adjoint")
def circuit_adjoint(a):
    qml.RX(a[0], wires=0)
    qml.CNOT(wires=(0, 1))
    qml.RY(a[1], wires=1)
    qml.RZ(a[2], wires=1)
    return qml.expval(M)

print(qml.grad(circuit_adjoint)(x))

##############################################################################
# Performance
# --------------
#
# The algorithm gives us the correct answers, but is it worth using? Parameter-shift
# gradients require at least two executions per parameter, so that method gets more
# and more expensive with the size of the circuit, especially on simulators.
# Backpropagation demonstrates decent time scaling, but requires more and more
# memory as the circuit gets larger.  Simulation of large circuits is already
# RAM-limited, and backpropagation constrains the size of possible circuits even more.
# PennyLane also achieves backpropagation derivatives from a Python simulator and interface-specific functions.
# The ``"lightning.qubit"`` device does not support
# backpropagation, so backpropagation derivatives lose the speedup from an optimized
# simulator.
#
# With adjoint differentiation on ``"lightning.qubit"``, you can get the best of both worlds: fast and
# memory efficient.
#
# But how fast? The provided script `here <https://pennylane.ai/qml/demos/adjoint_diff_benchmarking>`__
# generated the following images on a mid-range laptop.
# The backpropagation times were produced with the Python simulator ``"default.qubit"``, while parameter-shift
# and adjoint differentiation times were calculated with ``"lightning.qubit"``.
# The adjoint method clearly wins out for performance.
#
# .. figure:: ../_static/demonstration_assets/adjoint_diff/scaling.png
#     :width: 80%
#     :align: center
#
#
# Conclusions
# -----------
#
# So what have we learned? Adjoint differentiation is an efficient method for differentiating
# quantum circuits with state vector simulation.  It scales nicely in time without
# excessive memory requirements. Now that you've seen how the algorithm works, you can
# better understand what is happening when you select adjoint differentiation from one
# of PennyLane's simulators.
#
#
# Bibliography
# -------------
#
# Jones and Gacon. Efficient calculation of gradients in classical simulations of variational quantum algorithms.
# `https://arxiv.org/abs/2009.02823 <https://arxiv.org/abs/2009.02823>`__
#
# Xiu-Zhe Luo, Jin-Guo Liu, Pan Zhang, and Lei Wang. Yao.jl: `Extensible, efficient framework for quantum
# algorithm design <https://quantum-journal.org/papers/q-2020-10-11-341/>`__ , 2019
#
