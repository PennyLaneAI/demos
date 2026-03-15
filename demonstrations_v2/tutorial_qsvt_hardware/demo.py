r"""How to implement QSVT on hardware
=======================================

The :doc:`quantum singular value transform (QSVT) <demos/tutorial_intro_qsvt>`
is a quantum algorithm that allows us to perform polynomial
transformations on matrices or operators, and it is rapidly becoming
a go-to algorithm for :doc:`quantum application research <demos/tutorial_apply_qsvt>`
in the `ISQ era <https://pennylane.ai/blog/2023/06/from-nisq-to-isq/>`__.

In this how-to guide, we will show how we can implement the QSVT
subroutine in a hardware-compatible way, taking your application research
to the next level.

.. figure:: ../_static/demo_thumbnails/opengraph_demo_thumbnails/OGthumbnail_qsvt_hardware.png
    :align: center
    :width: 50%
    :target: javascript:void(0)


Calculating angles
------------------

Our goal is to apply a polynomial transformation to a given Hamiltonian, i.e., :math:`p(\mathcal{H}).` To achieve this, we must consider the two
fundamental components of the QSVT algorithm:

- **Projection angles**: A list of angles that will determine the coefficients of the polynomial to be applied.
- **Block encoding**: The strategy used to encode the Hamiltonian. We will use the :doc:`linear combinations of unitaries <demos/tutorial_lcu_blockencoding>` approach via the PennyLane :class:`~.qml.PrepSelPrep` operation.

Calculating angles is not a trivial task, but we can use the PennyLane function :func:`~.pennylane.poly_to_angles`
to obtain them. There are also tools such as `pyqsp <https://github.com/ichuang/pyqsp/tree/master/pyqsp>`_ that can do the job for us.
Let's try both tools to calculate the angles for applying the
polynomial :math:`p(x) = -x + \frac{x^3}{2}+ \frac{x^5}{2}`.

The :func:`~.pennylane.poly_to_angles` function in PennyLane accepts the coefficients of the
polynomial, ordered from lowest to highest power, as input. We also need to define the routine for
which the angles are computed, which is ``'QSVT'`` here.
"""
import pennylane as qml
qml.capture.enable()
poly = [0, -1.0, 0, 1/2, 0, 1/2]
angles_pl = qml.poly_to_angles(poly, "QSVT")
print(angles_pl)

######################################################################
# To find the angles with ``pyqsp`` we can run this code:
#
# .. code-block:: python
#
#    from pyqsp.angle_sequence import QuantumSignalProcessingPhases
#    import numpy as np
#
#    ang_seq = QuantumSignalProcessingPhases(np.array(poly), signal_operator="Wx")
#
# The angles obtained after execution are as follows:

ang_seq = [
    -1.5115007723754004,
    0.6300762184670975,
    0.8813995564082947,
    -2.2601930971815003,
    3.7716688720568885,
    0.059295554419495855,
]

######################################################################
# The ``pyqsp`` angles are obtained in the
# context of QSP and are not the same as the ones we have to use in QSVT.
# We can use the :func:`~.pennylane.transform_angles` function to transform the angles:

angles_pyqsp = qml.transform_angles(ang_seq, "QSP", "QSVT")
print(angles_pyqsp)

######################################################################
# Note that these angles are not exactly the same as those obtained with
# :func:`~.pennylane.poly_to_angles`, but they will both produce the same polynomial transformation.
# Using the angles computed with :func:`~.pennylane.poly_to_angles` or ``pyqsp``, we can now start
# working with the template.
#
# QSVT on hardware
# -----------------
#
# The :class:`~.qml.QSVT` template expects two inputs. The first one is the block encoding operator, :class:`~.qml.PrepSelPrep`,
# and the second one is a set of projection operators, :class:`~.qml.PCPhase`, that encode the angles properly.
# We will see how to apply them later, but first let's define
# a Hamiltonian and manually apply the polynomial of interest:

import pennylane as qml
qml.capture.enable()
import numpy as np
from numpy.linalg import matrix_power as mpow

coeffs = np.array([0.2, -0.7, -0.6])
coeffs /= np.linalg.norm(coeffs, ord=1)  # Normalize the coefficients

obs = [qml.X(3), qml.X(3) @ qml.Z(4), qml.Z(3) @ qml.Y(4)]

H = qml.dot(coeffs, obs)

H_mat = qml.matrix(H, wire_order=[3, 4])

# We calculate p(H) = -H + 0.5 * H^3 + 0.5 * H^5
H_poly = -H_mat + 0.5 * mpow(H_mat, 3) + 0.5 * mpow(H_mat, 5)

print(np.round(H_poly, 4))

######################################################################
# Now that we know what the target result is, let's see how to apply the polynomial with a quantum circuit instead.
# We start by defining the proper input operators for the :class:`~.qml.QSVT` template.

# We need |log2(len(coeffs))| = 2 control wires to encode the Hamiltonian
control_wires = [1, 2]
block_encode = qml.PrepSelPrep(H, control=control_wires)

projectors = [
    qml.PCPhase(angles_pl[i], dim=2 ** len(H.wires), wires=control_wires + H.wires)
    for i in range(len(angles_pl))
]


dev = qml.device("default.qubit")

@qml.qnode(dev)
def circuit():

    qml.Hadamard(0)
    qml.ctrl(qml.QSVT, control=0, control_values=[1])(block_encode, projectors)
    qml.ctrl(qml.adjoint(qml.QSVT), control=0, control_values=[0])(block_encode, projectors)
    qml.Hadamard(0)

    return qml.state()


matrix = qml.matrix(circuit, wire_order=[0] + control_wires + H.wires)()
print(np.round(matrix[: 2 ** len(H.wires), : 2 ** len(H.wires)], 4))

######################################################################
# The matrix obtained using QSVT is the same as the one obtained by applying the polynomial
# directly to the Hamiltonian! That means the circuit is encoding :math:`p(\mathcal{H})` correctly.
# The great advantage of this approach is that all the building blocks used in the circuit can be
# decomposed into basic gates easily, allowing this circuit
# to be easily executed on hardware devices with PennyLane.
#
# Please also note that QSVT encodes the desired polynomial :math:`p(\mathcal{H})` as well as
# a polynomial :math:`i q(\mathcal{H}).` To isolate :math:`p(\mathcal{H}),` we have used an auxiliary qubit and considered that
# the sum of a complex number and its conjugate gives us twice its real part. We
# recommend :doc:`this demo <demos/tutorial_apply_qsvt>` to learn more about the structure
# of the circuit.
#
# Conclusion
# ----------
# In this brief how-to we demonstrated applying QSVT on a sample Hamiltonian. Note that the algorithm is sensitive to
# the block-encoding method, so please make sure that the projection angles are converted to the proper format.
# This how-to serves as a guide for running your own workflows and experimenting with more advanced Hamiltonians and polynomial functions.
#
# References
# ----------
#
# .. [#unification]
#
#     John M. Martyn, Zane M. Rossi, Andrew K. Tan, Isaac L. Chuang.
#     "A Grand Unification of Quantum Algorithms".
#     `arXiv preprint arXiv:2105.02859 <https://arxiv.org/abs/2105.02859>`__.
#
