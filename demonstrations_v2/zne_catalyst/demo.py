r"""
Digital zero-noise extrapolation (ZNE) with Catalyst
====================================================

In this tutorial, you will learn how to use :doc:`error mitigation <demos/tutorial_error_mitigation>`, and in particular 
the zero-noise extrapolation (ZNE) technique, in combination with 
`Catalyst <https://docs.pennylane.ai/projects/catalyst>`_, a framework for quantum 
just-in-time (JIT) compilation with PennyLane. 
We'll demonstrate how to generate noise-scaled circuits, execute them on a noisy quantum
simulator, and use extrapolation techniques to estimate the zero-noise result, all while
leveraging JIT compilation through Catalyst.

.. image:: ../_static/demo_thumbnails/regular_demo_thumbnails/thumbnail_zne_catalyst.png
    :width: 70%
    :align: center

The demo :doc:`Error mitigation with Mitiq and PennyLane <demos/tutorial_error_mitigation>`
shows how ZNE, along with other error mitigation techniques, can be carried out in PennyLane
by using `Mitiq <https://github.com/unitaryfund/mitiq>`__, a Python library developed 
by `Unitary Fund <https://unitary.fund/>`__.

ZNE in particular is also offered out of the box in PennyLane as a *differentiable* error mitigation technique,
for usage in combination with variational workflows. More on this in the tutorial
:doc:`Differentiating quantum error mitigation transforms <demos/tutorial_diffable-mitigation>`.

On top of the error mitigation routines offered in PennyLane, ZNE is also available for just-in-time
(JIT) compilation. In this tutorial we see how an error mitigation routine can be
integrated in a Catalyst workflow.

At the end of the tutorial, we will compare the execution time of ZNE routines in
pure PennyLane vs. PennyLane and Catalyst with JIT.

What is zero-noise extrapolation (ZNE)
--------------------------------------
Zero-noise extrapolation (ZNE) is a technique used to mitigate the effect of noise on quantum
computations. First introduced in [#zne-2017]_, it helps improve the accuracy of quantum
results by running circuits at varying noise levels and extrapolating back to a hypothetical
zero-noise case. While this tutorial won't delve into the theory behind ZNE in detail (for which we
recommend reading the `Mitiq docs <https://mitiq.readthedocs.io/en/stable/guide/zne-5-theory.html>`_
and the references, including Mitiq's whitepaper [#mitiq-2022]_), let's first review what happens when using the protocol in practice.

Stage 1: Generating noise-scaled circuits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ZNE works by generating circuits with **increased** noise. Catalyst implements the unitary folding
framework introduced in [#dzne-2020]_ for generating noise-scaled circuits. In particular, 
the following two methods are available:

1. **Global folding**: If a circuit implements a global unitary :math:`U`, global folding applies
   :math:`U(U^\dagger U)^n` for some integer :math:`n`,
   effectively scaling the noise in the entire circuit.
2. **Local folding**: Individual gates are repeated (or folded) in contrast with the entire
   circuit.

Stage 2: Running the circuits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once noise-scaled circuits are created, they need to be run! These can be executed on either real
quantum hardware or a noisy quantum simulator. In this tutorial, we'll use the
`Qrack quantum simulator <https://qrack.readthedocs.io/>`_, which is both compatible with Catalyst,
and implements a noise model. For more about the integration of Qrack and Catalyst, see
the demo :doc:`QJIT compilation with Qrack and Catalyst <demos/qrack>`.

Stage 3: Combining the results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After executing the noise-scaled circuits, an extrapolation on the results is performed  
to estimate the zero-noise limit---the result we would expect in a noise-free scenario. 
Catalyst provides **polynomial** and **exponential** extrapolation methods.

These three stages illustrate what happens behind the scenes when using a ZNE routine. 
However, from the user's perspective, one only needs to define the initial circuit, 
the noise scaling method, and the extrapolation method. The rest is taken care of by Catalyst.
"""

##############################################################################
# .. note ::
#
#    To follow along with this demonstration, it is required to
#    `install Catalyst <https://docs.pennylane.ai/projects/catalyst/en/stable/dev/installation.html>`__,
#    as well as the `PennyLane-Qrack plugin <https://pennylane-qrack.readthedocs.io/en/latest/installation.html>`__.
#
#    .. code-block:: bash
#
#      pip install -U pennylane-catalyst pennylane-qrack 
# 

##############################################################################
# Defining the mirror circuit
# ---------------------------
# The first step for demoing an error mitigation routine is to define a circuit. 
# Here we build a simple mirror circuit starting off a `unitary 2-design <https://en.wikipedia.org/wiki/Quantum_t-design>`__. 
# This is a typical construction for a randomized benchmarking circuit, which is used in many tasks
# in quantum computing. Given such circuit, we measure the expectation value :math:`\langle Z\rangle` 
# on the state of the first qubit, and by construction of the circuit, we expect this value to be
# equal to 1.

import numpy as np
import pennylane as qml
qml.capture.enable()
from catalyst import mitigate_with_zne

import warnings
from pennylane.exceptions import PennyLaneDeprecationWarning
warnings.filterwarnings("ignore", category=PennyLaneDeprecationWarning)

n_wires = 3

np.random.seed(42)

n_layers = 5
template = qml.SimplifiedTwoDesign
weights_shape = template.shape(n_layers, n_wires)
w1, w2 = [2 * np.pi * np.random.random(s) for s in weights_shape]

def circuit(w1, w2):
    template(w1, w2, wires=range(n_wires))
    qml.adjoint(template)(w1, w2, wires=range(n_wires))
    return qml.expval(qml.PauliZ(0))

##############################################################################
# As a sanity check, we first execute the circuit on the Qrack simulator without any noise.

noiseless_device = qml.device("qrack.simulator", n_wires, noise=0)

ideal_value = qml.QNode(circuit, device=noiseless_device)(w1, w2)
print(f"Ideal value: {ideal_value}")

##############################################################################
# In the noiseless scenario, the expectation value of the Pauli-Z measurement
# is equal to 1, since the first qubit is back in the :math:`|0\rangle` state.
#
# Mitigating the noisy circuit
# ----------------------------
# Let's now run the circuit through a noisy scenario. The Qrack simulator models noise by
# applying single-qubit depolarizing noise channels to all qubits in all gates of the circuit.
# The probability of error is specified by the value of the ``noise`` constructor argument.

NOISE_LEVEL = 0.01
noisy_device = qml.device("qrack.simulator", n_wires, shots=1000, noise=NOISE_LEVEL)

noisy_qnode = qml.QNode(circuit, device=noisy_device, mcm_method="one-shot")
noisy_value = noisy_qnode(w1, w2)
print(f"Error without mitigation: {abs(ideal_value - noisy_value):.3f}")

##############################################################################
# Again expected, we obtain a noisy value that diverges from the ideal value we obtained above.
# Fortunately, we have error mitigation to the rescue! We can apply ZNE, however we are still
# missing some necessary parameters. In particular we still need to specify:
#
# 1. The method for scaling this noise up (in Catalyst there are two options: ``global`` and
#    ``local``).
# 2. The noise scaling factors (i.e. how much to increase the depth of the circuit).
# 3. The extrapolation technique used to estimate the ideal value (available in Catalyst are
#    polynomial and exponential extrapolation).
#
# First, we choose a method to scale the noise. This needs to be specified as a Python string.

folding_method = "global"

##############################################################################
# Next, we pick a list of scale factors. At the time of writing this tutorial,
# Catalyst supports only odd integer scale factors. In the global folding setting,
# a scale factor  :math:`s` correspond to the circuit being folded
# :math:`\frac{s - 1}{2}` times.
scale_factors = [1, 3, 5]

##############################################################################
# Finally, we'll choose the extrapolation technique. Both exponential and polynomial extrapolation
# is available in the :mod:`qml.noise <pennylane.noise>` module, and both of these functions can be passed directly
# into Catalyst's :func:`catalyst.mitigate_with_zne` function. In this tutorial we use polynomial extrapolation,
# which we hypothesize best models the behavior of the noise scenario we are considering.

from pennylane.noise import poly_extrapolate
from functools import partial

extrapolation_method = partial(poly_extrapolate, order=2)

##############################################################################
# We're now ready to run our example using ZNE with Catalyst! Putting these all together we're able
# to define a very simple :func:`~.QNode`, which represents the mitigated version of the original circuit.


@qml.qjit
def mitigated_circuit_qjit(w1, w2):
    return mitigate_with_zne(
        noisy_qnode,
        scale_factors=scale_factors,
        extrapolate=extrapolation_method,
        folding=folding_method,
    )(w1, w2)


zne_value = mitigated_circuit_qjit(w1, w2)

print(f"Error with ZNE in Catalyst: {abs(ideal_value - zne_value):.3f}")

##############################################################################
# It's crucial to note that we can use the :func:`~.qjit` decorator here, as all the functions used
# to define the node are compatible with Catalyst, and we can therefore
# exploit the potential of just-in-time compilation.
#
# Benchmarking
# ------------
# For comparison, let's define a very similar :func:`~.qnode`, but this time we don't decorate the node
# as just-in-time compilable.
# When it comes to the parameters, the only difference here (due to an implementation technicality)
# is the type of the ``folding`` argument. Despite the type being different, however,
# the value of the folding method is the same, i.e., global folding.


def mitigated_circuit(w1, w2):
    return qml.noise.mitigate_with_zne(
        noisy_qnode,
        scale_factors=scale_factors,
        extrapolate=extrapolation_method,
        folding=qml.noise.fold_global,
    )(w1, w2)


zne_value = mitigated_circuit(w1, w2)

print(f"Error with ZNE in PennyLane: {abs(ideal_value - zne_value):.3f}")

##############################################################################
# To showcase the impact of JIT compilation, we use Python's ``timeit`` module
# to measure execution time of ``mitigated_circuit_qjit`` vs. ``mitigated_circuit``.
# 
# Note: for the purpose of this last example, we reduce the number of shots of the simulator to 100,
# since we don't need the accuracy required for the previous demonstration. We do so in order to 
# reduce the running time of this tutorial, while still showcasing the performance differences.  
import timeit

noisy_device = qml.device("qrack.simulator", n_wires, shots=100, noise=NOISE_LEVEL)
noisy_qnode = qml.QNode(circuit, device=noisy_device, mcm_method="one-shot")

@qml.qjit
def mitigated_circuit_qjit(w1, w2):
    return mitigate_with_zne(
        noisy_qnode,
        scale_factors=scale_factors,
        extrapolate=extrapolation_method,
        folding=folding_method,
    )(w1, w2)

repeat = 5  # number of timing runs
number = 5  # number of loops executed in each timing run

times = timeit.repeat("mitigated_circuit(w1, w2)", globals=globals(), number=number, repeat=repeat)

print(f"mitigated_circuit running time (best of {repeat}): {min(times):.3f}s")

times = timeit.repeat(
    "mitigated_circuit_qjit(w1, w2)", globals=globals(), number=number, repeat=repeat
)

print(f"mitigated_circuit_qjit running time (best of {repeat}): {min(times):.3f}s")

##############################################################################
# Already with the simple circuit we started with, and with the simple parameters in our example,
# we can appreciate the performance differences. That was at the cost of very minimal syntax change.
#
# There are still reasons to use ZNE in PennyLane without :func:`~.qjit`, for instance,
# whenever the device of choice is not supported by Catalyst. To help,
# we conclude with a landscape of the QEM techniques available in the PennyLane ecosystem.
#
# .. list-table::
#     :widths: 30 20 20 20 20 30
#     :align: center
#     :header-rows: 0
#
#     * - **Framework**
#       - **ZNE folding**
#       - **ZNE extrapolation**
#       - **Differentiable**
#       - **JIT**
#       - **Other QEM techniques**
#     * - PennyLane + Mitiq
#       - global, local, random
#       - polynomial, exponential
#       - –
#       - –
#       - ✅
#     * - PennyLane transforms
#       - global, local
#       - polynomial, exponential
#       - ✅
#       - –
#       - –
#     * - Catalyst (experimental)
#       - global, local
#       - polynomial, exponential
#       - ✅
#       - ✅
#       - –


##############################################################################
#
# References
# ----------
#
# .. [#zne-2017] K. Temme, S. Bravyi, J. M. Gambetta
#     `"Error Mitigation for Short-Depth Quantum Circuits" <https://arxiv.org/abs/1612.02058>`_,
#     Phys. Rev. Lett. 119, 180509 (2017).
#
# .. [#dzne-2020] Tudor Giurgica-Tiron, Yousef Hindy, Ryan LaRose, Andrea Mari, and William J. Zeng, 
#     `"Digital zero noise extrapolation for quantum error mitigation" <https://arxiv.org/abs/2005.10921v2>`__, 
#     IEEE International Conference on Quantum Computing and Engineering (2020).
#
# .. [#mitiq-2022]
#     Ryan LaRose and Andrea Mari and Sarah Kaiser and Peter J. Karalekas and Andre A. Alves and 
#     Piotr Czarnik and Mohamed El Mandouh and Max H. Gordon and Yousef Hindy and Aaron Robertson 
#     and Purva Thakre and Misty Wahl and Danny Samuel and Rahul Mistri and Maxime Tremblay 
#     and Nick Gardner and Nathaniel T. Stemen and Nathan Shammah and William J. Zeng, 
#     `"Mitiq: A software package for error mitigation on noisy quantum computers" <https://doi.org/10.22331/q-2022-08-11-774>`__, 
#     Quantum (2022).
