r"""How to use noise models in PennyLane
------------------------------------
"""

######################################################################
# Noise models are essential for understanding and describing the effects of physical errors
# in a quantum computation. They allow for simulating the imperfections in state evolution
# arising from environment-based `errors <https://pennylane.ai/codebook/07-error-correction/>`__,
# state preparation routines, measurements, and more.
#
# Here, we show how to use the features provided in PennyLane's :mod:`~.pennylane.noise`
# module to construct and manipulate noise models, enabling noisy simulation (see the `noise module documentation <https://docs.pennylane.ai/en/stable/code/qml_noise.html>`__ for more details). In PennyLane,
# noise models are constructed from two main components:
#
# 1. Boolean conditions (referred to as *conditionals*) that dictate whether noise is inserted into the circuit.
# 2. Callables (called noise *functions*) that apply noise operations when a corresponding condition is satisfied.
#
# The following figure is an example that shows how a noise model transforms a sample circuit by inserting
# amplitude and phase damping errors for :class:`~.pennylane.RX`
# and :class:`~.pennylane.RY` gates, respectively.
#

######################################################################
# .. figure:: ../_static/demonstration_assets/noise_models/noise_model_long.jpg
#    :align: center
#    :width: 85%
#
#    ..
#

######################################################################
# In the upcoming sections, we will first cover the underlying components of
# noise models and learn how to use them to construct desired noise models.
# Finally, we will use our noise model to perform noisy simulations.
#

######################################################################
# Conditionals
# ~~~~~~~~~~~~
#
# We implement conditions as Boolean functions that accept an operation and evaluate it
# to return a Boolean output. In PennyLane, such objects are referred to as **conditionals**.
# They are constructed as instances of :class:`~.pennylane.BooleanFn` and can be combined
# using standard bitwise operations such as ``&``, ``|``, ``^``, or ``~``. PennyLane supports
# the following types of conditionals:
#
# 1. **Operation-based conditionals:** They evaluate whether a gate operation is a
#    specific type of operation or belongs to a specified set of operations. They are
#    built using :func:`~.pennylane.noise.op_eq` and :func:`~.pennylane.noise.op_in`, respectively.
# 2. **Wire-based conditionals:** They evaluate whether a gate operation's wires are
#    equal to or are contained in a specified set of wires. They are built using
#    :func:`~.pennylane.noise.wires_eq` and :func:`~.pennylane.noise.wires_in`, respectively.
# 3. **Arbitrary conditionals:** Custom conditionals can be defined as a function wrapped
#    with a :class:`~.pennylane.BooleanFn` decorator. The signature for such conditionals
#    must be ``cond_fn(operation: Operation) -> bool``.
#
# For example, here's how we would define a conditional that checks for :math:`R_X(\phi)`
# gate operations with :math:`|\phi| < 1.0` and wires :math:`\in \{0, 1\}`:
#

import pennylane as qml
qml.capture.enable()
import numpy as np

@qml.BooleanFn
def rx_cond(op):
    return isinstance(op, qml.RX) and np.abs(op.parameters[0]) < 1.0

# Combine this arbitrary conditional with a wire-based conditional
rx_and_wires_cond = rx_cond & qml.noise.wires_in([0, 1])
for op in [qml.RX(0.05, wires=[0]), qml.RX(2.34, wires=[1])]:
    print(f"Result for {op}: {rx_and_wires_cond(op)}")

######################################################################
# Noise functions
# ~~~~~~~~~~~~~~~
#
# Callables that apply noise operations are referred to as **noise functions** and have the
# signature ``fn(op, **metadata) -> None``. Their definition has no return statement and
# contains the error operations that are *inserted* when a gate operation in the circuit
# satisfies corresponding conditional. There are a few ways to construct noise functions:
#
# 1. **Single-instruction noise functions:** To add a single-operation noise, we can use
#    :func:`~pennylane.noise.partial_wires`. It performs a partial initialization of the
#    noise operation and queues it on the ``wires`` of the gate operation.
# 2. **User-defined noise functions:** For adding more sophisticated and custom noise,
#    we can define our own quantum function with the signature specified above.
#
# For example, one can use the following to insert a depolarization error and show
# the error that gets queued with an example gate operation:
#

depol_error = qml.noise.partial_wires(qml.DepolarizingChannel, 0.01)

op = qml.X('w1') # Example gate operation
print(f"Error for {op}: {depol_error(op)}")

######################################################################
# Creating a noise model
# ~~~~~~~~~~~~~~~~~~~~~~
#
# We can now create a PennyLane :class:`~.pennylane.NoiseModel` by stitching together
# multiple condition-callable pairs, where noise operations are inserted into the
# circuit when their corresponding condition is satisfied. For the first pair, we
# will use the previously constructed conditional and callable to insert a depolarization
# error after :class:`~.pennylane.RX` gates that satisfy :math:`|\phi| < 1.0` and that
# act on the wires :math:`\in \{0, 1\}`.
#

fcond1, noise1 = rx_and_wires_cond, depol_error

######################################################################
# Next, we construct a pair to mimic thermal relaxation errors that are encountered
# during the state preparation via :class:`~.pennylane.ThermalRelaxationError`:

fcond2 = qml.noise.op_eq(qml.StatePrep)

def noise2(op, **kwargs):
    for wire in op.wires:
        qml.ThermalRelaxationError(0.1, kwargs["t1"], kwargs["t2"], kwargs["tg"], wire)

######################################################################
# By default, noise operations specified by a noise function will be inserted *after* the
# gate operation that satisfies the conditional. However, they can be inserted in a custom
# order by manually queuing the evaluated gate operation via :func:`~pennylane.apply`
# within the function definition. For example, we can add a sandwiching constant-valued
# rotation error for :class:`~.pennylane.Hadamard` gates on the wires :math:`\in \{0, 1\}`:
#

fcond3 = qml.noise.op_eq("Hadamard") & qml.noise.wires_in([0, 1])

def noise3(op, **kwargs):
    qml.RX(np.pi / 16, op.wires)
    qml.apply(op)
    qml.RY(np.pi / 8, op.wires)

######################################################################
# Finally, we can build the noise model with some required ``metadata`` for ``noise2``:
#

metadata = dict(t1=0.02, t2=0.03, tg=0.001)  # times unit: sec
noise_model = qml.NoiseModel(
    {fcond1: noise1, fcond2: noise2, fcond3: noise3}, **metadata
)
print(noise_model)

######################################################################
# Adding noise models to your workflow
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Now that we have built our noise model, we can learn how to use it.
# A noise model can be applied to a circuit or device via the
# :func:`~pennylane.add_noise` transform. For example, consider
# the following circuit that performs the evolution and de-evolution
# of a given initial state based on some parameters:
#

from matplotlib import pyplot as plt

qml.drawer.use_style("pennylane")
dev = qml.device("default.mixed", wires=3)
init_state = np.random.RandomState(42).rand(2 ** len(dev.wires))
init_state /= np.linalg.norm(init_state)

def circuit(theta, phi):
    # State preparation
    qml.StatePrep(init_state, wires=[0, 1, 2])

    # Evolve state
    qml.Hadamard(0)
    qml.RX(theta, 1)
    qml.RX(phi, 2)
    qml.CNOT([1, 2])
    qml.CNOT([0, 1])

    # De-evolve state
    qml.CNOT([0, 1])
    qml.CNOT([1, 2])
    qml.RX(-phi, 2)
    qml.RX(-theta, 1)
    qml.Hadamard(0)
    return qml.state()

theta, phi = 0.21, 0.43
ideal_circuit = qml.QNode(circuit, dev)
qml.draw_mpl(ideal_circuit)(theta, phi)
plt.show()

######################################################################
# To attach the ``noise_model`` to this quantum circuit, we use the
# :func:`~.pennylane.add_noise` transform:
#

noisy_circuit = qml.add_noise(ideal_circuit, noise_model)
qml.draw_mpl(noisy_circuit)(theta, phi)
plt.show()

######################################################################
# We can then use the ``noisy_circuit`` to run noisy simulations as shown below:
#

init_dm = np.outer(init_state, init_state) # density matrix for the init_state
ideal_res = np.round(qml.math.fidelity(ideal_circuit(theta, phi), init_dm), 8)
noisy_res = np.round(qml.math.fidelity(noisy_circuit(theta, phi), init_dm), 8)

print(f"Ideal v/s Noisy: {ideal_res} and {noisy_res}")

######################################################################
# The fidelity for the state obtained from the ideal circuit is :math:`\approx 1.0`,
# which is expected since our circuit effectively does nothing to the initial state.
# We see that this is not the case for the result obtained from the noisy simulation,
# due to the error operations inserted in the circuit.
#


######################################################################
# Conclusion
# ~~~~~~~~~~
#
# Noise models provide a succinct way to describe the impact of the environment on
# quantum computation. In PennyLane, we define such models as mapping between conditionals
# that select the target operation and their corresponding noise operations. These
# can be constructed with utmost flexibility as we showed here.
#
# Should you have any questions about using noise models in PennyLane, you can consult the
# `noise module documentation <https://docs.pennylane.ai/en/stable/code/qml_noise.html>`__,
# the PennyLane Codebook module on
# `Noisy Quantum Theory <https://pennylane.ai/codebook/#06-noisy-quantum-theory>`__
# on noise, or create a post on the `PennyLane Discussion Forum <https://discuss.pennylane.ai>`__.
# You can also follow us on `X (formerly Twitter) <https://twitter.com/PennyLaneAI>`__
# or `LinkedIn <https://www.linkedin.com/company/pennylaneai/>`__ to stay up-to-date with
# the latest and greatest from PennyLane!
#
