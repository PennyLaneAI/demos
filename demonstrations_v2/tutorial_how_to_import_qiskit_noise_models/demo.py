r"""How to import noise models from Qiskit
==========================================

Noise models describe how a quantum system interacts with its environment.
These models are typically represented by a set of
`Kraus operators <https://pennylane.ai/qml/demos/tutorial_noisy_circuits/#noisy-operations>`_
that encapsulates the probabilistic nature of quantum errors. ⚡
Interestingly, different sets of Kraus operators can represent the same quantum noise process.
The non-unique nature of these representations allows quantum computing libraries to use
different approaches for storing and building Kraus operators to construct noise models. 
In this how-to guide, we will first compare the construction of noise models in
`Qiskit <https://docs.quantum.ibm.com/>`_ and
`PennyLane <https://docs.pennylane.ai/en/stable/code/qml.html>`_. Then, we will learn how to
convert a Qiskit noise model into an equivalent PennyLane one, allowing users to import any
custom user-defined or fake backend-based noise models.
"""

######################################################################
# Noise models in Qiskit and PennyLane
# ------------------------------------
#
# The noise models in Qiskit are built using the tools available in the
# `noise module <https://qiskit.github.io/qiskit-aer/apidocs/aer_noise.html>`_
# of the ``Qiskit-Aer`` package. Each model is represented by a `NoiseModel
# <https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.noise.NoiseModel.html>`_
# object that contains `QuantumError
# <https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.noise.QuantumError.html>`_
# to describe the errors encountered in gate operations. Optionally, it may also have a
# `ReadoutError <https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.noise.ReadoutError.html>`_
# that describes the classical readout errors.
#
# Let's build a Qiskit noise model that inserts *depolarization* errors for single-qubit gates,
# *bit-flip* errors for the target qubit of the two-qubit gates,
# and *amplitude damping* errors for each measurement:
#

import numpy as np
from qiskit_aer.noise import (
    amplitude_damping_error, depolarizing_error, pauli_error, NoiseModel
)

# Building the Qiskit noise model
model_qk = NoiseModel()

# Depolarization error for single-qubit gates
prob_depol = 0.2
error_gate1 = depolarizing_error(prob_depol, 1)
model_qk.add_all_qubit_quantum_error(error_gate1, ["u1", "u2", "u3"])

# Bit flip errors for two-qubit gate
prob_bit_flip = 0.1
error_gate2 = pauli_error([('X', prob_bit_flip), ('I', 1 - prob_bit_flip)]).tensor(
    pauli_error([('I', 1)])
)
model_qk.add_all_qubit_quantum_error(error_gate2, ["cx"])

# Amplitude damping error for measurements
n_qubits = 3
exc_population = 0.2
prob_ampl_damp = np.random.default_rng(42).uniform(0, 0.2, n_qubits)
for qubit in range(n_qubits):
    error_meas = amplitude_damping_error(prob_ampl_damp[qubit], exc_population)
    model_qk.add_quantum_error(error_meas, "measure", [qubit])

print(model_qk)

######################################################################
# In contrast, the noise models in PennyLane are :class:`~.pennylane.NoiseModel`
# objects with Boolean conditions that select the operation for which
# we want to apply noise. These conditions are mapped to noise functions
# that apply (or queue) the corresponding noise for the selected operation
# or measurement process based on user-provided metadata. This allows
# for a more functional construction, as we can see by recreating the
# above noise model as shown below. For more information on this, check out our
# :doc:`how-to for noise models in PennyLane <demos/tutorial_how_to_use_noise_models>`. 🧑‍🏫
#

import pennylane as qml
qml.capture.enable()

# Depolarization error for single-qubit gates
gate1_fcond = qml.noise.op_in(["U1", "U2", "U3"]) & qml.noise.wires_in(range(n_qubits))
gate1_noise = qml.noise.partial_wires(qml.DepolarizingChannel, prob_depol)

# Bit flip errors for two-qubit gate
gate2_fcond = qml.noise.op_eq("CNOT")
def gate2_noise(op, **metadata):
    qml.BitFlip(prob_bit_flip, op.wires[1])

# Readout errors for measurements
rmeas_fcond = qml.noise.meas_eq(qml.counts)
def rmeas_noise(op, **metadata):
    for wire in op.wires:
        qml.GeneralizedAmplitudeDamping(prob_ampl_damp[wire], exc_population, wire)

# Building the PennyLane noise model
model_pl = qml.NoiseModel(
    {gate1_fcond: gate1_noise, gate2_fcond: gate2_noise}, {rmeas_fcond: rmeas_noise},
)

print(model_pl)

######################################################################
# It is important to verify whether these noise models work the intended way.
# For this purpose, we will use them while simulating a
# `GHZ state <https://en.wikipedia.org/wiki/Greenberger–Horne–Zeilinger_state>`_ using the
# `default.mixed <https://docs.pennylane.ai/en/stable/code/api/pennylane.devices.default_mixed.html>`_
# and `qiskit.aer <https://docs.pennylane.ai/projects/qiskit/en/latest/devices/aer.html>`_
# devices. Note that we require :func:`~.pennylane.add_noise` transform for
# adding the PennyLane noise model but the Qiskit noise model is provided in
# the device definition itself:
#

# Preparing the devices
n_shots = int(2e6)
dev_pl_ideal = qml.device("default.mixed", wires=n_qubits)
dev_qk_noisy = qml.device("qiskit.aer", wires=n_qubits, noise_model=model_qk)

def GHZcircuit():
    qml.U2(0, np.pi, wires=[0])
    for wire in range(n_qubits-1):
        qml.CNOT([wire, wire + 1])
    return qml.counts(wires=range(n_qubits), all_outcomes=True)

# Preparing the circuits
pl_ideal_circ = qml.set_shots(qml.QNode(GHZcircuit, dev_pl_ideal), shots = n_shots)
pl_noisy_circ = qml.add_noise(pl_ideal_circ, noise_model=model_pl)
qk_noisy_circ = qml.set_shots(qml.QNode(GHZcircuit, dev_qk_noisy), shots = n_shots)

# Preparing the results
pl_noisy_res, qk_noisy_res = pl_noisy_circ(), qk_noisy_circ()

######################################################################
# Now let's look at the results to compare the two noise models:
#

pl_probs = np.array(list(pl_noisy_res.values())) / n_shots
qk_probs = np.array(list(qk_noisy_res.values())) / n_shots

print("PennyLane Results: ", np.round(pl_probs, 3))
print("Qiskit Results:    ", np.round(qk_probs, 3))
print("Are results equal? ", np.allclose(pl_probs, qk_probs, atol=1e-2))

######################################################################
# As the results are equal within a targeted tolerance,
# we can confirm that the two noise models are equivalent.
# Note that this tolerance can be further suppressed by
# increasing the number of shots (``n_shots``) in the simulation.
#

######################################################################
# Importing Qiskit noise models
# -----------------------------
#
# PennyLane provides the :func:`~.pennylane.from_qiskit_noise` function to
# easily convert a Qiskit noise model into an equivalent PennyLane noise model.
# Let's look at an example of a noise model based on the `GenericBackendV2
# <https://docs.quantum.ibm.com/api/qiskit/qiskit.providers.fake_provider.GenericBackendV2>`_
# backend that gets instantiated with the error data generated and sampled randomly from
# historical IBM backend data.
#

from qiskit.providers.fake_provider import GenericBackendV2

backend = GenericBackendV2(num_qubits=2, seed=42)
qk_noise_model = NoiseModel.from_backend(backend)
print(qk_noise_model)

######################################################################
# To import this noise model as a PennyLane one, we simply do:
#

pl_noise_model = qml.from_qiskit_noise(qk_noise_model)
print(pl_noise_model)

######################################################################
# This conversion leverages the standard Kraus representation of the errors
# stored in the ``qk_noise_model``. Internally, this is done in a smart three-step process:
#
# 1. First, all the basis gates from the noise model are mapped to the corresponding PennyLane
#    gate `operations <https://docs.pennylane.ai/en/stable/introduction/operations.html>`_.
# 2. Next, the operations with noise are mapped to the corresponding error channels
#    defined via :class:`~.pennylane.QubitChannel`.
# 3. Finally, the `Boolean conditionals <https://docs.pennylane.ai/en/stable/code/qml_noise.html#boolean-functions>`_
#    are constructed and combined based on their associated errors.
#
# This can be done for any noise model defined in Qiskit with a minor catch that
# the classical readout errors are not supported yet in PennyLane.
# However, we can easily re-insert quantum readout errors into our converted noise model.
# Here's an example that adds ``rmeas_fcond`` and ``rmeas_noise`` (defined earlier) to
# ``pl_noise_model``:
#

pl_noise_model += {"meas_map": {rmeas_fcond: rmeas_noise}}
print(pl_noise_model.meas_map)

######################################################################
# Conclusion
# ----------
#

######################################################################
# Qiskit provides noise models and tools that could be used to mirror the behaviour of quantum
# devices. Integrating them into PennyLane is a powerful way to enable users to perform
# differentiable noisy simulations that help them study the effects of noise on quantum circuits
# and develop noise-robust quantum algorithms. In this how-to guide, we learned how to construct
# PennyLane noise models from Qiskit ones by manually building one-to-one mappings
# for each kind of error and also by using the :func:`~.pennylane.from_qiskit_noise` function
# to convert the Qiskit noise model automatically. 💪
#
# Should you have any questions about using noise models in PennyLane, you can consult the
# `noise module documentation <https://docs.pennylane.ai/en/stable/code/qml_noise.html>`_,
# the `PennyLane Codebook module on Noisy Quantum Theory
# <https://pennylane.ai/codebook/#06-noisy-quantum-theory>`_,
# or create a post on the `PennyLane Discussion Forum <https://discuss.pennylane.ai>`_.
#
