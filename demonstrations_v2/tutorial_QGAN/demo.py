"""
.. _quantum_GAN:

Quantum generative adversarial networks with Cirq + TensorFlow
==============================================================

.. meta::
    :property="og:description": This demo constructs and trains a Quantum
        Generative Adversarial Network (QGAN) using PennyLane, Cirq, and TensorFlow.
    :property="og:image": https://pennylane.ai/qml/_static/demonstration_assets/qgan3.png


.. warning::
    This demo uses TensorFlow, which is a deprecated interface with PennyLane v0.42.
    Interfacing with TensorFlow will no longer be supported with PennyLane v0.43 and higher.
    Consider switching to a different machine learning interface with PennyLane, like
    :doc:`PyTorch <demos/tutorial_qnn_module_torch>` or
    :doc:`JAX <demos/tutorial_How_to_optimize_QML_model_using_JAX_and_Optax>`.

This demo constructs a Quantum Generative Adversarial Network (QGAN)
(`Lloyd and Weedbrook
(2018) <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.121.040502>`__,
`Dallaire-Demers and Killoran
(2018) <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.98.012324>`__)
using two subcircuits, a *generator* and a *discriminator*. The
generator attempts to generate synthetic quantum data to match a pattern
of "real" data, while the discriminator tries to discern real data from
fake data (see image below). The gradient of the discriminator’s output provides a
training signal for the generator to improve its fake generated data.

|

.. figure:: ../_static/demonstration_assets/QGAN/qgan.png
    :align: center
    :width: 75%
    :target: javascript:void(0)

|


"""

##############################################################################
# Using Cirq + TensorFlow
# -----------------------
# PennyLane allows us to mix and match quantum devices and classical machine
# learning software. For this demo, we will link together
# Google's `Cirq <https://cirq.readthedocs.io/en/stable/>`_ and `TensorFlow <https://www.tensorflow.org/>`_ libraries.
#
# We begin by importing PennyLane, NumPy, and TensorFlow.

import numpy as np
import pennylane as qml
qml.capture.enable()
import tensorflow as tf

##############################################################################
# We also declare a 3-qubit simulator device running in Cirq.

dev = qml.device('cirq.simulator', wires=3)


##############################################################################
# Generator and Discriminator
# ---------------------------
#
# In classical GANs, the starting point is to draw samples either from
# some "real data" distribution, or from the generator, and feed them to
# the discriminator. In this QGAN example, we will use a quantum circuit
# to generate the real data.
#
# For this simple example, our real data will be a qubit that has been
# rotated (from the starting state :math:`\left|0\right\rangle`) to some
# arbitrary, but fixed, state.

def real(angles, **kwargs):
    qml.Hadamard(wires=0)
    qml.Rot(*angles, wires=0)


##############################################################################
# For the generator and discriminator, we will choose the same basic
# circuit structure, but acting on different wires.
#
# Both the real data circuit and the generator will output on wire 0,
# which will be connected as an input to the discriminator. Wire 1 is
# provided as a workspace for the generator, while the discriminator’s
# output will be on wire 2.

def generator(w, **kwargs):
    qml.Hadamard(wires=0)
    qml.RX(w[0], wires=0)
    qml.RX(w[1], wires=1)
    qml.RY(w[2], wires=0)
    qml.RY(w[3], wires=1)
    qml.RZ(w[4], wires=0)
    qml.RZ(w[5], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RX(w[6], wires=0)
    qml.RY(w[7], wires=0)
    qml.RZ(w[8], wires=0)


def discriminator(w):
    qml.Hadamard(wires=0)
    qml.RX(w[0], wires=0)
    qml.RX(w[1], wires=2)
    qml.RY(w[2], wires=0)
    qml.RY(w[3], wires=2)
    qml.RZ(w[4], wires=0)
    qml.RZ(w[5], wires=2)
    qml.CNOT(wires=[0, 2])
    qml.RX(w[6], wires=2)
    qml.RY(w[7], wires=2)
    qml.RZ(w[8], wires=2)


##############################################################################
# We create two QNodes. One where the real data source is wired up to the
# discriminator, and one where the generator is connected to the
# discriminator.

@qml.qnode(dev)
def real_disc_circuit(phi, theta, omega, disc_weights):
    real([phi, theta, omega])
    discriminator(disc_weights)
    return qml.expval(qml.PauliZ(2))


@qml.qnode(dev)
def gen_disc_circuit(gen_weights, disc_weights):
    generator(gen_weights)
    discriminator(disc_weights)
    return qml.expval(qml.PauliZ(2))


##############################################################################
# QGAN cost functions
# -------------------
#
# There are two cost functions of interest, corresponding to the two
# stages of QGAN training. These cost functions are built from two pieces:
# the first piece is the probability that the discriminator correctly
# classifies real data as real. The second piece is the probability that the
# discriminator classifies fake data (i.e., a state prepared by the
# generator) as real.
#
# The discriminator is trained to maximize the probability of
# correctly classifying real data, while minimizing the probability of
# mistakenly classifying fake data.
#
# .. math:: 
# 
#     Cost_D = \mathrm{Pr}(real|\mathrm{fake}) - \mathrm{Pr}(real|\mathrm{real})
#
# The generator is trained to maximize the probability that the
# discriminator accepts fake data as real.
#
# .. math:: 
# 
#     Cost_G = - \mathrm{Pr}(real|\mathrm{fake})
#

def prob_real_true(disc_weights):
    true_disc_output = real_disc_circuit(phi, theta, omega, disc_weights)
    # convert to probability
    prob_real_true = (true_disc_output + 1) / 2
    return prob_real_true


def prob_fake_true(gen_weights, disc_weights):
    fake_disc_output = gen_disc_circuit(gen_weights, disc_weights)
    # convert to probability
    prob_fake_true = (fake_disc_output + 1) / 2
    return prob_fake_true


def disc_cost(disc_weights):
    cost = prob_fake_true(gen_weights, disc_weights) - prob_real_true(disc_weights)
    return cost


def gen_cost(gen_weights):
    return -prob_fake_true(gen_weights, disc_weights)


##############################################################################
# Training the QGAN
# -----------------
#
# We initialize the fixed angles of the "real data" circuit, as well as
# the initial parameters for both generator and discriminator. These are
# chosen so that the generator initially prepares a state on wire 0 that
# is very close to the :math:`\left| 1 \right\rangle` state.

phi = np.pi / 6
theta = np.pi / 2
omega = np.pi / 7
np.random.seed(0)
eps = 1e-2
init_gen_weights = np.array([np.pi] + [0] * 8) + \
                   np.random.normal(scale=eps, size=(9,))
init_disc_weights = np.random.normal(size=(9,))

gen_weights = tf.Variable(init_gen_weights)
disc_weights = tf.Variable(init_disc_weights)


##############################################################################
# We begin by creating the optimizer:

opt = tf.keras.optimizers.SGD(0.4)
opt.build([disc_weights, gen_weights])

##############################################################################
# In the first stage of training, we optimize the discriminator while
# keeping the generator parameters fixed.

cost = lambda: disc_cost(disc_weights)

for step in range(50):
    with tf.GradientTape() as tape:
        loss_value = cost()
    gradients = tape.gradient(loss_value, [disc_weights])
    opt.apply_gradients(zip(gradients, [disc_weights]))
    if step % 5 == 0:
        cost_val = loss_value.numpy()
        print("Step {}: cost = {}".format(step, cost_val))


##############################################################################
# At the discriminator’s optimum, the probability for the discriminator to
# correctly classify the real data should be close to one.

print("Prob(real classified as real): ", prob_real_true(disc_weights).numpy())


##############################################################################
# For comparison, we check how the discriminator classifies the
# generator’s (still unoptimized) fake data:

print("Prob(fake classified as real): ", prob_fake_true(gen_weights, disc_weights).numpy())


##############################################################################
# In the adversarial game we now have to train the generator to better
# fool the discriminator. For this demo, we only perform one stage of the
# game. For more complex models, we would continue training the models in an
# alternating fashion until we reach the optimum point of the two-player
# adversarial game.

cost = lambda: gen_cost(gen_weights)

for step in range(50):
    with tf.GradientTape() as tape:
        loss_value = cost()
    gradients = tape.gradient(loss_value, [gen_weights])
    opt.apply_gradients(zip(gradients, [gen_weights]))
    if step % 5 == 0:
        cost_val = loss_value.numpy()
        print("Step {}: cost = {}".format(step, cost_val))


##############################################################################
# At the optimum of the generator, the probability for the discriminator
# to be fooled should be close to 1.

print("Prob(fake classified as real): ", prob_fake_true(gen_weights, disc_weights).numpy())


##############################################################################
# At the joint optimum the discriminator cost will be close to zero,
# indicating that the discriminator assigns equal probability to both real and
# generated data.

print("Discriminator cost: ", disc_cost(disc_weights).numpy())

##############################################################################
# The generator has successfully learned how to simulate the real data
# enough to fool the discriminator.
#
# Let's conclude by comparing the states of the real data circuit and the generator. We expect
# the generator to have learned to be in a state that is very close to the one prepared in the
# real data circuit. An easy way to access the state of the first qubit is through its
# `Bloch sphere <https://en.wikipedia.org/wiki/Bloch_sphere>`__ representation:

obs = [qml.PauliX(0), qml.PauliY(0), qml.PauliZ(0)]

@qml.qnode(dev)
def bloch_vector_real(angles):
    real(angles)
    return [qml.expval(o) for o in obs]

@qml.qnode(dev)
def bloch_vector_generator(angles):
    generator(angles)
    return [qml.expval(o) for o in obs]

print(f"Real Bloch vector: {bloch_vector_real([phi, theta, omega])}")
print(f"Generator Bloch vector: {bloch_vector_generator(gen_weights)}")
