r"""
Quantum advantage in learning from experiments
==============================================

.. meta::
    :property="og:description": Learn how quantum memory can boost quantum machine learning algorithms
    :property="og:image": https://pennylane.ai/qml/_static/demonstration_assets/learning_from_exp_thumbnail.png

This demo is based on the article `Quantum advantage in learning from
experiments <https://arxiv.org/abs/2112.00778>`__ `[1] <#ref1>`__ by
Hsin-Yuan Huang and co-authors. The article investigates the following
question:

*How useful is access to quantum memory for quantum machine learning?*

They show that access to quantum memory can make a big difference, and
prove that there exist learning problems for which algorithms with
quantum memory require *exponentially less resources* than those
without. We look at one learning task studied in `[1] <#ref1>`__ for
which this is the case.

The learning task
-----------------

The learning task we focus on involves deciding if a unitary is
time-reversal symmetric (we’ll call them T-symmetric) or not.
Mathematically, time-reversal symmetry in quantum mechanics involves
reversing the sense of :math:`i` so that :math:`i \rightarrow -i.`
Hence, a unitary :math:`U` is T-symmetric if

.. math:: U^*=U.

Now for the learning task. Let’s say we have a bunch of quantum circuits
:math:`U_1, \cdots, U_n,` some of which are T-symmetric and some not,
but we are not told which ones are which.

"""


##############################################################################
# .. figure:: ../_static/demonstration_assets/learning_from_experiments/fig1b.png
#    :align: center
#    :width: 50%


######################################################################
# The task is to design an algorithm to determine which of the
# :math:`U`\ ’s are T-symmetric and which are not, given query access to
# the unitaries. Note that we do not have any labels here, so this is an
# unsupervised learning task. To make things concrete, let’s consider
# unitaries acting on 8 qubits. We will also limit the number of times we
# can use each unitary:
#

qubits = 8  # the number of qubits on which the unitaries act
n_shots = 100  # the number of times we can use each unitary


######################################################################
# Experiments with and without a quantum memory
# ---------------------------------------------
#


######################################################################
# To tackle this task we consider experiments with and without quantum
# memory. We also assume that we have access to a single physical
# realization of each unitary; in other words, we do not have multiple
# copies of the devices that implement :math:`U_i.`
#
# An experiment without quantum memory can therefore only make use of a
# single query to :math:`U_i` in each circuit, since querying :math:`U_i`
# again would require storing the state of the first query in memory and
# re-using the unitary. In the paper these experiments are called
# **conventional experiments**.
#
# Experiments with quantum memory do not have the limitations of
# conventional experiments. This means that multiple queries can be made
# to :math:`U_i` in a single circuit, which can be realized in practice by
# using a quantum memory. These experiments are called **quantum-enhanced
# experiments**.
#
# Note that we are not comparing classical and quantum algorithms here,
# but rather two classes of quantum algorithms.
#

##############################################################################
# .. figure:: ../_static/demonstration_assets/learning_from_experiments/experiments.png
#    :align: center
#    :width: 60%


######################################################################
# The conventional way
# --------------------
#


######################################################################
# First, we will try to solve the task with a conventional experiment. Our
# strategy will be as follows:
#
# -  For each :math:`U_i,` we prepare ``n_shots`` copies of the state
#    :math:`U_i\vert0\rangle` and measure each state to generate
#    classical measurement data.
# -  Use an unsupervised classical machine learning algorithm (kernel
#    PCA), to try and separate the data into two clusters corresponding to
#    T-symmetric unitaries vs. the rest.
#
# If we succeed in clustering the data then we have successfully managed
# to discriminate the two classes!
#


##############################################################################
# .. figure:: ../_static/demonstration_assets/learning_from_experiments/fig2b.png
#    :align: center
#    :width: 70%


######################################################################
# To generate the measurement data, we will measure the states
# :math:`U_i\vert0\rangle` in the :math:`y` basis. The local expectation
# values take the form
#
# .. math:: E_i  = \langle 0\vert U^{\dagger}\sigma_y^{(i)} U \vert 0 \rangle.
#
# where :math:`\sigma_y^{(i)}` acts on the :math:`i^{\text{th}}` qubit.
#
# Using the fact that :math:`\sigma_y^*=-\sigma_y` and the property
# :math:`U^*=U` for T-symmetric unitaries, one finds
#
# .. math:: E_i^*=\langle 0\vert (U^{\dagger})^*(\sigma_y^{(i)})^* (U)^* \vert 0 \rangle = - \langle 0\vert U^{\dagger}\sigma_y^{(i)} U \vert 0 \rangle = - E_i.
#
# Since :math:`E_i` is a real number, the only solution to this is
# :math:`E_i=0,` which implies that all local expectations values are 0
# for this class.
#
# For general unitaries it is not the case that :math:`E_i=0,` and so it
# seems as though this will allow us to discriminate the two classes of
# circuits easily. However, for general random unitaries the local
# expectation values approach zero exponentially with the number of
# qubits: from finite measurement data it can still be very hard to see
# any difference! In fact, in the article `exponential separations between
# learning with and without quantum
# memory <https://arxiv.org/abs/2111.05881>`__ `[2] <#ref2>`__ it is
# proven that using conventional experiments, any successful algorithm
# *must* use the unitaries an exponential number of times.
#


######################################################################
# Let’s see how this looks in practice. First we define a function to
# generate random unitaries, making use of Pennylane’s
# `RandomLayers <https://pennylane.readthedocs.io/en/stable/code/api/pennylane.RandomLayers.html>`__
# template. For the time-symmetric case we will only allow for Y
# rotations, since these unitaries contain only real numbers, and
# therefore result in T-symmetric unitaries. For the other unitaries, we
# will allow rotations about X,Y, and Z.
#

import pennylane as qml
qml.capture.enable()
from pennylane.templates.layers import RandomLayers
import numpy as np

np.random.seed(234087)

layers, gates = 10, 10  # the number of layers and gates used in RandomLayers


def generate_circuit(shots):
    """
    generate a random circuit that returns a number of measuement samples
    given by shots
    """
    dev = qml.device("lightning.qubit", wires=qubits)

    @qml.set_shots(shots)
    @qml.qnode(dev)
    def circuit(ts=False):

        if ts == True:
            ops = [qml.RY]  # time-symmetric unitaries
        else:
            ops = [qml.RX, qml.RY, qml.RZ]  # general unitaries

        weights = np.random.rand(layers, gates) * np.pi
        RandomLayers(weights, wires=range(qubits), rotations=ops, seed=np.random.randint(0, 10000))

        return [qml.sample(op=qml.PauliY(q)) for q in range(qubits)]

    return circuit


######################################################################
# let’s check if that worked:
#

# the measurement outcomes for the first 3 shots
circuit = generate_circuit(n_shots)
print(np.array(circuit(ts=True))[:, 0:3])
print("\n")
print(np.array(circuit(ts=False))[:, 0:3])


######################################################################
# Now we can generate some data. The first 30 circuits in the data set are
# T-symmetric and the second 30 circuits are not. Since we are in an
# unsupervised setting, the algorithm will not know this information.
#

circuits = 30  # the number of circuits in each data set

raw_data = []

for ts in [True, False]:
    for __ in range(circuits):
        circuit = generate_circuit(n_shots)
        raw_data.append(circuit(ts=ts))


######################################################################
# Before feeding the data to a clustering algorithm, we will process it a
# little. For each circuit, we calculate the mean and the variance of each
# output bit and store this in a vector of size ``2*qubits``. These
# vectors make up our classical data set.
#


def process_data(raw_data):
    "convert raw data to vectors of means and variances of each qubit"

    raw_data = np.array(raw_data)
    nc = len(raw_data)  # the number of circuits used to generate the data
    nq = len(raw_data[0])  # the number of qubits in each circuit
    new_data = np.zeros([nc, 2 * nq])

    for k, outcomes in enumerate(raw_data):
        means = [np.mean(outcomes[q, :]) for q in range(nq)]
        variances = [np.var(outcomes[q, :]) for q in range(nq)]
        new_data[k] = np.array(means + variances)

    return new_data


data = process_data(raw_data)


######################################################################
# Now we use scikit-learn’s `kernel
# PCA <https://en.wikipedia.org/wiki/Kernel_principal_component_analysis>`__
# package to try and cluster the data. This performs principal component
# analysis in a high dimensional feature space defined by a kernel (below
# we use the radial basis function kernel).
#

from sklearn.decomposition import KernelPCA
from sklearn import preprocessing

kernel_pca = KernelPCA(
    n_components=None, kernel="rbf", gamma=None, fit_inverse_transform=True, alpha=0.1
)

# rescale the data so it has unit standard deviation and zero mean.
scaler = preprocessing.StandardScaler().fit(data)
data = scaler.transform(data)
# try to cluster the data
fit = kernel_pca.fit(data).transform(data)


######################################################################
# Let’s plot the result. Here we look at the first two principal
# components.
#

import matplotlib.pyplot as plt

# make a colour map for the points
c = np.array([0 for __ in range(circuits)] + [1 for __ in range(circuits)])

plt.scatter(fit[:, 0], fit[:, 1], c=c)
plt.show()


######################################################################
# Looks like the algorithm failed to cluster the data. We can try to get a
# separation by increasing the number of shots. Let’s increase the number
# of shots by 100 and see what happens.
#

n_shots = 10000  # 100 x more shots

raw_data = []

for ts in [True, False]:
    for __ in range(circuits):
        circuit = generate_circuit(n_shots)
        raw_data.append(circuit(ts=ts))

data = process_data(raw_data)
scaler = preprocessing.StandardScaler().fit(data)
data = scaler.transform(data)

fit = kernel_pca.fit(data).transform(data)

plt.scatter(fit[:, 0], fit[:, 1], c=c)
plt.show()

######################################################################
# Now we have a separation, however we required a lot of shots from the
# quantum circuit. As we increase the number of qubits, the number of
# shots we need will scale exponentially (as shown in `[2] <#ref2>`__),
# and so conventional strategies cannot learn to separate the data
# efficiently.
#


######################################################################
# The quantum-enhanced way
# ------------------------
#
# Now let’s see what difference having a quantum memory can make. Instead
# of using a single unitary to generate measurement data, we will make use
# of twice the number of qubits, and apply the unitary twice:
#


##############################################################################
# .. figure:: ../_static/demonstration_assets/learning_from_experiments/fig3b.png
#    :align: center
#    :width: 70%

######################################################################
# In practice, this could be done by storing the output state from the
# first unitary in quantum memory and preparing the same state by using
# the unitary again. Let’s define a function ``enhanced_circuit()`` to
# implement that. Note that since we now have twice as many qubits, we use
# half the number of shots as before so that the total number of uses of
# the unitary is unchanged.
#

n_shots = 50
qubits = 8

dev = qml.device("lightning.qubit", wires=qubits * 2)


def CNOT_sequence(control_wires, target_wires):
    """Apply CNOTs in sequence using the provided control and target wires"""
    for c_wire, t_wire in zip(control_wires, target_wires):
        qml.CNOT([c_wire, t_wire])


@qml.set_shots(n_shots)
@qml.qnode(dev)
def enhanced_circuit(ts=False):
    "implement the enhanced circuit, using a random unitary"

    if ts == True:
        ops = [qml.RY]
    else:
        ops = [qml.RX, qml.RY, qml.RZ]

    weights = np.random.rand(layers, n_shots) * np.pi
    seed = np.random.randint(0, 10000)

    for q in range(qubits):
        qml.Hadamard(wires=q)

    CNOT_sequence(control_wires=range(qubits), target_wires=range(qubits, 2 * qubits))
    RandomLayers(weights, wires=range(0, qubits), rotations=ops, seed=seed)
    RandomLayers(weights, wires=range(qubits, 2 * qubits), rotations=ops, seed=seed)
    CNOT_sequence(control_wires=range(qubits), target_wires=range(qubits, 2 * qubits))

    for q in range(qubits):
        qml.Hadamard(wires=q)

    return [qml.sample(op=qml.PauliZ(q)) for q in range(2 * qubits)]


######################################################################
# Now we generate some raw measurement data, and calculate the mean and
# variance of each qubit as before. Our data vectors are now twice as long
# since we have twice the number of qubits.
#

raw_data = []

for ts in [True, False]:
    for __ in range(circuits):
        raw_data.append(enhanced_circuit(ts))

data = process_data(raw_data)


######################################################################
# Let’s throw that into Kernel PCA again and plot the result.
#

kernel_pca = KernelPCA(
    n_components=None, kernel="rbf", gamma=None, fit_inverse_transform=True, alpha=0.1
)

scaler = preprocessing.StandardScaler().fit(data)
data = scaler.transform(data)

fit = kernel_pca.fit(data).transform(data)

c = np.array([0 for __ in range(circuits)] + [1 for __ in range(circuits)])
plt.scatter(fit[:, 0], fit[:, 1], c=c)
plt.show()


######################################################################
# Kernel PCA has perfectly separated the two classes! In fact, all the
# T-symmetric unitaries have been mapped to the same point. This is
# because the circuit is actually equivalent to performing
# :math:`U^TU\otimes \mathbb{I}\vert 0 \rangle,` which for T-symmetric
# unitaries is just the identity operation.
#
# To see this, note that the Hadamard and CNOT gates before
# :math:`U_i\otimes U_i` map the :math:`\vert0\rangle` state to the
# maximally entanged state
# :math:`\vert \Phi^+\rangle = \frac{1}{\sqrt{2}}(\vert 00...0\rangle+ \vert11...1\rangle,`
# and the gates after :math:`U_i\otimes U_i` are just the inverse
# transformation. The probability that all measurement outcomes give the
# result :math:`+1` is therefore.
#
# .. math:: p(11\cdots 1) = \langle \Phi^+ \vert U_i \otimes U_i \vert\Phi^+ \rangle.
#
# A well known fact about the maximally entanged state is that
# :math:`U\otimes \mathbb{I}\vert\Phi^+\rangle= \mathbb{I}\otimes U^T\vert\Phi^+\rangle.`
# The probabilty is therefore
#
# .. math:: p(11\cdots 1) = \langle \Phi^+ \vert U_i^T U_i \otimes \mathbb{I} \vert\Phi^+ \rangle.
#
# For T-symmetric unitaries :math:`U_i^T=U_i^\dagger,` so this probability
# is equal to one: the :math:`11\cdots 1` outcome is always obtained.
#
# If we look at the raw measurement data for the T-symmetric unitaries:
#

np.array(raw_data[0])[:, 0:5]  # outcomes of first 5 shots of the first T-symmetric circuit


######################################################################
# We see that indeed this is the only measurement outcome.
#
# To make things a bit more interesting, let’s add some noise to the
# circuit. We will define a function ``noise_layer(epsilon)`` that adds
# some random single qubit rotations, where the maximum rotation angle is
# ``epsilon``.
#


def noise_layer(epsilon):
    "apply a random rotation to each qubit"
    for q in range(2 * qubits):
        angles = (2 * np.random.rand(3) - 1) * epsilon
        qml.Rot(angles[0], angles[1], angles[2], wires=q)


######################################################################
# We redefine our ``enhanced_circuit()`` function with a noise layer
# applied after the unitaries
#


@qml.set_shots(n_shots)
@qml.qnode(dev)
def enhanced_circuit(ts=False):
    "implement the enhanced circuit, using a random unitary with a noise layer"

    if ts == True:
        ops = [qml.RY]
    else:
        ops = [qml.RX, qml.RY, qml.RZ]

    weights = np.random.rand(layers, n_shots) * np.pi
    seed = np.random.randint(0, 10000)

    for q in range(qubits):
        qml.Hadamard(wires=q)

    CNOT_sequence(control_wires=range(qubits), target_wires=range(qubits, 2 * qubits))
    RandomLayers(weights, wires=range(0, qubits), rotations=ops, seed=seed)
    RandomLayers(weights, wires=range(qubits, 2 * qubits), rotations=ops, seed=seed)
    noise_layer(np.pi / 4)  # added noise layer
    CNOT_sequence(control_wires=range(qubits, 2 * qubits), target_wires=range(qubits))

    for q in range(qubits):
        qml.Hadamard(wires=qubits + q)

    return [qml.sample(op=qml.PauliZ(q)) for q in range(2 * qubits)]


######################################################################
# Now we generate the data and feed it to kernel PCA again.
#

raw_data = []

for ts in [True, False]:
    for __ in range(circuits):
        raw_data.append(enhanced_circuit(ts))

data = process_data(raw_data)

kernel_pca = KernelPCA(
    n_components=None, kernel="rbf", gamma=None, fit_inverse_transform=True, alpha=0.1
)
scaler = preprocessing.StandardScaler().fit(data)
data = scaler.transform(data)
fit = kernel_pca.fit(data).transform(data)

c = np.array([0 for __ in range(circuits)] + [1 for __ in range(circuits)])
plt.scatter(fit[:, 0], fit[:, 1], c=c)
plt.show()


######################################################################
# Nice! Even in the presence of noise we still have a clean separation of
# the two classes. This shows that using entanglement can make a big
# difference to learning.
#


######################################################################
# References
# ----------
#
# [1] *Quantum advantage in learning from experiments*, Hsin-Yuan Huang
# et. al., `arxiv:2112.00778 <https://arxiv.org/pdf/2112.00778.pdf>`__
# (2021)
#
# [2] *Exponential separations between learning with and without quantum
# memory*, Sitan Chen, Jordan Cotler, Hsin-Yuan Huang, Jerry Li,
# `arxiv:2111.05881 <https://arxiv.org/abs/2111.05881>`__ (2021)
#
